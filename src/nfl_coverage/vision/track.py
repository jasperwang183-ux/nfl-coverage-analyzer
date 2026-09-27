"""Player tracking across clip frames -> canonical 10 Hz tracks (rung 3 proper).

Two unlinked stills can say what the shell looked like at two instants; linked
tracks say who moved where, which is what the movement features, the run model
and the per-defender disguise evidence need. This module:

  1. runs YOLO with a multi-object tracker (ByteTrack via ultralytics) over the
     clip's native frames so identities persist, keeping one frame per `step_s`;
  2. carries one anchor calibration to every kept frame by chaining
     frame-to-frame homographies (small motions, so each transfer is reliable;
     keyframe calibrations can re-anchor the chain);
  3. assigns each track a side once, on the snap frame, from jersey color and
     formation geometry, and carries it through the play;
  4. assembles the canonical frame schema (`data/standardize.py`) at 10 Hz with
     speeds from finite differences, so the tracking-data models and the
     disguise components run on film unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nfl_coverage.config import FIELD_LENGTH, FIELD_WIDTH
from nfl_coverage.data.standardize import finalize
from nfl_coverage.vision.clip import transfer_calibration
from nfl_coverage.vision.homography import FieldCalibration, on_field
from nfl_coverage.vision.still import line_player_mask
from nfl_coverage.vision.teams import (
    TeamColors,
    cluster_teams,
    referee_mask,
    sides_from_clusters,
    torso_colors,
)

SIDELINE_Y_MARGIN = -0.5
# At the snap almost nobody lines up near a sideline: of 107K snap-frame positions in
# Big Data Bowl weeks 1-4 (2018 + 2023), 15 were within 4 yd of it (59 within 5).
# Bodies there are the line judge, the down judge or the chain crew. Calibration near
# a cropped sideline is extrapolated, so an official standing on it can land yards
# inside: on 15-34-46 the head linesman and line judge measured 2.6 and 3.8 yd in and
# were sided as the widest defenders. (On clip 1 a line judge 0.6 yd inside the far
# sideline took the corner's place and turned a corner bail into a 1.0 "rotation_up".)
SNAP_SIDELINE_CLEAR = 4.0


@dataclass
class TrackedFrame:
    index: int
    t: float
    image: np.ndarray
    boxes: np.ndarray  # (N, 4) x1 y1 x2 y2
    ids: np.ndarray  # (N,) track ids, -1 where the tracker gave none
    confs: np.ndarray


def track_players(
    video: Path,
    start_s: float,
    end_s: float,
    step_s: float = 0.1,
    yolo: str = "yolo11s.pt",
    conf: float = 0.3,
    imgsz: int = 1280,
    tracker: str = "bytetrack.yaml",
    progress: Callable[[float], None] | None = None,
) -> list[TrackedFrame]:
    """Track people through [start_s, end_s]; return one frame per `step_s` with ids.

    `progress` is called with the fraction of native frames processed (0-1).
    """
    from ultralytics import YOLO  # deferred: heavy import

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"could not open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(yolo)
    first = round(start_s * fps)
    last = round(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    kept: list[TrackedFrame] = []
    next_keep = start_s
    for fi in range(first, last + 1):
        ok, img = cap.read()
        if not ok:
            break
        t = fi / fps
        if progress is not None:
            progress((fi - first + 1) / max(1, last - first + 1))
        res = model.track(
            img, persist=True, tracker=tracker, classes=[0], conf=conf, imgsz=imgsz, verbose=False
        )[0]
        if t + 1e-6 < next_keep:
            continue
        next_keep += step_s
        b = res.boxes
        boxes = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        ids = (
            b.id.cpu().numpy().astype(int)
            if (len(b) and b.id is not None)
            else -np.ones(len(b), int)
        )
        confs = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        kept.append(TrackedFrame(len(kept), round(t, 3), img, boxes, ids, confs))
    cap.release()
    return kept


@dataclass
class ChainedCalibration:
    calibrations: list[FieldCalibration]
    notes: list[str] = field(default_factory=list)


def chain_calibrations(
    frames: list[TrackedFrame],
    anchor: FieldCalibration,
    anchor_index: int,
    keyframes: dict[int, FieldCalibration] | None = None,
) -> ChainedCalibration:
    """One calibration per frame, chained frame to frame from the anchor.

    Each step transfers the previous frame's calibration through the homography
    between consecutive frames (a fraction of a second of camera motion, so the
    match is dense and reliable). `keyframes` re-anchor the chain where a
    directly fitted calibration exists. A failed step reuses the previous
    calibration and is noted.
    """
    keyframes = dict(keyframes or {})
    keyframes[anchor_index] = anchor
    n = len(frames)
    calibs: list[FieldCalibration | None] = [None] * n
    notes: list[str] = []
    for i, c in keyframes.items():
        if 0 <= i < n:
            calibs[i] = c
    order = list(range(anchor_index + 1, n)) + list(range(anchor_index - 1, -1, -1))
    for i in order:
        if calibs[i] is not None:
            continue
        prev = i - 1 if i > anchor_index else i + 1
        try:
            calibs[i] = transfer_calibration(
                calibs[prev], frames[prev].image, frames[i].image
            ).calibration
        except ValueError as e:
            calibs[i] = calibs[prev]
            notes.append(f"frame {i}: transfer failed ({e}); reused frame {prev}")
    return ChainedCalibration([c for c in calibs], notes)


@dataclass
class SnapBodies:
    """Every tracked, unclipped, on-field body on the snap frame, with its jersey color."""

    ids: np.ndarray
    boxes: np.ndarray
    xy: np.ndarray  # field coordinates (raw)
    colors: np.ndarray
    labels: np.ndarray  # 2-means jersey-color cluster
    is_player: np.ndarray  # False: officials (striped), color outliers, sideline personnel
    n_offfield: int


def snap_bodies(frame: TrackedFrame, calib: FieldCalibration) -> SnapBodies:
    """The snap frame's bodies in field space with team colors: input to sides and formation."""
    keep = unclipped_mask_boxes(frame.boxes, frame.image.shape[0]) & (frame.ids >= 0)
    boxes = frame.boxes[keep]
    ids = frame.ids[keep]
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    xy = calib.to_field(feet.astype(np.float32))
    onf = on_field(xy, y_margin=SIDELINE_Y_MARGIN)
    boxes, ids, xy = boxes[onf], ids[onf], xy[onf]
    colors = torso_colors(frame.image, [tuple(b) for b in boxes])
    labels, is_player = cluster_teams(colors)
    is_player &= ~referee_mask(frame.image, [tuple(b) for b in boxes])
    is_player &= np.minimum(xy[:, 1], FIELD_WIDTH - xy[:, 1]) >= SNAP_SIDELINE_CLEAR
    return SnapBodies(ids, boxes, xy, colors, labels, is_player, int((~onf).sum()))


@dataclass
class SideAssignment:
    side_by_id: dict[int, str]
    line_ids: set[int]
    n_referee: int
    n_offfield: int
    teams: TeamColors | None = None  # snap-frame team colors, to side ids that appear later


def assign_sides(
    frame: TrackedFrame,
    calib: FieldCalibration,
    los: float,
    offense: str,
    bodies: SnapBodies | None = None,
) -> SideAssignment:
    """Side per track id from the snap frame: color clusters voted by formation geometry.

    Also names the line players (dropped downstream so the population matches
    the coverage-only training data) using the same LOS-band rule as stills.
    `bodies` reuses a snap-frame analysis already done (for the formation).
    """
    b = bodies if bodies is not None else snap_bodies(frame, calib)
    isp = b.is_player
    sides = sides_from_clusters(b.labels[isp], b.xy[isp, 0], los, offense, colors=b.colors[isp])
    sides = cap_sides(sides, b.xy[isp, 0], los, offense)
    # bodies ruled out at the snap stay out: "-" keeps a later frame from re-siding them
    side_by_id = {int(i): "-" for i in b.ids[~isp]}
    side_by_id |= {int(i): s for i, s in zip(b.ids[isp], sides)}
    # line players via the still rule on a one-frame canonical table
    pts = b.xy[isp].copy()
    if offense == "right":
        pts[:, 0] = FIELD_LENGTH - pts[:, 0]
        pts[:, 1] = FIELD_WIDTH - pts[:, 1]
    one = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "side": sides})
    line = line_player_mask(one)
    line_ids = {int(i) for i, ln in zip(b.ids[isp], line.to_numpy()) if ln}
    teams = TeamColors.from_frame(b.colors, b.labels, isp, sides)
    return SideAssignment(side_by_id, line_ids, int((~isp).sum()), b.n_offfield, teams)


TEAM_SIZE = 11
CONTESTED_BAND = 2.5  # yd either side of the LOS where sides come from jersey color


def cap_sides(sides: np.ndarray, x: np.ndarray, los: float, offense: str) -> np.ndarray:
    """At most 11 per side: move the surplus back across, from the color-decided bodies.

    Away from the ball geometry decides a body's side; within CONTESTED_BAND of
    the LOS jersey color does, and similar kits (powder blue vs white on the
    Dolphins-Chargers clip) can put 13 on one side. The surplus is taken from the
    contested bodies standing furthest toward the other team's side of the ball
    (a defender a yard onto the offense's side is the likeliest miscall).
    """
    sides = np.asarray(sides).astype(object)
    toward_off = (los - x) if offense == "right" else (x - los)  # > 0 = defense side
    for side, other in (("Defense", "Offense"), ("Offense", "Defense")):
        # move only into room on the other side (a 12th body there would just come back)
        surplus = min(
            int((sides == side).sum()) - TEAM_SIZE, TEAM_SIZE - int((sides == other).sum())
        )
        if surplus <= 0:
            continue
        cand = np.flatnonzero((sides == side) & (np.abs(x - los) < CONTESTED_BAND))
        # likeliest miscalls first: for a defense surplus the bodies furthest toward the
        # offense, for an offense surplus those furthest toward the defense
        key = toward_off[cand] if side == "Defense" else -toward_off[cand]
        for i in cand[np.argsort(key)][:surplus]:
            sides[i] = other
    return sides.astype(str)


def unclipped_mask_boxes(boxes: np.ndarray, image_height: int, edge_px: int = 4) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    return boxes[:, 3] < image_height - edge_px


def _sideline_stray(xy: np.ndarray, side: str, los: float | None, offense: str) -> bool:
    """A body first seen by the sideline in the offense's backfield is not a defender:
    the chain crew and officials stand there, defenders never do."""
    if los is None or side != "Defense":
        return False
    behind = (xy[0] - los) if offense == "right" else (los - xy[0])  # > 0: offense's side
    return behind > 0 and min(xy[1], FIELD_WIDTH - xy[1]) < SNAP_SIDELINE_CLEAR


def canonical_tracks(
    frames: list[TrackedFrame],
    calibs: list[FieldCalibration],
    sides: SideAssignment,
    snap_index: int,
    offense: str,
    step_s: float = 0.1,
    dataset: str = "clip",
    game_id: int = 0,
    play_id: int = 0,
    drop_line: bool = True,
    los: float | None = None,
) -> pd.DataFrame:
    """Canonical frames at the sampling rate: offense -> right, rel_frame 0 at the snap.

    Speeds and accelerations come from finite differences of position; `dir` is
    the direction of motion (degrees, tracking-data convention), `o` is NaN
    (no body orientation from a box). `los` (raw field x) lets a body the tracker
    re-identifies later be ruled out as a sideline stray.
    """
    rows = []
    side_by_id = dict(sides.side_by_id)
    for fr, calib in zip(frames, calibs):
        keep = unclipped_mask_boxes(fr.boxes, fr.image.shape[0]) & (fr.ids >= 0)
        if not keep.any():
            continue
        boxes, ids = fr.boxes[keep], fr.ids[keep]
        feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]).astype(np.float32)
        xy = calib.to_field(feet)
        # a tracker re-identifies a lost body with a fresh id: side it by jersey color
        # against the snap-frame team reference (post-snap geometry says nothing)
        new = [k for k, tid in enumerate(ids) if int(tid) not in side_by_id]
        if new and sides.teams is not None and fr.index > 0:
            nb = [tuple(boxes[k]) for k in new]
            colors = torso_colors(fr.image, nb)
            striped = referee_mask(fr.image, nb)
            got, ok = sides.teams.assign(colors)
            onf = on_field(xy[new], y_margin=SIDELINE_Y_MARGIN)
            for k, sd, good, st, on in zip(new, got, ok, striped, onf):
                side_by_id[int(ids[k])] = (
                    sd
                    if (good and not st and on and not _sideline_stray(xy[k], sd, los, offense))
                    else "-"
                )
        for tid, (x, y) in zip(ids, xy):
            side = side_by_id.get(int(tid))
            if side in (None, "-") or (drop_line and int(tid) in sides.line_ids):
                continue
            if offense == "right":
                x, y = FIELD_LENGTH - x, FIELD_WIDTH - y
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": play_id,
                    "nfl_id": int(tid),
                    "frame_id": fr.index + 1,
                    "rel_frame": fr.index - snap_index,
                    "x": float(x),
                    "y": float(y),
                    "side": side,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no tracked players with a side; check sides/calibration")
    df = df.sort_values(["nfl_id", "rel_frame"]).reset_index(drop=True)
    g = df.groupby("nfl_id")
    dx = g["x"].diff()
    dy = g["y"].diff()
    dt = g["rel_frame"].diff() * step_s
    speed = np.hypot(dx, dy) / dt
    df["s"] = speed.fillna(0.0)
    df["a"] = (g["s"].diff() / dt).fillna(0.0) if "s" in df else 0.0
    df["o"] = np.nan
    df["dir"] = np.degrees(np.arctan2(dx, dy)) % 360  # tracking convention: 0 = +y
    df["position"] = None
    df["coverage_type"] = None
    df["man_zone"] = None
    snap = df[df["rel_frame"] == 0]
    off = snap[snap["side"] == "Offense"]
    if len(off):
        qb_id = off.loc[off["x"].idxmin(), "nfl_id"]  # deepest offensive player at the snap
        df.loc[df["nfl_id"] == qb_id, "position"] = "QB"
    return finalize(df, dataset=dataset)
