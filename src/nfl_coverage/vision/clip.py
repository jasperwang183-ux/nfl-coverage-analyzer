"""Clip ingestion: an mp4 -> sampled still frames -> per-frame field analysis.

Rung 3, thin slice. There is no multi-object tracking yet, so a clip is treated
as a set of unlinked stills: the analyst names which sampled frame is the
pre-snap look and which one is the post-snap read, each goes through the still
pipeline (detect -> homography -> jersey-color sides -> line filter ->
canonical frame), and the shell reader compares the two. One calibration
serves the whole clip, so this assumes a fixed camera (all-22 / elevated end
zone); a panning broadcast camera needs per-frame calibration, which is not
built. The field-visibility guard runs on every analyzed frame regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nfl_coverage.analysis.shell import shell_state
from nfl_coverage.config import FIELD_LENGTH
from nfl_coverage.features.alignment import KEYS, _los_estimate
from nfl_coverage.vision.detect import dedupe_mask, detect_people, foot_points, unclipped_mask
from nfl_coverage.vision.homography import FieldCalibration, on_field
from nfl_coverage.vision.still import frame_from_points, line_player_mask
from nfl_coverage.vision.teams import (
    TeamColors,
    cluster_teams,
    referee_mask,
    sides_from_clusters,
    torso_colors,
)
from nfl_coverage.vision.visibility import check_visibility, defender_count_warnings

SIDELINE_Y_MARGIN = -0.5  # bench feet sit on the paint: keep only bodies half a yard inside


@dataclass
class SampledFrame:
    index: int
    t: float  # seconds into the clip
    image: np.ndarray  # BGR


def sample_frames(
    video: Path, every_s: float = 0.5, start_s: float = 0.0, end_s: float | None = None
) -> list[SampledFrame]:
    """Grab one frame every `every_s` seconds between start and end (inclusive)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"could not open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_total / fps if n_total > 0 else None
    if end_s is None:
        end_s = duration if duration is not None else start_s
    out = []
    t = start_s
    idx = 0
    while t <= end_s + 1e-9:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        ok, img = cap.read()
        if not ok:
            break
        out.append(SampledFrame(idx, round(t, 3), img))
        idx += 1
        t += every_s
    cap.release()
    return out


def save_frames(samples: list[SampledFrame], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in samples:
        p = out_dir / f"frame_{s.index:03d}_{s.t:05.2f}s.png"
        cv2.imwrite(str(p), s.image)
        paths.append(p)
    return paths


def motion_scores(samples: list[SampledFrame], size: tuple[int, int] = (160, 90)) -> list[float]:
    """Mean absolute gray-level change vs. the previous sample (0 for the first).

    A hint, not a detector: the snap shows up as the first big jump after a
    quiet stretch, and a camera cut as a huge one.
    """
    prev = None
    scores = []
    for s in samples:
        g = cv2.cvtColor(cv2.resize(s.image, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
        scores.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    return scores


def contact_sheet(
    samples: list[SampledFrame], scores: list[float], cols: int = 4, width: int = 420
) -> np.ndarray:
    """Thumbnails in a grid, each stamped with its index, time and motion score."""
    h0, w0 = samples[0].image.shape[:2]
    th = int(width * h0 / w0)
    rows = (len(samples) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * width, 3), dtype=np.uint8)
    for s, sc in zip(samples, scores):
        r, c = divmod(s.index, cols)
        thumb = cv2.resize(s.image, (width, th))
        text = f"#{s.index}  {s.t:.1f}s  motion {sc:.0f}"
        cv2.putText(thumb, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            thumb, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1, cv2.LINE_AA
        )
        sheet[r * th : (r + 1) * th, c * width : (c + 1) * width] = thumb
    return sheet


@dataclass
class CameraMotion:
    """How the camera moved between two frames, from matched field features."""

    shift_px: float  # translation in original pixels
    scale: float  # 1.0 = no zoom
    n_inliers: int
    moved: bool
    reason: str = ""


def camera_motion(
    img_a: np.ndarray,
    img_b: np.ndarray,
    max_shift_frac: float = 0.02,
    max_scale_dev: float = 0.03,
    min_inliers: int = 20,
) -> CameraMotion:
    """Did the camera pan, tilt or zoom between two frames of the same clip?

    One calibration only holds while the camera holds. ORB features on the
    field (lines, numbers, logos dominate; moving players are RANSAC outliers)
    give a similarity transform between the frames; a shift over
    `max_shift_frac` of the width or a zoom over `max_scale_dev` means the
    second frame needs its own calibration. Too few inliers is treated as moved
    (a cut to another camera leaves nothing to match).
    """
    w = img_a.shape[1]
    pa, pb = _match_features(img_a, img_b)
    if len(pa) < 8:
        return CameraMotion(np.nan, np.nan, 0, True, "too few features to match")
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    n_inl = int(inl.sum()) if inl is not None else 0
    if M is None or n_inl < min_inliers:
        return CameraMotion(np.nan, np.nan, n_inl, True, "frames do not match (camera cut?)")
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    shift = float(np.hypot(M[0, 2], M[1, 2]))
    moved = shift > max_shift_frac * w or abs(scale - 1.0) > max_scale_dev
    reason = f"shift {shift:.0f} px, zoom x{scale:.3f}, {n_inl} inliers" if moved else ""
    return CameraMotion(shift, scale, n_inl, bool(moved), reason)


def _match_features(
    img_a: np.ndarray, img_b: np.ndarray, work_w: int = 960, keep: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """ORB matches between two frames, as (N, 2) pixel arrays in original coordinates."""
    f = work_w / img_a.shape[1]
    ga = cv2.cvtColor(cv2.resize(img_a, None, fx=f, fy=f), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(img_b, None, fx=f, fy=f), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    matches = sorted(
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda m: m.distance
    )[:keep]
    pa = np.float32([ka[m.queryIdx].pt for m in matches]) / f
    pb = np.float32([kb[m.trainIdx].pt for m in matches]) / f
    return pa, pb


@dataclass
class TransferredCalibration:
    calibration: FieldCalibration
    n_inliers: int
    frame_homography: np.ndarray  # (3, 3) pixels in frame A -> pixels in frame B
    reprojection_error: float  # yards, of the transferred landmarks under the refit


def transfer_calibration(
    calib_a: FieldCalibration, img_a: np.ndarray, img_b: np.ndarray, min_inliers: int = 30
) -> TransferredCalibration:
    """Carry frame A's field calibration to frame B through the camera's motion.

    A broadcast camera on a fixed mount pans, tilts and zooms about its optical
    center, and such a motion maps frame A to frame B by an exact homography.
    Matched field features (lines, numbers, logos; players are RANSAC
    outliers) give that homography; frame A's clicked landmarks are pushed
    through it and refit against the same field coordinates, so frame B gets a
    calibration without anyone clicking on it. The refit's reprojection error
    is the sanity number: it should stay near frame A's.
    """
    pa, pb = _match_features(img_a, img_b)
    if len(pa) < min_inliers:
        raise ValueError(f"only {len(pa)} feature matches between frames; cannot transfer")
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    n_inl = int(inl.sum()) if inl is not None else 0
    if H is None or n_inl < min_inliers:
        raise ValueError(f"frame-to-frame homography unreliable ({n_inl} inliers)")
    moved_px = cv2.perspectiveTransform(
        calib_a.pixel_points.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)
    calib_b = FieldCalibration.from_points(moved_px, calib_a.field_points)
    return TransferredCalibration(calib_b, n_inl, H, calib_b.reprojection_error())


@dataclass
class FrameAnalysis:
    """One still through the vision pipeline, ready for the shell reader."""

    frame: pd.DataFrame  # canonical rel_frame-0 frame, all players
    analysis: pd.DataFrame  # coverage-relevant players only (lines dropped)
    detections: list
    status: np.ndarray  # per detection: Offense / Defense / Line / Referee
    caveats: list[str] = field(default_factory=list)
    visibility: object = None
    teams: TeamColors | None = None  # learned here (pre-snap) or inherited (post-snap)
    line_center_y: float | None = None  # canonical y of the ball, from the line players


def analyze_image(
    image_path: Path,
    calib: FieldCalibration,
    los: float,
    offense: str,
    yolo: str = "yolo11s.pt",
    conf: float = 0.3,
    reference: TeamColors | None = None,
    imgsz: int = 1280,
) -> FrameAnalysis:
    """The still pipeline (scripts/analyze_still.py) as a function, no model call.

    Without `reference` (a pre-snap frame) sides come from jersey-color clusters
    voted by geometry. With it (a post-snap frame, where geometry lies because
    receivers have crossed the LOS) sides come from the nearest team color only.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"could not read {image_path}")
    vis = check_visibility(calib, img.shape, los, offense)

    detections = detect_people(image_path, yolo, conf, imgsz=imgsz)
    detections = [d for d, k in zip(detections, unclipped_mask(detections, img.shape[0])) if k]
    keep = on_field(calib.to_field(foot_points(detections)), y_margin=SIDELINE_Y_MARGIN)
    detections = [d for d, k in zip(detections, keep) if k]
    field_xy = calib.to_field(foot_points(detections))
    unique = dedupe_mask(field_xy, [d.conf for d in detections])
    detections = [d for d, k in zip(detections, unique) if k]
    field_xy = field_xy[unique]

    boxes = [d.box for d in detections]
    colors = torso_colors(img, boxes)
    striped = referee_mask(img, boxes)
    if reference is None:
        labels, is_player = cluster_teams(colors)
        is_player &= ~striped
        sides = sides_from_clusters(
            labels[is_player], field_xy[is_player, 0], los, offense, colors=colors[is_player]
        )
        teams = TeamColors.from_frame(colors, labels, is_player, sides)
    else:
        all_sides, is_player = reference.assign(colors)
        is_player &= ~striped
        sides = all_sides[is_player]
        teams = reference
    status = np.where(is_player, "", "Referee").astype(object)
    status[is_player] = sides

    frame = frame_from_points(field_xy[is_player], los, offense, side=sides)
    # post-snap the offense has left the line, so the line filter gets the snap
    # LOS explicitly (canonical coords: frame_from_points mirrors 'right' plays)
    canonical_los = FIELD_LENGTH - los if offense == "right" else los
    line = line_player_mask(frame, los=canonical_los if reference is not None else None)
    status[np.flatnonzero(is_player)[line.to_numpy()]] = "Line"
    analysis = frame[~line.to_numpy()]
    n_def = int((analysis["side"] == "Defense").sum())
    caveats = list(vis.warnings) + defender_count_warnings(n_def)
    # the ball sits in the middle of the two lines: a better formation center
    # than the deepest offensive player, who on film is often an offset back
    line_center = float(frame.loc[line.to_numpy(), "y"].median()) if line.sum() >= 3 else None
    return FrameAnalysis(frame, analysis, detections, status, caveats, vis, teams, line_center)


def clip_shells(pre: FrameAnalysis, run: FrameAnalysis, run_deep_depth: float) -> dict:
    """Shown shell from the pre-snap frame, run shell from the post-snap frame.

    The LOS and formation center come from the pre-snap frame and are imposed on
    the post-snap read (the offense has moved by then). Both frames are
    canonical rel_frame-0 frames with the same dummy play key.
    """
    pre_off = pre.analysis[pre.analysis["side"] == "Offense"]
    if len(pre_off) == 0 or (pre.analysis["side"] == "Defense").sum() == 0:
        counts = dict(zip(*np.unique(pre.status, return_counts=True)))
        raise ValueError(
            "pre-snap frame has no usable offense/defense after filtering "
            f"(status counts: {counts}); check --imgsz, the calibration, and the annotated frame"
        )
    los = float(_los_estimate(pre_off).iloc[0])
    qb = pre_off[pre_off["position"] == "QB"]
    if pre.line_center_y is not None:
        center = pre.line_center_y
    else:
        center = float(qb["y"].iloc[0]) if len(qb) else float(pre_off["y"].mean())
    shown = shell_state(pre.analysis, 0, los=los, center=center).iloc[0]
    played = shell_state(run.analysis, 0, deep_depth=run_deep_depth, los=los, center=center).iloc[0]
    return {
        "los": los,
        "center": center,
        "shown": shown.drop(KEYS).to_dict(),
        "run": played.drop(KEYS).to_dict(),
    }
