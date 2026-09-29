"""A film clip, end to end: tracked frames (cached) -> the read under the video (rung 3).

Two stages, so a person can correct the read without waiting on the tracker:

  track_frames  YOLO + ByteTrack over the play window and one anchor calibration
                chained to every kept frame. Minutes on a CPU; `TrackedClip.save`
                writes it to disk (frames as JPEG, boxes/ids/confs and the chained
                homographies as JSON).
  read_tracked  snap, formation, sides, canonical 10 Hz tracks, shown and played
                structure, the disguise components, the variant model and the
                `CoverageReport`. Seconds, so a snap time, LOS, offense side or ball
                position typed over the inferred one re-reads immediately.

Every read runs from the saved frames (the first one too), so a re-read with
no overrides reproduces the first exactly. `scripts/track_clip.py` is the CLI
over this module; the web page (`nfl_coverage.web`) is the other client.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nfl_coverage.analysis.disguise import RUN_FRAME, declared_frame, shell_rotation
from nfl_coverage.analysis.mechanisms import HIT_COLUMNS, mechanism_hits
from nfl_coverage.analysis.report import (
    CoverageReport,
    ModelRead,
    StructuralRead,
    disguise_text,
    model_read,
    structural_read,
    variant_confidence,
)
from nfl_coverage.config import FIELD_LENGTH, FIELD_WIDTH
from nfl_coverage.features.alignment import snapshot_features
from nfl_coverage.features.movement import post_snap_features
from nfl_coverage.features.phases import defender_phases
from nfl_coverage.vision.formation import Formation, infer_formation
from nfl_coverage.vision.homography import FieldCalibration, on_field
from nfl_coverage.vision.snap import SnapDetection, detect_snap, field_positions
from nfl_coverage.vision.track import (
    ChainedCalibration,
    SideAssignment,
    SnapBodies,
    TrackedFrame,
    assign_sides,
    canonical_tracks,
    chain_calibrations,
    snap_bodies,
    track_players,
)
from nfl_coverage.vision.visibility import check_visibility

SHOWN_DEEP = 9.0  # yd: deep at the snap (the tracking-data threshold)
RUN_DEEP_FILM = 12.0  # yd: deep in the played read on film (see HANDOFF: under review)
PLAYED_MAX_S = 5.0  # s after the snap: the latest frame the played read may be taken at
JPEG_QUALITY = 92


class ReadError(ValueError):
    """The clip cannot be read as asked; `needs` names the value a person can give."""

    def __init__(self, message: str, needs: str):
        super().__init__(message)
        self.needs = needs  # 'snap' / 'offense'


@dataclass
class TrackedClip:
    frames: list[TrackedFrame]
    calibrations: list[FieldCalibration]
    step: float
    notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)  # carried into every read's report

    @property
    def times(self) -> np.ndarray:
        return np.array([f.t for f in self.frames])

    def index_at(self, t: float) -> int:
        return int(np.argmin(np.abs(self.times - t)))

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"step": self.step, "notes": self.notes, "caveats": self.caveats, "frames": []}
        for fr, cal in zip(self.frames, self.calibrations):
            name = f"frame_{fr.index:03d}.jpg"
            cv2.imwrite(str(out_dir / name), fr.image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            meta["frames"].append(
                {
                    "index": fr.index,
                    "t": fr.t,
                    "image": name,
                    "boxes": np.asarray(fr.boxes, float).tolist(),
                    "ids": np.asarray(fr.ids, int).tolist(),
                    "confs": np.asarray(fr.confs, float).tolist(),
                    "matrix": np.asarray(cal.matrix, float).tolist(),
                    # the points too: the visibility check takes the camera side from them
                    "pixel_points": np.asarray(cal.pixel_points, float).tolist(),
                    "field_points": np.asarray(cal.field_points, float).tolist(),
                }
            )
        (out_dir / "tracked.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, out_dir: Path) -> TrackedClip:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "tracked.json").read_text())
        frames, calibs = [], []
        empty = np.zeros((0, 2), np.float32)
        for m in meta["frames"]:
            img = cv2.imread(str(out_dir / m["image"]))
            if img is None:
                raise FileNotFoundError(out_dir / m["image"])
            boxes = np.asarray(m["boxes"], float).reshape(-1, 4)
            frames.append(
                TrackedFrame(
                    m["index"],
                    m["t"],
                    img,
                    boxes,
                    np.asarray(m["ids"], int),
                    np.asarray(m["confs"], float),
                )
            )
            calibs.append(
                FieldCalibration(
                    np.asarray(m.get("pixel_points", empty), np.float32).reshape(-1, 2),
                    np.asarray(m.get("field_points", empty), np.float32).reshape(-1, 2),
                    np.asarray(m["matrix"], float),
                )
            )
        return cls(frames, calibs, meta["step"], meta.get("notes", []), meta.get("caveats", []))


def _frame_at(video: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"no frame at {t:.2f} s")
    return img


CARRY_STEP = 0.25  # s between hops when carrying a calibration along the clip


def carry_calibration(
    video: Path, calib: FieldCalibration, t_from: float, t_to: float, step: float = CARRY_STEP
) -> tuple[FieldCalibration, list[str]]:
    """Carry a calibration fitted at `t_from` to the frame at `t_to`.

    The camera pans and zooms between the frame a person calibrated (a still
    pre-snap look) and the play window, so the calibration hops through frames
    `step` s apart, each hop through the frame-to-frame homography of matched
    field features (`clip.transfer_calibration`). At the destination it is
    refit on that frame's own markings: its yard lines are labeled by the
    carried calibration (nearest 5 yd, consistent with their spacing) and
    `search_calibration` fits them, so hop errors do not accumulate. Raises
    ValueError when a hop cannot be matched (a cut or a jump the features do not
    survive): calibrate a frame inside the window then.
    """
    from nfl_coverage.vision.clip import transfer_calibration
    from nfl_coverage.vision.fieldlines import refit_from_paint

    notes: list[str] = []
    if abs(t_to - t_from) < 1e-3:
        return calib, notes
    sign = 1.0 if t_to > t_from else -1.0
    ts = list(np.arange(t_from + sign * step, t_to, sign * step)) + [t_to]
    prev_t, prev = t_from, _frame_at(video, t_from)
    cur = calib
    for t in ts:
        img = _frame_at(video, t)
        try:
            cur = transfer_calibration(cur, prev, img).calibration
        except ValueError as e:
            raise ValueError(
                f"could not follow the camera from {prev_t:.2f} s to {t:.2f} s ({e}); "
                "calibrate a frame inside the play window instead"
            ) from e
        prev_t, prev = t, img
    notes.append(f"calibration carried {t_from:.1f} s -> {t_to:.1f} s in {len(ts)} hops")
    try:
        refit = refit_from_paint(prev, cur)
    except ValueError as e:
        notes.append(f"kept the carried calibration (refit at {t_to:.1f} s failed: {e})")
        return cur, notes
    notes.append(
        f"refit at {t_to:.1f} s on {len(refit.best.pixel)} landmarks ({refit.n_rows} field rows)"
    )
    if refit.warning:
        notes.append(refit.warning)
    return refit.best.calibration, notes


REANCHOR_EVERY = 1.0  # s between drift checks along the tracked window
REANCHOR_DRIFT = 0.4  # yd off the 5-yd grid that triggers a refit on the frame's own paint
REANCHOR_GAIN = 0.1  # yd the refit must improve on the chain to replace it


def reanchored_chain(
    frames: list[TrackedFrame],
    anchor: FieldCalibration,
    anchor_i: int,
    keyframes: dict[int, FieldCalibration],
    step: float,
) -> ChainedCalibration:
    """Chain the anchor through the window, refitting wherever the chain drifts.

    Frame-to-frame transfers accumulate error (clip 1531a slid 2.2 yd in 5 s).
    Walking out from the anchor about once a second, a frame whose own yard lines
    sit REANCHOR_DRIFT off the chained grid is refit on its paint, named by the
    chained calibration, and becomes a keyframe the chain restarts from. Checking
    often keeps the drift far under the 2.5 yd at which the naming would slip.

    Only the along-field position is corrected: that is what the yard lines
    measure. A refit on a hash-only frame can move the depth toward the sidelines
    by a yard with nothing to check it against. The drift built up gradually, so
    the correction is spread back linearly over the frames since the previous
    anchor. Applied in one frame it moves every body at once, and the snap
    detector reads that as the line firing.
    """
    from nfl_coverage.vision.fieldlines import find_marks, grid_drift, refit_from_paint

    kf = dict(keyframes)
    chain = chain_calibrations(frames, anchor, anchor_i, kf)
    every = max(1, round(REANCHOR_EVERY / step))
    walks = [range(anchor_i + every, len(frames), every), range(anchor_i - every, -1, -every)]
    corrections: list[tuple[int, int, np.ndarray]] = []  # (previous anchor, keyframe, C)
    notes: list[str] = []
    for walk in walks:
        prev = anchor_i
        for i in walk:
            if i in kf:
                prev = i
                continue
            img = frames[i].image
            marks = find_marks(img)
            drift = grid_drift(img, chain.calibrations[i], marks)
            if not drift >= REANCHOR_DRIFT:  # NaN: no lines to check against
                continue
            try:
                refit = refit_from_paint(img, chain.calibrations[i], marks).best.calibration
            except ValueError as e:
                notes.append(f"{frames[i].t:.1f} s drifts {drift:.2f} yd; refit failed ({e})")
                continue
            c = along_field_correction(chain.calibrations[i], refit, img.shape[:2])
            fixed = corrected(chain.calibrations[i], c)
            after = grid_drift(img, fixed, marks)
            if not after < drift - REANCHOR_GAIN:
                continue
            corrections.append((prev, i, c))
            kf[i] = fixed
            chain = chain_calibrations(frames, anchor, anchor_i, kf)
            notes.append(f"re-anchored at {frames[i].t:.1f} s: drift {drift:.2f} -> {after:.2f} yd")
            prev = i
    calibs = list(chain.calibrations)
    for prev, i, c in corrections:
        n = abs(i - prev)
        for k in range(1, n):
            j = prev + k if i > prev else prev - k
            w = k / n
            calibs[j] = corrected(calibs[j], (1 - w) * np.eye(3) + w * c)
    return ChainedCalibration(calibs, chain.notes + notes)


def along_field_correction(
    old: FieldCalibration, new: FieldCalibration, shape: tuple[int, int]
) -> np.ndarray:
    """Field-to-field map x -> a x + b y + c (y kept) taking `old` to `new` over the frame."""
    h, w = shape
    gx, gy = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, 12), np.linspace(0.05 * h, 0.95 * h, 9))
    px = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    a, b = old.to_field(px), new.to_field(px)
    ok = on_field(a) & on_field(b)
    if ok.sum() < 6:
        return np.eye(3)
    design = np.column_stack([a[ok], np.ones(int(ok.sum()))])
    coef, *_ = np.linalg.lstsq(design, b[ok, 0], rcond=None)
    return np.array([coef, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def corrected(calib: FieldCalibration, c: np.ndarray) -> FieldCalibration:
    """`calib` followed by a field-to-field correction; its landmarks move with it."""
    m = c @ calib.matrix
    fp = cv2.perspectiveTransform(calib.pixel_points.reshape(-1, 1, 2), m).reshape(-1, 2)
    return FieldCalibration(calib.pixel_points, fp.astype(np.float32), m)


def track_frames(
    video: Path,
    start_s: float,
    end_s: float,
    anchor: FieldCalibration,
    anchor_t: float,
    keyframes: dict[float, FieldCalibration] | None = None,
    step: float = 0.1,
    yolo: str = "yolo11s.pt",
    conf: float = 0.3,
    imgsz: int = 1920,
    progress: Callable[[float], None] | None = None,
) -> TrackedClip:
    """Track the play window and chain the anchor calibration to every kept frame.

    `anchor_t` (and each keyframe time) must fall inside [start_s, end_s];
    `progress` gets the tracking fraction (0-1) as frames are processed.
    """
    if not start_s <= anchor_t <= end_s:
        raise ValueError(f"anchor {anchor_t:.1f} s is outside the window {start_s}-{end_s} s")
    frames = track_players(video, start_s, end_s, step, yolo, conf, imgsz, progress=progress)
    if not frames:
        raise ValueError("no frames tracked: is the window inside the video?")
    times = np.array([f.t for f in frames])
    anchor_i = int(np.argmin(np.abs(times - anchor_t)))
    kf = {int(np.argmin(np.abs(times - t))): c for t, c in (keyframes or {}).items()}
    chain = reanchored_chain(frames, anchor, anchor_i, kf, step)
    return TrackedClip(frames, chain.calibrations, step, list(chain.notes))


@dataclass
class ClipRead:
    snap: SnapDetection
    snap_index: int
    snap_t: float
    snap_source: str  # 'detected' / 'given'
    formation: Formation  # inferred at the snap frame, no hints
    offense: str
    los: float  # field x (raw) actually used
    ball_y: float | None
    used: dict[str, str]  # 'inferred' / 'given' per value
    warnings: list[str]
    sides: SideAssignment
    bodies: SnapBodies
    tracks: pd.DataFrame
    declared_frame: int
    rotation: dict
    report: CoverageReport
    features: dict[str, float]
    step: float

    def to_dict(self) -> dict:
        """JSON-able: the contract for the page (`result.json`)."""
        f = self.formation
        return {
            "snap": {
                "t": self.snap_t,
                "index": self.snap_index,
                "source": self.snap_source,
                "detected_t": self.snap.t,
                "confidence": self.snap.confidence,
                "score": self.snap.score,
                "reasons": self.snap.reasons,
            },
            "formation": {
                "inferred": {
                    "offense": f.offense,
                    "offense_confidence": f.offense_confidence,
                    "los": f.los,
                    "los_range": list(f.los_range),
                    "los_confidence": f.los_confidence,
                    "ball_y": f.ball_y,
                    "ball_y_confidence": f.ball_y_confidence,
                    "reasons": f.reasons,
                },
                "used": {"offense": self.offense, "los": self.los, "ball_y": self.ball_y},
                "source": self.used,
            },
            "warnings": self.warnings,
            "sides": {
                "offense": sum(s == "Offense" for s in self.sides.side_by_id.values()),
                "defense": sum(s == "Defense" for s in self.sides.side_by_id.values()),
                "line": len(self.sides.line_ids),
                "non_players": self.sides.n_referee,
            },
            "played_frame": {
                "rel_frame": self.declared_frame,
                "t": self.snap_t + self.declared_frame * self.step,
            },
            "rotation": _jsonable(self.rotation),
            "report": _jsonable(self.report.to_dict()),
            "features": _jsonable(self.features),
            "n_players_tracked": int(self.tracks["nfl_id"].nunique()),
        }


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None  # NaN / inf are not JSON; the page reads null as "not available"
    return x


def _canonical(offense: str, los: float, ball_y: float | None) -> tuple[float, float | None]:
    """LOS and middle-of-field center in canonical coordinates (offense moving right)."""
    los_c = FIELD_LENGTH - los if offense == "right" else los
    center = None
    if ball_y is not None:
        ball_c = FIELD_WIDTH - ball_y if offense == "right" else ball_y
        # a post safety on a hash-mark ball sits between the ball and the field's middle
        center = (ball_c + FIELD_WIDTH / 2) / 2
    return los_c, center


def sequence_reads(
    bundle, clip: TrackedClip, sides, snap_i: int, offense: str, step: float, los=None
) -> tuple[ModelRead | None, ModelRead | None]:
    """The sequence model's pre-snap and whole-play calls, or (None, None) if it can't read.

    The model trained on all 22 players at 10 Hz, so it gets tracks with the line kept
    (its LOS estimate comes from the offense) and only a 0.1 s grid.
    """
    if bundle is None or abs(step - 0.1) > 1e-6:
        return None, None
    from nfl_coverage.models import sequence  # deferred: torch import

    tracks = canonical_tracks(
        clip.frames, clip.calibrations, sides, snap_i, offense, step, drop_line=False, los=los
    )
    try:
        p = sequence.predict(bundle, tracks)
    except ValueError:
        return None, None
    source = bundle.meta.get("args", {}).get("name", "sequence model")
    pre = model_read(source, p["pre_rel_frame"] * step, p["pre_variant"], p["pre_man_zone"])
    post = model_read(source, p["last_rel_frame"] * step, p["variant"], p["man_zone"])
    return pre, post


def read_tracked(
    clip: TrackedClip,
    snap_t: float | None = None,
    los: float | None = None,
    offense: str | None = None,
    ball_y: float | None = None,
    run_deep: float = RUN_DEEP_FILM,
    bundle: dict | None = None,
    sequence_bundle=None,
) -> ClipRead:
    """Snap, formation, sides, tracks and the report; given values override inferred ones."""
    frames, calibs, step = clip.frames, clip.calibrations, clip.step
    det = detect_snap(field_positions(frames, calibs), los=los, step_s=step)
    if snap_t is not None:
        snap_i, snap_source = clip.index_at(snap_t), "given"
    elif det.frame is not None:
        snap_i, snap_source = det.frame, "detected"
    else:
        raise ReadError(
            "no snap found (" + "; ".join(det.reasons) + "): give the snap time, or start "
            "the window at least 0.5 s before the snap",
            needs="snap",
        )

    bodies = snap_bodies(frames[snap_i], calibs[snap_i])
    form = infer_formation(bodies.xy, bodies.labels, bodies.is_player)
    if form.suspect_official is not None:
        # a lone body deep behind the offense: an official the striped-shirt check missed
        bodies.is_player[form.suspect_official] = False
    warnings = []
    for name, given, inferred in (
        ("offense side", offense, form.offense),
        ("LOS", los, form.los),
        ("ball y", ball_y, form.ball_y),
    ):
        if given is None or inferred is None:
            continue
        if isinstance(given, str):
            if given != inferred:
                warnings.append(f"{name} given as {given}; the formation reads {inferred}")
        elif abs(given - inferred) >= 0.5:
            warnings.append(f"{name} given as {given:.1f}; the formation reads {inferred:.1f}")
    used = {
        "offense": "given" if offense else "inferred",
        "los": "given" if los is not None else "inferred",
        "ball_y": "given" if ball_y is not None else "inferred",
    }
    offense = offense or form.offense
    if offense is None:
        raise ReadError(
            "the offense side is not readable from the formation ("
            + form.reasons[0]
            + "): give it",
            needs="offense",
        )
    los = los if los is not None else form.los
    ball_y = ball_y if ball_y is not None else form.ball_y

    sides = assign_sides(frames[snap_i], calibs[snap_i], los, offense, bodies)
    tracks = canonical_tracks(frames, calibs, sides, snap_i, offense, step, los=los)
    los_c, center = _canonical(offense, los, ball_y)

    declared = declared_frame(
        tracks, RUN_FRAME, run_deep, center=center, los=los_c, max_frame=round(PLAYED_MAX_S / step)
    )
    rot = shell_rotation(
        tracks, run_frame=declared, run_deep_depth=run_deep, center=center, los=los_c
    ).iloc[0]
    snap = tracks[tracks["rel_frame"] == 0]
    center_or_off = (
        center if center is not None else snap.loc[snap["side"] == "Offense", "y"].mean()
    )

    shown_caveats = check_visibility(
        calibs[snap_i], frames[snap_i].image.shape[:2], los, offense
    ).warnings
    shown = structural_read(snap, los_c, center_or_off, SHOWN_DEEP, "snap", list(shown_caveats))
    rf = int(rot["run_frame"])
    later = tracks[tracks["rel_frame"] == rf]
    played: StructuralRead | None = None
    played_caveats: list[str] = []
    if len(later):
        pi = min(max(snap_i + rf, 0), len(frames) - 1)
        played_caveats = list(
            check_visibility(calibs[pi], frames[pi].image.shape[:2], los, offense).warnings
        )
        played = structural_read(
            later,
            los_c,
            center_or_off,
            run_deep,
            f"+{rf * step:.1f} s",
            played_caveats,
            rel_frame=rf,
        )

    variant: list[tuple[str, float]] = []
    if bundle is not None and len(snap):
        from nfl_coverage.models.persist import predict_proba  # deferred: sklearn import

        ranked = predict_proba(bundle, snapshot_features(snap)).iloc[0]
        variant = [(str(k), float(v)) for k, v in ranked.sort_values(ascending=False).items()]
    pre_model, post_model = sequence_reads(
        sequence_bundle, clip, sides, snap_i, offense, step, los=los
    )
    label = rot["shell_rotation_label"] or ""
    score = float(rot["shell_rotation_score"])
    phases = defender_phases(tracks, los=los_c, center=center_or_off, step=step)
    hits = mechanism_hits(phases)[HIT_COLUMNS]
    report = CoverageReport(
        shown=shown,
        played=played,
        variant=variant,
        variant_confidence=variant_confidence(variant),
        disguise_label=label,
        disguise_score=score,
        disguise_text=disguise_text(label, shown, played),
        caveats=list(clip.caveats) + list(shown_caveats) + played_caveats,
        pre_model=pre_model,
        post_model=post_model,
        mechanisms=[_jsonable(h) for h in hits.to_dict("records")],
    )
    feats = post_snap_features(tracks, horizon=rf).iloc[0]
    features = {k: float(v) for k, v in feats.items() if k.startswith("post_")}
    rotation = {
        k: _jsonable(v) for k, v in rot.to_dict().items() if not isinstance(v, (list, dict))
    }
    return ClipRead(
        snap=det,
        snap_index=snap_i,
        snap_t=float(frames[snap_i].t),
        snap_source=snap_source,
        formation=form,
        offense=offense,
        los=float(los),
        ball_y=None if ball_y is None else float(ball_y),
        used=used,
        warnings=warnings,
        sides=sides,
        bodies=bodies,
        tracks=tracks,
        declared_frame=int(declared),
        rotation=rotation,
        report=report,
        features=features,
        step=step,
    )


SIDE_COLORS = {  # BGR
    "Offense": (235, 150, 40),
    "Defense": (40, 60, 230),
    "line": (170, 170, 170),
    "other": (255, 255, 255),
}


def annotate_snap(clip: TrackedClip, read: ClipRead) -> np.ndarray:
    """The snap frame with every body boxed by side, the LOS drawn and the ball marked.

    The one-glance check of the formation read: blue offense, red defense, gray
    line players (dropped from the coverage read), white officials and others.
    """
    fr = clip.frames[read.snap_index]
    calib = clip.calibrations[read.snap_index]
    out = fr.image.copy()
    hinv = np.linalg.inv(calib.matrix)

    def px(x: float, y: float) -> tuple[int, int]:
        p = hinv @ np.array([x, y, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])

    b = read.bodies
    for tid, box, isp in zip(b.ids, b.boxes, b.is_player):
        side = read.sides.side_by_id.get(int(tid))
        key = (
            "other"
            if not isp or side is None
            else "line"
            if int(tid) in read.sides.line_ids
            else side
        )
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), SIDE_COLORS[key], 3)
    los_pts = [px(read.los, y) for y in np.linspace(0, FIELD_WIDTH, 20)]
    for a, c in pairwise(los_pts):
        cv2.line(out, a, c, (0, 230, 255), 3, cv2.LINE_AA)
    if read.ball_y is not None:
        cv2.circle(out, px(read.los, read.ball_y), 12, (0, 230, 255), -1)
    text = f"snap {read.snap_t:.1f}s | offense {read.offense} | LOS x={read.los:.1f}" + (
        f" | ball y={read.ball_y:.1f}" if read.ball_y is not None else ""
    )
    cv2.putText(out, text, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(out, text, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 255), 2, cv2.LINE_AA)
    return out
