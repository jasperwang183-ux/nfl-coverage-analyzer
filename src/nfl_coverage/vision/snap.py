"""Snap detection from field-space tracks (rung 3).

Where it sits in the clip pipeline:

    track people -> chain the calibration (anchored on any still frame)
        -> **detect the snap** -> assign sides at the snap frame -> canonical tracks

It runs on unsided field-space positions: sides and line players are assigned
*at* the snap frame, so the detector cannot depend on either.

The primary signal is the offensive line firing. Five bodies are stationary by
rule for at least a second before the snap and all fire in the same frame;
nothing else in a play looks like that. A man in motion is one body; a shift is
one or two. The defense is excluded on purpose: their pre-snap movement is the
disguise the rest of the pipeline measures. Both lines straddle the line of
scrimmage, so the "line" here is every body in the LOS band (offense and
defense) - more bodies, same event.

Working in yards rather than pixels makes the thresholds camera-invariant: a
zoom changes pixel motion, not yards, and a pan during the dwell moves every
pixel but no player.

The ball, when a detector exists, is a confirmation signal (`ball_xy`): it
leaves the center's hands at the snap. It is optional because a pretrained
person detector does not see it and it is occluded under center anyway.

The detector is a rule on a per-frame feature series (`snap_series`) - the
series a learned model would consume once enough confirmed clips exist. Keep
the interface (series in, snap frame + confidence out) when that model lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nfl_coverage.vision.homography import FieldCalibration, on_field

# defaults (yards, frames at the 10 Hz sampling step)
STILL_FRAMES = 5  # trailing window the line must have been set over (0.5 s)
BODY_DISP = 0.5  # a body has "moved" when it is this far from its window mean
FIRE_FRAC = 0.6  # this fraction of the set line moving together = the snap
SUSTAIN = 2  # the fire must hold this many consecutive frames
MIN_LINE = 4  # fewer set bodies in the LOS band = no formation to read
LINE_DEPTH = 2.5  # LOS band half-depth (same rule as still.line_player_mask)
LINE_HALFWIDTH = 4.5  # LOS band half-width about the line's y center
MIN_PRESENCE = 0.6  # a body must be tracked in this share of the window
SIDELINE_Y_MARGIN = -0.5


def field_positions(frames, calibs: list[FieldCalibration]) -> pd.DataFrame:
    """Foot points of every tracked, unclipped, on-field body in field coordinates.

    `frames` are `track.TrackedFrame`s. Columns: frame, t, nfl_id (track id), x, y.
    Field coordinates are raw (not offense-standardized); the snap is direction-free.
    """
    rows = []
    for fr, calib in zip(frames, calibs):
        if len(fr.boxes) == 0:
            continue
        keep = (fr.boxes[:, 3] < fr.image.shape[0] - 4) & (fr.ids >= 0)
        if not keep.any():
            continue
        boxes, ids = fr.boxes[keep], fr.ids[keep]
        feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]).astype(np.float32)
        xy = calib.to_field(feet)
        onf = on_field(xy, y_margin=SIDELINE_Y_MARGIN)
        for tid, (x, y) in zip(ids[onf], xy[onf]):
            rows.append({"frame": fr.index, "t": fr.t, "nfl_id": int(tid), "x": x, "y": y})
    cols = ["frame", "t", "nfl_id", "x", "y"]
    return pd.DataFrame(rows, columns=cols).astype({"x": float, "y": float})


def estimate_los(x: np.ndarray, depth: float = LINE_DEPTH) -> float | None:
    """The x with the most bodies within `depth` yards: both lines straddle the LOS."""
    if len(x) == 0:
        return None
    counts = np.array([(np.abs(x - c) < depth).sum() for c in x])
    best = x[counts == counts.max()]
    group = x[np.abs(x - np.median(best)) < depth]
    return float(np.median(group))


@dataclass
class _LineWindow:
    """The set line at one frame: bodies in the LOS band over the trailing window."""

    ids: np.ndarray
    ref: np.ndarray  # (n, 2) window-mean positions
    still: np.ndarray  # (n,) window position spread below the body threshold
    los: float


def _line_window(
    pos: pd.DataFrame,
    i: int,
    los: float | None,
    window: int,
    body_disp: float,
    depth: float,
    halfwidth: float,
    min_presence: float,
) -> _LineWindow | None:
    w = pos[(pos["frame"] >= i - window) & (pos["frame"] < i)]
    if w.empty:
        return None
    g = w.groupby("nfl_id")
    presence = g.size() / window
    mean = g[["x", "y"]].mean()
    spread = g[["x", "y"]].std().fillna(0.0)
    ok = presence >= min_presence
    mean, spread = mean[ok], spread[ok]
    if mean.empty:
        return None
    los_ = los if los is not None else estimate_los(mean["x"].to_numpy(), depth)
    if los_ is None:
        return None
    near = (mean["x"] - los_).abs() < depth
    if not near.any():
        return None
    cy = mean.loc[near, "y"].median()
    band = near & ((mean["y"] - cy).abs() < halfwidth)
    if not band.any():
        return None
    ids = mean.index[band].to_numpy()
    ref = mean.loc[band, ["x", "y"]].to_numpy()
    still = (np.hypot(spread.loc[band, "x"], spread.loc[band, "y"]) < body_disp).to_numpy()
    return _LineWindow(ids, ref, still, los_)


def _displacements(pos_i: pd.DataFrame, lw: _LineWindow) -> np.ndarray:
    """Per set-line body: distance at this frame from its window mean (NaN if not tracked)."""
    here = pos_i.set_index("nfl_id")[["x", "y"]]
    out = np.full(len(lw.ids), np.nan)
    for k, tid in enumerate(lw.ids):
        if tid in here.index:
            out[k] = float(np.hypot(*(here.loc[tid].to_numpy() - lw.ref[k])))
    return out


def snap_series(
    pos: pd.DataFrame,
    los: float | None = None,
    ball_xy: dict[int, tuple[float, float]] | None = None,
    window: int = STILL_FRAMES,
    body_disp: float = BODY_DISP,
    depth: float = LINE_DEPTH,
    halfwidth: float = LINE_HALFWIDTH,
    min_presence: float = MIN_PRESENCE,
) -> pd.DataFrame:
    """Per-frame snap features from field positions (the input a learned detector would take).

    Columns, one row per frame index present in `pos`:
      n_line          set-line bodies (in the LOS band over the trailing window, tracked now)
      n_still         of those, bodies whose window spread is below `body_disp`
      line_disp       mean displacement of the set line from its window mean (yd)
      line_moving     fraction of the set line displaced more than `body_disp`
      all_disp        mean displacement of every tracked body from its window mean (yd)
      ball_disp       ball displacement from its window mean (yd), NaN without a ball
      los             LOS used (given, or estimated from the densest band)
    """
    frames = sorted(pos["frame"].unique()) if not pos.empty else []
    by_frame = {f: d for f, d in pos.groupby("frame")}
    rows = []
    for i in frames:
        row = {
            "frame": i,
            "t": float(by_frame[i]["t"].iloc[0]),
            "n_line": 0,
            "n_still": 0,
            "line_disp": np.nan,
            "line_moving": np.nan,
            "all_disp": np.nan,
            "ball_disp": np.nan,
            "los": np.nan,
        }
        w = pos[(pos["frame"] >= i - window) & (pos["frame"] < i)]
        if not w.empty:
            means = w.groupby("nfl_id")[["x", "y"]].mean()
            here = by_frame[i].set_index("nfl_id")[["x", "y"]]
            common = means.index.intersection(here.index)
            if len(common):
                d = np.hypot(*(here.loc[common].to_numpy() - means.loc[common].to_numpy()).T)
                row["all_disp"] = float(d.mean())
        lw = _line_window(pos, i, los, window, body_disp, depth, halfwidth, min_presence)
        if lw is not None:
            d = _displacements(by_frame[i], lw)
            got = ~np.isnan(d)
            row["n_line"] = int(got.sum())
            row["n_still"] = int((got & lw.still).sum())
            row["los"] = lw.los
            if got.any():
                row["line_disp"] = float(d[got].mean())
                row["line_moving"] = float((d[got] > body_disp).mean())
        if ball_xy:
            prev = [ball_xy[j] for j in range(i - window, i) if j in ball_xy]
            if prev and i in ball_xy:
                ref = np.mean(np.array(prev, dtype=float), axis=0)
                row["ball_disp"] = float(np.hypot(*(np.array(ball_xy[i], dtype=float) - ref)))
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "frame",
            "t",
            "n_line",
            "n_still",
            "line_disp",
            "line_moving",
            "all_disp",
            "ball_disp",
            "los",
        ],
    )


def fit_onset(disp: np.ndarray) -> int:
    """Change point of a flat-then-ramp series: index j minimizing the least-squares error
    of `disp[m] = 0 for m <= j, c * (m - j)^2 for m > j` (constant acceleration from rest).

    Returns the last flat index; the ramp starts right after it.
    """
    n = len(disp)
    best_j, best_sse = 0, np.inf
    for j in range(n):
        flat = disp[: j + 1]
        ramp = disp[j + 1 :]
        sse = float((flat**2).sum())
        if len(ramp):
            k = np.arange(1, len(ramp) + 1, dtype=float) ** 2
            c = max(0.0, float((ramp * k).sum() / (k * k).sum()))
            sse += float(((ramp - c * k) ** 2).sum())
        if sse < best_sse - 1e-12:
            best_j, best_sse = j, sse
    return best_j


@dataclass
class SnapDetection:
    frame: int | None  # kept-frame index of the snap (last still frame), None if not found
    t: float | None
    confidence: str  # high / medium / low
    score: float
    fire_frame: int | None  # first frame the set line was moving together
    n_line: int
    reasons: list[str] = field(default_factory=list)
    series: pd.DataFrame | None = None


def detect_snap(
    pos: pd.DataFrame,
    los: float | None = None,
    ball_xy: dict[int, tuple[float, float]] | None = None,
    step_s: float = 0.1,
    window: int = STILL_FRAMES,
    body_disp: float = BODY_DISP,
    fire_frac: float = FIRE_FRAC,
    sustain: int = SUSTAIN,
    min_line: int = MIN_LINE,
    depth: float = LINE_DEPTH,
    halfwidth: float = LINE_HALFWIDTH,
) -> SnapDetection:
    """First frame where a set line fires together; the snap is the last still frame before it.

    A set line = at least `min_line` bodies in the LOS band that were tracked over
    the trailing `window` frames. It fires when `fire_frac` of them are more than
    `body_disp` yards from their window mean for `sustain` consecutive frames -
    simultaneity across bodies is what separates the snap from detection jitter
    (independent per body), a man in motion or a shift (one or two bodies).
    Within the fire's window the snap is placed by a flat-then-ramp fit of the
    line's mean displacement, so rel_frame 0 is the frame the line is still at
    and moving right after, as on tracking data.
    """
    series = snap_series(pos, los, ball_xy, window, body_disp, depth, halfwidth)
    frames = series["frame"].to_numpy()
    moving = series["line_moving"].to_numpy()
    n_line = series["n_line"].to_numpy()
    by_frame = {f: d for f, d in pos.groupby("frame")} if not pos.empty else {}

    fire_k = None
    for k in range(len(series) - sustain + 1):
        ok = True
        for m in range(sustain):
            if n_line[k + m] < min_line or not (moving[k + m] >= fire_frac):
                ok = False
                break
        if ok:
            fire_k = k
            break
    if fire_k is None:
        n_max = int(n_line.max()) if len(n_line) else 0
        why = (
            f"no frame with {min_line}+ set line bodies"
            if n_max < min_line
            else f"the set line never fired together (max moving fraction "
            f"{np.nanmax(moving) if len(moving) else 0:.2f} < {fire_frac})"
        )
        return SnapDetection(None, None, "low", 0.0, None, n_max, [why], series)

    fire = int(frames[fire_k])
    # place the snap inside the fire's window with the line set that fired
    lw = _line_window(pos, fire, los, window, body_disp, depth, halfwidth, MIN_PRESENCE)
    span = [f for f in range(fire - 2 * window, fire + sustain) if f in by_frame]
    if lw is not None and span:
        # reference = the earliest frames of the span, not the window mean (the
        # mean already contains the first frames of the ramp). One onset per body,
        # the snap is their median: a body that was already walking (a shift, a
        # man in motion) or a lost track is one outlier, not a bias.
        early = span[: max(1, min(3, len(span) - sustain - 1))]
        table = np.stack(
            [by_frame[f].set_index("nfl_id")[["x", "y"]].reindex(lw.ids).to_numpy() for f in span]
        )  # (frames, bodies, 2)
        ref_pts = np.nanmean(table[: len(early)], axis=0)  # (bodies, 2)
        disp = np.hypot(*(table - ref_pts).transpose(2, 0, 1))  # (frames, bodies)
        present = np.isfinite(disp).mean(axis=0) >= 0.8
        onsets = [fit_onset(np.nan_to_num(disp[:, k])) for k in np.flatnonzero(present)]
        if len(onsets) >= 2:
            snap = span[int(np.median(onsets))]
        else:
            mean_disp = np.nan_to_num(np.nanmean(disp, axis=1)) if present.any() else disp[:, 0]
            snap = span[fit_onset(np.nan_to_num(mean_disp))]
        snap = min(snap, fire - 1) if fire - 1 in by_frame else snap
    else:
        snap = fire - 1
    snap = int(snap)
    t = float(pos.loc[pos["frame"] == snap, "t"].iloc[0]) if snap in by_frame else None

    # confidence: how many bodies, how completely they fired, how still they were before
    n_at = int(n_line[fire_k])
    frac = float(np.nanmax(moving[fire_k : fire_k + sustain]))
    pre = series.loc[(series["frame"] < fire) & (series["frame"] >= fire - window), "line_disp"]
    stillness = 1.0 - min(1.0, float(pre.mean()) / body_disp) if len(pre) else 0.5
    terms = [min(1.0, max(0.0, (n_at - 3) / 4.0)), frac, stillness]
    reasons = [
        (
            f"{n_at} bodies set in the LOS band (LOS x={series['los'].iloc[fire_k]:.1f}), "
            f"{frac:.0%} fired together at frame {fire} ({series['t'].iloc[fire_k]:.1f} s)"
        ),
        f"line displacement over the {window * step_s:.1f} s before: {pre.mean():.2f} yd mean"
        if len(pre)
        else "no still stretch before the fire",
    ]
    if ball_xy:
        after = series.loc[(series["frame"] >= fire) & (series["frame"] < fire + 5), "ball_disp"]
        if after.notna().any():
            moved = float(after.max())
            if moved >= 1.0:
                terms.append(1.0)
                reasons.append(f"ball moved {moved:.1f} yd within 0.5 s of the fire: confirms")
            else:
                terms.append(0.0)
                reasons.append(f"ball moved only {moved:.1f} yd after the fire: contradicts")
    score = float(np.mean(terms))
    confidence = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return SnapDetection(snap, t, confidence, score, fire, n_at, reasons, series)
