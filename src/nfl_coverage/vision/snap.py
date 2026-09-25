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
BODY_DISP = 0.3  # a body has "moved" when it is this far from its window mean
FIRE_DISP = 0.3  # floor on the line's mean displacement that counts as firing
NOISE_K = 4.0  # ... raised to the pre-fire mean + NOISE_K sd on a jittery clip
FIRE_FRAC = 0.4  # this share of the line past BODY_DISP, at the fire or the frame after
SUSTAIN = 2  # the fire must hold this many consecutive frames
MIN_LINE = 4  # set bodies in the LOS band the frame before the fire
MIN_TRACKED = 3  # line bodies still tracked while it fires (they merge once engaged)
NOISE_LAG = 3  # frames between the noise window and a fire candidate
STILL_SHARE = 0.5  # the line counts as still below this share of the fire threshold
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
    fire_disp: float = FIRE_DISP,
    noise_k: float = NOISE_K,
    fire_frac: float = FIRE_FRAC,
    sustain: int = SUSTAIN,
    min_line: int = MIN_LINE,
    min_tracked: int = MIN_TRACKED,
    depth: float = LINE_DEPTH,
    halfwidth: float = LINE_HALFWIDTH,
) -> SnapDetection:
    """First frame where a set line fires together; the snap is the last still frame before it.

    A set line = at least `min_line` bodies in the LOS band, tracked over the
    trailing `window` frames, the frame before the fire. It fires when the line's
    mean displacement from its window mean jumps past `fire_disp` (or past its own
    pre-fire noise, mean + `noise_k` sd, on a jittery clip) and stays there for
    `sustain` frames, with `fire_frac` of the bodies individually past
    `body_disp`. The mean catches the first 0.1-0.2 s of a pass set, when each
    lineman has moved only a few tenths of a yard; the per-body share keeps one
    walking body (a shift, a man in motion) from passing as the line. Once the
    lines engage their boxes merge, so only `min_tracked` bodies need to remain.
    The snap is then the last frame before the fire at which the line's mean
    displacement was still below half the fire threshold, so rel_frame 0 is the
    frame the line is still at and moving right after, as on tracking data.
    """
    series = snap_series(pos, los, ball_xy, window, body_disp, depth, halfwidth)
    frames = series["frame"].to_numpy()
    moving = series["line_moving"].to_numpy()
    disp_s = series["line_disp"].to_numpy()
    n_line = series["n_line"].to_numpy()
    by_frame = {f: d for f, d in pos.groupby("frame")} if not pos.empty else {}

    def threshold_noise(k: int) -> float:
        # noise from a window that ends NOISE_LAG frames back: the frames right
        # before a candidate may already hold the start of the ramp, and a
        # threshold estimated on them climbs as fast as the signal
        prior = disp_s[max(0, k - NOISE_LAG - window) : max(0, k - NOISE_LAG)]
        prior = prior[np.isfinite(prior)]
        if len(prior) < 3:
            return 0.0
        med = float(np.median(prior))
        sd = 1.4826 * float(np.median(np.abs(prior - med)))  # robust to one spike
        return med + noise_k * sd

    def threshold(k: int) -> float:
        return max(fire_disp, threshold_noise(k))

    fire_k = None
    thr = fire_disp
    for k in range(1, len(series) - sustain + 1):
        if n_line[k - 1] < min_line:
            continue
        thr = threshold(k)
        run = range(k, k + sustain)
        if any(n_line[m] < min_tracked or not (disp_s[m] >= thr) for m in run):
            continue
        if not np.nanmax(np.r_[moving[k : k + sustain], 0.0]) >= fire_frac:
            continue
        fire_k = k
        break
    if fire_k is None:
        n_max = int(n_line.max()) if len(n_line) else 0
        why = (
            f"no frame with {min_line}+ set line bodies"
            if n_max < min_line
            else f"the set line never fired together (max mean displacement "
            f"{np.nanmax(np.r_[disp_s, 0.0]):.2f} yd, max moving share "
            f"{np.nanmax(np.r_[moving, 0.0]):.2f})"
        )
        return SnapDetection(None, None, "low", 0.0, None, n_max, [why], series)

    fire = int(frames[fire_k])
    # the snap = the last frame before the fire at which the line was still:
    # walk back while its mean displacement is above the still level (half the
    # fire threshold, or the pre-fire noise). A shape fit per body (flat, then a
    # quadratic ramp) was tried first and placed clip 2 0.4 s early: three or
    # four points per body cannot carry a shape against tracker noise.
    lw = _line_window(pos, fire, los, window, body_disp, depth, halfwidth, MIN_PRESENCE)
    still_level = max(STILL_SHARE * fire_disp, threshold_noise(fire_k))
    snap_k = fire_k - 1
    while snap_k > 0 and np.isfinite(disp_s[snap_k]) and disp_s[snap_k] > still_level:
        snap_k -= 1
    snap = int(frames[snap_k])
    t = float(pos.loc[pos["frame"] == snap, "t"].iloc[0]) if snap in by_frame else None

    # confidence: how many bodies, how completely they fired, how still they were
    # before the snap (not before the fire: the fire can lag the snap by several
    # frames while linemen set, and those frames are already moving)
    n_at = len(lw.ids) if lw is not None else int(n_line[fire_k - 1])
    frac = float(np.nanmax(moving[fire_k : fire_k + sustain]))
    pre = series.loc[
        (series["frame"] <= snap) & (series["frame"] > snap - window), "line_disp"
    ].dropna()
    stillness = 1.0 - min(1.0, float(pre.mean()) / thr) if len(pre) else 0.5
    terms = [min(1.0, max(0.0, (n_at - 3) / 4.0)), frac, stillness]
    reasons = [
        (
            f"{n_at} bodies set in the LOS band (LOS x={series['los'].iloc[fire_k]:.1f}), "
            f"{frac:.0%} fired together at frame {fire} ({series['t'].iloc[fire_k]:.1f} s), "
            f"mean displacement {disp_s[fire_k]:.2f} yd past the {thr:.2f} yd threshold"
        ),
        f"line displacement over the {window * step_s:.1f} s up to the snap: "
        f"{pre.mean():.2f} yd mean"
        if len(pre)
        else "no still stretch before the snap",
        f"the fire confirmed {fire - snap} frame(s) after the snap frame",
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
