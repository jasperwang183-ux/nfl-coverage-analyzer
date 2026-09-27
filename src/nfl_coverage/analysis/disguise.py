"""Disguise components: canonical frames in, one scored row per play out.

Every component has the same shape so the harness can run, compare and
aggregate them without knowing what they measure:

    component(frames, **deps) -> DataFrame[KEYS + <name>_score, <name>_label, evidence...]

`<name>_score` is in [0, 1] (NaN when the play cannot be scored), `<name>_label`
names the mechanism seen ('' when none). Each mechanism a DC uses to disguise
gets its own component (see docs/DISGUISE_PLAN.md); this module holds the two
that exist so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_coverage.analysis.shell import DEEP_DEPTH, deep_defenders, shell_state
from nfl_coverage.features.alignment import KEYS
from nfl_coverage.models.persist import predict_proba

RUN_FRAME = 20  # 2.0 s post-snap (or the throw if earlier): calibrated in docs/DISGUISE_PLAN.md
MIN_RUN_FRAME = 10  # a play that ends before 1.0 s cannot be read
RUN_DEEP_DEPTH = 14.0  # by 2 s every zone dropper is past 9 yd; deep responsibility is deeper
ROTATION_MIN_MOVE = 3.0  # yards a defender must move for a shell change to count as a rotation

# score per label: real shell lies score 1, mechanics a QB expects score low
ROTATION_SCORES = {
    "rotate_to_mof_closed": 1.0,  # two-high look -> post safety (Cover 3 sky/buzz, Cover 1)
    "rotate_to_mof_open": 1.0,  # single-high look -> split safeties (inverted Cover 2 / Tampa)
    "rotation_down": 1.0,  # a deep defender comes down, deep count drops (robber, buzz)
    "rotation_up": 1.0,  # a non-corner bails from the box/slot into a deep zone
    "corner_bail": 0.25,  # corners bail to thirds: resolves Cover 3 vs 1, not a shell lie
    "threshold_flicker": 0.25,  # shell label changed but nobody moved 3 yd
    "": 0.0,
}


def _run_frame_per_play(frames: pd.DataFrame, run_frame: int = RUN_FRAME) -> pd.Series:
    """The frame to read the played shell at: RUN_FRAME, or the last one if shorter."""
    last = frames.groupby(KEYS)["rel_frame"].max()
    return last.clip(upper=run_frame).rename("run_frame")


def _at_run_frame(frames: pd.DataFrame, run_frames: pd.Series, reader, **kw) -> pd.DataFrame:
    """Apply a per-frame reader at each play's own run frame."""
    parts = []
    for rf, keys in run_frames.groupby(run_frames).groups.items():
        sub = frames.set_index(KEYS).loc[list(keys)].reset_index()
        parts.append(reader(sub, rel_frame=int(rf), **kw))
    return pd.concat(parts, ignore_index=True) if parts else reader(frames.iloc[0:0], **kw)


def shell_rotation(
    frames: pd.DataFrame,
    run_frame: int = RUN_FRAME,
    run_deep_depth: float = RUN_DEEP_DEPTH,
    center: float | None = None,
    los: float | None = None,
) -> pd.DataFrame:
    """Shell at the snap vs. shell at +1.5 s; the most common disguise.

    Evidence per play: shown/run shell, who moved most (`rotator_id`,
    `rotator_dd`), and how many deep-at-snap defenders came down
    (`n_came_down`), non-corners bailed deep (`n_bailed_inside`) and corners
    bailed deep (`n_corner_bail`). The label is the first matching mechanism in
    ROTATION_SCORES order; the score is that label's weight.
    """
    run_frames = _run_frame_per_play(frames, run_frame)
    shown = shell_state(frames, 0, center=center, los=los).set_index(KEYS)
    run = _at_run_frame(
        frames, run_frames, shell_state, deep_depth=run_deep_depth, center=center, los=los
    ).set_index(KEYS)

    d0 = deep_defenders(frames, 0, center=center, los=los).set_index(KEYS + ["nfl_id"])
    d1 = _at_run_frame(
        frames, run_frames, deep_defenders, deep_depth=run_deep_depth, center=center, los=los
    ).set_index(KEYS + ["nfl_id"])
    both = d0[["depth", "deep", "widest"]].join(
        d1[["depth", "deep"]].rename(columns={"depth": "depth1", "deep": "deep1"}), how="inner"
    )
    both["dd"] = both["depth1"] - both["depth"]
    both["came_down"] = both["deep"] & (both["dd"] <= -ROTATION_MIN_MOVE)
    both["bailed"] = ~both["deep"] & both["deep1"] & (both["dd"] >= ROTATION_MIN_MOVE)
    both["bailed_inside"] = both["bailed"] & ~both["widest"]
    both["corner_bail"] = both["bailed"] & both["widest"]
    flat = both.reset_index()
    per_play = flat.groupby(KEYS).agg(
        n_came_down=("came_down", "sum"),
        n_bailed_inside=("bailed_inside", "sum"),
        n_corner_bail=("corner_bail", "sum"),
        max_move=("dd", lambda s: float(s.abs().max())),
    )
    biggest = flat.loc[flat.groupby(KEYS)["dd"].apply(lambda s: s.abs().idxmax())]
    biggest = biggest.set_index(KEYS)[["nfl_id", "dd"]].rename(
        columns={"nfl_id": "rotator_id", "dd": "rotator_dd"}
    )

    out = (
        shown[["shell", "n_deep", "mof_closed"]]
        .add_prefix("shown_")
        .join(run[["shell", "n_deep", "mof_closed"]].add_prefix("run_"), how="left")
        .join(run_frames)
        .join(per_play)
        .join(biggest)
    )
    changed = (out["shown_n_deep"] != out["run_n_deep"]) | (
        out["shown_mof_closed"] != out["run_mof_closed"]
    )
    moved = out["max_move"] >= ROTATION_MIN_MOVE
    few = out["run_n_deep"] <= 3  # with four deep the middle is shared: no MOF verdict to flip
    label = np.select(
        [
            changed & moved & few & ~out["shown_mof_closed"] & out["run_mof_closed"],
            changed & moved & few & out["shown_mof_closed"] & ~out["run_mof_closed"],
            changed & (out["n_came_down"] > 0),
            changed & (out["n_bailed_inside"] > 0),
            changed & (out["n_corner_bail"] > 0),
            changed,
        ],
        [
            "rotate_to_mof_closed",
            "rotate_to_mof_open",
            "rotation_down",
            "rotation_up",
            "corner_bail",
            "threshold_flicker",
        ],
        default="",
    )
    score = np.array([ROTATION_SCORES[lb] for lb in label])
    unreadable = (
        out["run_frame"].isna() | (out["run_frame"] < MIN_RUN_FRAME) | out["run_shell"].isna()
    )
    out["shell_rotation_score"] = np.where(unreadable, np.nan, score)
    out["shell_rotation_label"] = np.where(unreadable, "insufficient_frames", label)
    return out.reset_index()


def declared_frame(
    frames: pd.DataFrame,
    run_frame: int = RUN_FRAME,
    deep_depth: float = RUN_DEEP_DEPTH,
    min_defenders: int = 5,
    center: float | None = None,
    los: float | None = None,
    max_frame: int | None = None,
) -> int:
    """The rel_frame at which the played shell has declared: most deep defenders.

    Coverage responsibilities appear over a window, not at an instant — a corner
    bailing to a third crosses the deep line later than a safety who was already
    there. Among frames from `run_frame` to the end that still carry at least
    `min_defenders` tracked defenders (tracks get lost as the camera zooms),
    take the one with the most defenders past `deep_depth`; ties go to the
    earliest; none past `max_frame`. Film reads use this; the tracking-data
    harness keeps the fixed 2.0 s read it was calibrated on.
    """
    last = int(frames["rel_frame"].max())
    if max_frame is not None:
        last = min(last, max_frame)
    best_rf, best_n = run_frame, -1
    for rf in range(min(run_frame, last), last + 1):
        d = deep_defenders(frames, rf, deep_depth=deep_depth, center=center, los=los)
        if len(d) < min_defenders:
            continue
        n = int(d["deep"].sum())
        if n > best_n:
            best_rf, best_n = rf, n
    return best_rf


def shell_change_label(
    shown_n_deep: int, shown_mof_closed: bool, run_n_deep: int, run_mof_closed: bool
) -> tuple[str, float]:
    """Rotation label from two unlinked shell reads (a clip without tracking).

    Same vocabulary as `shell_rotation`, but nobody can be named as the rotator
    and a threshold flicker cannot be told from a real move, so deep-count-only
    changes score 0.5 instead of 1.0. MOF flips score 1.0 either way.
    """
    if shown_mof_closed != run_mof_closed:
        return ("rotate_to_mof_closed" if run_mof_closed else "rotate_to_mof_open"), 1.0
    if run_n_deep < shown_n_deep:
        return "rotation_down", 0.5
    if run_n_deep > shown_n_deep:
        return "rotation_up", 0.5
    return "", 0.0


def model_disagreement(
    features: pd.DataFrame, bundle: dict, label_col: str = "coverage_type"
) -> pd.DataFrame:
    """The shown model (snap features -> run label) confident and wrong.

    Score is the model's confidence in its top class when that class is not the
    coverage played, else 0; label is 'SHOWN->RAN'. This is the statistical
    reading of disguise (the alignment usually means something else), kept as a
    component beside the football reading, not as the definition.
    """
    proba = predict_proba(bundle, features)
    top = proba.idxmax(axis=1).to_numpy()
    conf = proba.max(axis=1).to_numpy()
    ran = features[label_col].to_numpy()
    wrong = (top != ran) & pd.notna(ran)
    out = features[KEYS].copy()
    out["shown_pred"] = top
    out["shown_conf"] = conf
    out["model_disagreement_score"] = np.where(pd.notna(ran), np.where(wrong, conf, 0.0), np.nan)
    out["model_disagreement_label"] = np.where(
        wrong, pd.Series(top).str.cat(pd.Series(ran).astype(str), sep="->"), ""
    )
    return out


COMPONENTS = ["shell_rotation", "model_disagreement"]
SNAP_DEEP_DEPTH = DEEP_DEPTH
