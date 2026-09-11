"""Movement features over frame windows.

post_snap_features: how the coverage *reveals* itself (zone drops, man mirroring,
safety rotation) between the snap and ~2.5 s later. Works for both datasets.

pre_snap_features: what moved in the second before the snap (late rotation, motion
response). Only meaningful for bdb2021, which has rel_frame < 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["dataset", "game_id", "play_id"]


def _window(frames: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return frames[(frames["rel_frame"] >= lo) & (frames["rel_frame"] <= hi)]


def _per_defender_delta(win: pd.DataFrame) -> pd.DataFrame:
    """First->last displacement per defender within the window."""
    d = win[win["side"] == "Defense"].sort_values("rel_frame")
    g = d.groupby(KEYS + ["nfl_id"])
    out = pd.DataFrame(
        {
            "dx": g["x"].last() - g["x"].first(),
            "dy": g["y"].last() - g["y"].first(),
            "mean_s": g["s"].mean(),
        }
    )
    out["disp"] = np.hypot(out["dx"], out["dy"])
    return out.reset_index()


def _man_mirror_frac(win: pd.DataFrame) -> pd.Series:
    """Fraction of defenders whose nearest non-QB offensive player stays close.

    Man defenders keep a small min-distance to their assignment throughout; zone
    defenders separate. Computed as: defender's mean over frames of the distance to
    the nearest offensive player, then the per-play fraction with mean < 4 yd.
    """
    defense = win[win["side"] == "Defense"][KEYS + ["rel_frame", "nfl_id", "x", "y"]]
    offense = win[(win["side"] == "Offense") & (win["position"] != "QB")][
        KEYS + ["rel_frame", "x", "y"]
    ]
    merged = defense.merge(offense, on=KEYS + ["rel_frame"], suffixes=("", "_o"))
    merged["dist"] = np.hypot(merged["x"] - merged["x_o"], merged["y"] - merged["y_o"])
    per_frame_min = merged.groupby(KEYS + ["nfl_id", "rel_frame"])["dist"].min()
    per_def_mean = per_frame_min.groupby(KEYS + ["nfl_id"]).mean()
    return per_def_mean.groupby(KEYS).apply(lambda s: (s < 4.0).mean())


def post_snap_features(frames: pd.DataFrame, horizon: int = 25) -> pd.DataFrame:
    """Per-play features from rel_frame 1..horizon (default 2.5 s post-snap)."""
    win = _window(frames, 0, horizon)
    deltas = _per_defender_delta(win)
    g = deltas.groupby(KEYS)
    feats = pd.DataFrame(
        {
            "post_max_depth_gain": g["dx"].max(),
            "post_n_dropping_5": g.apply(lambda d: (d.dx > 5).sum(), include_groups=False),
            "post_n_dropping_10": g.apply(lambda d: (d.dx > 10).sum(), include_groups=False),
            "post_mean_disp": g["disp"].mean(),
            "post_mean_speed": g["mean_s"].mean(),
            "post_lateral_std": g["dy"].std(),
        }
    )
    feats["post_man_mirror_frac"] = _man_mirror_frac(win).reindex(feats.index)
    return feats.reset_index()


def pre_snap_features(frames: pd.DataFrame, depth: int = 10) -> pd.DataFrame:
    """Per-play features from rel_frame -depth..0 (bdb2021: the last second)."""
    win = _window(frames, -depth, 0)
    deltas = _per_defender_delta(win)
    g = deltas.groupby(KEYS)
    feats = pd.DataFrame(
        {
            "pre_max_disp": g["disp"].max(),
            "pre_mean_disp": g["disp"].mean(),
            "pre_max_depth_change": g["dx"].max(),
            "pre_min_depth_change": g["dx"].min(),
            "pre_n_moving_2": g.apply(lambda d: (d.disp > 2).sum(), include_groups=False),
            "pre_max_speed": g["mean_s"].max(),
        }
    )
    return feats.reset_index()
