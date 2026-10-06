"""Static alignment features from a single frame (default: the snap, rel_frame 0).

These answer "what is the defense showing?" — safety shell, depth structure, press
looks — and are computable for both datasets (both have the snap frame).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["dataset", "game_id", "play_id"]


def _los_estimate(offense: pd.DataFrame) -> pd.Series:
    """Line of scrimmage per play ≈ 75th percentile of offensive x at the frame.

    Receivers align on the LOS while the QB/backs sit behind it, so a high quantile
    tracks the LOS well without needing ball tracking (absent in bdb2026).
    """
    return offense.groupby(KEYS)["x"].quantile(0.75).rename("los")


def snap_los(offense_snap: pd.DataFrame, ball_spots: pd.DataFrame | None = None) -> pd.Series:
    """LOS per play: the ball spot when `ball_spots` is given, else the stand-in.

    `ball_spots` is `data.ball_spots.load_ball_spots()` (KEYS + ball_x). A play missing
    from it gets NaN rather than the stand-in, so the two zeros never mix in one table.
    """
    stand_in = _los_estimate(offense_snap)
    if ball_spots is None:
        return stand_in
    return ball_spots.set_index(KEYS)["ball_x"].reindex(stand_in.index).rename("los")


def snapshot_features(frames: pd.DataFrame, rel_frame: int = 0) -> pd.DataFrame:
    """Per-play alignment features at one rel_frame. Returns one row per play."""
    snap = frames[frames["rel_frame"] == rel_frame]
    offense = snap[snap["side"] == "Offense"]
    defense = snap[snap["side"] == "Defense"].copy()

    los = _los_estimate(offense)
    defense = defense.join(los, on=KEYS)
    defense["depth"] = defense["x"] - defense["los"]

    g = defense.groupby(KEYS)
    feats = pd.DataFrame(
        {
            "n_defenders": g.size(),
            "n_deep_8": g.apply(lambda d: (d.depth >= 8).sum(), include_groups=False),
            "n_deep_12": g.apply(lambda d: (d.depth >= 12).sum(), include_groups=False),
            "n_box_5": g.apply(lambda d: (d.depth < 5).sum(), include_groups=False),
            "max_depth": g["depth"].max(),
            "mean_depth": g["depth"].mean(),
            "depth_2nd": g["depth"].apply(lambda s: s.nlargest(2).min()),
            "deep_y_spread": g.apply(
                lambda d: d.loc[d.depth >= 8, "y"].std() if (d.depth >= 8).sum() >= 2 else 0.0,
                include_groups=False,
            ),
            "def_y_std": g["y"].std(),
        }
    )

    # press count: defenders within 1.5 yd of any non-QB offensive player
    non_qb = offense[offense["position"] != "QB"]
    press = (
        _pairwise_min_dist(defense, non_qb)
        .groupby(KEYS)
        .apply(lambda s: (s < 1.5).sum(), include_groups=False)
    )
    feats["n_press"] = press.reindex(feats.index).fillna(0)

    prefix = f"align{rel_frame}_" if rel_frame != 0 else "snap_"
    return feats.add_prefix(prefix).reset_index()


def _pairwise_min_dist(defense: pd.DataFrame, offense: pd.DataFrame) -> pd.Series:
    """Distance from each defender to the nearest given offensive player (per play)."""
    merged = defense[KEYS + ["nfl_id", "x", "y"]].merge(
        offense[KEYS + ["x", "y"]], on=KEYS, suffixes=("", "_o")
    )
    merged["dist"] = np.hypot(merged["x"] - merged["x_o"], merged["y"] - merged["y_o"])
    return merged.groupby(KEYS + ["nfl_id"])["dist"].min().droplevel("nfl_id")
