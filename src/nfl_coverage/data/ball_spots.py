"""Per-play ball spot at the snap, as a side table keyed like the canonical frames.

The line of scrimmage is the line through the ball. The cached frames have no ball (2018
drops the football rows, 2023 never tracked it), so depths were measured from a stand-in,
the 75th percentile of offensive x, which sits a median 1.7 yd behind the ball in 2023.
`ball_x` is stored separately (`data/processed/ball_spots.parquet`) instead of rewriting the
frames, in canonical field x (offense moving left to right), like `releases.py`.
"""

from __future__ import annotations

import pandas as pd

from nfl_coverage.config import FIELD_LENGTH
from nfl_coverage.data.loaders import _BDB2021_RENAMES, bdb2021_snap_frames
from nfl_coverage.data.standardize import flip_left_plays

BALL_SPOT_COLUMNS = ["dataset", "game_id", "play_id", "ball_x"]

_KEYS = ["game_id", "play_id"]


def ball_spots_bdb2021(tracking: pd.DataFrame) -> pd.DataFrame:
    """The football's tracked x at the snap, per play of one raw BDB 2021 week.

    The snap is the loader's (`loaders.bdb2021_snap_frames`, the frame most rows agree on),
    not the football row's own `ball_snap` tag, which is frame-shifted on a few plays.
    Plays with no football row at that frame keep NaN.
    """
    df = tracking.rename(columns=_BDB2021_RENAMES)
    snap_frames = bdb2021_snap_frames(df)
    ball = df.loc[df["team"] == "football", [*_KEYS, "frame_id", "play_direction", "x", "y"]]
    at_snap = ball.merge(
        snap_frames.rename(columns={"snap_frame": "frame_id"}), on=[*_KEYS, "frame_id"]
    ).drop_duplicates(_KEYS)
    at_snap = flip_left_plays(at_snap).rename(columns={"x": "ball_x"})
    out = snap_frames[_KEYS].merge(at_snap[[*_KEYS, "ball_x"]], on=_KEYS, how="left")
    return _finish(out, "bdb2021")


def ball_spots_bdb2026(tracking: pd.DataFrame) -> pd.DataFrame:
    """The recorded spot (`absolute_yardline_number`, whole yards) per play of one raw week.

    It is in raw field x, so a play moving left is mirrored like its players (x = 120 - x).
    """
    first = tracking.loc[
        tracking["frame_id"] == tracking.groupby(_KEYS)["frame_id"].transform("min"),
        [*_KEYS, "play_direction", "absolute_yardline_number"],
    ].drop_duplicates(_KEYS)
    spot = first["absolute_yardline_number"]
    left = first["play_direction"].str.lower().eq("left")
    out = first[_KEYS].assign(ball_x=spot.where(~left, FIELD_LENGTH - spot))
    return _finish(out, "bdb2026")


def _finish(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = out.copy()
    out["dataset"] = dataset
    out["ball_x"] = out["ball_x"].astype(float)
    return out[BALL_SPOT_COLUMNS].sort_values(_KEYS).reset_index(drop=True)
