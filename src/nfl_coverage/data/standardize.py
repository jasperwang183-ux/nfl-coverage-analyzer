"""Coordinate and frame standardization shared by all tracking datasets.

Canonical frame schema produced by the loaders (one row per player per frame):

    dataset       'bdb2021' | 'bdb2026'
    game_id, play_id, nfl_id, frame_id
    rel_frame     frames relative to the snap (snap = 0; pre-snap < 0)
    x, y          yards, standardized so the offense always moves left -> right
    s, a          speed (yd/s), acceleration (yd/s^2)
    o, dir        orientation / movement direction, degrees, standardized with x/y
    side          'Offense' | 'Defense'
    position      roster position string (CB, FS, WR, ...) where available
    coverage_type canonical label (COVER_3_ZONE, ...) or NaN
    man_zone      MAN_COVERAGE | ZONE_COVERAGE or NaN
"""

from __future__ import annotations

import pandas as pd

from nfl_coverage.config import FIELD_LENGTH, FIELD_WIDTH

CANONICAL_COLUMNS = [
    "dataset",
    "game_id",
    "play_id",
    "nfl_id",
    "frame_id",
    "rel_frame",
    "x",
    "y",
    "s",
    "a",
    "o",
    "dir",
    "side",
    "position",
    "coverage_type",
    "man_zone",
]


def flip_left_plays(df: pd.DataFrame, direction_col: str = "play_direction") -> pd.DataFrame:
    """Standardize all plays to offense moving left -> right.

    For plays marked 'left', mirrors x and y and rotates the angle columns 180°.
    Returns a copy; the direction column is dropped from the result.
    """
    df = df.copy()
    left = df[direction_col].str.lower().eq("left")
    df.loc[left, "x"] = FIELD_LENGTH - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_WIDTH - df.loc[left, "y"]
    for angle in ("o", "dir"):
        if angle in df.columns:
            df.loc[left, angle] = (df.loc[left, angle] + 180.0) % 360.0
    return df.drop(columns=[direction_col])


def add_rel_frame(df: pd.DataFrame, snap_frames: pd.DataFrame) -> pd.DataFrame:
    """Add rel_frame = frame_id - snap frame of that play.

    `snap_frames` has columns game_id, play_id, snap_frame. Plays without a known
    snap frame are dropped (they cannot be aligned).
    """
    out = df.merge(snap_frames, on=["game_id", "play_id"], how="inner", validate="m:1")
    out["rel_frame"] = out["frame_id"] - out["snap_frame"]
    return out.drop(columns=["snap_frame"])


def finalize(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Tag with the dataset name, order canonical columns, and sort."""
    df = df.copy()
    df["dataset"] = dataset
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"canonical columns missing for {dataset}: {missing}")
    return (
        df[CANONICAL_COLUMNS]
        .sort_values(["game_id", "play_id", "frame_id", "nfl_id"])
        .reset_index(drop=True)
    )
