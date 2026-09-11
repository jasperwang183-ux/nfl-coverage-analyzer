"""Assemble the per-play feature table from the canonical parquet cache."""

from __future__ import annotations

import pandas as pd

from nfl_coverage.config import DATA_PROCESSED
from nfl_coverage.features.alignment import snapshot_features
from nfl_coverage.features.movement import post_snap_features, pre_snap_features

FRAMES_DIR = DATA_PROCESSED / "frames"
FEATURES_PATH = DATA_PROCESSED / "features.parquet"
KEYS = ["dataset", "game_id", "play_id"]


def _week_features(frames: pd.DataFrame) -> pd.DataFrame:
    meta = (
        frames.groupby(KEYS)
        .agg(coverage_type=("coverage_type", "first"), man_zone=("man_zone", "first"))
        .reset_index()
    )
    out = meta.merge(snapshot_features(frames), on=KEYS, how="left")
    out = out.merge(post_snap_features(frames), on=KEYS, how="left")
    if (frames["rel_frame"] < 0).any():
        out = out.merge(pre_snap_features(frames), on=KEYS, how="left")
    return out


def build_features(force: bool = False) -> pd.DataFrame:
    """Build (or load cached) per-play features for every processed week."""
    if FEATURES_PATH.exists() and not force:
        return pd.read_parquet(FEATURES_PATH)

    parts = []
    for path in sorted(FRAMES_DIR.glob("*.parquet")):
        frames = pd.read_parquet(path)
        feats = _week_features(frames)
        feats["week"] = int(path.stem.split("_w")[1])
        parts.append(feats)
        print(f"{path.name}: {len(feats)} plays")
    table = pd.concat(parts, ignore_index=True)
    table.to_parquet(FEATURES_PATH, index=False)
    return table
