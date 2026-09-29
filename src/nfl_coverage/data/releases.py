"""Per-play pass-release frame, as a side table keyed like the canonical frames.

The canonical frames drop `event`, so the release frame is rebuilt here from raw tracking
and stored separately (`data/processed/releases.parquet`) instead of rewriting the cached
frames. `release_rel_frame` uses the canonical `rel_frame` convention (snap = 0, 10 Hz);
it is float so plays without a release can carry NaN.
"""

from __future__ import annotations

import pandas as pd

from nfl_coverage.data.loaders import _BDB2021_RENAMES, bdb2021_snap_frames
from nfl_coverage.data.standardize import add_rel_frame, majority_event_frame

RELEASE_COLUMNS = ["dataset", "game_id", "play_id", "release_rel_frame"]

# pass_shovel is a forward pass logged under its own event; qb_spike is excluded (no coverage
# is played) and handoff/lateral events are not releases.
BDB2021_RELEASE_EVENTS = ("pass_forward", "pass_shovel")

_KEYS = ["game_id", "play_id"]


def releases_bdb2021(tracking: pd.DataFrame) -> pd.DataFrame:
    """Release frame per play for one raw BDB 2021 week (camelCase, as read from weekN.csv).

    The snap is the loader's (`loaders.bdb2021_snap_frames`, plays without one dropped) so
    `release_rel_frame` lines up with the cached canonical frames. The release uses the same
    majority rule (`standardize.majority_event_frame`) against frame-shifted players.
    """
    df = tracking.rename(columns=_BDB2021_RENAMES)
    snap_frames = bdb2021_snap_frames(df)
    release = majority_event_frame(df, BDB2021_RELEASE_EVENTS)
    release = add_rel_frame(release, snap_frames).rename(columns={"rel_frame": "release_rel_frame"})
    out = snap_frames[_KEYS].merge(
        release[_KEYS + ["release_rel_frame"]], on=_KEYS, how="left", validate="1:1"
    )
    return _finish(out, "bdb2021")


def releases_bdb2026(tracking: pd.DataFrame) -> pd.DataFrame:
    """Release frame per play for one raw BDB 2026 input week (snake_case input_2023_wNN.csv).

    The input files stop at the pass release (the ball-in-air frames live in output_*.csv) and
    frame 1 is the snap, so the release is the play's last frame, as `frame_id - 1`
    (the same shift `loaders._canonicalize_bdb2026` applies).
    """
    out = tracking.groupby(_KEYS, as_index=False)["frame_id"].max()
    out["release_rel_frame"] = out["frame_id"] - 1
    return _finish(out, "bdb2026")


def _finish(out: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = out.copy()
    out["dataset"] = dataset
    out["release_rel_frame"] = out["release_rel_frame"].astype(float)
    return out[RELEASE_COLUMNS].sort_values(_KEYS).reset_index(drop=True)
