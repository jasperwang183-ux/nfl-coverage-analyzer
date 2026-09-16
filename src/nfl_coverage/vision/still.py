"""Bridge a still image's field-space player positions into the canonical frame schema.

This is the rung-2 contract: whatever the vision stack produced (detections ->
homography -> field coordinates), it becomes one canonical rel_frame-0 frame that
the same feature/model code consumes as tracking data does. A still carries no
movement, so s/a are 0 and o/dir are NaN; only snap_* alignment features are
meaningful downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_coverage.config import FIELD_LENGTH, FIELD_WIDTH
from nfl_coverage.data.standardize import finalize
from nfl_coverage.features.alignment import snapshot_features
from nfl_coverage.models.persist import predict_proba


def frame_from_points(
    field_xy: np.ndarray, los: float, offense: str, side: np.ndarray | None = None
) -> pd.DataFrame:
    """Build a canonical single-frame DataFrame from field-space player positions.

    field_xy : (N, 2) player positions in yards (same frame as the calibration).
    los      : line of scrimmage, x in [0, 120].
    offense  : 'left' or 'right' — which side of the LOS the offense occupies
               in these coordinates. Output is standardized to offense -> right,
               matching the tracking datasets.
    side     : optional per-player 'Offense'/'Defense' labels (e.g. from jersey
               color clustering, `vision.teams`). Preferred: the fallback is a
               hard split at the LOS (ties go to offense), which homography
               error near the line can flip.
    """
    if offense not in ("left", "right"):
        raise ValueError(f"offense must be 'left' or 'right', got {offense!r}")
    pts = np.asarray(field_xy, dtype=float).reshape(-1, 2).copy()
    if len(pts) < 4:
        raise ValueError(f"need at least a few players to read a formation, got {len(pts)}")

    # Standardize to offense moving left -> right (offense on the low-x side of LOS).
    if offense == "right":
        pts[:, 0] = FIELD_LENGTH - pts[:, 0]
        pts[:, 1] = FIELD_WIDTH - pts[:, 1]
        los = FIELD_LENGTH - los

    if side is None:
        side = np.where(pts[:, 0] <= los, "Offense", "Defense")
    else:
        side = np.asarray(side)
        if side.shape[0] != len(pts):
            raise ValueError(f"side has {side.shape[0]} labels for {len(pts)} points")

    df = pd.DataFrame(
        {
            "game_id": 0,
            "play_id": 0,
            "nfl_id": np.arange(1, len(pts) + 1),
            "frame_id": 0,
            "rel_frame": 0,
            "x": pts[:, 0],
            "y": pts[:, 1],
            "s": 0.0,
            "a": 0.0,
            "o": np.nan,
            "dir": np.nan,
            "side": side,
            "position": None,
            "coverage_type": None,
            "man_zone": None,
        }
    )
    # Deepest offensive player is almost always the QB (or the back next to him);
    # tagging him keeps the press feature from counting QB proximity.
    off = df["side"] == "Offense"
    if off.any():
        df.loc[df.loc[off, "x"].idxmin(), "position"] = "QB"
    return finalize(df, dataset="still")


def line_player_mask(frame: pd.DataFrame, depth: float = 2.5, halfwidth: float = 4.5) -> pd.Series:
    """Mark players on either line of scrimmage (OL, DL, under-center QB).

    The tracking datasets the models trained on carry coverage-relevant players
    only — no linemen on either side (training mean ~7 defenders vs 11 on the
    field) — so a still must drop the box bodies to match that distribution.
    A line player is within `depth` yards of the LOS *and* within `halfwidth`
    yards of the line cluster's y-center (receivers and press corners are on the
    LOS too, but wide of it).
    """
    los = frame.loc[frame["side"] == "Offense", "x"].quantile(0.75)
    near = (frame["x"] - los).abs() < depth
    if not near.any():
        return pd.Series(False, index=frame.index)
    center = frame.loc[near, "y"].median()
    return near & ((frame["y"] - center).abs() < halfwidth)


def predict_shown(frame: pd.DataFrame, bundle: dict) -> pd.Series:
    """Score one canonical still frame; returns class -> probability, descending."""
    feats = snapshot_features(frame)
    proba = predict_proba(bundle, feats)
    return proba.iloc[0].sort_values(ascending=False)
