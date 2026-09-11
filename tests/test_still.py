"""Still-image bridge: field points -> canonical frame -> snap features."""

import numpy as np
import pytest

from nfl_coverage.data.standardize import CANONICAL_COLUMNS
from nfl_coverage.features.alignment import snapshot_features
from nfl_coverage.vision.still import frame_from_points

# Synthetic two-high shell, offense on the left of a LOS at x=50 (offense -> right
# convention already): 7 offense, 11 defense, two safeties 13 yd deep.
OFFENSE = [(43.0, 26.6)] + [(49.5, y) for y in (5, 12, 20, 30, 40, 48)]
DEFENSE = (
    [(52.0, y) for y in (8, 16, 24, 32, 40)]  # box / underneath
    + [(50.5, 5.2), (50.5, 47.8)]  # press corners (~1 yd off the WRs)
    + [(56.0, 12.0), (56.0, 41.0)]  # apex defenders
    + [(63.0, 18.0), (63.0, 35.0)]  # two deep safeties
)


def test_canonical_schema_and_sides():
    frame = frame_from_points(np.array(OFFENSE + DEFENSE), los=50.0, offense="left")
    assert list(frame.columns) == CANONICAL_COLUMNS
    assert (frame["rel_frame"] == 0).all()
    assert (frame["side"] == "Offense").sum() == len(OFFENSE)
    assert (frame["side"] == "Defense").sum() == len(DEFENSE)
    # deepest offensive player tagged QB
    qb = frame[frame["position"] == "QB"]
    assert len(qb) == 1 and qb["x"].iloc[0] == pytest.approx(43.0)


def test_offense_right_is_mirrored_to_canonical():
    mirrored = [(120 - x, 53.3 - y) for x, y in OFFENSE + DEFENSE]
    frame = frame_from_points(np.array(mirrored), los=70.0, offense="right")
    direct = frame_from_points(np.array(OFFENSE + DEFENSE), los=50.0, offense="left")
    assert np.allclose(
        frame.sort_values(["x", "y"])[["x", "y"]].values,
        direct.sort_values(["x", "y"])[["x", "y"]].values,
        atol=1e-9,
    )


def test_snap_features_read_the_shell():
    frame = frame_from_points(np.array(OFFENSE + DEFENSE), los=50.0, offense="left")
    feats = snapshot_features(frame).iloc[0]
    assert feats["snap_n_defenders"] == len(DEFENSE)
    assert feats["snap_n_deep_12"] == 2  # the two-high safeties
    assert feats["snap_n_press"] >= 2  # press corners within 1.5 yd of WRs
    assert feats["snap_max_depth"] == pytest.approx(63.0 - 49.5, abs=0.5)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        frame_from_points(np.array([(50.0, 20.0)]), los=50.0, offense="left")
    with pytest.raises(ValueError):
        frame_from_points(np.array(OFFENSE + DEFENSE), los=50.0, offense="north")
