"""Still-image bridge: field points -> canonical frame -> snap features."""

import numpy as np
import pytest

from nfl_coverage.data.standardize import CANONICAL_COLUMNS
from nfl_coverage.features.alignment import snapshot_features
from nfl_coverage.vision.still import frame_from_points, line_player_mask

# Synthetic two-high shell, offense on the left of a LOS at x=50 (offense -> right
# convention already): 7 offense, 11 defense, two safeties 13 yd deep.
# receivers kept wide of the box: the line filter is a y-band around the box
# center, so a tight slot (within ~4.5 yd of it) would be dropped with the OL
OFFENSE = [(43.0, 26.6)] + [(49.5, y) for y in (5, 12, 20, 35, 40, 48)]
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


def test_explicit_sides_override_los_split():
    # a press corner whose projected x drifted just past the LOS: the color-based
    # side label must win over the geometric split
    pts = np.array(OFFENSE + DEFENSE)
    sides = np.array(["Offense"] * len(OFFENSE) + ["Defense"] * len(DEFENSE))
    drifted = pts.copy()
    drifted[1] = (50.4, 5.0)  # WR now 0.4 yd past the LOS
    frame = frame_from_points(drifted, los=50.0, offense="left", side=sides)
    assert (frame["side"] == "Offense").sum() == len(OFFENSE)
    with pytest.raises(ValueError, match="labels"):
        frame_from_points(pts, los=50.0, offense="left", side=sides[:-1])


def test_line_player_mask_drops_both_lines_only():
    ol = [(49.0, y) for y in (22, 24, 26, 28, 30)]  # five OL just behind the LOS
    dl = [(51.0, y) for y in (23, 25, 27, 29)]  # four down linemen
    pts = np.array(OFFENSE + ol + DEFENSE + dl)
    sides = np.array(
        ["Offense"] * (len(OFFENSE) + len(ol)) + ["Defense"] * (len(DEFENSE) + len(dl))
    )
    frame = frame_from_points(pts, los=50.0, offense="left", side=sides)
    mask = line_player_mask(frame)
    dropped = frame[mask]
    kept = frame[~mask]
    assert len(dropped) == len(ol) + len(dl)
    assert (dropped["x"].between(48.5, 51.5)).all()
    # receivers on the LOS but wide, and all coverage defenders, survive
    assert (kept["side"] == "Defense").sum() == len(DEFENSE)
    assert (kept["side"] == "Offense").sum() == len(OFFENSE)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        frame_from_points(np.array([(50.0, 20.0)]), los=50.0, offense="left")
    with pytest.raises(ValueError):
        frame_from_points(np.array(OFFENSE + DEFENSE), los=50.0, offense="north")
