import pandas as pd
import pytest

from nfl_coverage.data.standardize import add_rel_frame, flip_left_plays


def test_flip_left_plays_mirrors_coordinates_and_angles():
    df = pd.DataFrame(
        {
            "x": [20.0, 100.0],
            "y": [10.0, 43.3],
            "o": [90.0, 270.0],
            "dir": [350.0, 10.0],
            "play_direction": ["left", "right"],
        }
    )
    out = flip_left_plays(df)

    # left play mirrored
    assert out.loc[0, "x"] == pytest.approx(100.0)
    assert out.loc[0, "y"] == pytest.approx(43.3)
    assert out.loc[0, "o"] == pytest.approx(270.0)
    assert out.loc[0, "dir"] == pytest.approx(170.0)
    # right play untouched
    assert out.loc[1, "x"] == pytest.approx(100.0)
    assert out.loc[1, "y"] == pytest.approx(43.3)
    assert out.loc[1, "o"] == pytest.approx(270.0)
    assert "play_direction" not in out.columns


def test_flip_left_plays_does_not_mutate_input():
    df = pd.DataFrame({"x": [20.0], "y": [10.0], "play_direction": ["left"]})
    flip_left_plays(df)
    assert df.loc[0, "x"] == 20.0


def test_add_rel_frame_aligns_snap_to_zero_and_drops_unmatched():
    frames = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2],
            "play_id": [10, 10, 10, 99],
            "frame_id": [1, 11, 20, 5],
        }
    )
    snaps = pd.DataFrame({"game_id": [1], "play_id": [10], "snap_frame": [11]})
    out = add_rel_frame(frames, snaps)

    assert list(out["rel_frame"]) == [-10, 0, 9]  # play (2, 99) dropped
    assert "snap_frame" not in out.columns
