import numpy as np
import pandas as pd

from nfl_coverage.data.loaders import _canonicalize_bdb2021, _canonicalize_bdb2026
from nfl_coverage.data.standardize import CANONICAL_COLUMNS


def _bdb2021_fixture():
    """Two frames x two players + a football row, one play, offense = home team (KC)."""
    tracking = pd.DataFrame(
        {
            "gameId": [1] * 5,
            "playId": [10] * 5,
            "nflId": [100.0, 200.0, np.nan, 100.0, 200.0],
            "frameId": [1, 1, 1, 2, 2],
            "team": ["home", "away", "football", "home", "away"],
            "event": [None, None, None, "ball_snap", "ball_snap"],
            "x": [50.0, 55.0, 52.0, 50.5, 55.5],
            "y": [20.0, 25.0, 22.0, 20.5, 25.5],
            "s": [0.1] * 5,
            "a": [0.0] * 5,
            "o": [90.0] * 5,
            "dir": [180.0] * 5,
            "position": ["WR", "CB", None, "WR", "CB"],
            "playDirection": ["left"] * 5,
        }
    )
    plays = pd.DataFrame({"gameId": [1], "playId": [10], "possessionTeam": ["KC"]})
    games = pd.DataFrame({"gameId": [1], "homeTeamAbbr": ["KC"], "visitorTeamAbbr": ["BUF"]})
    labels = pd.DataFrame(
        {
            "gameId": [1],
            "playId": [10],
            "coverage_type": ["COVER_1_MAN"],
            "man_zone": ["MAN_COVERAGE"],
        }
    )
    return tracking, plays, games, labels


def test_canonicalize_bdb2021_schema_side_and_snap():
    out = _canonicalize_bdb2021(*_bdb2021_fixture())

    assert list(out.columns) == CANONICAL_COLUMNS
    assert (out["dataset"] == "bdb2021").all()
    # football row dropped: 2 players x 2 frames
    assert len(out) == 4
    # home team (KC) has possession -> Offense; away -> Defense
    assert set(out.loc[out["nfl_id"] == 100.0, "side"]) == {"Offense"}
    assert set(out.loc[out["nfl_id"] == 200.0, "side"]) == {"Defense"}
    # snap at frameId 2 -> rel_frame -1 then 0
    assert sorted(out["rel_frame"].unique()) == [-1, 0]
    # left play flipped
    assert out.loc[(out["nfl_id"] == 100.0) & (out["frame_id"] == 1), "x"].iloc[0] == 70.0
    # labels joined
    assert (out["coverage_type"] == "COVER_1_MAN").all()


def test_canonicalize_bdb2021_missing_label_is_nan():
    tracking, plays, games, labels = _bdb2021_fixture()
    labels = labels.iloc[0:0]  # empty
    out = _canonicalize_bdb2021(tracking, plays, games, labels)
    assert out["coverage_type"].isna().all()


def test_canonicalize_bdb2026_rel_frame_and_labels():
    tracking = pd.DataFrame(
        {
            "game_id": [5, 5],
            "play_id": [7, 7],
            "nfl_id": [300, 300],
            "frame_id": [1, 2],
            "player_side": ["Defense", "Defense"],
            "player_position": ["FS", "FS"],
            "x": [30.0, 31.0],
            "y": [26.65, 26.65],
            "s": [0.0, 1.0],
            "a": [0.0, 1.0],
            "o": [0.0, 0.0],
            "dir": [0.0, 0.0],
            "play_direction": ["right", "right"],
        }
    )
    supplementary = pd.DataFrame(
        {
            "game_id": [5],
            "play_id": [7],
            "team_coverage_type": ["COVER_3_ZONE"],
            "team_coverage_man_zone": ["ZONE_COVERAGE"],
        }
    )
    out = _canonicalize_bdb2026(tracking, supplementary)

    assert list(out.columns) == CANONICAL_COLUMNS
    assert list(out["rel_frame"]) == [0, 1]  # frame 1 = snap
    assert (out["coverage_type"] == "COVER_3_ZONE").all()
    assert (out["side"] == "Defense").all()
