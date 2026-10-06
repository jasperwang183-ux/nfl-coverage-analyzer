import numpy as np
import pandas as pd
import pytest

from nfl_coverage.data.ball_spots import BALL_SPOT_COLUMNS, ball_spots_bdb2021, ball_spots_bdb2026


def _bdb2021_play(play_id: int, snap: int, direction: str = "right", n_frames: int = 6):
    """Two players + a football that moves 1 yd per frame from x=40; ball_snap on `snap`."""
    rows = []
    for frame in range(1, n_frames + 1):
        for nfl_id, team in ((100.0, "home"), (200.0, "away"), (np.nan, "football")):
            rows.append(
                {
                    "gameId": 1,
                    "playId": play_id,
                    "nflId": nfl_id,
                    "frameId": frame,
                    "team": team,
                    "playDirection": direction,
                    "event": "ball_snap" if frame == snap else None,
                    "x": 40.0 + frame if team == "football" else 30.0,
                    "y": 26.0,
                }
            )
    return pd.DataFrame(rows)


def test_bdb2021_takes_the_football_x_at_the_snap():
    out = ball_spots_bdb2021(_bdb2021_play(10, snap=3))
    assert list(out.columns) == BALL_SPOT_COLUMNS
    assert out.iloc[0].to_dict() == {
        "dataset": "bdb2021",
        "game_id": 1,
        "play_id": 10,
        "ball_x": 43.0,
    }


def test_bdb2021_mirrors_a_play_moving_left():
    out = ball_spots_bdb2021(_bdb2021_play(10, snap=3, direction="left"))
    assert out["ball_x"].iloc[0] == pytest.approx(120.0 - 43.0)


def test_bdb2021_a_stray_snap_tag_on_the_football_row_does_not_move_the_spot():
    play = _bdb2021_play(10, snap=4)
    ball = play["team"].eq("football")
    play.loc[ball, "event"] = None
    play.loc[ball & play["frameId"].eq(1), "event"] = "ball_snap"
    # the players agree on frame 4, so the spot is the football there, not at its own tag
    assert ball_spots_bdb2021(play)["ball_x"].iloc[0] == 44.0


def test_bdb2021_no_football_at_the_snap_leaves_nan():
    play = _bdb2021_play(10, snap=3)
    play = play[~(play["team"].eq("football") & play["frameId"].eq(3))]
    assert ball_spots_bdb2021(play)["ball_x"].isna().all()


def test_bdb2026_takes_the_recorded_spot_and_mirrors_left_plays():
    tracking = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "play_id": [10, 10, 11, 11],
            "frame_id": [1, 2, 1, 2],
            "play_direction": ["right", "right", "left", "left"],
            "absolute_yardline_number": [65, 65, 42, 42],
        }
    )
    out = ball_spots_bdb2026(tracking).set_index("play_id")["ball_x"]
    assert out.to_dict() == {10: 65.0, 11: 78.0}
