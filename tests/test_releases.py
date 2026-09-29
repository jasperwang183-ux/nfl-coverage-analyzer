import numpy as np
import pandas as pd

from nfl_coverage.data.releases import RELEASE_COLUMNS, releases_bdb2021, releases_bdb2026


def _bdb2021_play(play_id: int, events: dict[int, str], n_frames: int = 8) -> pd.DataFrame:
    """Two players + a football row per frame; `events` maps frameId -> event on that frame."""
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
                    "event": events.get(frame),
                }
            )
    return pd.DataFrame(rows)


def test_bdb2021_pass_forward_rel_to_snap():
    tracking = _bdb2021_play(10, {3: "ball_snap", 7: "pass_forward", 8: "pass_arrived"})
    out = releases_bdb2021(tracking)

    assert list(out.columns) == RELEASE_COLUMNS
    assert out.iloc[0].to_dict() == {
        "dataset": "bdb2021",
        "game_id": 1,
        "play_id": 10,
        "release_rel_frame": 4.0,
    }


def test_bdb2021_shovel_counts_and_first_release_wins():
    tracking = pd.concat(
        [
            _bdb2021_play(10, {2: "ball_snap", 5: "pass_shovel"}),
            _bdb2021_play(11, {2: "ball_snap", 4: "pass_forward", 6: "pass_forward"}),
        ]
    )
    out = releases_bdb2021(tracking).set_index("play_id")["release_rel_frame"]
    assert out.to_dict() == {10: 3.0, 11: 2.0}


def test_bdb2021_frame_shifted_player_does_not_move_the_release():
    tracking = _bdb2021_play(10, {4: "ball_snap", 7: "pass_forward"})
    stray = tracking["nflId"].eq(200.0)
    tracking.loc[stray, "event"] = None
    tracking.loc[stray & tracking["frameId"].eq(2), "event"] = "ball_snap"
    tracking.loc[stray & tracking["frameId"].eq(3), "event"] = "pass_forward"

    out = releases_bdb2021(tracking)
    # the loader snaps on the first ball_snap (frame 2), so rel_frame counts from there
    assert out["release_rel_frame"].iloc[0] == 5.0


def test_bdb2021_no_release_is_nan_and_no_snap_is_dropped():
    tracking = pd.concat(
        [
            _bdb2021_play(10, {2: "ball_snap", 6: "qb_sack"}),
            _bdb2021_play(11, {5: "pass_forward"}),  # no ball_snap: loader drops it too
        ]
    )
    out = releases_bdb2021(tracking)
    assert list(out["play_id"]) == [10]
    assert np.isnan(out["release_rel_frame"].iloc[0])


def test_bdb2026_release_is_last_frame():
    tracking = pd.DataFrame(
        {
            "game_id": [5] * 5 + [5] * 2,
            "play_id": [7] * 5 + [8] * 2,
            "nfl_id": [300, 300, 300, 400, 400, 300, 300],
            "frame_id": [1, 2, 3, 1, 2, 1, 2],
        }
    )
    out = releases_bdb2026(tracking)

    assert list(out.columns) == RELEASE_COLUMNS
    assert (out["dataset"] == "bdb2026").all()
    # frame 1 = snap (rel 0), so a 3-frame play releases at rel_frame 2
    assert out.set_index("play_id")["release_rel_frame"].to_dict() == {7: 2.0, 8: 1.0}
