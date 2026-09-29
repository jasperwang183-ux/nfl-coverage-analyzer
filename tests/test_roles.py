"""Geometric snap roles on tiny synthetic plays (offense -> right, LOS 49.5, ball y 26.6)."""

import numpy as np
import pandas as pd
import pytest

from nfl_coverage.analysis.roles import ROLES, snap_roles
from nfl_coverage.data.standardize import finalize

OFFENSE = {"QB": (43.0, 26.6), "WR1": (49.5, 5.0), "WR2": (49.5, 48.0), "TE": (49.5, 35.0)}
DEFENSE = {
    "CB1": (50.5, 5.2),  # press
    "CB2": (57.0, 47.0),  # off
    "NT": (51.0, 26.6),  # on the ball: box, never the mike
    "DE": (51.0, 33.0),  # edge rusher outside the tackle box, still the front
    "MIKE": (54.5, 25.0),
    "WILL": (54.0, 30.0),
    "NB": (55.5, 12.0),  # slot overhang
    "FS": (63.0, 18.0),
    "SS": (63.0, 35.0),
}


def _play(defense=None, positions=True, game_id=1, play_id=1):
    rows = []
    for i, (name, (x, y)) in enumerate({**OFFENSE, **(defense or DEFENSE)}.items()):
        rows.append(
            {
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": i + 1,
                "frame_id": 1,
                "rel_frame": 0,
                "x": x,
                "y": y,
                "s": 0.0,
                "a": 0.0,
                "o": np.nan,
                "dir": np.nan,
                "side": "Offense" if name in OFFENSE else "Defense",
                "position": name if positions else None,
                "coverage_type": None,
                "man_zone": None,
            }
        )
    return finalize(pd.DataFrame(rows), dataset="test")


def _by_name(frames, roles, defense=None):
    names = {i + 1: n for i, n in enumerate({**OFFENSE, **(defense or DEFENSE)})}
    return roles.assign(name=roles["nfl_id"].map(names)).set_index("name")


def test_two_high_look():
    frames = _play()
    r = _by_name(frames, snap_roles(frames))
    assert set(r["role"]) <= set(ROLES)
    assert sorted(r.index[r["role"] == "safety"]) == ["FS", "SS"]
    assert sorted(r.index[r["role"] == "corner"]) == ["CB1", "CB2"]
    assert sorted(r.index[r["role"] == "box"]) == ["DE", "MIKE", "NT", "WILL"]
    assert r.loc["NB", "role"] == "apex"
    assert r.loc["FS", "depth"] == pytest.approx(13.5)
    assert r.loc["FS", "lateral"] == pytest.approx(-8.6)
    assert r.loc["SS", "dist_to_ball_lat"] == pytest.approx(8.4)


def test_one_mike_is_the_off_ball_box_defender_nearest_the_ball():
    frames = _play()
    r = _by_name(frames, snap_roles(frames))
    assert r["is_mike"].sum() == 1
    assert r.index[r["is_mike"]].tolist() == ["MIKE"]  # the nose is nearer but on the line


def test_film_frame_without_positions():
    tracking = snap_roles(_play())
    film = snap_roles(_play(positions=False), los=49.5, center=26.6)
    assert film["role"].tolist() == tracking["role"].tolist()
    assert film["is_mike"].tolist() == tracking["is_mike"].tolist()


def test_no_box_defender_no_mike():
    defense = {k: DEFENSE[k] for k in ("CB1", "CB2", "NB", "FS", "SS")}
    frames = _play(defense)
    r = snap_roles(frames)
    assert "box" not in set(r["role"])
    assert not r["is_mike"].any()


def test_cropped_corners_leave_safeties_as_safeties():
    # corners out of frame: the widest defenders are the two-high safeties inside the hashes
    defense = {"MIKE": DEFENSE["MIKE"], "FS": (63.0, 20.0), "SS": (63.0, 33.0)}
    frames = _play(defense)
    r = _by_name(frames, snap_roles(frames), defense)
    assert r.loc["FS", "role"] == "safety" and r.loc["SS", "role"] == "safety"


def test_mike_per_play():
    frames = pd.concat([_play(play_id=1), _play(play_id=2)], ignore_index=True)
    r = snap_roles(frames)
    assert r.groupby("play_id")["is_mike"].sum().tolist() == [1, 1]
