"""The front and the rush (analysis.front) on hand-built canonical tracks."""

import pandas as pd

from nfl_coverage.analysis.front import front_read, rush_class

LOS, CENTER = 50.0, 26.65


def _tracks(defense, moves=None, frames=16):
    """`defense`: (depth, lat) at the snap per defender; `moves`: id -> depth change per frame."""
    moves = moves or {}
    rows = []
    for i, (depth, lat) in enumerate(defense):
        for f in range(frames):
            dd = moves.get(i, 0.0) * f
            rows.append((i, f, LOS + depth + dd, CENTER + lat, "Defense"))
    return pd.DataFrame(rows, columns=["nfl_id", "rel_frame", "x", "y", "side"])


# 4 down linemen, 2 linebackers, 2 corners wide, 2 safeties deep, a nickel at 5 yd wide
FOUR_TWO = [(1.0, -4.5), (1.2, -1.5), (1.1, 1.5), (0.9, 4.5), (4.5, -2.0), (4.8, 2.5),
            (6.0, -18.0), (6.5, 17.0), (12.0, -8.0), (13.0, 8.0), (5.0, 11.0)]  # fmt: skip


def test_a_four_man_front_rushing_four_is_a_standard_rush():
    f = front_read(_tracks(FOUR_TWO), LOS, CENTER)
    assert (f.front, f.n_line, f.n_second, f.n_box) == ("4-2", 4, 2, 6)
    assert (f.rushers, f.dropped, f.rush_class) == (4, 0, "standard rush")
    assert f.unsure  # nothing on film checks the rush


def test_six_at_the_line_with_two_dropping_is_simulated_pressure():
    six = [*FOUR_TWO[:4], (1.3, -6.5), (1.4, 6.5), *FOUR_TWO[6:]]  # LBs walked up: a 6-0 look
    six.append((4.6, 0.0))  # one linebacker stays off: 6-1
    f = front_read(_tracks(six[:11]), LOS, CENTER)
    assert f.front == "6-0" and f.rush_class == "blitz"  # 6 shown and all 6 came
    dropping = front_read(_tracks(six[:11], {4: 0.3, 5: 0.3}), LOS, CENTER)  # 4.5 yd by +1.5 s
    assert (dropping.dropped, dropping.rushers, dropping.rush_class) == (
        2,
        4,
        "simulated pressure",
    )


def test_a_linebacker_who_comes_to_the_line_counts_as_a_rusher():
    f = front_read(_tracks(FOUR_TWO, {4: -0.4}), LOS, CENTER)  # 4.5 yd -> under 2.5 by +1.0 s
    assert (f.rushers, f.rush_class) == (5, "blitz")


def test_broken_sides_are_unreadable_not_a_front():
    f = front_read(_tracks(FOUR_TWO + [(1.0, 0.0), (1.0, 2.0)]), LOS, CENTER)  # 13 defenders
    assert f.front == "" and "13" in f.unsure


def test_rush_classes():
    assert rush_class(4, 4) == "standard rush"
    assert rush_class(4, 5) == "blitz"
    assert rush_class(6, 4) == "simulated pressure"
