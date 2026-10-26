"""The coverage diagram (analysis.zones) on hand-built canonical tracks."""

import pandas as pd
import pytest

from nfl_coverage.analysis.zones import coverage_key, diagram, template

LOS, BALL = 50.0, 26.65


def _tracks(defense, offense=(), frames=31):
    """Each player moves in a straight line from `start` (depth, y) at the snap to `end` at
    +2.0 s, then holds; `defense` / `offense`: lists of (start, end)."""
    rows = []
    for side, players, base in (("Defense", defense, 0), ("Offense", offense, 100)):
        for i, ((d0, y0), (d1, y1)) in enumerate(players):
            for f in range(frames):
                k = min(f / 20, 1.0)
                rows.append((base + i, f, LOS + d0 + k * (d1 - d0), y0 + k * (y1 - y0), side))
    return pd.DataFrame(rows, columns=["nfl_id", "rel_frame", "x", "y", "side"])


# Cover 3 from a two-high look: corners bail to the outside thirds, a safety spins to the
# middle, the other buzzes to a curl; two linebackers and a nickel fill the rest
COVER_3 = [((7, 5), (16, 8)), ((7, 48), (16, 45)), ((13, 20), (17, 26)), ((13, 33), (8, 36)),
           ((5, 22), (8, 18)), ((5, 30), (3, 6)), ((4, 44), (4, 47))]  # fmt: skip


def test_cover_3_defenders_land_in_their_zones():
    d = diagram(_tracks(COVER_3), LOS, BALL, "Cover 3 zone (three-deep shell).")
    assert (d.coverage, d.kind, len(d.zones)) == ("Cover 3 zone", "zone", 7)
    zone_of = {a["nfl_id"]: d.zones[a["zone"]]["name"] for a in d.assigned}
    assert zone_of[2] == "deep middle third" and zone_of[3] == "hook / curl"
    assert all(a["reached"] for a in d.assigned)
    assert d.fit == "7 of 7 defenders reached their Cover 3 zones by +3.0 s."


def test_a_call_the_defenders_do_not_fit_is_questioned():
    everyone_short = [((6, y), (3, y)) for y in (6, 15, 22, 30, 38, 47)]  # all sit at 3 yd
    d = diagram(_tracks(everyone_short), LOS, BALL, "Cover 3 zone (three-deep shell).")
    assert "check the call" in d.fit and "deep middle third" in d.fit


def test_rushers_get_no_zone():
    d = diagram(_tracks([*COVER_3, ((1, 26), (-2, 26))]), LOS, BALL, "Cover 3 zone.")
    assert 7 not in {a["nfl_id"] for a in d.assigned} and "1 within a yard of the line" in d.fit


def test_one_player_on_two_track_ids_fills_one_zone():
    twin = ((13, 20.4), (17, 26.4))  # the middle safety's second id, 0.4 yd off
    d = diagram(_tracks([*COVER_3, twin]), LOS, BALL, "Cover 3 zone.")
    assert len(d.assigned) == 7 and "no zone left" not in d.fit


RECEIVERS = [((0, 5), (12, 5)), ((0, 40), (8, 46))]


def test_cover_1_draws_the_deep_middle_and_each_man_on_his_receiver():
    on_them = [((1.5, 6), (12, 6)), ((1.5, 41), (8, 45)), ((14, 26), (16, 26))]
    d = diagram(_tracks(on_them, RECEIVERS), LOS, BALL, "Cover 1 man (single-high shell).")
    assert d.kind == "man" and [z["name"] for z in d.zones] == ["deep middle"]
    assert {(m["nfl_id"], m["receiver"]) for m in d.matchups} == {(0, 100), (1, 101)}
    assert d.fit == (
        "The deep middle defender reached his zone by +3.0 s. "
        "2 of 2 underneath defenders stayed within 3.5 yd of one receiver."
    )


def test_a_man_call_nobody_plays_is_questioned():
    sitting = [((5, 15), (6, 15)), ((5, 35), (6, 35)), ((14, 26), (16, 26))]  # zone droppers
    d = diagram(_tracks(sitting, RECEIVERS), LOS, BALL, "Cover 1 man (single-high shell).")
    assert "0 of 2 underneath" in d.fit and "check the call" in d.fit


def test_man_without_tracked_receivers_says_it_cannot_check():
    on_them = [((1.5, 6), (12, 6)), ((14, 26), (16, 26))]
    d = diagram(_tracks(on_them), LOS, BALL, "Cover 1 man (single-high shell).")
    assert "Too few receivers" in d.fit and "check the call" not in d.fit


@pytest.mark.parametrize(
    ("call", "key"),
    [
        ("Tampa 2 (Cover 2 shell).", "tampa2"),
        ("Cover 2 trap (Cover 2 shell).", "cover2"),
        ("Cover 2 man (four-deep shell).", "cover2man"),
        ("Cover 4 zone (four-deep shell).", "cover4"),
        ("Cover 6 zone.", "cover6"),
        ("Cover 2 shell (two safeties deep, middle of the field open).", ""),
        ("Single-high shell (one safety deep, over the middle).", ""),
    ],
)
def test_coverage_key(call, key):
    assert coverage_key(call) == key


def test_a_shell_draws_nothing():
    d = diagram(_tracks(COVER_3), LOS, BALL, "Rotated two-high shell (two deep).")
    assert (d.coverage, d.zones) == ("", []) and "none is drawn" in d.fit


def test_cover_6_puts_the_quarters_where_the_defenders_are():
    # two quarter players deep on the low side, one half player high, a cloud corner high
    six = [((8, 5), (15, 6)), ((12, 20), (16, 19)), ((12, 38), (17, 40)), ((5, 50), (3, 48)),
           ((5, 18), (8, 18)), ((5, 32), (8, 32))]  # fmt: skip
    d = diagram(_tracks(six), LOS, BALL, "Cover 6 zone (split three-deep shell).")
    halves = [z for z in d.zones if z["name"] == "deep half"]
    assert halves[0]["y0"] > 20 and all(a["reached"] for a in d.assigned)


def test_every_template_stays_on_the_field():
    for key in ("cover2", "tampa2", "cover3", "cover4", "cover6", "cover1", "cover2man"):
        for z in template(key, 10.0):  # ball on the near hash's outside
            assert 0 <= z.y0 <= z.y1 <= 53.3 or z.name.startswith("curl"), (key, z)
