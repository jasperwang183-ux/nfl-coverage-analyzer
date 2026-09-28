"""Tendency tables on tiny synthetic play lists."""

import numpy as np
import pandas as pd
import pytest

from nfl_coverage.analysis import tendencies as td


def _plays(rows):
    cols = ["season", "defense", "play_caller", "shown", "situation", "coverage_type", "week"]
    return pd.DataFrame(rows, columns=cols)


def _league_and_team():
    # league: MOF open plays Cover 4 half the time, Cover 1 half; team AAA always rotates to Cover 1
    rows = []
    for i in range(50):
        cov = "COVER_4_ZONE" if i % 2 else "COVER_1_MAN"
        rows.append((2023, "BBB", "", "MOF open", "early down", cov, 1))
    for _ in range(10):
        rows.append((2023, "AAA", "", "MOF open", "early down", "COVER_1_MAN", 1))
    return _plays(rows)


def _row(tables, defense, situation):
    t = tables[
        (tables["defense"] == defense)
        & (tables["shown"] == "MOF open")
        & (tables["situation"] == situation)
    ]
    assert len(t) == 1
    return t.iloc[0]


def test_situation_buckets():
    down = pd.Series([1, 2, 3, 3, 3, 4])
    togo = pd.Series([10, 2, 2, 5, 12, 1])
    assert td.situation(down, togo).tolist() == [
        "early down",
        "early down",
        "3rd/4th & short",
        "3rd/4th & medium",
        "3rd/4th & long",
        "3rd/4th & short",
    ]


def test_shown_state():
    assert td.shown_state(0, False) == "0-high"
    assert td.shown_state(1, True) == "MOF closed"
    assert td.shown_state(2, False) == "MOF open"


def test_team_cell_is_shrunk_toward_the_league():
    tables = td.build_tables(_league_and_team(), k=10)
    league = _row(tables, td.LEAGUE, td.ANY)
    assert league["n"] == 60
    assert league["COVER_1_MAN"] == pytest.approx(35 / 60)
    team = _row(tables, "AAA", td.ANY)
    # 10 own plays (all Cover 1) + 10 pseudo-plays of the league mix
    assert team["n"] == 10
    assert team["COVER_1_MAN"] == pytest.approx((10 + 10 * 35 / 60) / 20)
    assert team[td.CLASSES].sum() == pytest.approx(1.0)


def test_k_trades_team_evidence_for_the_league():
    loose = _row(td.build_tables(_league_and_team(), k=1), "AAA", td.ANY)
    tight = _row(td.build_tables(_league_and_team(), k=1000), "AAA", td.ANY)
    assert loose["COVER_1_MAN"] > 0.95
    assert tight["COVER_1_MAN"] == pytest.approx(35 / 60, abs=0.01)


def test_situation_cell_inherits_the_team_and_the_league_tilt():
    tables = td.build_tables(_league_and_team(), k=10)
    team_any = _row(tables, "AAA", td.ANY)
    team_sit = _row(tables, "AAA", "early down")
    # every play is early down, so the tilt is 1 and the situation cell sits between
    # the team's own plays and its whole-season row
    assert team_sit["COVER_1_MAN"] == pytest.approx((10 + 10 * team_any["COVER_1_MAN"]) / 20)
    assert team_sit[td.CLASSES].sum() == pytest.approx(1.0)


def test_prevent_and_unlabeled_plays_are_left_out():
    plays = pd.concat(
        [
            _league_and_team(),
            _plays([(2023, "AAA", "", "MOF open", "early down", "PREVENT", 1)]),
            _plays([(2023, "AAA", "", "MOF open", "early down", None, 1)]),
        ]
    )
    assert _row(td.build_tables(plays, k=10), "AAA", td.ANY)["n"] == 10


def test_log_loss_falls_back_to_the_league_for_an_unseen_team():
    tables = td.build_tables(_league_and_team(), k=10)
    unseen = _plays([(2023, "ZZZ", "", "MOF open", "early down", "COVER_4_ZONE", 2)])
    assert td.log_loss(tables, unseen, "team_sit") == pytest.approx(-np.log(25 / 60))
    seen = _plays([(2023, "AAA", "", "MOF open", "early down", "COVER_1_MAN", 2)])
    assert td.log_loss(tables, seen, "team") < td.log_loss(tables, seen, "league")


def test_plays_it_sums_the_classes_that_keep_the_look():
    probs = pd.Series(
        dict.fromkeys(td.CLASSES, 0.0)
        | {"COVER_0_MAN": 0.1, "COVER_1_MAN": 0.3, "COVER_4_ZONE": 0.4, "COVER_6_ZONE": 0.2}
    )
    assert td.plays_it(probs, "MOF open") == pytest.approx(0.6)
    assert td.plays_it(probs, "MOF closed") == pytest.approx(0.3)
    assert td.plays_it(probs, "0-high") == pytest.approx(0.1)


def test_play_callers_split_a_season_at_the_change():
    plays = pd.DataFrame({"season": [2023] * 3, "defense": ["WAS"] * 3, "week": [1, 9, 10]})
    callers = pd.DataFrame(
        {
            "season": [2023, 2023],
            "team": ["WAS", "WAS"],
            "first_week": [1, 10],
            "last_week": [9, 18],
            "play_caller": ["Del Rio", "Rivera"],
        }
    )
    assert td.assign_play_callers(plays, callers).tolist() == ["Del Rio", "Del Rio", "Rivera"]
    assert td.assign_play_callers(plays, None).tolist() == ["", "", ""]


def test_lookup_reports_the_teams_own_n_and_plays_it():
    tables = td.build_tables(_league_and_team(), k=10)
    t = td.lookup(tables, 2023, "AAA", "MOF open")
    assert t.n == 10
    assert t.plays_it == pytest.approx(t.probs["COVER_4_ZONE"])
    assert td.lookup(tables, 2023, "AAA", "MOF closed") is None


def test_the_play_caller_table_covers_every_week_of_every_team_season():
    from itertools import pairwise
    from pathlib import Path

    path = Path(td.__file__).parents[1] / "data" / "defensive_play_callers.csv"
    callers = pd.read_csv(path)
    last = {2018: 17, 2023: 18}
    assert set(callers["season"]) == set(last)
    for (season, team), g in callers.groupby(["season", "team"]):
        weeks = g.sort_values("first_week")[["first_week", "last_week"]].to_numpy()
        assert weeks[0][0] == 1 and weeks[-1][1] == last[season], (season, team)
        assert all(a[1] + 1 == b[0] for a, b in pairwise(weeks)), (season, team)
    assert callers.groupby("season")["team"].nunique().tolist() == [32, 32]
    assert set(callers["confidence"]) <= {"high", "medium", "low"}
