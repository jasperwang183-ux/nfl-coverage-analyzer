"""Defender phases and the pole-run mechanism on synthetic plays."""

import numpy as np
import pandas as pd
import pytest
from test_disguise import DEFENSE, _play

from nfl_coverage.analysis.mechanisms import mechanism_component, mechanism_hits
from nfl_coverage.features.phases import defender_phases

# LOS at 49.5 (75th pct of offense x), center 26.6 (the QB): the MIKE is 5.5 yd off the ball
TAMPA = {**DEFENSE, "MIKE": (55.0, 26.0)}
POLE = {"MIKE": (61.0, 27.0), "FS": (66.0, 12.0), "SS": (66.0, 41.0)}  # 6 yd in 2 s, halves
HOOK = {"MIKE": (57.0, 20.0), "FS": (66.0, 12.0), "SS": (66.0, 41.0)}  # hook-curl drop


def _mike(phases):
    return phases[phases["is_mike"]].iloc[0]


def test_phases_read_the_trigger_and_the_landmark():
    p = defender_phases(_play(1, 1, POLE, n_frames=21, base=TAMPA))
    m = _mike(p)
    assert m["role"] == "box" and m["align_depth"] == pytest.approx(5.5)
    assert m["commit_v_depth"] == pytest.approx(3.0)  # 6 yd over 2 s, linear
    assert m["landmark_t"] == pytest.approx(2.0)
    assert m["landmark_depth"] == pytest.approx(11.5)
    assert m["first_move_t"] == pytest.approx(0.4)  # 1 yd of a 6.1 yd path at 20 frames


def test_the_landmark_stops_at_the_throw():
    releases = pd.DataFrame(
        {"dataset": ["test"], "game_id": [1], "play_id": [1], "release_rel_frame": [12.0]}
    )
    m = _mike(defender_phases(_play(1, 1, POLE, n_frames=21, base=TAMPA), releases))
    assert m["landmark_t"] == pytest.approx(1.2)
    assert m["landmark_depth"] == pytest.approx(5.5 + 6.0 * 12 / 20)


def test_a_play_too_short_for_the_commit_window_has_no_trigger():
    m = _mike(defender_phases(_play(1, 1, POLE, n_frames=10, base=TAMPA)))
    assert np.isnan(m["commit_v_depth"])


def test_the_pole_run_fires_on_the_middle_runner_and_not_on_a_hook_drop():
    plays = pd.concat(
        [_play(1, 1, POLE, n_frames=21, base=TAMPA), _play(1, 2, HOOK, n_frames=21, base=TAMPA)]
    )
    phases = defender_phases(plays)
    hits = mechanism_hits(phases)
    assert hits["play_id"].tolist() == [1]
    mike_id = _mike(phases[phases["play_id"] == 1])["nfl_id"]
    assert hits.iloc[0]["nfl_id"] == mike_id
    assert "deep middle" in hits.iloc[0]["text"]

    comp = mechanism_component(phases).set_index("play_id")
    assert comp.loc[1, "pole_run_score"] == 1.0 and comp.loc[1, "pole_run_label"] == "pole_run"
    assert comp.loc[2, "pole_run_score"] == 0.0 and comp.loc[2, "pole_run_label"] == ""


def test_an_unreadable_play_scores_nan_not_zero():
    comp = mechanism_component(defender_phases(_play(1, 1, POLE, n_frames=10, base=TAMPA)))
    assert np.isnan(comp.iloc[0]["pole_run_score"])


def test_a_deep_middle_run_with_nobody_deep_at_the_snap_is_a_late_post():
    zero = {**TAMPA, "FS": (57.0, 18.0), "SS": (57.0, 35.0)}  # no safety 9 yd deep
    hits = mechanism_hits(defender_phases(_play(1, 1, {"MIKE": (61.0, 27.0)}, 21, zero)))
    assert hits["mechanism"].tolist() == ["late_post"]
    assert "Cover 1 or Cover 3" in hits.iloc[0]["text"]
    two_high = mechanism_hits(defender_phases(_play(1, 1, POLE, n_frames=21, base=TAMPA)))
    assert two_high["mechanism"].tolist() == ["pole_run"]
