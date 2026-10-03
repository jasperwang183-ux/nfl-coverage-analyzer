"""Defender phases and the pole-run mechanism on synthetic plays."""

import numpy as np
import pandas as pd
import pytest
from test_disguise import DEFENSE, HOLD, OFFENSE, _play

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


def test_the_pole_runner_is_not_a_deep_safety_and_the_verdict_leads_with_him():
    from nfl_coverage.analysis.report import disguise_verdict, played_read, structural_read

    deep_pole = {**POLE, "MIKE": (63.0, 27.0)}  # 13.5 yd deep at +2.0 s, past the count line
    play = _play(1, 1, deep_pole, n_frames=21, base=TAMPA)
    phases = defender_phases(play)
    hits = mechanism_hits(phases).to_dict("records")
    as_safety = played_read(play, 49.5, 26.6, 0.1, [])
    as_runner = played_read(play, 49.5, 26.6, 0.1, [], hit=hits[0])
    assert (as_safety.n_deep, as_safety.mof_closed) == (3, True)  # reads like Cover 3
    assert (as_runner.n_deep, as_runner.mof_closed) == (2, False)
    assert as_runner.family.startswith("Tampa 2")

    shown = structural_read(play[play["rel_frame"] == 0], 49.5, 26.6, 9.0, "snap", [])
    label, text = disguise_verdict("rotate_to_mof_closed", shown, as_runner, hits)
    assert label == "pole_run" and "Tampa 2" in text
    label, text = disguise_verdict("threshold_flicker", shown, as_runner, [])
    assert label == "threshold_flicker" and text.startswith("no disguise")


def test_a_late_post_short_of_the_count_line_is_still_the_single_high_safety():
    from nfl_coverage.analysis.report import played_read

    zero = {**TAMPA, "FS": (57.0, 18.0), "SS": (57.0, 35.0)}
    play = _play(1, 1, {"MIKE": (60.0, 27.0)}, n_frames=21, base=zero)  # 10.5 yd at +2.0 s
    hits = mechanism_hits(defender_phases(play)).to_dict("records")
    assert hits[0]["mechanism"] == "late_post"
    read = played_read(play, 49.5, 26.6, 0.1, [], hit=hits[0])
    assert (read.n_deep, read.mof_closed) == (1, True)
    assert "Cover 1 or Cover 3" in read.family


# Corner roll-up: CB2 aligned 8.5 yd off bails 2.5 yd/s for 0.6 s, then drives downhill to
# 3.5 yd deep at +2.0 s, staying wide (20 yd off the center). Far from every threshold the
# fit sweeps, so these plays keep their meaning when the thresholds move.
SOFT = {**DEFENSE, "CB2": (58.0, 47.0)}


def _roll_up(play, name="CB2", start=58.0, end=53.0):
    """Move one defender's x along a bail-then-drive path (the linear `_play` cannot)."""
    nid = list({**OFFENSE, **SOFT}).index(name) + 1
    t = play["rel_frame"] / 10
    turn = start + 2.5 * 0.6
    x = np.where(t <= 0.6, start + 2.5 * t, turn + (end - turn) * (t - 0.6) / 1.4)
    rows = play["nfl_id"] == nid
    play.loc[rows, "x"] = x[rows]
    return play


def test_the_corner_roll_up_fires_on_a_soft_corner_who_drives_downhill():
    bail = {"CB2": (64.0, 47.0)}  # a Cover 3 corner bailing to his third
    plays = pd.concat(
        [_roll_up(_play(1, 1, HOLD, n_frames=21, base=SOFT)), _play(1, 2, bail, 21, SOFT)]
    )
    phases = defender_phases(plays)
    cb2 = phases[(phases["play_id"] == 1) & (phases["role"] == "corner")]
    cb2 = cb2.loc[cb2["align_depth"].idxmax()]
    assert cb2["first_v_depth"] > 0 and cb2["commit_v_depth"] < -3
    assert cb2["landmark_depth"] == pytest.approx(3.5)

    hits = mechanism_hits(phases)
    assert hits["play_id"].tolist() == [1]
    assert hits.iloc[0]["nfl_id"] == cb2["nfl_id"]
    assert "flat" in hits.iloc[0]["text"]


def test_the_roll_up_names_the_corner_who_drove_downhill_fastest():
    both_soft = {**SOFT, "CB1": (58.0, 5.2)}
    play = _play(1, 1, HOLD, n_frames=21, base=both_soft)
    play = _roll_up(_roll_up(play, "CB1", end=55.0), "CB2", end=53.0)
    phases = defender_phases(play)
    hits = mechanism_hits(phases)
    cb2_id = list({**OFFENSE, **both_soft}).index("CB2") + 1
    assert hits["nfl_id"].tolist() == [cb2_id]


def test_a_roll_up_names_cover_2_behind_two_halves():
    from nfl_coverage.analysis.report import disguise_verdict, played_read, structural_read

    two = _roll_up(_play(1, 1, HOLD, n_frames=21, base=SOFT))
    hit = mechanism_hits(defender_phases(two)).to_dict("records")[0]
    read = played_read(two, 49.5, 26.6, 0.1, [], hit=hit)
    assert (read.n_deep, read.mof_closed) == (2, False)
    assert read.family.startswith("Cover 2 trap, or 2-Read")
    assert any("rolled up" in r for r in read.reasons)

    shown = structural_read(two[two["rel_frame"] == 0], 49.5, 26.6, 9.0, "snap", [])
    label, text = disguise_verdict("", shown, read, [hit])
    assert label == "corner_roll_up" and "squat the flat" in text
