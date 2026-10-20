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

    # 12.5 yd at +2.0 s, inside the pole run's 13 yd cap and past the count line
    deep_pole = {**POLE, "MIKE": (62.0, 27.0)}
    play = _play(1, 1, deep_pole, n_frames=21, base=TAMPA)
    phases = defender_phases(play)
    hits = mechanism_hits(phases).to_dict("records")
    as_safety = played_read(play, 49.5, 26.6, 0.1, [])
    as_runner = played_read(play, 49.5, 26.6, 0.1, [], hit=hits[0])
    assert as_safety.n_deep == 3  # counted as a third deep defender
    assert (as_runner.n_deep, as_runner.mof_closed) == (2, False)
    assert as_runner.family.startswith("Tampa 2")

    shown = structural_read(play[play["rel_frame"] == 0], 49.5, 26.6, 9.0, "snap", [])
    label, text = disguise_verdict("rotate_to_mof_closed", shown, as_runner, hits)
    assert label == "pole_run" and "Tampa 2" in text
    label, text = disguise_verdict("threshold_flicker", shown, as_runner, [])
    assert label == "threshold_flicker" and text.startswith("no disguise")


def test_a_runner_past_13_yd_is_the_safety_not_a_pole_runner():
    from nfl_coverage.analysis.report import played_read

    past_cap = {"MIKE": (66.0, 27.0)}  # 16.5 yd at +2.0 s
    play = _play(1, 1, {**POLE, **past_cap}, n_frames=21, base=TAMPA)
    assert mechanism_hits(defender_phases(play)).empty
    read = played_read(play, 49.5, 26.6, 0.1, [])
    assert (read.n_deep, read.mof_closed) == (3, True)

    zero = {**TAMPA, "FS": (57.0, 18.0), "SS": (57.0, 35.0)}
    hits = mechanism_hits(defender_phases(_play(1, 1, past_cap, n_frames=21, base=zero)))
    assert hits["mechanism"].tolist() == ["late_post"]


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


def test_ball_spots_move_every_depth_and_no_speed():
    play = _play(1, 1, POLE, n_frames=21, base=TAMPA)
    spots = pd.DataFrame({"dataset": ["test"], "game_id": [1], "play_id": [1], "ball_x": [50.5]})
    old, new = _mike(defender_phases(play)), _mike(defender_phases(play, ball_spots=spots))
    # the stand-in LOS is 49.5, the ball 1 yd further on: depths shrink by 1, speeds do not
    assert new["align_depth"] == pytest.approx(old["align_depth"] - 1.0)
    assert new["landmark_depth"] == pytest.approx(old["landmark_depth"] - 1.0)
    assert new["commit_v_depth"] == pytest.approx(old["commit_v_depth"])


# Buzz: SS comes down 6 yd in 2 s (-3 yd/s) to the curl-flat, 7.5 yd deep and 13.4 yd wide,
# while FS drifts to 16.5 yd deep, 1.6 yd off the middle. The corners' mean cushion is 7.5 yd
# off, or 1.0 yd pressed.
OFF = {**DEFENSE, "CB1": (57.0, 5.2)}
PRESSED = {**DEFENSE, "CB2": (50.5, 47.0)}
BUZZ = {"FS": (66.0, 25.0), "SS": (57.0, 40.0)}


def test_a_safety_buzzing_wide_behind_off_corners_is_cover_3():
    play = _play(1, 1, BUZZ, n_frames=21, base=OFF)
    hits = mechanism_hits(defender_phases(play))
    assert hits["mechanism"].tolist() == ["buzz_soft"]
    assert hits.iloc[0]["nfl_id"] == list({**OFFENSE, **OFF}).index("SS") + 1
    assert hits.iloc[0]["text"].endswith("a buzz, leans Cover 3")


@pytest.mark.parametrize(
    ("moves", "base"),
    [
        (BUZZ, PRESSED),  # pressed corners: the dropped Cover 1 row, no call
        ({**BUZZ, "SS": (57.0, 27.0)}, OFF),  # to the middle hole: the dropped robber rows
        ({**BUZZ, "FS": (66.0, 12.0)}, OFF),  # the other safety holds his half
    ],
)
def test_the_buzz_does_not_fire_on_the_shapes_it_dropped(moves, base):
    assert mechanism_hits(defender_phases(_play(1, 1, moves, n_frames=21, base=base))).empty


def test_the_buzz_leads_the_verdict_over_the_rotation():
    from nfl_coverage.analysis.report import disguise_verdict, played_read, structural_read

    play = _play(1, 1, BUZZ, n_frames=21, base=OFF)
    hits = mechanism_hits(defender_phases(play)).to_dict("records")
    played = played_read(play, 49.5, 26.6, 0.1, [], hit=hits[0])
    shown = structural_read(play[play["rel_frame"] == 0], 49.5, 26.6, 9.0, "snap", [])
    label, text = disguise_verdict("rotate_to_mof_closed", shown, played, hits)
    assert label == "buzz_soft" and "a buzz, leans Cover 3" in text


# Pressure (2018 rosters): both LBs mugged 2 yd off the stand-in LOS (49.5), inside the
# tackles (1.6 and 1.9 yd off the center), or off the ball 4.5 yd deep as in DEFENSE.
POSITIONS = {"CB1": "CB", "CB2": "CB", "LB1": "ILB", "LB2": "ILB", "NB": "CB", "FS": "FS"}
MUGGED = {**DEFENSE, "LB1": (51.5, 25.0), "LB2": (51.5, 28.5)}


def _rostered(play, base, dataset="bdb2021"):
    names = list({**OFFENSE, **base})
    for name, pos in {**POSITIONS, "SS": "SS"}.items():
        play.loc[play["nfl_id"] == names.index(name) + 1, "position"] = pos
    return play.assign(dataset=dataset)  # the rush count, and so the rows, are 2018's


def _pressure_hits(moves, base, dataset="bdb2021"):
    play = _rostered(_play(1, 1, moves, n_frames=21, base=base), base, dataset)
    return mechanism_hits(defender_phases(play))


def test_a_mugged_linebacker_who_bails_is_a_front_bluff():
    hits = _pressure_hits({"LB2": (57.5, 28.5)}, MUGGED)  # bails 6 yd in 2 s
    assert hits["mechanism"].tolist() == ["front_bluff"]
    assert hits.iloc[0]["nfl_id"] == list({**OFFENSE, **MUGGED}).index("LB2") + 1
    assert hits.iloc[0]["text"].endswith("likely 4 or fewer rush")


@pytest.mark.parametrize(
    ("moves", "base"),
    [
        ({"LB2": (47.5, 28.5)}, MUGGED),  # both mugged, nobody bails: the heat the look shows
        ({"LB2": (57.5, 32.0)}, {**DEFENSE, "LB2": (51.5, 28.5)}),  # only one mugged
        ({"LB2": (57.5, 28.5)}, {**MUGGED, "SS": (63.0, 27.0)}),  # single-high: SS in the post
        ({"LB2": (57.5, 28.5), "LB1": (47.5, 25.0)}, MUGGED),  # LB1 shoots the gap: a live blitz
    ],
)
def test_a_front_bluff_needs_two_mugged_two_high_a_bail_and_no_partner_rushing(moves, base):
    assert _pressure_hits(moves, base).empty


def test_an_off_ball_linebacker_coming_downhill_is_a_second_level_rusher():
    hits = _pressure_hits({"LB1": (44.0, 22.0)}, DEFENSE)  # 10 yd toward the line in 2 s
    assert hits["mechanism"].tolist() == ["second_level_rush"]
    assert hits.iloc[0]["nfl_id"] == list({**OFFENSE, **DEFENSE}).index("LB1") + 1
    assert hits.iloc[0]["text"].endswith("a second-level rusher, likely 5+ rush")


def test_a_second_level_rusher_needs_nobody_mugged():
    base = {**DEFENSE, "LB2": (51.5, 28.5)}
    assert _pressure_hits({"LB1": (44.0, 22.0)}, base).empty


def test_pressure_rows_need_a_roster_position():
    play = _play(1, 1, {"LB1": (44.0, 22.0)}, n_frames=21, base=DEFENSE)  # film: no positions
    assert mechanism_hits(defender_phases(play.assign(dataset="bdb2021"))).empty


@pytest.mark.parametrize(
    ("moves", "base"),
    [({"LB2": (57.5, 28.5)}, MUGGED), ({"LB1": (44.0, 22.0)}, DEFENSE)],
)
def test_pressure_rows_stay_off_2023_which_has_no_rush_count(moves, base):
    assert _pressure_hits(moves, base, dataset="bdb2026").empty


# A single-high look with an off corner: the FS 19 yd deep and 5 yd off the center, the
# widest corner 9.5 yd deep (technique, not a deep third, by the page's 12 yd corner line),
# the SS in the box, two LBs mugged and one bailing.
OFF_CORNER_SINGLE_HIGH = {**MUGGED, "FS": (68.5, 31.6), "CB2": (59.0, 47.0), "SS": (55.0, 35.0)}


@pytest.mark.xfail(
    strict=True,
    reason="QA 2026-10-20: _two_high counts corners 9-12 yd deep; the page needs 12 for the "
    "widest defender, so it calls this two-high. Open definition question.",
)
def test_the_bluffs_two_high_agrees_with_the_shown_read_on_an_off_corner():
    from nfl_coverage.analysis.report import structural_read

    base = OFF_CORNER_SINGLE_HIGH
    play = _rostered(_play(1, 1, {"LB2": (57.5, 28.5)}, n_frames=21, base=base), base)
    shown = structural_read(play[play["rel_frame"] == 0], 49.5, 26.6, 9.0, "snap", [])
    assert shown.shell == "1-high MOF-closed"
    assert mechanism_hits(defender_phases(play)).empty  # not two high, so no front bluff
