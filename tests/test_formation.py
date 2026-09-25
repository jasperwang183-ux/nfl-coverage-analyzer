"""Formation inference at the snap: offense side, LOS from the two fronts, ball y."""

import numpy as np
import pytest

from nfl_coverage.vision.formation import cluster_sides, infer_formation, split_fronts

LOS = 60.0
BALL_Y = 26.3
OFF, DEF = 0, 1


def _formation(shotgun: bool = True, goal_line: bool = False):
    """A 21-body snap: offense on the right of x=60 (color 0), defense on the left (color 1).

    Foot points as film gives them: the offensive line ~0.8 yd behind the ball,
    the defensive line ~1.0 yd off it.
    """
    bodies = []
    for dy in (-4.0, -2.0, 0.0, 2.0, 4.0):  # offensive line; the center at the ball's y
        bodies.append((LOS + 0.8, BALL_Y + dy, OFF))
    if shotgun:
        bodies.append((LOS + 5.0, BALL_Y + 0.2, OFF))  # QB behind the center
        bodies.append((LOS + 5.2, BALL_Y - 1.8, OFF))  # RB beside him
    else:
        bodies.append((LOS + 1.6, BALL_Y + 0.1, OFF))  # QB under center: inside the line cluster
        bodies.append((LOS + 7.0, BALL_Y - 3.5, OFF))  # offset back
    bodies += [(LOS + 0.8, 10.0, OFF), (LOS + 0.8, 45.0, OFF), (LOS + 0.8, BALL_Y + 6.0, OFF)]
    for dy in (-3.0, -1.0, 1.0, 3.0):  # defensive line
        bodies.append((LOS - 1.0, BALL_Y + dy, DEF))
    lb, cb, sf = (4.0, 4.0, 4.0) if goal_line else (5.0, 6.0, 13.0)
    bodies += [(LOS - lb, 22.0, DEF), (LOS - lb, 31.0, DEF)]  # linebackers
    bodies += [(LOS - cb, 10.0, DEF), (LOS - cb, 45.0, DEF)]  # corners
    bodies += [(LOS - sf, 17.0, DEF), (LOS - sf, 36.0, DEF)]  # safeties
    arr = np.array(bodies)
    return arr[:, :2], arr[:, 2].astype(int), np.ones(len(arr), dtype=bool)


def _mirror(xy):
    m = xy.copy()
    m[:, 0] = 120.0 - m[:, 0]
    return m


def test_shotgun_formation_reads_side_los_and_ball():
    xy, lab, isp = _formation()
    f = infer_formation(xy, lab, isp)
    assert f.offense == "right" and f.offense_confidence == "high"
    assert abs(f.los - LOS) < 0.3  # middle of the 1.8 yd neutral zone, 0.1 off the ball
    assert f.los_range == pytest.approx((LOS - 1.0, LOS + 0.8))
    assert f.los_confidence == "high"
    assert f.ball_y == pytest.approx(BALL_Y + 0.2)
    assert f.ball_y_confidence == "medium"


def test_mirrored_play_flips_the_side_and_mirrors_the_los():
    xy, lab, isp = _formation()
    f = infer_formation(_mirror(xy), lab, isp)
    assert f.offense == "left"
    assert abs(f.los - (120.0 - LOS)) < 0.3


def test_under_center_falls_back_to_the_center_for_ball_y():
    xy, lab, isp = _formation(shotgun=False)
    f = infer_formation(xy, lab, isp)
    assert f.offense == "right"
    assert f.ball_y_confidence == "low"
    assert abs(f.ball_y - BALL_Y) < 1.0


def test_referee_behind_the_offense_is_ignored_when_flagged():
    xy, lab, isp = _formation()
    # the referee 16 yd behind the offense would out-deep the safeties
    xy = np.vstack([xy, [LOS + 16.0, 30.0]])
    lab = np.append(lab, OFF)
    flagged = np.append(isp, False)
    assert infer_formation(xy, lab, flagged).offense == "right"
    # unflagged (the striped-shirt check misses small officials), the second-deepest
    # body still reads the side, at low confidence and with the reason given
    unflagged = infer_formation(xy, lab, np.append(isp, True))
    assert unflagged.offense == "right" and unflagged.offense_confidence == "low"
    assert "official" in unflagged.reasons[0]
    assert unflagged.suspect_official == len(xy) - 1  # the referee, set aside


def test_shaded_defenders_in_the_white_cluster_do_not_break_the_team_split():
    xy, lab, isp = _formation()
    # a corner and a linebacker cluster with the offense: half the clear bodies
    # of the white cluster now stand on the defense side, which is why clusters
    # are anchored on the deep defenders rather than on where their bodies stand
    for k, (x, y) in enumerate(xy):
        if (x, y) in {(LOS - 6.0, 10.0), (LOS - 5.0, 22.0)}:
            lab[k] = OFF
    f = infer_formation(xy, lab, isp)
    assert f.offense == "right"
    assert abs(f.los - LOS) < 0.3


def test_goal_line_defense_without_deep_players_is_not_called():
    xy, lab, isp = _formation(goal_line=True)
    # every defender within 4 yd; the offense's QB and back 5 yd back: no deep side
    f = infer_formation(xy, lab, isp)
    assert f.offense is None
    assert f.ball_y is None
    assert "too close to call" in f.reasons[0]


def test_hints_are_used_as_given():
    xy, lab, isp = _formation()
    f = infer_formation(xy, lab, isp, los_hint=59.5, offense_hint="right")
    assert f.los == 59.5 and f.los_confidence == "given"
    assert f.offense == "right" and f.offense_confidence == "given"


def test_cluster_sides_anchors_on_deep_defenders():
    xy, lab, _ = _formation()
    assert cluster_sides(xy[:, 0], lab, LOS, "right") == {DEF: "Defense", OFF: "Offense"}
    assert cluster_sides(xy[:, 0], lab, LOS, "left") is None  # no deep "defenders" on the right


def test_split_fronts_counts_violations_and_prefers_the_widest_gap():
    s = np.array([-1.0, -0.8, 0.9, 1.1, 1.3])
    is_off = np.array([False, False, True, False, True])  # one mislabeled defender behind
    viol, lo, hi = split_fronts(s, is_off)
    assert viol == 1 and (lo, hi) == (-0.8, 0.9)


def test_too_few_players_raises():
    with pytest.raises(ValueError):
        infer_formation(np.zeros((3, 2)), np.zeros(3, int), np.ones(3, bool))
