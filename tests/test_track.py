"""Tracking assembly: chained calibrations, side carry-over, canonical 10 Hz tracks."""

import cv2
import numpy as np
import pytest

from nfl_coverage.data.standardize import CANONICAL_COLUMNS
from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.track import (
    SideAssignment,
    TrackedFrame,
    canonical_tracks,
    cap_sides,
    chain_calibrations,
    snap_bodies,
)


def _textured(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), 40, dtype=np.uint8)
    for _ in range(200):
        x, y = int(rng.integers(0, 600)), int(rng.integers(0, 340))
        cv2.rectangle(
            img, (x, y), (x + 12, y + 8), tuple(int(v) for v in rng.integers(80, 255, 3)), -1
        )
    for x in range(40, 640, 60):
        cv2.line(img, (x, 0), (x, 359), (230, 230, 230), 2)
    return img


def _identity_calib():
    px = np.float32([[40, 30], [600, 30], [600, 330], [40, 330]])
    field = np.float32([[20, 5], [80, 5], [80, 45], [20, 45]])
    return FieldCalibration.from_points(px, field)


def test_chain_follows_a_slow_pan_and_reanchors():
    base = _textured()
    frames = []
    for i in range(5):
        m = np.float32([[1, 0, 8.0 * i], [0, 1, 0]])  # 8 px per step to the right
        frames.append(
            TrackedFrame(
                i,
                i * 0.1,
                cv2.warpAffine(base, m, (640, 360)),
                np.zeros((0, 4)),
                np.zeros(0, int),
                np.zeros(0),
            )
        )
    anchor = _identity_calib()
    chain = chain_calibrations(frames, anchor, 0)
    assert len(chain.calibrations) == 5 and not chain.notes
    # a field point seen at pixel p in frame 0 sits at p + 32 px in frame 4
    p0 = np.float32([[300.0, 200.0]])
    p4 = p0 + np.float32([[32.0, 0.0]])
    assert np.allclose(chain.calibrations[4].to_field(p4), anchor.to_field(p0), atol=0.5)
    # a keyframe replaces the chained one where given
    chain2 = chain_calibrations(frames, anchor, 0, keyframes={3: anchor})
    assert chain2.calibrations[3] is anchor


def test_canonical_tracks_schema_speeds_and_mirroring():
    calib = _identity_calib()

    # two tracked players: id 7 (offense) standing still, id 9 (defense) moving 1 yd per frame in x
    def frame(i):
        boxes = np.array([[300, 150, 320, 200], [400 + 9.33 * i, 150, 420 + 9.33 * i, 200]], float)
        return TrackedFrame(
            i, i * 0.1, np.zeros((360, 640, 3), np.uint8), boxes, np.array([7, 9]), np.ones(2)
        )

    frames = [frame(i) for i in range(4)]
    sides = SideAssignment({7: "Offense", 9: "Defense"}, set(), 0, 0)
    tr = canonical_tracks(frames, [calib] * 4, sides, snap_index=1, offense="left")
    assert list(tr.columns) == CANONICAL_COLUMNS
    assert sorted(tr["rel_frame"].unique()) == [-1, 0, 1, 2]
    d9 = tr[tr["nfl_id"] == 9].sort_values("rel_frame")
    assert d9["s"].iloc[1:].mean() == pytest.approx(10.0, rel=0.05)  # 1 yd / 0.1 s
    assert (tr.loc[tr["nfl_id"] == 7, "s"] == 0).all()
    assert (tr.loc[tr["nfl_id"] == 7, "position"] == "QB").all()  # deepest (only) offensive player
    mirrored = canonical_tracks(frames, [calib] * 4, sides, 1, offense="right")
    assert np.allclose(
        mirrored.loc[mirrored["nfl_id"] == 7, "x"], 120 - tr.loc[tr["nfl_id"] == 7, "x"]
    )


def test_line_players_and_unknown_ids_are_dropped():
    calib = _identity_calib()
    boxes = np.array([[300, 150, 320, 200], [400, 150, 420, 200], [500, 150, 520, 200]], float)
    fr = TrackedFrame(
        0, 0.0, np.zeros((360, 640, 3), np.uint8), boxes, np.array([1, 2, -1]), np.ones(3)
    )
    sides = SideAssignment({1: "Offense", 2: "Defense"}, {2}, 0, 0, None)
    tr = canonical_tracks([fr], [calib], sides, 0, "left")
    assert tr["nfl_id"].tolist() == [1]
    with pytest.raises(ValueError, match="no tracked players"):
        canonical_tracks([fr], [calib], SideAssignment({}, set(), 0, 0, None), 0, "left")


def test_a_body_within_4_yd_of_a_sideline_at_the_snap_is_not_a_player():
    # 7.5 px per yard of field width, field y 0 (a sideline) at image row 30
    px = np.float32([[40, 30], [600, 30], [600, 330], [40, 330]])
    calib = FieldCalibration.from_points(px, np.float32([[20, 0], [80, 0], [80, 40], [20, 40]]))
    img = np.full((360, 640, 3), (40, 120, 40), np.uint8)
    feet = [(150, 15.0), (250, 18.0), (350, 21.0), (150, 25.0), (250, 28.0), (350, 31.0)]
    feet.append((450, 3.8))  # the line judge as measured on 15-34-46
    boxes = []
    for k, (col, y) in enumerate(feet):
        row = 30 + 7.5 * y
        box = (col - 8, row - 40, col + 8, row)
        color = (40, 40, 220) if k in (0, 1, 2, 6) else (220, 60, 40)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, -1)
        boxes.append(box)
    fr = TrackedFrame(0, 0.0, img, np.array(boxes, float), np.arange(1, 8), np.ones(7))
    b = snap_bodies(fr, calib)
    assert b.ids[~b.is_player].tolist() == [7]


def test_officials_stay_out_when_the_tracker_sees_them_again():
    from nfl_coverage.vision.track import _sideline_stray

    # ruled out at the snap: marked "-", so a later frame does not re-side the id
    sides = SideAssignment({1: "Offense", 2: "Defense", 3: "-"}, set(), 1, 0, None)
    calib = _identity_calib()
    boxes = np.array([[300, 150, 320, 200], [400, 150, 420, 200], [500, 150, 520, 200]], float)
    fr = TrackedFrame(
        1, 0.1, np.zeros((360, 640, 3), np.uint8), boxes, np.array([1, 2, 3]), np.ones(3)
    )
    tr = canonical_tracks([fr], [calib], sides, 0, "left")
    assert sorted(tr["nfl_id"].unique()) == [1, 2]
    # a new "defender" by the sideline on the offense's side of the ball; the offense's
    # backfield is x < LOS for offense "left" and x > LOS for "right" (as in cap_sides)
    los = 50.0
    assert _sideline_stray(np.array([40.0, 1.9]), "Defense", los, "left")
    assert _sideline_stray(np.array([60.0, 51.5]), "Defense", los, "right")
    assert not _sideline_stray(np.array([60.0, 1.9]), "Defense", los, "left")  # defense side
    assert not _sideline_stray(np.array([40.0, 20.0]), "Defense", los, "left")  # mid-field
    assert not _sideline_stray(np.array([40.0, 1.9]), "Offense", los, "left")
    assert not _sideline_stray(np.array([40.0, 1.9]), "Defense", None, "left")


def test_cap_sides_moves_the_surplus_back_across_the_line():
    los = 50.0  # offense on the right (x > 50)
    x = np.array([60.0] * 8 + [50.8, 51.5] + [40.0] * 9 + [49.8, 49.0, 50.4])
    sides = np.array(["Offense"] * 10 + ["Defense"] * 12)  # 12 defense, 10 offense
    capped = cap_sides(sides, x, los, "right")
    assert (capped == "Defense").sum() == 11 and (capped == "Offense").sum() == 11
    # the defense body standing on the offense's side of the ball (x=50.4) moved
    assert capped[-1] == "Offense"
    # nothing moves when both sides are legal
    ok = np.array(["Offense"] * 11 + ["Defense"] * 11)
    assert (cap_sides(ok, x[:22], los, "right") == ok).all()


def test_a_line_defender_who_drops_into_coverage_is_kept():
    calib = _identity_calib()  # 9.33 px per yd along x; offense left, so deeper is +x

    def frame(i):
        boxes = np.array(
            [
                [300, 150, 320, 200],  # 1: offense, off the line
                [380, 150, 400, 200],  # 4: offensive lineman
                [420 + 9.33 * i, 150, 440 + 9.33 * i, 200],  # 2: mugged, drops 1 yd a frame
                [430, 200, 450, 250],  # 3: defensive tackle, holds
            ],
            float,
        )
        return TrackedFrame(
            i, i * 0.1, np.zeros((360, 640, 3), np.uint8), boxes, np.array([1, 4, 2, 3]), np.ones(4)
        )

    sides = SideAssignment(
        {1: "Offense", 4: "Offense", 2: "Defense", 3: "Defense"}, {2, 3, 4}, 0, 0, None
    )
    tr = canonical_tracks([frame(i) for i in range(7)], [calib] * 7, sides, 0, "left")
    assert sorted(tr["nfl_id"].unique()) == [1, 2]


def test_film_centers_keep_each_rule_on_its_tuned_reference():
    from nfl_coverage.vision.pipeline import _canonical, film_centers

    # offense lined up on the right: canonical mirrors the ball to the other hash
    los_c, ball_c = _canonical("right", 70.0, 29.73)
    assert los_c == pytest.approx(50.0) and ball_c == pytest.approx(23.57)
    c = film_centers(ball_c, offense_y=30.0)
    assert c.mechanisms == pytest.approx(23.57) and c.mechanisms_from == "the ball"
    assert c.middle == pytest.approx((23.57 + 26.65) / 2)  # halfway to the field's middle
    assert c.middle_from.startswith("halfway")
    none = film_centers(None, offense_y=30.0)
    assert none.middle == none.mechanisms == 30.0
    assert "no ball spot" in none.middle_from
