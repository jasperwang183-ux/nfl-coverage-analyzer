"""Structural reads with confidence, separate from the model's variant probabilities."""

import numpy as np
import pandas as pd

from nfl_coverage.analysis.report import (
    CoverageReport,
    disguise_text,
    model_read,
    played_read,
    structural_read,
    variant_confidence,
)
from nfl_coverage.vision.still import frame_from_points

# two-high open shell, offense -> right, LOS 50 (as in test_still)
OFFENSE = [
    (43.0, 26.6),
    (49.5, 5.0),
    (49.5, 12.0),
    (49.5, 20.0),
    (49.5, 35.0),
    (49.5, 40.0),
    (49.5, 48.0),
]
DEFENSE = (
    [(52.0, y) for y in (8, 16, 24, 32, 40)]
    + [(50.5, 5.2), (50.5, 47.8)]
    + [
        (56.0, 12.0),
        (56.0, 41.0),
        (63.0, 18.0),
        (63.0, 35.0),
    ]
)


def _frame(defense=DEFENSE):
    sides = np.array(["Offense"] * len(OFFENSE) + ["Defense"] * len(defense))
    return frame_from_points(np.array(OFFENSE + defense), los=50.0, offense="left", side=sides)


def test_two_high_open_reads_high_confidence():
    r = structural_read(_frame(), los=49.5, center=26.6, deep_depth=9.0, frame="snap", caveats=[])
    assert r.n_deep == 2 and not r.mof_closed
    assert r.shell == "2-high MOF-open"
    assert "Cover 2 / Cover 4 / Cover 6" in r.family
    assert r.corners_pressed == (True, True)
    assert r.confidence == "high" and r.confidence_why == []
    assert any("middle open" in x for x in r.reasons)


def test_caveats_and_thin_margins_lower_confidence():
    r = structural_read(
        _frame(), 49.5, 26.6, 9.0, "snap", caveats=["deep zone beyond 14 yd is off-frame"]
    )
    assert r.confidence == "medium"
    assert r.confidence_why == ["deep zone beyond 14 yd is off-frame"]
    # a safety sitting right on the deep threshold: the count could flip either way
    edgy = [d for d in DEFENSE if d != (63.0, 35.0)] + [(58.7, 35.0)]
    r2 = structural_read(_frame(edgy), 49.5, 26.6, 9.0, "snap", caveats=[])
    assert r2.depth_margin < 0.5 and r2.confidence == "low"
    assert any("from the deep line" in w for w in r2.confidence_why)


def test_post_safety_reads_closed():
    post = [d for d in DEFENSE if d not in ((63.0, 18.0), (63.0, 35.0))] + [(64.0, 27.0)]
    r = structural_read(_frame(post), 49.5, 26.6, 9.0, "+2 s", caveats=[])
    assert r.n_deep == 1 and r.mof_closed
    assert "Cover 1 or Cover 3" in r.family


def _two_frames(before, after):
    """A play read at +2.0 s whose defenders were at `before` 0.3 s earlier."""
    a, b = _frame(before), _frame(after)
    a["rel_frame"], b["rel_frame"] = 17, 20
    return pd.concat([a, b], ignore_index=True)


def test_count_reads_where_a_defender_is_headed():
    base = [d for d in DEFENSE if d not in ((63.0, 18.0), (63.0, 35.0))]
    # deep safety settled at 15 yd; a corner bailing 9.1 -> 10.0 yd (3 yd/s, headed to 10.6);
    # a corner squatting at 10.0 yd; a robber driving down 12 -> 11 yd (headed to 10.3)
    before = base + [(64.5, 27.0), (58.6, 5.0), (59.5, 48.0), (61.5, 22.0)]
    after = base + [(64.5, 27.0), (59.5, 5.0), (59.5, 48.0), (60.5, 22.0)]
    r = played_read(_two_frames(before, after), 49.5, 26.6, 0.1, [])
    assert r.deep_depths == [15.0, 10.0]
    assert any("10.0 yd, +3.0 yd/s -> headed to 10.6 yd -> deep" in x for x in r.reasons)
    assert any("11.0 yd, -3.3 yd/s -> headed to 10.3 yd -> not deep" in x for x in r.reasons)
    plain = played_read(_two_frames(before, after), 49.5, 26.6, 0.1, [], count_test=None)
    assert plain.deep_depths == [15.0, 11.0]  # a plain line keeps the robber, drops the corner


def test_variant_confidence_and_disguise_text():
    assert variant_confidence([("COVER_4_ZONE", 0.75), ("COVER_3_ZONE", 0.1)]) == "high"
    assert variant_confidence([("COVER_4_ZONE", 0.48), ("COVER_3_ZONE", 0.21)]) == "medium"
    assert variant_confidence([("COVER_1_MAN", 0.44), ("COVER_3_ZONE", 0.3)]) == "low"
    shown = structural_read(_frame(), 49.5, 26.6, 9.0, "snap", [])
    post = [d for d in DEFENSE if d not in ((63.0, 18.0), (63.0, 35.0))] + [(66.0, 27.0)]
    played = structural_read(_frame(post), 49.5, 26.6, 14.0, "+2 s", [])
    text = disguise_text("rotate_to_mof_closed", shown, played)
    assert "rotated to a post safety" in text and "Cover 1 or Cover 3" in text
    rep = CoverageReport(
        shown,
        played,
        [("COVER_4_ZONE", 0.5), ("COVER_3_ZONE", 0.2)],
        "medium",
        "rotate_to_mof_closed",
        1.0,
        text,
        [],
    )
    s = rep.summary()
    assert "SHOWN" in s and "PLAYED +2 s" in s and "DISGUISE" in s
    assert rep.to_dict()["shown"]["shell"] == "2-high MOF-open"


def test_model_read_ranks_and_levels_a_head():
    r = model_read(
        "sequence_v1",
        2.96,
        {"COVER_3_ZONE": 0.1, "COVER_4_ZONE": 0.8, "COVER_1_MAN": 0.1},
        {"MAN_COVERAGE": 0.2, "ZONE_COVERAGE": 0.8},
    )
    assert r.t == 3.0
    assert [c for c, _ in r.variant] == ["COVER_4_ZONE", "COVER_3_ZONE", "COVER_1_MAN"]
    assert r.confidence == "high"
    assert r.man_zone[0][0] == "ZONE_COVERAGE"
