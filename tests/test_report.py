"""Structural reads with confidence, separate from the model's variant probabilities."""

import numpy as np

from nfl_coverage.analysis.report import (
    CoverageReport,
    disguise_text,
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
    assert r.confidence == "high"
    assert any("middle open" in x for x in r.reasons)


def test_caveats_and_thin_margins_lower_confidence():
    r = structural_read(
        _frame(), 49.5, 26.6, 9.0, "snap", caveats=["deep zone beyond 14 yd is off-frame"]
    )
    assert r.confidence == "medium"
    # a safety sitting right on the deep threshold: the count could flip either way
    edgy = [d for d in DEFENSE if d != (63.0, 35.0)] + [(58.7, 35.0)]
    r2 = structural_read(_frame(edgy), 49.5, 26.6, 9.0, "snap", caveats=[])
    assert r2.depth_margin < 0.5 and r2.confidence == "low"


def test_post_safety_reads_closed():
    post = [d for d in DEFENSE if d not in ((63.0, 18.0), (63.0, 35.0))] + [(64.0, 27.0)]
    r = structural_read(_frame(post), 49.5, 26.6, 9.0, "+2 s", caveats=[])
    assert r.n_deep == 1 and r.mof_closed
    assert "Cover 1 or Cover 3" in r.family


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
