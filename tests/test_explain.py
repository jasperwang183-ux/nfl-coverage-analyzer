"""The cards' four lines (analysis.explain) from hand-built reads."""

from nfl_coverage.analysis.explain import _Glosser, cards, coverage_name
from nfl_coverage.analysis.report import CoverageReport, ModelRead, StructuralRead

MIDDLE = "from the middle of the field"


def _read(n_deep=2, closed=False, depths=(14.7, 13.7), offsets=(4.2, 11.0), **kw):
    base = {
        "frame": "snap",
        "n_deep": n_deep,
        "mof_closed": closed,
        "shell": "2-high MOF-open",
        "family": "Cover 2 / Cover 4 / Cover 6 (two-high, middle open)",
        "n_defenders": 11,
        "deep_depths": list(depths),
        "deep_offsets": list(offsets),
        "corner_depths": (6.0, 6.0),
        "corners_pressed": (False, False),
        "depth_margin": 3.0,
        "mof_margin": 0.7,
        "confidence": "low",
        "confidence_why": ["26% of the analysis region is off-frame"],
    }
    base.update(kw)
    return StructuralRead(**base)


def _model(*ranked):
    return ModelRead("sequence_v1", 2.0, list(ranked), "medium", [("ZONE_COVERAGE", 0.9)])


POLE = {
    "mechanism": "pole_run",
    "kind": "coverage",
    "nfl_id": 13,
    "t": 2.0,
    "text": "a box defender ran the deep middle",
    "align_depth": 6.5,
    "landmark_depth": 11.2,
    "landmark_lat": 2.8,
    "commit_v_depth": 1.8,
}


def _report(hits=(), label="", post=None):
    played = _read(
        frame="+1.5 s (middle) / +2.0 s (count)",
        depths=(18.0, 18.0),
        family="Tampa 2 (Cover 2 zone, a runner down the middle)" if hits else "x",
        depth_margin=0.2,
    )
    return CoverageReport(
        shown=_read(),
        played=played,
        variant=[],
        variant_confidence="low",
        disguise_label=label,
        disguise_score=0.0,
        disguise_text="",
        caveats=[],
        pre_model=_model(("COVER_2_ZONE", 0.41), ("COVER_2_MAN", 0.26)),
        post_model=post or _model(("COVER_2_ZONE", 0.95), ("COVER_4_ZONE", 0.01)),
        mechanisms=list(hits),
    )


def test_pre_card_says_the_call_why_how_sure_and_could_flip():
    c = cards(_report(), {}, MIDDLE)["pre"]
    assert c["call"] == "Two safeties deep, middle of the field open (Cover 2, 4 or 6 shell)."
    assert "(14.7 and 13.7 yd)" in c["why"] and "4.2 yd from the middle of the field" in c["why"]
    assert c["sure"] == "Low. 26% of the analysis region is off-frame."
    assert c["flip"] == "The middle-open call is 0.7 yd from flipping to closed."
    assert not c["disagree"] and not c["guess"]


def test_a_threshold_within_half_a_yard_is_the_reason_ahead_of_caveats():
    rep = _report()
    rep.shown.mof_margin = 0.3
    assert cards(rep, {}, MIDDLE)["pre"]["sure"].startswith("Low. The middle call is 0.3 yd")


def test_pole_run_names_tampa_2_with_its_odds_and_the_runner_up():
    c = cards(_report([POLE], label="pole_run"), {}, MIDDLE)
    assert c["post"]["call"].startswith("Two deep halves, a linebacker running the deep middle")
    d = c["disguise"]
    assert d["call"] == "Tampa 2 from a two-high look."
    assert d["also"].startswith("Likeliest: Tampa 2")  # glossed already on the post card
    assert "78% of past plays" in d["also"] and "Can also be Cover 3" in d["also"]
    assert "2.8 yd off the ball" in d["why"] and d["sure"].startswith("Likely.")
    # nearest of the rule's lines: 11.2 deep vs the 10 yd floor (1.2) beats 2.8 vs 4 (1.2 tie,
    # the depth sorts first) and 13 - 11.2 (1.8)
    assert d["flip"].startswith("He had 1.2 yd to spare on")
    assert not d["guess"]


def test_no_rule_fired_is_a_marked_best_guess_with_the_model_odds():
    rep = _report(label="", post=_model(("COVER_2_ZONE", 0.6), ("COVER_4_ZONE", 0.3)))
    d = cards(rep, {"max_move": 0.5}, MIDDLE)["disguise"]
    assert d["guess"] and d["call"].startswith("Best guess: no disguise.")
    assert d["sure"].startswith("Guess.")
    assert "Nobody moved 3 yd" in d["why"]
    assert "Likeliest: Cover 2" in d["also"] and "Can also be Cover 4" in d["also"]


def test_the_runner_up_needs_ten_percent():
    rep = _report(post=_model(("COVER_2_ZONE", 0.95), ("COVER_4_ZONE", 0.01)))
    assert "Can also be" not in cards(rep, {}, MIDDLE)["disguise"]["also"]


def test_the_model_disagreeing_with_the_rule_read_gets_a_line():
    rep = _report(post=_model(("COVER_3_ZONE", 0.7), ("COVER_1_MAN", 0.2)))
    post = cards(rep, {}, MIDDLE)["post"]
    assert post["disagree"].startswith("The model disagrees: it leans Cover 3 (70%)")


def test_names_are_glossed_on_first_use_only():
    g = _Glosser()
    assert g("Tampa 2") == "Tampa 2 (Cover 2 with a linebacker running the deep middle)"
    assert g("Tampa 2") == "Tampa 2"
    assert g("Cover 2, 4 or 6 shell") == "Cover 2, 4 or 6 shell"  # a list is not glossed
    assert coverage_name("COVER_2_MAN") == "Cover 2 man"
    assert coverage_name("COVER_3_ZONE") == "Cover 3"
