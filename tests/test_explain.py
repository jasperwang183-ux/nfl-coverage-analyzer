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


def test_pre_card_leads_with_the_shell_term():
    c = cards(_report(), {}, MIDDLE)["pre"]
    assert c["call"] == "Cover 2 shell (two safeties deep, middle of the field open)."
    assert c["gloss"] == "Cover 2: two deep safeties, each with a half"
    assert "(14.7 and 13.7 yd)" in c["why"] and "4.2 yd from the middle of the field" in c["why"]
    assert c["sure"] == "Low. 26% of the analysis region is off-frame."
    assert c["flip"] == "The middle-open call is 0.7 yd from flipping to closed."
    assert not c["disagree"] and not c["guess"] and c["grades_key"] == "pre"


def test_two_deep_sharing_the_middle_is_not_worded_as_no_one_over_it():
    rep = _report()
    rep.shown.deep_offsets = [1.1, 7.7]  # play 62: the partner sits under the 8 yd split
    why = cards(rep, {}, MIDDLE)["pre"]["why"]
    assert "no one is over the middle" not in why
    assert "1.1 yd from the middle of the field but the other only 7.7 yd" in why


def test_a_threshold_within_half_a_yard_is_the_reason_ahead_of_caveats():
    rep = _report()
    rep.shown.mof_margin = 0.3
    assert cards(rep, {}, MIDDLE)["pre"]["sure"].startswith("Low. The middle call is 0.3 yd")


def test_post_card_leads_with_the_model_coverage_and_its_structure():
    rep = _report()
    rep.played.mof_closed = True  # the rule reads the middle closed, the model Cover 2 zone
    post = cards(rep, {}, MIDDLE)["post"]
    assert post["call"] == "Cover 2 zone (rotated two-high shell)."
    assert post["grades_key"] == "post_exact"
    assert post["disagree"].startswith("The rule read disagrees: it reads the middle closed")


def test_pole_run_leads_the_disguise_card_with_the_tactic():
    c = cards(_report([POLE], label="pole_run"), {}, MIDDLE)
    assert c["post"]["call"] == "Tampa 2 (Cover 2 shell)." and c["post"]["grades_key"] == "post"
    d = c["disguise"]
    assert d["call"] == "Pole run: Tampa 2 from a Cover 2 shell."
    assert d["also"].startswith("Likeliest: Tampa 2")  # glossed already on the post card
    assert "78% of past plays" in d["also"] and "Can also be Cover 3 zone" in d["also"]
    assert "2.8 yd off the ball" in d["why"] and d["sure"].startswith("Likely.")
    # nearest of the rule's lines: 11.2 deep vs the 10 yd floor (1.2) beats 2.8 vs 4 (1.2 tie,
    # the depth sorts first) and 13 - 11.2 (1.8)
    assert d["flip"].startswith("He had 1.2 yd to spare on")
    assert not d["guess"] and d["grades_key"] == "disguise"


def test_no_rule_fired_names_the_rotation_tactic_as_a_best_guess():
    rep = _report(label="", post=_model(("COVER_2_ZONE", 0.6), ("COVER_4_ZONE", 0.3)))
    d = cards(rep, {"max_move": 0.5}, MIDDLE)["disguise"]
    assert d["guess"] and d["call"] == "Best guess: No disguise (the Cover 2 shell held)."
    assert d["sure"].startswith("Guess.")
    assert "Nobody moved 3 yd" in d["why"]
    assert "Likeliest: Cover 2 zone" in d["also"] and "Can also be Cover 4 zone" in d["also"]
    rot = _report(label="rotate_to_mof_closed")
    call = cards(rot, {"max_move": 4.9, "rotator_dd": -4.9}, MIDDLE)["disguise"]["call"]
    assert call == "Best guess: Spin (Cover 2 shell spun to single-high)."


def test_the_runner_up_needs_ten_percent():
    rep = _report(post=_model(("COVER_2_ZONE", 0.95), ("COVER_4_ZONE", 0.01)))
    assert "Can also be" not in cards(rep, {}, MIDDLE)["disguise"]["also"]


def test_the_model_disagreeing_with_the_pre_snap_read_gets_a_line():
    rep = _report()
    rep.pre_model = _model(("COVER_3_ZONE", 0.61), ("COVER_1_MAN", 0.15))
    pre = cards(rep, {}, MIDDLE)["pre"]
    assert pre["disagree"].startswith("The model disagrees: it leans Cover 3 zone (61%)")


def test_names_are_glossed_on_first_use_only():
    g = _Glosser()
    assert g("Tampa 2") == "Tampa 2 (Cover 2 with a linebacker running the deep middle)"
    assert g("Tampa 2") == "Tampa 2"
    assert g("Cover 2, 4 or 6 shell") == "Cover 2, 4 or 6 shell"  # a list is not glossed
    assert g.note("Cover 3 zone") == "Cover 3: three deep defenders, each with a third"
    assert g.note("Cover 3 zone") == ""
    assert coverage_name("COVER_2_MAN") == "Cover 2 man"
    assert coverage_name("COVER_3_ZONE") == "Cover 3 zone"


def test_the_disguise_card_carries_the_front_and_rush_marked_unsure():
    rep = _report()
    rep.front = {
        "front": "6-1", "n_line": 6, "n_second": 1, "n_box": 7, "rushers": 4, "dropped": 2,
        "rush_class": "simulated pressure", "unsure": "nothing on film checks who rushed",
        "reasons": [],
    }  # fmt: skip
    d = cards(rep, {}, MIDDLE)["disguise"]
    assert d["pressure"] == (
        "Simulated pressure from a 6-1 front: 6 at the line, 2 dropped, 4 rushed; 7 in the box."
    )
    assert d["pressure_sure"].startswith("Unsure: nothing on film checks who rushed.")
    rep.front = {"front": "", "unsure": "the defense counts 13 at the snap, so the sides are off"}
    assert cards(rep, {}, MIDDLE)["disguise"]["pressure"] == "Front unreadable."


def test_a_lowercase_shell_term_mid_sentence():
    rep = _report(label="")
    rep.shown.n_deep, rep.shown.mof_closed = 1, True
    call = cards(rep, {"max_move": 0.5}, MIDDLE)["disguise"]["call"]
    assert call == "Best guess: No disguise (the single-high shell held)."
