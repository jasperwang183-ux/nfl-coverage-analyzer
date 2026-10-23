"""The cards in words: Call, Why, How sure, Could flip (the analyst's template, 2026-10-22).

Every card says the same four things in the same order:
  - call: the answer in plain words, the football name in parentheses;
  - why: the two or three numbers that decided it, each tied to a landmark;
  - sure: one level plus the single biggest reason;
  - flip: the nearest threshold as a distance, the line a reader can check on film.
`also` names the likeliest coverage and the runner-up when a call has two readings,
`disagree` says when the sequence model leans another way, `guess` marks a disguise call no
rule fired for ("Best guess:"). Football names are glossed on first use across the cards.
Mechanism distances are "off the ball"; the middle call's are from its own reference
(`lateral`, the film's middle-of-field point).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from nfl_coverage.analysis.report import CoverageReport, ModelRead, StructuralRead

GLOSS = {
    "Cover 0": "no deep safety, man across with extra rushers",
    "Cover 1": "man coverage with one deep safety in the middle",
    "Cover 2": "two deep safeties, each with a half",
    "Cover 3": "three deep defenders, each with a third",
    "Cover 4": "four deep defenders, each with a quarter",
    "Cover 6": "quarters to one side, a deep half to the other",
    "Tampa 2": "Cover 2 with a linebacker running the deep middle",
    "Cover 2 trap": "a soft corner jumps the flat instead of bailing",
    "Cover 3 buzz": "a safety rotates down to the curl-flat",
}

# (deep count, middle closed) -> plain words, the shell's football name
SHELL_WORDS = {
    (0, False): ("Nobody deep", "Cover 0"),
    (0, True): ("Nobody deep", "Cover 0"),
    (1, True): ("One safety deep, over the middle", "Cover 1 or Cover 3 shell"),
    (1, False): ("One deep defender, off the middle", "a shaded single-high shell"),
    (2, False): ("Two safeties deep, middle of the field open", "Cover 2, 4 or 6 shell"),
    (2, True): ("Two deep, one of them over the middle", "a rotated two-high shell"),
    (3, True): ("Three deep, middle closed", "Cover 3"),
    (3, False): ("Three deep, middle open", "a quarters-halves rotation"),
    (4, True): ("Four deep", "Cover 4, quarters"),
    (4, False): ("Four deep", "Cover 4, quarters"),
}
# plain words for a played family a mechanism names
NAMED_WORDS = {
    "Tampa 2": "Two deep halves, a linebacker running the deep middle",
    "Cover 2 trap": "Two deep halves, a corner squatting the flat",
}
LOOK_WORDS = {"two-high": "two-high", "single-high": "single-high", "0-high": "zero-high"}

# What each coverage mechanism was on the tuning weeks (2018 and 2023 weeks 1-13, ball-spot
# phases, `data/processed/phases_tuning_ball.parquet`, 2026-10-22), by the look at the
# snap: (plays, [(label, share)...]). The likeliest names the call; the next is "can also".
MECHANISM_ODDS = {
    ("pole_run", "two-high"): (344, [("COVER_2_ZONE", 78), ("COVER_3_ZONE", 12)]),
    ("pole_run", "single-high"): (206, [("COVER_2_ZONE", 52), ("COVER_3_ZONE", 37)]),
    ("late_post", "0-high"): (38, [("COVER_3_ZONE", 42), ("COVER_1_MAN", 29)]),
    ("corner_roll_up", "two-high"): (176, [("COVER_2_ZONE", 59), ("COVER_6_ZONE", 32)]),
    ("corner_roll_up", "single-high"): (106, [("COVER_2_ZONE", 55), ("COVER_1_MAN", 20)]),
    ("buzz_soft", "two-high"): (356, [("COVER_3_ZONE", 86), ("COVER_1_MAN", 12)]),
    ("buzz_soft", "single-high"): (91, [("COVER_3_ZONE", 86), ("COVER_1_MAN", 11)]),
}
# a mechanism's own name for a label it points to
MECHANISM_NAMES = {
    ("pole_run", "COVER_2_ZONE"): "Tampa 2",
    ("pole_run", "COVER_3_ZONE"): "Cover 3 (a hook defender carrying the seam)",
    ("corner_roll_up", "COVER_2_ZONE"): "Cover 2 trap",
    ("buzz_soft", "COVER_3_ZONE"): "Cover 3 buzz",
}
MOF_CLOSED = {"COVER_1_MAN": True, "COVER_3_ZONE": True, "COVER_2_MAN": False,
              "COVER_2_ZONE": False, "COVER_4_ZONE": False, "COVER_6_ZONE": False}  # fmt: skip
NUMBERS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
           "Eleven"]  # fmt: skip
ALSO_MIN = 10  # %: a runner-up under this is not worth "can also be"
LIST_MAX = 4  # depths listed in a Why line before "and N more"
NEAR = 0.5  # yd: a threshold this close is the reason a call is unsure, ahead of caveats
HIGH_MARGIN = 99.0  # report.structural_read's stand-in for "no threshold near"


@dataclass
class Card:
    call: str
    why: str
    sure: str
    flip: str
    grades: str  # what the card's Correct / Wrong judges
    also: str = ""
    disagree: str = ""
    guess: bool = False


class _Glosser:
    """Glosses a football name the first time the cards use it."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, name: str, inside: bool = False) -> str:
        """`inside`: the name already sits in parentheses, so the gloss follows a colon."""
        if " or " in name:  # a list of coverages: glossing one would mislead
            return name
        key = next(
            (
                k
                for k in sorted(GLOSS, key=len, reverse=True)
                if name == k or name.startswith((k + " ", k + ","))
            ),
            None,
        )
        if key is None or key in self.seen:
            return name
        self.seen.add(key)
        if name != key:
            return f"{name}; {GLOSS[key]}"
        return f"{name}: {GLOSS[key]}" if inside else f"{name} ({GLOSS[key]})"


def coverage_name(label: str) -> str:
    """COVER_2_MAN -> Cover 2 man; COVER_3_ZONE -> Cover 3."""
    parts = label.split("_")
    if len(parts) == 3 and parts[0] == "COVER":
        name = f"Cover {parts[1]}"
        return f"{name} man" if parts[2] == "MAN" and parts[1] == "2" else name
    return label.replace("_", " ").title()


def _shell(read: StructuralRead) -> tuple[str, str]:
    return SHELL_WORDS[(min(read.n_deep, 4), read.mof_closed)]


def _look(read: StructuralRead) -> str:
    if read.n_deep == 0:
        return "0-high"
    return "two-high" if read.n_deep >= 2 and not read.mof_closed else "single-high"


def _yds(values: list[float]) -> str:
    v = [f"{x:.1f}" for x in values]
    if len(v) > LIST_MAX:
        return ", ".join(v[:LIST_MAX]) + f" and {len(v) - LIST_MAX} more"
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + f" and {v[-1]}"


def _middle_words(read: StructuralRead, lateral: str) -> str:
    """Where the deep players sit relative to the middle, in one clause."""
    off = read.deep_offsets
    if read.n_deep == 1:
        where = "so he covers the middle" if read.mof_closed else "so the middle is open"
        return f"{off[0]:.1f} yd {lateral}, {where}"
    if read.mof_closed:
        return (
            f"the closest is {off[0]:.1f} yd {lateral} and the next {off[1]:.1f} yd away, so "
            "one of them is over the middle"
        )
    return f"the closest is {off[0]:.1f} yd {lateral}, so no one is over the middle"


def _sure(read: StructuralRead) -> str:
    level = read.confidence.capitalize()
    if read.confidence == "high":
        return f"{level}. Every fact sits 2 yd or more from its line and the field is in view."
    if read.depth_margin < NEAR:
        why = f"a defender is {read.depth_margin:.1f} yd from the deep line"
    elif read.mof_margin < NEAR:
        why = f"the middle call is {read.mof_margin:.1f} yd from flipping"
    else:
        why = (read.confidence_why or ["the reads sit near their lines"])[0]
    return f"{level}. {why[0].upper()}{why[1:]}."


def _flip(read: StructuralRead) -> str:
    if min(read.depth_margin, read.mof_margin) >= HIGH_MARGIN:
        return "Nothing is close to a line."
    if read.depth_margin <= read.mof_margin:
        return (
            f"A defender is {read.depth_margin:.1f} yd from the deep line, so the deep count "
            "could change."
        )
    side, other = ("closed", "open") if read.mof_closed else ("open", "closed")
    return f"The middle-{side} call is {read.mof_margin:.1f} yd from flipping to {other}."


def _disagree(read: StructuralRead, model: ModelRead | None) -> str:
    if model is None or not model.variant:
        return ""
    label, p = model.variant[0]
    closed = MOF_CLOSED.get(label)
    if label == "COVER_0_MAN":
        differs = read.n_deep > 0
    elif closed is None or read.n_deep == 0:
        differs = closed is not None
    else:
        differs = closed != (read.mof_closed and read.n_deep < 4)
    if not differs:
        return ""
    return (
        f"The model disagrees: it leans {coverage_name(label)} ({100 * p:.0f}%), a "
        f"{'middle-closed' if closed else 'middle-open' if closed is False else 'no-deep'} "
        "coverage."
    )


def _rotation_words(rotation: dict) -> str:
    move = rotation.get("max_move")
    if move is None or move < 3:
        return ""
    dd = rotation.get("rotator_dd") or 0.0
    parts = [f"one defender {'dropped' if dd >= 0 else 'came up'} {abs(dd):.1f} yd"]
    for key, one, many in (
        ("n_came_down", "a deep defender came down", "deep defenders came down"),
        ("n_bailed_inside", "a box or slot defender bailed deep", "box or slot defenders bailed"),
        ("n_corner_bail", "a corner bailed deep", "corners bailed deep"),
    ):
        n = int(rotation.get(key) or 0)
        if n:
            parts.append(one if n == 1 else f"{n} {many}")
    return ", ".join(parts)


def _read_card(
    read: StructuralRead,
    model: ModelRead | None,
    lateral: str,
    gloss: _Glosser,
    when: str,
    deep_line: str,
    grades: str,
    extra: str = "",
    name: str | None = None,
    middle: str | None = None,
) -> Card:
    words, shell_name = _shell(read)
    words = NAMED_WORDS.get(name, words)
    call = f"{words} ({gloss(name or shell_name, inside=True)})."
    if read.n_deep == 0:
        why = f"Nobody is {deep_line} deep {when}."
    else:
        n = NUMBERS[read.n_deep] if read.n_deep < len(NUMBERS) else str(read.n_deep)
        verb = "is" if read.n_deep == 1 else "are"
        why = (
            f"{n} defender{'s' if read.n_deep != 1 else ''} {verb} {deep_line} deep {when} "
            f"({_yds(read.deep_depths)} yd), and {middle or _middle_words(read, lateral)}."
        )
    if extra:
        why += f" {extra[0].upper()}{extra[1:]}."
    return Card(call, why, _sure(read), _flip(read), grades, disagree=_disagree(read, model))


# ------------------------------------------------------------------ the disguise card

# What each mechanism's runner must clear (mirrors analysis.mechanisms): field, limit,
# "max"/"min", the limit in words, and the way he would have to move to break it.
THRESHOLDS = {
    "pole_run": [
        ("landmark_lat", 4.0, "max", "the 4 yd limit off the ball", "wider"),
        ("landmark_depth", 10.0, "min", "the 10 yd depth he must reach", "shallower"),
        ("landmark_depth", 13.0, "max", "the 13 yd cap, past which he is a safety", "deeper"),
    ],
    "late_post": [
        ("landmark_lat", 4.0, "max", "the 4 yd limit off the ball", "wider"),
        ("landmark_depth", 10.0, "min", "the 10 yd depth he must reach", "shallower"),
    ],
    "corner_roll_up": [
        ("align_depth", 5.5, "min", "the 5.5 yd cushion that makes him soft", "tighter"),
        ("landmark_depth", 8.0, "max", "the 8 yd depth of a flat squat", "deeper"),
        ("landmark_lat", 16.0, "min", "the 16 yd width of the flat", "narrower"),
    ],
    "buzz_soft": [
        ("landmark_lat", 7.5, "min", "the 7.5 yd width of the curl-flat", "narrower"),
        ("landmark_depth", 10.0, "max", "the 10 yd depth of a buzz", "deeper"),
    ],
}


def _story(h: dict) -> str:
    m = h["mechanism"]
    t, lat = h.get("t", 0.0), abs(h.get("landmark_lat", 0.0))
    a, d, v = h.get("align_depth", 0.0), h.get("landmark_depth", 0.0), h.get("commit_v_depth", 0.0)
    if m == "pole_run":
        return (
            f"A box defender aligned {a:.1f} yd deep ran to {d:.1f} yd deep and "
            f"{lat:.1f} yd off the ball by +{t:.1f} s ({v:.1f} yd/s): he took the deep middle "
            "between the safeties."
        )
    if m == "late_post":
        return (
            f"With nobody deep at the snap, a box defender aligned {a:.1f} yd deep ran "
            f"to {d:.1f} yd deep in the middle by +{t:.1f} s: he is the post safety, hidden in "
            "the box."
        )
    if m == "corner_roll_up":
        return (
            f"A corner aligned {a:.1f} yd deep came downhill ({abs(v):.1f} yd/s) to {d:.1f} yd "
            f"deep and {lat:.1f} yd off the ball by +{t:.1f} s: he squats the flat instead of "
            "bailing to a deep third."
        )
    if m == "buzz_soft":
        return (
            f"A safety deep at the snap came down ({abs(v):.1f} yd/s) to {d:.1f} yd deep and "
            f"{lat:.1f} yd off the ball by +{t:.1f} s, with the corners off and the other safety "
            "deep in the middle: he buzzed down to the curl-flat."
        )
    return h.get("text", "")


def _mechanism_flip(h: dict) -> str:
    rows = []
    for col, limit, kind, words, way in THRESHOLDS.get(h["mechanism"], []):
        if h.get(col) is None:
            continue
        v = abs(h[col]) if col == "landmark_lat" else h[col]
        rows.append((limit - v if kind == "max" else v - limit, words, way))
    if not rows:
        return "The rule fired; its margins were not recorded."
    m, words, way = min(rows)
    return f"He had {m:.1f} yd to spare on {words}: {m:.1f} yd {way} and the rule would not fire."


def _disguise_card(rep: CoverageReport, rotation: dict, gloss: _Glosser) -> Card:
    shown, played = rep.shown, rep.played
    look = _look(shown)
    shown_words = f"{LOOK_WORDS[look]} look"
    grades = "the disguise call"
    hit = next((h for h in rep.mechanisms if h.get("kind") == "coverage"), None)
    model = rep.post_model
    if hit is not None:
        m = hit["mechanism"]
        n, odds = MECHANISM_ODDS.get((m, look), (0, []))
        names = [(MECHANISM_NAMES.get((m, lab), coverage_name(lab)), p) for lab, p in odds]
        best = names[0][0] if names else "A named move"
        also = ""
        if names:
            also = f"Likeliest: {gloss(names[0][0])}, {names[0][1]}% of past plays with this move."
            if len(names) > 1 and names[1][1] >= ALSO_MIN:
                also += f" Can also be {gloss(names[1][0])}, {names[1][1]}%."
        pct = names[0][1] if names else 0
        level = "Likely" if pct >= 75 else "Leans" if pct >= 50 else "Lean only"
        sure = (
            f"{level}. When this move came from a {LOOK_WORDS[look]} look on past plays it was "
            f"{best} {pct}% of the time ({n} plays)."
            if names
            else "Lean only. This move was too rare from this look to count."
        )
        disagree = ""
        if model and model.variant and odds and model.variant[0][0] not in dict(odds):
            label, p = model.variant[0]
            disagree = f"The model disagrees: it leans {coverage_name(label)} ({100 * p:.0f}%)."
        return Card(
            call=f"{best} from a {shown_words}.",
            why=_story(hit),
            sure=sure,
            flip=_mechanism_flip(hit),
            grades=grades,
            also=also,
            disagree=disagree,
        )

    played_words = _shell(played)[0].lower() if played else "no post-snap read"
    label = rep.disguise_label
    if not label or label in ("corner_bail", "threshold_flicker"):
        call = f"Best guess: no disguise. The {shown_words} held ({played_words})."
    else:
        call = f"Best guess: showed {_shell(shown)[0].lower()}, played {played_words}."
    moved = _rotation_words(rotation)
    why = (f"After the snap {moved}. " if moved else "Nobody moved 3 yd after the snap. ") + (
        "No disguise rule fired in the first second."
    )
    also = ""
    if model and model.variant:
        (a, pa), *rest = model.variant
        also = f"Likeliest: {gloss(coverage_name(a))}, {100 * pa:.0f}% per the sequence model."
        if rest and 100 * rest[0][1] >= ALSO_MIN:
            also += f" Can also be {gloss(coverage_name(rest[0][0]))}, {100 * rest[0][1]:.0f}%."
    flip = _flip(played) if played else "No post-snap read to flip."
    return Card(
        call=call,
        why=why,
        sure="Guess. No named rule fired, so this is read from where the defenders ended up.",
        flip=flip,
        grades=grades,
        also=also,
        guess=True,
    )


def cards(
    rep: CoverageReport, rotation: dict, lateral: str, played_when: str = "by +2.0 s"
) -> dict[str, dict]:
    """The three cards of a report, as JSON-able dicts: pre, post, disguise."""
    gloss = _Glosser()
    pre = _read_card(
        rep.shown, rep.pre_model, lateral, gloss, "at the snap", "9 yd or more",
        "the pre-snap shell (how many deep, middle open or closed)",
    )  # fmt: skip
    out = {"pre": pre}
    if rep.played is not None:
        hit = next((h for h in rep.mechanisms if h.get("kind") == "coverage"), None)
        name = None
        if hit and ("Tampa 2" in rep.played.family or "trap" in rep.played.family):
            name = "Tampa 2" if "Tampa 2" in rep.played.family else "Cover 2 trap"
        mid = rep.played.frame.split(" (middle)")[0]
        middle = f"the middle reads {'closed' if rep.played.mof_closed else 'open'} at {mid}"
        out["post"] = _read_card(
            rep.played, rep.post_model, lateral, gloss, played_when, "at or headed past 10.5 yd",
            "the played shell", extra=_rotation_words(rotation), name=name, middle=middle,
        )  # fmt: skip
    out["disguise"] = _disguise_card(rep, rotation, gloss)
    return {k: asdict(v) for k, v in out.items()}
