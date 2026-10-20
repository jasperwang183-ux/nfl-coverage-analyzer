"""Disguise mechanisms: named movements, read from each defender's phases.

A mechanism is a row, not a branch: who can run it (a role at the snap), what his
trigger and landmark must look like (`features.phases`), which coverage families it
points to, and how to say it on a card. `mechanism_hits` runs every row over a phases
table; `mechanism_component` turns the hits into the harness's component shape.

Thresholds are fitted on tuning weeks (2018 and 2023 weeks 1-13, every week outside the
holdout) against the coverage labels, never taken from articles; each row records what it
bought there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from nfl_coverage.analysis.report import CORNER_DEEP_DEPTH
from nfl_coverage.analysis.roles import LINE_DEPTH
from nfl_coverage.analysis.shell import DEEP_DEPTH, MOF_HALFWIDTH, MOF_PARTNER_MIN
from nfl_coverage.features.alignment import KEYS

Kind = Literal["coverage", "pressure", "technique"]


@dataclass(frozen=True)
class Mechanism:
    name: str
    kind: Kind
    implies: frozenset[str]  # the labels a hit points to
    fires: Callable[[pd.DataFrame], pd.Series]  # per defender row of defender_phases
    text: Callable[[pd.Series], str]
    downhill: bool = False  # the defender who ran it is the fastest toward the LOS, not away


# yd past the ball at the landmark: in the deep middle. Refit from the ball on tuning weeks
# 1-13 (scripts/fit_mechanism_depths.py, 2026-10-13): the pole run's lift holds within
# 2 SE of its best at 10-10.5 yd in both seasons (5.04 / 5.62 at 10, 581 plays; 4.25 /
# 5.31 at 9, 1,065 plays). With the pole run's 13 yd cap the plateau shrinks to 12 alone,
# set by 36 plays of 2018 at the grid edge (54 in all); 10 kept: too few plays to trust.
# The late post fires too rarely to fit and shares it.
RUN_LANDMARK_MIN = 10.0
# yd past the ball at the landmark: past this the runner is the safety, not a pole runner.
# Tuning weeks, a safety deep at the snap: runners landing 13+ yd deep lift Cover 2 zone
# only 4.17 / 2.57 (17 / 15 plays); capping at 13 keeps the lift (5.08 / 5.84, 368 / 182
# plays). Pole run only: with nobody deep, 13+ yd runners are Cover 1/3 on 17 of 17 plays,
# the late post's own safety.
POLE_LANDMARK_MAX = 13.0


def _deep_middle_run(p: pd.DataFrame, landmark_min: float = RUN_LANDMARK_MIN) -> pd.Series:
    """A box defender who commits deep and is in the deep middle by the landmark.

    A mugged linebacker counts (Warner on SF's Tampa 2 of 2023 wk 2 aligned 1.9 yd off
    the ball): on weeks 1-4 the 26 runners aligned inside 3 yd were all linebackers or
    safeties, and keeping them raised the lift (4.39 -> 4.44).
    """
    return (
        (p["role"] == "box")
        & (p["commit_v_depth"] >= 1.5)
        & (p["landmark_depth"] >= landmark_min)
        & (p["landmark_lat"].abs() <= 4.0)
    )


def _safeties_at_snap(p: pd.DataFrame) -> pd.Series:
    return (p["role"] == "safety").groupby([p[k] for k in KEYS]).transform("sum")


def _pole_run(
    p: pd.DataFrame,
    landmark_min: float = RUN_LANDMARK_MIN,
    landmark_max: float = POLE_LANDMARK_MAX,
) -> pd.Series:
    return (
        _deep_middle_run(p, landmark_min)
        & (p["landmark_depth"] < landmark_max)
        & (_safeties_at_snap(p) > 0)
    )


def _speed(r: pd.Series) -> str:
    return (
        f"({r['commit_v_depth']:.1f} yd/s by +1.2 s, "
        f"{r['landmark_depth']:.0f} yd deep at +{r['landmark_t']:.1f} s)"
    )


# The same run means two things. With a safety already deep it splits the safeties:
# Tampa 2's pole runner, or a Cover 3 hook defender carrying the seam. With nobody deep
# at the snap it becomes the only deep defender: a post safety hidden in the box.
# Weeks 1-4 (7,630 labeled plays), with a safety deep at the snap: from a single-high look
# Cover 2 zone 6% -> 34% and Cover 1 42% -> 4%, from two-high Cover 2 zone 13% -> 64%.
# Holdout (scored once): single-high 7.5% -> 42%, Cover 1 39% -> 3%.
POLE_RUN = Mechanism(
    name="pole_run",
    kind="coverage",
    implies=frozenset({"COVER_2_ZONE"}),
    fires=_pole_run,
    text=lambda r: (
        f"a box defender ran the deep middle between the safeties {_speed(r)}: Tampa 2's pole "
        "runner, or a Cover 3 hook defender carrying the seam; not man"
    ),
)

# 0-high looks where a box defender ran the deep middle: 10 plays on weeks 1-4, 21 on the
# holdout, Cover 1 or Cover 3 on 80-88% of them. Too few to fit; the analyst's 1533
# (0-high look, Cover 1 man) is one.
LATE_POST = Mechanism(
    name="late_post",
    kind="coverage",
    implies=frozenset({"COVER_1_MAN", "COVER_3_ZONE"}),
    fires=lambda p: _deep_middle_run(p) & (_safeties_at_snap(p) == 0),
    text=lambda r: (
        f"with nobody deep at the snap, a box defender ran to the deep middle {_speed(r)}: "
        "a post safety hidden in the box, Cover 1 or Cover 3"
    ),
)

# Cover 2 trap: a corner aligned soft, reading like a Cover 3 or 4 bail, comes downhill to
# squat the flat. Its opposite, press to bail, is Cover 3 technique, not a disguise, so it
# is not a row. Fitted by `scripts/fit_corner_roll_up.py` on weeks 1-4 (7,620 plays): soft
# Cover 2 corners never bail first (first-step velocity p25 -0.2 yd/s vs +0.4 in Cover 3),
# so the first-step condition is off. Fires on 78 plays (1.0%): Cover 2 zone 68% (base 12%),
# Cover 6 zone 21% (the half side), Cover 3 zone 3% (base 34%: no Cloud), Cover 1 man 5%.
# By the look at the snap: from two-high Cover 2/6 zone 30% -> 97% and Cover 4 24% -> 3%
# (59 plays, the trap read as quarters), from single-high 9% -> 67% (18 plays, Cover 1 22%).
# Holdout (scored once): lift 3.1 (0.9% of plays); single-high 12% -> 68% and Cover 1
# 37% -> 14% (28 plays); two-high 33% -> 74% but Cover 4 stays 20% (46 plays).
# Refit from the ball on tuning weeks 1-13 (2026-10-13): the roll-up's lift is flat over
# cushions of 4-8 yd and landmarks of 6.5-10 yd (3.5-4.1); its tell is the downhill drive
# into the flat, not these depths. The middle of that plateau:
ROLL_UP_ALIGN_MIN = 5.5  # yd off the ball at the snap: soft, not press
ROLL_UP_FIRST_MIN = None  # yd/s over +0.3-0.7 s; None: the data drops it
ROLL_UP_COMMIT_MAX = -0.5  # yd/s over +0.7-1.2 s: coming toward the LOS
ROLL_UP_LANDMARK_MAX = 8.0  # yd deep at the landmark: a flat squat, not a deep third
ROLL_UP_LAT_MIN = 16.0  # yd off the formation center at the landmark: wide, in the flat


def _corner_roll_up(
    p: pd.DataFrame,
    align_min: float = ROLL_UP_ALIGN_MIN,
    first_min: float | None = ROLL_UP_FIRST_MIN,
    commit_max: float = ROLL_UP_COMMIT_MAX,
    landmark_max: float = ROLL_UP_LANDMARK_MAX,
    lat_min: float | None = ROLL_UP_LAT_MIN,
) -> pd.Series:
    """A corner aligned soft who drives downhill into the flat. None skips a condition."""
    fires = (
        (p["role"] == "corner")
        & (p["align_depth"] >= align_min)
        & (p["commit_v_depth"] <= commit_max)
        & (p["landmark_depth"] <= landmark_max)
    )
    if first_min is not None:
        fires &= p["first_v_depth"] >= first_min
    if lat_min is not None:
        fires &= p["landmark_lat"].abs() >= lat_min
    return fires


CORNER_ROLL_UP = Mechanism(
    name="corner_roll_up",
    kind="coverage",
    implies=frozenset({"COVER_2_ZONE", "COVER_6_ZONE"}),
    fires=_corner_roll_up,
    text=lambda r: (
        f"a corner aligned {r['align_depth']:.0f} yd off came downhill "
        f"({r['commit_v_depth']:.1f} yd/s by +1.2 s) to squat the flat "
        f"{r['landmark_depth']:.0f} yd deep at +{r['landmark_t']:.1f} s: a Cover 2 trap corner "
        "(or Cover 6's half side), not a deep third"
    ),
    downhill=True,
)

# Buzz: a safety deep at the snap comes down wide toward the curl-flat while the corners
# play off and the other safety ends deep in the middle: Cover 3. Fitted on tuning weeks
# 1-13 (scripts/fit_mechanism_depths.py, 2026-10-19) by the analyst's rule: from the most
# precise setting whose seasons agree within 5 pts, loosen an edge while it buys at least
# 2.5 right calls per wrong one, looking past one short step (3 lands on the same setting),
# then require 85% (no season under 82%) on 200+ plays with the seasons within 5 pts, and
# neighbours that also pass. 447 plays, Cover 3 85.7%
# (84.7 / 86.8). Its sideways speed adds nothing past where he lands, so it is not a
# condition. Dropped by the same rule (Cover 1 / 3 from the same drop): to the middle
# hole (press corners 70 stable plays, under the 75 floor; off corners best 68%), and wide
# with press corners (best 55% on 200+ plays). Those plays keep the rotation line.
# Holdout pass marks, set before it was scored (analyst, 2026-10-19), on the pooled share
# right when it fires: 82% or more stays strong, 70-82% turns lean, under 70% drops it.
# Holdout (scored once, 2026-10-19; never re-score): 182 plays, Cover 3 81.3% (80.5 /
# 82.1), Cover 1 on 26 of them, so the card says it leans.
LOW_COMMIT_MAX = -0.5  # yd/s over +0.7-1.2 s: coming toward the LOS
PARTNER_DEPTH_MIN = 10.0  # yd past the ball at the landmark
PARTNER_LAT_MAX = 6.0  # yd off the formation center at the landmark
BUZZ_LAT_MIN = 7.5  # yd off the formation center at the landmark: wide
BUZZ_DEPTH_MAX = 10.0  # yd past the ball at the landmark: shallow
BUZZ_SOFT_MIN = 4.0  # yd: the corners' mean depth off the ball at the snap


def _corner_cushion(p: pd.DataFrame) -> pd.Series:
    corners = p["align_depth"].where(p["role"] == "corner")
    return corners.groupby([p[k] for k in KEYS]).transform("mean")


def _partner_deep_middle(p: pd.DataFrame) -> pd.Series:
    """Another safety of the play is deep in the middle at the landmark."""
    deep = (
        (p["role"] == "safety")
        & (p["landmark_depth"] >= PARTNER_DEPTH_MIN)
        & (p["landmark_lat"].abs() <= PARTNER_LAT_MAX)
    ).astype(int)
    return deep.groupby([p[k] for k in KEYS]).transform("sum") - deep > 0


def _buzz(
    p: pd.DataFrame, lat_min: float = BUZZ_LAT_MIN, soft_min: float = BUZZ_SOFT_MIN
) -> pd.Series:
    return (
        (p["role"] == "safety")
        & (p["commit_v_depth"] <= LOW_COMMIT_MAX)
        & (p["landmark_lat"].abs() > lat_min)
        & (p["landmark_depth"] < BUZZ_DEPTH_MAX)
        & (_corner_cushion(p) >= soft_min)
        & _partner_deep_middle(p)
    )


BUZZ_SOFT = Mechanism(
    name="buzz_soft",
    kind="coverage",
    implies=frozenset({"COVER_3_ZONE"}),
    fires=_buzz,
    text=lambda r: (
        f"a safety came down wide to the curl-flat {_speed(r)} with the corners off and the "
        "other safety deep in the middle: a buzz, leans Cover 3"
    ),
    downhill=True,
)

# Pressure: who rushes, not the coverage. A pressure row points to the rush count
# (RUSH_LE4 / RUSH_GE5, from plays.numberOfPassRushers), which only 2018 has: 2023
# tracking carries no rushers. 2018 tracks linebackers and DBs but almost no linemen,
# and film has no positions. 2023 tracking does carry positions, so the rows are gated to
# PRESSURE_DATASET: they fire on 2018 tracking only, where they were fitted. Fitted on 2018
# tuning weeks 1-13 (13,058 plays; 74% rush 4 or fewer) by the analyst's walk
# (scripts/fit_mechanism_depths.py, 2026-10-19), weeks 1-7 vs 8-13 standing in for two
# seasons:
# - Front bluff: two or more LBs or DBs mugged within BLUFF_LAT_MAX of the ball (inside
#   the tackles: the A and B gaps), two high at the snap, and one of them bails (before
#   the QA's partner-rush kill, below). 140 plays,
#   4 or fewer rush 79.3% (80.2 / 77.6), strong against the analyst's 75% floor; the look
#   alone rushes 4 or fewer on 68.9% (334), and the same look and bail from single-high on
#   74.4% (195). Widening to 6 yd took 340 plays at 76.2%, but there a two-high mugged look
#   where nobody bails is already 73.2%, so the analyst kept the tackles (2026-10-19).
# - Second-level rusher: nobody mugged within RUSH_LAT_MAX and an off-ball LB or DB comes
#   downhill. 551 plays, 5 or more rush 73.7% (75.2 / 71.4), strong against 70%. The rest
#   are most likely a lineman dropping out (a sim), which 2018 cannot see. With the commit
#   window stopping at the throw (2026-10-20): 549 plays, 74.3% (76.1 / 71.7), still strong.
# Holdout pass marks, set before it was scored (analyst, 2026-10-19), on the share right
# when it fires, 3 points under each tuning floor: front bluff 72% or more strong, 60-72%
# lean, under 60% dropped; second-level rusher 67% strong, 55-67% lean, under 55% dropped.
# Holdout (2018 weeks 14-17, scored once, 2026-10-19; never re-score): front bluff 51
# plays, 4 or fewer rush 80.4% (base 71.8%); second-level rusher 167 plays, 5 or more
# rush 77.2% (base 28.2%). Both stay strong.
# Front bluff after the QA (analyst, 2026-10-20): this row achieved an official blind
# holdout score of 80.4% (51 plays) under the original v1 definition. Post-exam QA
# refactored the logic to enforce a strict partner-rush kill switch (PARTNER_RUSH_MAX: any
# other mugged defender coming downhill faster than 1 yd/s kills the row). This refactor
# isolated an 89.7% precision tell on tuning weeks (88.6 / 91.3), shrinking the volume to
# 58 plays. The historic 80.4% holdout score remains as a validated baseline for the
# structural shape, but carries an asterisk as it does not map 1:1 to this post-QA
# tightened rule.
# Then the two-high test moved to the page's 12 yd line for the widest defender, and the
# commit window to stop at the throw (both 2026-10-20). At 4 yd / bail 2 the row now fires
# on 53 plays, 4 or fewer rush 88.7% (87.1 / 90.9). Kept there by the analyst: it passes
# the elite precision exception (88%+ on the tuning weeks needs 50 plays, not 75;
# scripts/fit_mechanism_depths.py) with the halves 3.8 pts apart and a passing neighbour
# (3.5 yd: 52 plays, 88.5%). The walk itself does not stop there: it loosens to bail 0
# (77 plays, 85.7%, 88.9 / 81.2) and drops it on the 5-pt gap.
SECOND_LEVEL = frozenset({"OLB", "ILB", "MLB", "LB", "CB", "SS", "FS", "DB", "S"})
PRESSURE_DATASET = "bdb2021"  # the one season with a rush count to fit and score on
BLUFF_LAT_MAX = 4.0  # yd off the formation center at the snap: inside the tackles
BAIL_MIN = 2.0  # yd/s over +0.7-1.2 s away from the LOS
# yd/s over +0.7-1.2 s: a mugged partner coming toward the LOS faster than this is rushing,
# and kills the bluff (a live blitz that traded gaps, not a look dissolving into coverage).
# Partners split cleanly on the tuning weeks: downhill ones mostly at -2 to -4 yd/s, the
# rest bail or hold, so the line anywhere in -1..0 kept the same plays (56-58 of 140,
# before the two-high and throw changes).
PARTNER_RUSH_MAX = -1.0
RUSH_LAT_MAX = 3.0  # yd off the formation center at the snap: nobody mugged in here
RUSH_DOWNHILL_MAX = -4.5  # yd/s over +0.7-1.2 s toward the LOS


def _mugged(p: pd.DataFrame, lat_max: float) -> pd.Series:
    return (
        p["position"].isin(SECOND_LEVEL)
        & (p["align_depth"] < LINE_DEPTH)
        & (p["align_lat"].abs() <= lat_max)
    )


def _n_mugged(p: pd.DataFrame, lat_max: float) -> pd.Series:
    return _mugged(p, lat_max).astype(int).groupby([p[k] for k in KEYS]).transform("sum")


def _two_high(p: pd.DataFrame) -> pd.Series:
    """Two or more deep at the snap with the middle open (`shell.mof_is_closed`, per play).

    Deep as the page reads it (`report.structural_read`): the widest defender on each side
    is deep only past CORNER_DEEP_DEPTH (an off corner at 9-11 yd is technique, not a deep
    third), everyone else past DEEP_DEPTH. Before the QA (2026-10-20) corners counted at 9,
    so a single-high look with an off corner read two-high here and single-high on the page.
    """
    keys = [p[k] for k in KEYS]
    lat = p["align_lat"]
    widest = (lat == lat.groupby(keys).transform("min")) | (
        lat == lat.groupby(keys).transform("max")
    )
    line = np.where(widest, max(DEEP_DEPTH, CORNER_DEEP_DEPTH), DEEP_DEPTH)
    off = lat.abs().where(p["align_depth"] >= line)
    rank = off.groupby(keys).rank(method="first")
    nearest = off.where(rank == 1).groupby(keys).transform("max")
    partner = off.where(rank == 2).groupby(keys).transform("max")
    closed = (nearest <= MOF_HALFWIDTH + 0.05) & (partner >= MOF_PARTNER_MIN - 0.05)
    return (off.notna().groupby(keys).transform("sum") >= 2) & ~closed


def _front_bluff(
    p: pd.DataFrame,
    lat_max: float = BLUFF_LAT_MAX,
    bail_min: float = BAIL_MIN,
    partner_max: float = PARTNER_RUSH_MAX,
) -> pd.Series:
    """A mugged LB or DB who bails, with two or more mugged, two high at the snap, and no
    other mugged defender coming downhill."""
    mugged = _mugged(p, lat_max)
    # the bailer himself is never counted: he moves away from the LOS (bail_min >= 0)
    rushing = (mugged & (p["commit_v_depth"] < partner_max)).astype(int)
    return (
        (p["dataset"] == PRESSURE_DATASET)
        & mugged
        & (p["commit_v_depth"] > bail_min)
        & (_n_mugged(p, lat_max) >= 2)
        & (rushing.groupby([p[k] for k in KEYS]).transform("sum") == 0)
        & _two_high(p)
    )


def _second_level_rush(
    p: pd.DataFrame, lat_max: float = RUSH_LAT_MAX, downhill_max: float = RUSH_DOWNHILL_MAX
) -> pd.Series:
    """An off-ball LB or DB who comes downhill fast, with nobody mugged at the snap."""
    return (
        (p["dataset"] == PRESSURE_DATASET)
        & p["position"].isin(SECOND_LEVEL)
        & (p["align_depth"] >= LINE_DEPTH)
        & (p["commit_v_depth"] < downhill_max)
        & (_n_mugged(p, lat_max) == 0)
    )


FRONT_BLUFF = Mechanism(
    name="front_bluff",
    kind="pressure",
    implies=frozenset({"RUSH_LE4"}),
    fires=_front_bluff,
    text=lambda r: (
        f"from two high, two or more LBs or DBs were mugged at the line, a {r['position']} "
        f"bailed ({r['commit_v_depth']:.1f} yd/s by +1.2 s) and none came downhill: an A-gap "
        "bluff, likely 4 or fewer rush"
    ),
)

SECOND_LEVEL_RUSH = Mechanism(
    name="second_level_rush",
    kind="pressure",
    implies=frozenset({"RUSH_GE5"}),
    fires=_second_level_rush,
    text=lambda r: (
        f"with nobody mugged, a {r['position']} {r['align_depth']:.0f} yd off the ball came "
        f"downhill ({r['commit_v_depth']:.1f} yd/s by +1.2 s): a second-level rusher, "
        "likely 5+ rush"
    ),
    downhill=True,
)

# The film path reads only the first coverage hit, so the roll-up goes after the deep runs.
# The buzz never fires with the roll-up or the late post on the tuning weeks, and with the
# pole run on 2 plays. Pressure rows are never a coverage hit, so their place is free.
MECHANISMS: tuple[Mechanism, ...] = (
    POLE_RUN,
    LATE_POST,
    CORNER_ROLL_UP,
    BUZZ_SOFT,
    FRONT_BLUFF,
    SECOND_LEVEL_RUSH,
)
# Rows written but not yet shown to lift their families on the tuning weeks: the scripts
# can score them, the page does not run them.
CANDIDATES: tuple[Mechanism, ...] = ()

HIT_COLUMNS = ["mechanism", "kind", "nfl_id", "t", "text"]


def mechanism_hits(
    phases: pd.DataFrame, mechanisms: tuple[Mechanism, ...] = MECHANISMS
) -> pd.DataFrame:
    """KEYS + HIT_COLUMNS: one row per play and mechanism, the defender who ran it fastest."""
    parts = []
    for m in mechanisms:
        hit = phases[m.fires(phases).fillna(False)]
        if hit.empty:
            continue
        speed = -hit["commit_v_depth"] if m.downhill else hit["commit_v_depth"]
        best = hit.loc[speed.groupby([hit[k] for k in KEYS]).idxmax()]
        parts.append(
            best[KEYS].assign(
                mechanism=m.name,
                kind=m.kind,
                nfl_id=best["nfl_id"],
                t=best["landmark_t"],
                text=[m.text(r) for _, r in best.iterrows()],
            )
        )
    if not parts:
        return pd.DataFrame(columns=KEYS + HIT_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def mechanism_component(phases: pd.DataFrame, mechanism: Mechanism = POLE_RUN) -> pd.DataFrame:
    """The harness component shape for one mechanism: KEYS + <name>_score, <name>_label.

    Score 1 on a hit, 0 on a play whose phases were read, NaN when no defender of the
    play has a trigger (too short, or tracks dropped).
    """
    read = phases.groupby(KEYS)["commit_v_depth"].apply(lambda s: s.notna().any())
    hits = mechanism_hits(phases, (mechanism,)).set_index(KEYS)
    out = pd.DataFrame(index=read.index)
    out[f"{mechanism.name}_score"] = read.map({True: 0.0, False: float("nan")})
    out.loc[out.index.isin(hits.index), f"{mechanism.name}_score"] = 1.0
    out[f"{mechanism.name}_label"] = ""
    out.loc[out.index.isin(hits.index), f"{mechanism.name}_label"] = mechanism.name
    out[f"{mechanism.name}_id"] = hits["nfl_id"].reindex(out.index)
    return out.reset_index()
