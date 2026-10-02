"""Disguise mechanisms: named movements, read from each defender's phases.

A mechanism is a row, not a branch: who can run it (a role at the snap), what his
trigger and landmark must look like (`features.phases`), which coverage families it
points to, and how to say it on a card. `mechanism_hits` runs every row over a phases
table; `mechanism_component` turns the hits into the harness's component shape.

Thresholds are fitted on tuning weeks (2018 and 2023 weeks 1-4) against the coverage
labels, never taken from articles; each row records what it bought there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

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


def _deep_middle_run(p: pd.DataFrame) -> pd.Series:
    """A box defender who commits deep and is in the deep middle by the landmark.

    A mugged linebacker counts (Warner on SF's Tampa 2 of 2023 wk 2 aligned 1.9 yd off
    the ball): on weeks 1-4 the 26 runners aligned inside 3 yd were all linebackers or
    safeties, and keeping them raised the lift (4.39 -> 4.44).
    """
    return (
        (p["role"] == "box")
        & (p["commit_v_depth"] >= 1.5)
        & (p["landmark_depth"] >= 9.0)
        & (p["landmark_lat"].abs() <= 4.0)
    )


def _safeties_at_snap(p: pd.DataFrame) -> pd.Series:
    return (p["role"] == "safety").groupby([p[k] for k in KEYS]).transform("sum")


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
    fires=lambda p: _deep_middle_run(p) & (_safeties_at_snap(p) > 0),
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

# Cover 2 trap / Cloud: a corner aligned soft, reading like a Cover 3 or 4 bail, stops after
# his first steps and drives downhill to squat the flat. Its opposite, press to bail, is
# Cover 3 technique, not a disguise, so it is not a row.
# NOT FITTED YET: the thresholds below are placeholders until `scripts/fit_corner_roll_up.py`
# runs on the tuning weeks, so the row stays out of MECHANISMS until it shows lift there.
ROLL_UP_ALIGN_MIN = 6.0  # yd off the LOS at the snap: soft, not press
ROLL_UP_FIRST_MIN = 0.0  # yd/s over +0.3-0.7 s: he gave ground or held, the bail look
ROLL_UP_COMMIT_MAX = -1.0  # yd/s over +0.7-1.2 s: driving toward the LOS
ROLL_UP_LANDMARK_MAX = 6.0  # yd deep at the landmark: a flat squat, not a deep third
ROLL_UP_LAT_MIN = 8.0  # yd off the formation center at the landmark: outside, in the flat


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
    implies=frozenset({"COVER_2_ZONE"}),
    fires=_corner_roll_up,
    text=lambda r: (
        f"a corner aligned {r['align_depth']:.0f} yd off stopped his bail and drove downhill "
        f"({r['commit_v_depth']:.1f} yd/s by +1.2 s) to squat the flat "
        f"{r['landmark_depth']:.0f} yd deep at +{r['landmark_t']:.1f} s: a Cover 2 trap or "
        "Cloud corner, not a deep third"
    ),
    downhill=True,
)

MECHANISMS: tuple[Mechanism, ...] = (POLE_RUN, LATE_POST)
# Rows written but not yet shown to lift their families on the tuning weeks: the scripts
# can score them, the page does not run them.
CANDIDATES: tuple[Mechanism, ...] = (CORNER_ROLL_UP,)

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
