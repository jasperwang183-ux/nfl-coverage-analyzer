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

from nfl_coverage.analysis.roles import LINE_DEPTH
from nfl_coverage.features.alignment import KEYS

Kind = Literal["coverage", "pressure", "technique"]


@dataclass(frozen=True)
class Mechanism:
    name: str
    kind: Kind
    implies: frozenset[str]  # the labels a hit points to
    fires: Callable[[pd.DataFrame], pd.Series]  # per defender row of defender_phases
    text: Callable[[pd.Series], str]


def _deep_middle_run(p: pd.DataFrame) -> pd.Series:
    """A box defender who commits deep and is in the deep middle by the landmark."""
    return (
        (p["role"] == "box")
        & (p["align_depth"] >= LINE_DEPTH)
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

MECHANISMS: tuple[Mechanism, ...] = (POLE_RUN, LATE_POST)

HIT_COLUMNS = ["mechanism", "kind", "nfl_id", "t", "text"]


def mechanism_hits(
    phases: pd.DataFrame, mechanisms: tuple[Mechanism, ...] = MECHANISMS
) -> pd.DataFrame:
    """KEYS + HIT_COLUMNS: one row per play and mechanism, the defender who ran it deepest."""
    parts = []
    for m in mechanisms:
        hit = phases[m.fires(phases).fillna(False)]
        if hit.empty:
            continue
        best = hit.loc[hit.groupby(KEYS)["landmark_depth"].idxmax()]
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
