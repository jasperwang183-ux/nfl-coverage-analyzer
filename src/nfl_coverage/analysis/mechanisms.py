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


def _pole_run(p: pd.DataFrame) -> pd.Series:
    return (
        (p["role"] == "box")
        & (p["align_depth"] >= LINE_DEPTH)
        & (p["commit_v_depth"] >= 1.5)
        & (p["landmark_depth"] >= 9.0)
        & (p["landmark_lat"].abs() <= 4.0)
    )


# Weeks 1-4, 7,627 labeled plays: fires on 6.9%; 53% of hits are Cover 2 zone (base 12%).
# From a single-high look, Cover 2 zone goes 6% -> 34% and Cover 1 42% -> 4%; the rest is
# Cover 3 with a hook defender carrying the seam, which moves the same way.
POLE_RUN = Mechanism(
    name="pole_run",
    kind="coverage",
    implies=frozenset({"COVER_2_ZONE"}),
    fires=_pole_run,
    text=lambda r: (
        f"a box defender ran the deep middle ({r['commit_v_depth']:.1f} yd/s by +1.2 s, "
        f"{r['landmark_depth']:.0f} yd deep at +{r['landmark_t']:.1f} s): Tampa 2's pole "
        "runner, or a Cover 3 hook defender carrying the seam; not man"
    ),
)

MECHANISMS: tuple[Mechanism, ...] = (POLE_RUN,)

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
