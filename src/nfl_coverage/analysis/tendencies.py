"""Team disguise tendencies: how often a defense plays the look it shows.

P(played coverage | shown look, defense, situation), per season and defensive
play-caller. A team has ~15 plays in a shown x played cell a season, so every
team cell is shrunk toward a parent with Dirichlet pseudo-counts (`k` plays of
the parent's distribution):

  league | shown                raw (thousands of plays)
  league | shown, situation     -> league | shown
  team   | shown                -> league | shown
  team   | shown, situation     -> team | shown, tilted by the league's
                                   situation effect (league|shown,sit / league|shown)

`shown` is the middle-of-field state at the snap from `structural_read`, the
same call the page's pre-snap card makes, so a film read looks the tables up
directly. Tables are built from weeks outside the frozen holdout only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_coverage.analysis.harness import MOF_CLOSED_BY_COVERAGE
from nfl_coverage.analysis.report import structural_read
from nfl_coverage.analysis.shell import DEEP_DEPTH, _formation_center
from nfl_coverage.config import COVERAGE_CLASSES, DATA_PROCESSED
from nfl_coverage.features.alignment import KEYS, _los_estimate

TABLES_PATH = DATA_PROCESSED / "tendencies.parquet"
PLAYS_PATH = DATA_PROCESSED / "tendency_plays.parquet"

SEASON = {"bdb2021": 2018, "bdb2026": 2023}
SHOWN = ["0-high", "MOF closed", "MOF open"]
SITUATIONS = ["early down", "3rd/4th & short", "3rd/4th & medium", "3rd/4th & long"]
ANY = "any"
LEAGUE = "league"
# PREVENT is an end-of-half call, not a tendency; unlabeled plays carry nothing
CLASSES = [c for c in COVERAGE_CLASSES if c != "PREVENT"]
TABLE_KEYS = ["season", "defense", "play_caller", "shown", "situation"]


def shown_state(n_deep: int, mof_closed: bool) -> str:
    if n_deep == 0:
        return "0-high"
    return "MOF closed" if mof_closed else "MOF open"


def situation(down: pd.Series, yards_to_go: pd.Series) -> pd.Series:
    late = pd.cut(yards_to_go, [-1, 3, 7, 200], labels=SITUATIONS[1:]).astype(object)
    return late.where(down >= 3, SITUATIONS[0])


def shown_reads(frames: pd.DataFrame) -> pd.DataFrame:
    """The page's pre-snap structural read of every play at the snap (~8 ms a play)."""
    rows = []
    for key, snap in frames[frames["rel_frame"] == 0].groupby(KEYS, sort=False):
        off = snap[snap["side"] == "Offense"]
        try:
            los = float(_los_estimate(off).iloc[0])
            center = float(_formation_center(off).iloc[0])
            read = structural_read(snap, los, center, DEEP_DEPTH, "snap", [], 0)
        except (ValueError, IndexError, KeyError):
            continue
        rows.append((*key, read.n_deep, read.mof_closed, shown_state(read.n_deep, read.mof_closed)))
    return pd.DataFrame(rows, columns=KEYS + ["n_deep", "mof_closed", "shown"])


def assign_play_callers(plays: pd.DataFrame, callers: pd.DataFrame | None) -> pd.Series:
    """The defensive play-caller for each play's (season, defense, week); '' when unknown."""
    if callers is None or callers.empty:
        return pd.Series("", index=plays.index)
    m = (
        plays[["season", "defense", "week"]]
        .reset_index()
        .merge(callers.rename(columns={"team": "defense"}), on=["season", "defense"], how="left")
    )
    m = m[(m["week"] >= m["first_week"]) & (m["week"] <= m["last_week"])]
    return m.set_index("index")["play_caller"].reindex(plays.index).fillna("")


def _counts(plays: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    c = plays.groupby(keys + ["coverage_type"]).size().unstack("coverage_type", fill_value=0)
    return c.reindex(columns=CLASSES, fill_value=0).astype(float)


def _shrink(counts: pd.DataFrame, prior: pd.DataFrame, k: float) -> pd.DataFrame:
    p = (counts + k * prior.to_numpy()).div(counts.sum(axis=1) + k, axis=0)
    p.insert(0, "n", counts.sum(axis=1).astype(int))
    return p


def _parent(child: pd.DataFrame, parent: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    return child.reset_index()[on].merge(parent[CLASSES].reset_index(), on=on, how="left")[CLASSES]


def build_tables(plays: pd.DataFrame, k: float) -> pd.DataFrame:
    """One row per (season, defense, play_caller, shown, situation): n and P(class).

    `plays` needs season, defense, play_caller, shown, situation, coverage_type.
    League rows carry defense 'league'; rows over all situations carry 'any'.
    """
    plays = plays[plays["coverage_type"].isin(CLASSES)]
    team = ["season", "defense", "play_caller", "shown"]

    lc = _counts(plays, ["season", "shown"])
    league = lc.div(lc.sum(axis=1), axis=0)
    league.insert(0, "n", lc.sum(axis=1).astype(int))

    lsc = _counts(plays, ["season", "shown", "situation"])
    league_sit = _shrink(lsc, _parent(lsc, league, ["season", "shown"]), k)

    tc = _counts(plays, team)
    team_any = _shrink(tc, _parent(tc, league, ["season", "shown"]), k)

    tsc = _counts(plays, team + ["situation"])
    base = _parent(tsc, league, ["season", "shown"])
    # a class the league never plays from this look has no situation effect to carry
    tilt = (_parent(tsc, league_sit, ["season", "shown", "situation"]) / base).where(base > 0, 1.0)
    prior = _parent(tsc, team_any, team) * tilt
    team_sit = _shrink(tsc, prior.div(prior.sum(axis=1), axis=0), k)

    def keyed(t: pd.DataFrame, **fixed: str) -> pd.DataFrame:
        t = t.reset_index()
        for col, val in fixed.items():
            t[col] = val
        return t[TABLE_KEYS + ["n"] + CLASSES]

    return pd.concat(
        [
            keyed(league, defense=LEAGUE, play_caller="", situation=ANY),
            keyed(league_sit, defense=LEAGUE, play_caller=""),
            keyed(team_any, situation=ANY),
            keyed(team_sit),
        ],
        ignore_index=True,
    )


def log_loss(tables: pd.DataFrame, plays: pd.DataFrame, level: str) -> float:
    """Mean -log P(played coverage) of `plays` under one level of `tables`.

    level: 'league' (shown), 'league_sit', 'team' (shown), 'team_sit'. A play whose
    cell is missing from `tables` (a team or caller unseen when they were built)
    falls back to the league row for its shown state.
    """
    plays = plays[plays["coverage_type"].isin(CLASSES)].reset_index(drop=True)
    keyed = plays.assign(
        defense=plays["defense"] if level.startswith("team") else LEAGUE,
        play_caller=plays["play_caller"] if level.startswith("team") else "",
        situation=plays["situation"] if level.endswith("_sit") else ANY,
    )
    fallback = keyed.assign(defense=LEAGUE, play_caller="", situation=ANY)
    t = tables.set_index(TABLE_KEYS)[CLASSES]
    probs = t.reindex(pd.MultiIndex.from_frame(keyed[TABLE_KEYS]))
    probs = probs.fillna(
        t.reindex(pd.MultiIndex.from_frame(fallback[TABLE_KEYS])).set_axis(probs.index)
    )
    col = probs.columns.get_indexer(plays["coverage_type"])
    p = probs.to_numpy()[np.arange(len(plays)), col]
    return float(-np.log(np.clip(p, 1e-6, 1.0)).mean())


def plays_it(probs: pd.Series, shown: str) -> float:
    """P(the played coverage keeps the shown look): Cover 0 for a 0-high look, else MOF state."""
    if shown == "0-high":
        return float(probs["COVER_0_MAN"])
    closed = shown == "MOF closed"
    return float(sum(probs[c] for c in CLASSES if MOF_CLOSED_BY_COVERAGE[c] is closed))


@dataclass(frozen=True)
class Tendency:
    season: int
    defense: str
    play_caller: str
    shown: str
    situation: str
    n: int  # the defense's own plays in this cell; the rest is the league prior
    probs: dict[str, float]
    plays_it: float


def lookup(
    tables: pd.DataFrame,
    season: int,
    defense: str,
    shown: str,
    situation_: str = ANY,
    play_caller: str | None = None,
) -> Tendency | None:
    """The defense's tendency for one look; its busiest play-caller when none is given."""
    rows = tables[
        (tables["season"] == season)
        & (tables["defense"] == defense)
        & (tables["shown"] == shown)
        & (tables["situation"] == situation_)
    ]
    if play_caller is not None:
        rows = rows[rows["play_caller"] == play_caller]
    if rows.empty:
        return None
    row = rows.sort_values("n").iloc[-1] if play_caller is None else rows.iloc[0]
    probs = {c: float(row[c]) for c in CLASSES}
    return Tendency(
        season=season,
        defense=defense,
        play_caller=str(row["play_caller"]),
        shown=shown,
        situation=situation_,
        n=int(row["n"]),
        probs=probs,
        plays_it=plays_it(pd.Series(probs), shown),
    )
