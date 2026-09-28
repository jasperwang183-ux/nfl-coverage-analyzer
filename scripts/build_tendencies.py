"""Build the team disguise-tendency tables from the weeks outside the frozen holdout.

Reads every tuning week's snap with the page's shell reader (cached to
data/processed/tendency_plays.parquet), picks the shrinkage k by building on weeks
1-9 and scoring weeks 10-13, then writes data/processed/tendencies.parquet from
weeks 1-13 and prints the validation table and the face-validity views.

Usage:
  python scripts/build_tendencies.py            # reuse the cached shown reads
  python scripts/build_tendencies.py --refresh  # re-read every snap (~3 min)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_coverage.analysis import tendencies as td
from nfl_coverage.analysis.holdout import FRAMES_DIR, HOLDOUT, tuning_weeks
from nfl_coverage.data.context import load_play_context
from nfl_coverage.features.alignment import KEYS

CALLERS_PATH = Path(td.__file__).parents[1] / "data" / "defensive_play_callers.csv"
FIT_LAST_WEEK = 9  # k is chosen on weeks 1-9 -> 10-13; the holdout is never read
K_GRID = [1, 2, 5, 10, 20, 40, 80, 160, 320]
LEVELS = ["league", "league_sit", "team", "team_sit"]


def read_plays(refresh: bool) -> pd.DataFrame:
    if td.PLAYS_PATH.exists() and not refresh:
        return pd.read_parquet(td.PLAYS_PATH)
    parts = []
    for ds in td.SEASON:
        ctx = load_play_context(ds).drop(columns=["week"])
        for week in tuning_weeks(ds):
            assert week not in HOLDOUT[ds]
            frames = pd.read_parquet(FRAMES_DIR / f"{ds}_w{week:02d}.parquet")
            reads = td.shown_reads(frames)
            labels = frames.groupby(KEYS, as_index=False)["coverage_type"].first()
            part = reads.merge(labels, on=KEYS).merge(ctx, on=["game_id", "play_id"], how="left")
            part["week"] = week
            parts.append(part)
            print(f"  {ds} w{week:02d}: {len(part)} plays")
    plays = pd.concat(parts, ignore_index=True)
    plays.to_parquet(td.PLAYS_PATH, index=False)
    return plays


def prepare(plays: pd.DataFrame) -> pd.DataFrame:
    plays = plays.dropna(subset=["defense", "down", "yards_to_go"]).copy()
    plays["season"] = plays["dataset"].map(td.SEASON)
    plays["situation"] = td.situation(plays["down"], plays["yards_to_go"])
    callers = pd.read_csv(CALLERS_PATH) if CALLERS_PATH.exists() else None
    plays["play_caller"] = td.assign_play_callers(plays, callers)
    if callers is None:
        print(f"(no {CALLERS_PATH.name}: tables are keyed by season and team only)")
    return plays


def choose_k(plays: pd.DataFrame) -> float:
    fit = plays[plays["week"] <= FIT_LAST_WEEK]
    score = plays[plays["week"] > FIT_LAST_WEEK]
    rows = []
    for k in K_GRID:
        tables = td.build_tables(fit, k)
        rows.append({"k": k, **{lv: td.log_loss(tables, score, lv) for lv in LEVELS}})
    grid = pd.DataFrame(rows).set_index("k")
    labeled = score["coverage_type"].isin(td.CLASSES)
    mix = fit.loc[fit["coverage_type"].isin(td.CLASSES), "coverage_type"].value_counts(
        normalize=True
    )
    no_look = -np.log(mix.reindex(score.loc[labeled, "coverage_type"]).to_numpy()).mean()
    print(
        f"=== log loss on weeks {FIT_LAST_WEEK + 1}-13 (n = {int(labeled.sum())}), "
        f"tables from weeks 1-{FIT_LAST_WEEK} ==="
    )
    print(f"no look (league coverage mix): {no_look:.4f}")
    print(grid.round(4).to_string())
    best = float(grid["team_sit"].idxmin())
    print(f"k = {best:g} (lowest team x situation loss)\n")
    return best


def face_validity(tables: pd.DataFrame) -> None:
    league = tables[(tables["defense"] == td.LEAGUE) & (tables["situation"] == td.ANY)]
    print("=== league: played coverage by shown look ===")
    print(league.set_index(["season", "shown"])[["n", *td.CLASSES]].round(3).to_string())
    teams = tables[(tables["defense"] != td.LEAGUE) & (tables["situation"] == td.ANY)].copy()
    teams["plays_it"] = [td.plays_it(r[td.CLASSES], r["shown"]) for _, r in teams.iterrows()]
    for (season, shown), t in teams.groupby(["season", "shown"]):
        if shown == "0-high":
            continue
        t = t.sort_values("plays_it")
        print(f"\n=== {season}, shown {shown}: least and most likely to play it ===")
        cols = ["defense", "play_caller", "n", "plays_it"]
        print(pd.concat([t.head(4), t.tail(4)])[cols].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-read every snap")
    ap.add_argument("--k", type=float, default=None, help="skip the k search")
    args = ap.parse_args()

    plays = prepare(read_plays(args.refresh))
    print(plays.groupby(["season", "shown"]).size().unstack().to_string(), "\n")
    k = args.k if args.k is not None else choose_k(plays)
    tables = td.build_tables(plays, k)
    tables.attrs["k"] = k
    tables.to_parquet(td.TABLES_PATH, index=False)
    print(f"wrote {td.TABLES_PATH} ({len(tables)} rows, k = {k:g})\n")
    face_validity(tables)


if __name__ == "__main__":
    main()
