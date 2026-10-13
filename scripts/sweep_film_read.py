"""Sweep the film path's played read on the tuning weeks (never the holdout).

The count read asks where each defender is headed (`report.ProjectedDepth`): deep when
his depth plus a short look-ahead at his drop rate reaches the count line. This sweeps
the count line, the look-ahead, the window the rate is measured over, and the middle
read's line, at the fixed read times (MOF_READ, COUNT_READ). Every rule reads the same
seeded sample of labeled plays, cut to a film-length window (FILM_WINDOW_FRAMES after
the snap), so the table compares rules and not samples.

The plateau is every rule within 2 standard errors of each season's best middle
accuracy and best deep count. The count settings are the middle of the plateau (each
setting's median, then the nearest rule inside it). The middle line splits the seasons
(2018 peaks at 13-14, 2023 is flat within its noise), so it is the one with the smallest
worst-case loss, as for the rotation line (scripts/sweep_run_depth.py).

History (2026-10-13, 650 plays per season): plain middle/count/corner lines at 9-14 yd
gave a one-point plateau (13/10/10); a 10-13 yd band a defender must still be dropping
through tied with projected depth, which won as the simpler of the two; look-ahead 0 (a
plain line) costs 4 points of 2023 deep count. Results of each run are saved to
data/verify/sweeps/film_read.json. Score the chosen rule on the holdout once, with
`scripts/verify.py bdb --checks film_read`.

Usage: python scripts/sweep_film_read.py [--per-week 50]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings

import pandas as pd

from nfl_coverage.analysis.holdout import (
    FRAMES_DIR,
    FilmReadRule,
    film_read_eval,
    tuning_weeks,
)
from nfl_coverage.analysis.report import COUNT_READ, MOF_READ, ProjectedDepth
from nfl_coverage.config import ROOT
from nfl_coverage.data.ball_spots import load_ball_spots
from nfl_coverage.features.alignment import KEYS

MIDDLE = [11.0, 12.0, 13.0, 14.0, 15.0]
GRID = {
    "line": [10.0, 10.5, 11.0, 11.5, 12.0],
    "lookahead": [0.0, 0.1, 0.2, 0.3, 0.5],
    "window": [0.2, 0.3, 0.5],
}
DATASETS = ("bdb2021", "bdb2026")


def sample_tuning_plays(dataset: str, per_week: int, seed: int = 0) -> pd.DataFrame:
    parts = []
    for w in tuning_weeks(dataset):
        f = pd.read_parquet(FRAMES_DIR / f"{dataset}_w{w:02d}.parquet")
        labeled = f.dropna(subset=["coverage_type"])[KEYS].drop_duplicates()
        keep = labeled.sample(min(per_week, len(labeled)), random_state=seed + w)
        parts.append(f.merge(keep, on=KEYS))
    return pd.concat(parts, ignore_index=True)


def grid() -> tuple[pd.DataFrame, list[FilmReadRule]]:
    rows, rules = [], []
    for md, line, lookahead, window in itertools.product(MIDDLE, *GRID.values()):
        test = ProjectedDepth(line, lookahead, window)
        rule = FilmReadRule((MOF_READ[0], md), (COUNT_READ[0], line), test)
        rules.append(rule)
        rows.append(
            {
                "middle": md,
                "line": line,
                "lookahead": lookahead,
                "window": window,
                "rule": rule.name,
            }
        )
    return pd.DataFrame(rows), rules


def within_2se(acc: pd.Series, n: pd.Series) -> pd.Series:
    best = acc.max()
    se = math.sqrt(best * (1 - best) / n[acc.idxmax()])
    return best - acc < 2 * se


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-week", type=int, default=50)
    args = ap.parse_args()
    points, rules = grid()
    print(f"{len(rules)} rules")
    ball_spots = load_ball_spots()
    results = {}
    for ds in DATASETS:
        plays = sample_tuning_plays(ds, args.per_week)
        print(f"{ds}: {plays.groupby(KEYS).ngroups} plays from weeks {tuning_weeks(ds)}")
        results[ds] = film_read_eval(plays, rules, sample=0, ball_spots=ball_spots)
        scores = pd.DataFrame(results[ds]).T.astype(float).add_prefix(f"{ds} ")
        points = points.join(scores, on="rule")
    plateau = pd.Series(True, index=points.index)
    for ds in DATASETS:
        plateau &= within_2se(points[f"{ds} mof_accuracy"], points[f"{ds} n"])
        plateau &= within_2se(points[f"{ds} deep_count_accuracy"], points[f"{ds} n_deep_count"])
    points["plateau"] = plateau
    score_cols = [f"{ds} {m}" for ds in DATASETS for m in ("mof_accuracy", "deep_count_accuracy")]
    pd.set_option("display.width", 250)
    best = points.assign(mean=points[score_cols].mean(axis=1)).sort_values("mean", ascending=False)
    print("\ntop 15 by mean of the four scores")
    print(best[["middle", *GRID, *score_cols, "plateau"]].head(15).round(3).to_string(index=False))

    inside = points[points["plateau"]]
    print(f"\nplateau: {len(inside)} of {len(points)} rules hold in both seasons")
    out = {"per_week": args.per_week, "results": results}
    if inside.empty:
        print("no plateau: nothing holds in both seasons")
    else:
        for name, values in {"middle": MIDDLE, **GRID}.items():
            lo, hi = inside[name].min(), inside[name].max()
            edge = " (runs to the edge of the range)" if lo == values[0] or hi == values[-1] else ""
            print(f"  {name:<9} {lo:g}-{hi:g}, median {inside[name].median():g}{edge}")
        settings = list(GRID)
        median = inside[settings].median()
        spread = (inside[settings].max() - inside[settings].min()).replace(0, 1)
        count = inside.loc[((inside[settings] - median).abs() / spread).sum(axis=1).idxmin()]
        same_count = points[(points[settings] == count[settings]).all(axis=1)]
        loss = pd.Series(0.0, index=same_count.index)
        for ds in DATASETS:
            acc = same_count[f"{ds} mof_accuracy"]
            loss = loss.combine(acc.max() - acc, max)
        pick = same_count.loc[loss.idxmin()]
        print(
            f"\npick: middle {pick['middle']:g} yd (worst-case loss {loss.min():.3f}), "
            + ", ".join(f"{s} {pick[s]:g}" for s in settings)
        )
        print(pick[score_cols].round(3).to_string())
        out["pick"] = {s: float(pick[s]) for s in ["middle", *settings]}
    path = ROOT / "data" / "verify" / "sweeps" / "film_read.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
