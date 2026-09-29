"""Sweep the film path's played-read rule on the tuning weeks (never the holdout).

Every rule reads the same seeded sample of labeled plays, cut to a film-length
window (FILM_WINDOW_FRAMES after the snap), so the table compares rules and not
samples. Pick by MOF accuracy, tie-break on the deep count; a gap under 2
standard errors is a tie, and a tie goes to the simpler rule. Score the chosen
rule on the holdout once, with `scripts/verify.py bdb --checks film_read`.

Usage: python scripts/sweep_film_read.py [--per-week 50]
"""

from __future__ import annotations

import argparse
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
from nfl_coverage.analysis.report import COUNT_READ, MOF_READ
from nfl_coverage.config import ROOT
from nfl_coverage.features.alignment import KEYS

MOF_READS = [(1.2, 11.0), (1.5, 11.0), (1.5, 12.0), (1.8, 12.0)]
COUNT_READS = [(2.0, 12.0), (2.2, 12.0), (2.5, 12.0), (2.2, 13.0)]


def sample_tuning_plays(dataset: str, per_week: int, seed: int = 0) -> pd.DataFrame:
    parts = []
    for w in tuning_weeks(dataset):
        f = pd.read_parquet(FRAMES_DIR / f"{dataset}_w{w:02d}.parquet")
        labeled = f.dropna(subset=["coverage_type"])[KEYS].drop_duplicates()
        keep = labeled.sample(min(per_week, len(labeled)), random_state=seed + w)
        parts.append(f.merge(keep, on=KEYS))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-week", type=int, default=50)
    args = ap.parse_args()
    rules = [FilmReadRule(m, COUNT_READ) for m in MOF_READS] + [
        FilmReadRule(MOF_READ, c) for c in COUNT_READS if c != COUNT_READ
    ]
    results = {}
    for ds in ("bdb2021", "bdb2026"):
        plays = sample_tuning_plays(ds, args.per_week)
        print(f"{ds}: {plays.groupby(KEYS).ngroups} plays from weeks {tuning_weeks(ds)}")
        results[ds] = film_read_eval(plays, rules, sample=0)

    print(
        f"\n{'rule':<24}"
        + "".join(f"{ds + ' mof':>14}{'deep#':>8}" for ds in results)
        + f"{'mean mof':>10}"
    )
    table = []
    for r in rules:
        cells, mofs = "", []
        for ds, res in results.items():
            m = res[r.name]
            cells += f"{m['mof_accuracy']:>14.3f}{m['deep_count_accuracy']:>8.3f}"
            mofs.append(m["mof_accuracy"])
        mean = sum(mofs) / len(mofs)
        table.append((mean, r.name))
        print(f"{r.name:<24}{cells}{mean:>10.3f}")
    n = min(m["n"] for res in results.values() for m in res.values())
    best = max(table)
    se = math.sqrt(best[0] * (1 - best[0]) / (n * len(results)))
    ties = [name for mean, name in table if best[0] - mean < 2 * se]
    print(f"\nbest {best[1]} ({best[0]:.3f}); within 2 SE (+-{2 * se:.3f}): {', '.join(ties)}")
    out = ROOT / "data" / "verify" / "sweeps" / "film_read.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_week": args.per_week, "results": results}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
