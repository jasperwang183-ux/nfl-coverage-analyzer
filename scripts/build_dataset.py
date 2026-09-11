"""Process all raw tracking weeks into canonical parquet under data/processed/frames/.

Usage: python scripts/build_dataset.py [--dataset bdb2021|bdb2026]
"""

from __future__ import annotations

import argparse
import time

from nfl_coverage.config import DATA_PROCESSED
from nfl_coverage.data.loaders import load_bdb2021_week, load_bdb2026_week
from nfl_coverage.data.nflverse import load_coverage_labels

FRAMES_DIR = DATA_PROCESSED / "frames"

WEEKS = {"bdb2021": range(1, 18), "bdb2026": range(1, 19)}


def build(dataset: str) -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    labels_2018 = load_coverage_labels(2018) if dataset == "bdb2021" else None
    for week in WEEKS[dataset]:
        dest = FRAMES_DIR / f"{dataset}_w{week:02d}.parquet"
        if dest.exists():
            print(f"skip {dest.name} (exists)")
            continue
        t0 = time.time()
        if dataset == "bdb2021":
            df = load_bdb2021_week(week, labels=labels_2018)
        else:
            df = load_bdb2026_week(week)
        df.to_parquet(dest, index=False)
        print(
            f"{dest.name}: {len(df):,} rows, "
            f"{df.groupby(['game_id', 'play_id']).ngroups} plays, "
            f"{time.time() - t0:.0f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(WEEKS), default=None)
    args = parser.parse_args()
    for ds in [args.dataset] if args.dataset else list(WEEKS):
        build(ds)
