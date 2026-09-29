"""Build data/processed/releases.parquet: the pass-release rel_frame of every cached play.

Usage: python scripts/build_releases.py

Reads the raw tracking week behind every cached data/processed/frames/{dataset}_wNN.parquet
(the cached frames themselves are only read for their play keys, never rewritten).
"""

from __future__ import annotations

import re
import time

import pandas as pd

from nfl_coverage.config import BDB2021, BDB2026, DATA_PROCESSED, FRAMES_PER_SECOND
from nfl_coverage.data.releases import releases_bdb2021, releases_bdb2026

FRAMES_DIR = DATA_PROCESSED / "frames"
DEST = DATA_PROCESSED / "releases.parquet"
KEYS = ["game_id", "play_id"]


def _week_releases(dataset: str, week: int) -> pd.DataFrame:
    if dataset == "bdb2021":
        raw = pd.read_csv(
            BDB2021 / f"week{week}.csv", usecols=["gameId", "playId", "frameId", "event"]
        )
        return releases_bdb2021(raw)
    raw = pd.read_csv(
        BDB2026 / "train" / f"input_2023_w{week:02d}.csv", usecols=[*KEYS, "frame_id"]
    )
    return releases_bdb2026(raw)


def build() -> pd.DataFrame:
    parts = []
    for path in sorted(FRAMES_DIR.glob("*_w*.parquet")):
        dataset, week = re.fullmatch(r"(\w+)_w(\d+)", path.stem).groups()
        t0 = time.time()
        rel = _week_releases(dataset, int(week))
        cached = pd.read_parquet(path, columns=KEYS).drop_duplicates()
        # The cached frames define the play set (the loader drops plays it cannot join).
        kept = cached.merge(rel.drop(columns="dataset"), on=KEYS, how="left", validate="1:1")
        kept["dataset"] = dataset
        not_cached = len(rel) - len(kept.merge(rel[KEYS], on=KEYS))
        print(
            f"{path.stem}: {len(kept):,} cached plays, "
            f"{kept['release_rel_frame'].isna().sum()} without release, "
            f"{not_cached} raw plays not cached, {time.time() - t0:.0f}s"
        )
        parts.append(kept[rel.columns])
    return pd.concat(parts, ignore_index=True)


def report(releases: pd.DataFrame) -> None:
    for dataset, grp in releases.groupby("dataset"):
        secs = grp["release_rel_frame"].dropna() / FRAMES_PER_SECOND
        q = secs.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(
            f"\n{dataset}: {len(grp):,} plays, {grp['release_rel_frame'].isna().sum()} no release"
        )
        print("  release time (s) q05/q25/q50/q75/q95: " + " / ".join(f"{v:.1f}" for v in q))
        for cut in (1.0, 1.5, 2.0, 2.5):
            print(f"  released before {cut:.1f}s: {(secs < cut).mean():.1%}")
        print(f"  release at or before the snap: {(secs <= 0).sum()}")


if __name__ == "__main__":
    releases = build()
    releases.to_parquet(DEST, index=False)
    print(f"\nwrote {DEST} ({len(releases):,} plays)")
    report(releases)
