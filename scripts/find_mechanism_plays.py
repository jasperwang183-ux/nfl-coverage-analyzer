"""List the plays of given 2023 games where a mechanism fires, to pick one for a film check.

Prints quarter, clock, down and distance, the label and the play description from
supplementary_data.csv, plus the defender who ran it. Holdout weeks are refused: a film
check is still tuning.

Usage:
  python scripts/find_mechanism_plays.py corner_roll_up --games 2023091010 2023091709 \\
      --labels COVER_2_ZONE COVER_3_ZONE
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from nfl_coverage.analysis.holdout import FRAMES_DIR, HOLDOUT
from nfl_coverage.analysis.mechanisms import CANDIDATES, MECHANISMS, mechanism_hits
from nfl_coverage.config import BDB2026, DATA_PROCESSED
from nfl_coverage.features.alignment import KEYS
from nfl_coverage.features.phases import defender_phases

CONTEXT = [
    "game_id",
    "play_id",
    "week",
    "quarter",
    "game_clock",
    "down",
    "yards_to_go",
    "possession_team",
    "defensive_team",
    "play_description",
]


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    by_name = {m.name: m for m in MECHANISMS + CANDIDATES}
    ap = argparse.ArgumentParser()
    ap.add_argument("mechanism", choices=sorted(by_name))
    ap.add_argument("--games", type=int, nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="keep only these labels")
    args = ap.parse_args()

    sup = pd.read_csv(BDB2026 / "supplementary_data.csv", low_memory=False)
    sup = sup.loc[sup["game_id"].isin(args.games), [c for c in CONTEXT if c in sup.columns]]
    weeks = sorted(sup["week"].unique())
    if set(weeks) & set(HOLDOUT["bdb2026"]):
        ap.error(f"weeks {weeks} include holdout weeks {HOLDOUT['bdb2026']}")

    releases = pd.read_parquet(DATA_PROCESSED / "releases.parquet")
    rows = []
    for w in weeks:
        frames = pd.read_parquet(FRAMES_DIR / f"bdb2026_w{w:02d}.parquet")
        frames = frames[frames["game_id"].isin(args.games)]
        labels = frames.groupby(KEYS, as_index=False)["coverage_type"].first()
        hits = mechanism_hits(defender_phases(frames, releases), (by_name[args.mechanism],))
        rows.append(hits.merge(labels, on=KEYS))
    hits = pd.concat(rows, ignore_index=True)
    if args.labels:
        hits = hits[hits["coverage_type"].isin(args.labels)]
    out = hits.merge(sup.drop(columns="week"), on=["game_id", "play_id"], how="left")
    left = out["game_clock"].map(lambda c: 60 * int(c.split(":")[0]) + int(c.split(":")[1]))
    out = out.assign(left=left).sort_values(
        ["game_id", "quarter", "left"], ascending=[True, True, False]
    )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(f"{len(out)} plays where {args.mechanism} fires")
    cols = [c for c in CONTEXT if c in out.columns and c != "week"]
    print(out[cols[:2] + ["coverage_type", "nfl_id"] + cols[2:]].to_string(index=False))
    for _, r in out.iterrows():
        print(f"\n{r['game_id']} {r['play_id']}: {r['text']}")


if __name__ == "__main__":
    main()
