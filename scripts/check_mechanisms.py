"""Does each disguise mechanism point where it claims? Fire rate and lift by the shown look.

For every mechanism in `analysis.mechanisms.MECHANISMS`: how often it fires, and among
plays where it fires vs not, the share whose label is in the families it implies, split
by the look at the snap (0-high / MOF closed / MOF open, the page's own read). A
mechanism that does not lift its families is not a tell and should be dropped. A pressure
row's label is the 2018 rush count (RUSH_LE4 / RUSH_GE5), so it is scored on 2018 only.

Usage:
  python scripts/check_mechanisms.py              # tuning weeks 1-4 of both seasons
  python scripts/check_mechanisms.py --candidates # also the rows not yet in MECHANISMS
  python scripts/check_mechanisms.py --holdout    # the frozen holdout: score a finished rule ONCE
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from nfl_coverage.analysis.holdout import FRAMES_DIR, HOLDOUT, TUNING
from nfl_coverage.analysis.mechanisms import CANDIDATES, MECHANISMS, Mechanism, mechanism_hits
from nfl_coverage.analysis.tendencies import CLASSES, shown_reads
from nfl_coverage.config import DATA_PROCESSED
from nfl_coverage.data.ball_spots import load_ball_spots
from nfl_coverage.data.context import load_play_context
from nfl_coverage.features.alignment import KEYS
from nfl_coverage.features.phases import defender_phases


def load(
    weeks: dict[str, list[int]], mechanisms: tuple[Mechanism, ...] = MECHANISMS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    releases = pd.read_parquet(DATA_PROCESSED / "releases.parquet")
    spots = load_ball_spots()
    plays, hits = [], []
    for ds, ws in weeks.items():
        rush = load_play_context(ds)[["game_id", "play_id", "number_of_pass_rushers"]]
        for w in ws:
            frames = pd.read_parquet(FRAMES_DIR / f"{ds}_w{w:02d}.parquet")
            labels = frames.groupby(KEYS, as_index=False)["coverage_type"].first()
            shown = shown_reads(frames, spots)[KEYS + ["shown"]]
            plays.append(labels.merge(shown, on=KEYS).merge(rush, on=["game_id", "play_id"]))
            phases = defender_phases(frames, releases, ball_spots=spots)
            hits.append(mechanism_hits(phases, mechanisms))
            print(f"  {ds} w{w:02d}")
    plays = pd.concat(plays, ignore_index=True)
    n = plays["number_of_pass_rushers"]
    plays["rush_label"] = np.where(n.isna(), None, np.where(n <= 4, "RUSH_LE4", "RUSH_GE5"))
    return plays[plays["coverage_type"].isin(CLASSES)], pd.concat(hits, ignore_index=True)


def report(
    plays: pd.DataFrame, hits: pd.DataFrame, mechanisms: tuple[Mechanism, ...] = MECHANISMS
) -> None:
    pd.set_option("display.width", 200)
    for m in mechanisms:
        fired = hits.loc[hits["mechanism"] == m.name, KEYS].drop_duplicates()
        t = plays.merge(fired.assign(fired=True), on=KEYS, how="left")
        t["fired"] = t["fired"].fillna(False).astype(bool)
        target = "rush_label" if m.kind == "pressure" else "coverage_type"
        t = t[t[target].notna()]  # pressure: 2018 only, the one season with a rush count
        t["implied"] = t[target].isin(m.implies)
        base = t["implied"].mean()
        hit = t.loc[t["fired"], "implied"].mean()
        print(f"\n=== {m.name} ({m.kind}; implies {', '.join(sorted(m.implies))}) ===")
        print(
            f"n = {len(t)}, fires on {t['fired'].mean():.1%}; implied share {hit:.3f} when "
            f"fired vs {base:.3f} overall (lift {hit / base:.2f}); catches "
            f"{t.loc[t['implied'], 'fired'].mean():.1%} of implied plays"
        )
        view = t.groupby(["shown", "fired"]).agg(
            n=("implied", "size"),
            implied=("implied", "mean"),
            **{
                c.lower(): ("coverage_type", lambda s, c=c: (s == c).mean())
                for c in ("COVER_1_MAN", "COVER_2_ZONE", "COVER_3_ZONE", "COVER_4_ZONE")
            },
        )
        print(view.round(3).to_string())


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="score the frozen holdout (once)")
    ap.add_argument(
        "--candidates", action="store_true", help="also score CANDIDATES (tuning weeks only)"
    )
    args = ap.parse_args()
    if args.holdout and args.candidates:
        ap.error("a candidate is fitted on the tuning weeks; register it before the holdout")
    mechanisms = MECHANISMS + CANDIDATES if args.candidates else MECHANISMS
    report(*load(HOLDOUT if args.holdout else TUNING, mechanisms), mechanisms)


if __name__ == "__main__":
    main()
