"""Render one play from the canonical cache as a field diagram PNG.

Blue = offense, red = defense. Faint trail = pre-snap second (bdb2021 only),
solid trail = post-snap movement, dot = position at the snap.

Usage:
  python scripts/viz_play.py                          # random labeled bdb2021 play
  python scripts/viz_play.py --dataset bdb2026 --game 2023090700 --play 101
"""

from __future__ import annotations

import argparse

import pandas as pd

from nfl_coverage.analysis.viz import render_play
from nfl_coverage.config import DATA_PROCESSED

FRAMES_DIR = DATA_PROCESSED / "frames"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bdb2021")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--game", type=int)
    ap.add_argument("--play", type=int)
    ap.add_argument("--out", default="play.png")
    args = ap.parse_args()

    frames = pd.read_parquet(FRAMES_DIR / f"{args.dataset}_w{args.week:02d}.parquet")
    if args.game and args.play:
        play = frames[(frames.game_id == args.game) & (frames.play_id == args.play)]
    else:
        labeled = frames[frames.coverage_type.notna()]
        gid, pid = (
            labeled[["game_id", "play_id"]].drop_duplicates().sample(1, random_state=None).iloc[0]
        )
        play = frames[(frames.game_id == gid) & (frames.play_id == pid)]
    print(f"saved {render_play(play, args.out)}")


if __name__ == "__main__":
    main()
