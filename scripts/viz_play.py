"""Render one play from the canonical cache as a field diagram PNG.

Blue = offense, red = defense. Faint trail = pre-snap second (bdb2021 only),
solid trail = post-snap movement, dot = position at the snap.

Usage:
  python scripts/viz_play.py                          # random labeled bdb2021 play
  python scripts/viz_play.py --dataset bdb2026 --game 2023090700 --play 101
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from nfl_coverage.config import DATA_PROCESSED, FIELD_WIDTH

FRAMES_DIR = DATA_PROCESSED / "frames"


def draw_field(ax, x_min, x_max):
    ax.set_facecolor("#2e7d32")
    for x in range(int(x_min) // 5 * 5, int(x_max) + 5, 5):
        ax.axvline(x, color="white", lw=0.8 if x % 10 else 1.4, alpha=0.35)
    for x in range(int(x_min) // 10 * 10, int(x_max) + 10, 10):
        yard = x - 10 if x <= 60 else 110 - x
        if 0 < yard <= 50:
            ax.text(x, 2, str(yard), color="white", ha="center", fontsize=8, alpha=0.7)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks([])
    ax.set_yticks([])


def render(play: pd.DataFrame, out_path: str, note: str = "") -> None:
    snap = play[play.rel_frame == 0]
    label = play.coverage_type.iloc[0]
    mz = play.man_zone.iloc[0]
    pad = 8
    x_min, x_max = play.x.min() - pad, play.x.max() + pad

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax, x_min, x_max)

    for ((_, pdata),) in zip(play.groupby("nfl_id")):
        color = "#1565c0" if pdata.side.iloc[0] == "Offense" else "#c62828"
        pre = pdata[pdata.rel_frame <= 0].sort_values("rel_frame")
        post = pdata[pdata.rel_frame >= 0].sort_values("rel_frame")
        if len(pre) > 1:
            ax.plot(pre.x, pre.y, color=color, lw=1.2, alpha=0.35, ls=":")
        ax.plot(post.x, post.y, color=color, lw=1.8, alpha=0.9)
        s = pdata[pdata.rel_frame == 0]
        if len(s):
            ax.scatter(s.x, s.y, color=color, s=90, zorder=5, edgecolor="white", lw=1)
            pos = s.position.iloc[0]
            if pd.notna(pos):
                ax.annotate(
                    pos,
                    (s.x.iloc[0], s.y.iloc[0]),
                    color="white",
                    fontsize=6,
                    ha="center",
                    va="center",
                    zorder=6,
                )

    gid, pid = play.game_id.iloc[0], play.play_id.iloc[0]
    title = f"{play.dataset.iloc[0]}  game {gid}  play {pid}   |   coverage: {label} ({mz})"
    if note:
        title += f"   |   {note}"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


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
    render(play, args.out)


if __name__ == "__main__":
    main()
