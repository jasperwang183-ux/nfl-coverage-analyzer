"""Animate a page clip's film tracks as a top-down GIF with its three cards.

The GIF shows what the pipeline recovered from the film (field-space tracks in the
canonical frame, offense moving right), not the footage itself.

Usage:
  python scripts/render_demo.py data/web/187dcd12b5d2 --out docs/demo.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from nfl_coverage.analysis.viz import draw_field
from nfl_coverage.vision.pipeline import _canonical

STEP = 0.1  # s per rel_frame (10 Hz)
TRAIL = 8  # frames of trail behind each player
OFFENSE, DEFENSE, KEY = "#64b5f6", "#ef5350", "#ffd54f"
MOVER = {  # who the gold dot is, per mechanism
    "pole_run": "the pole runner",
    "late_post": "the late post safety",
    "corner_roll_up": "the rolled-up corner",
    "buzz_soft": "the buzz safety",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    ap.add_argument("--start", type=float, default=-1.2, help="s from the snap")
    ap.add_argument("--end", type=float, default=3.0, help="s from the snap")
    ap.add_argument("--hold", type=int, default=15, help="frames to hold the last frame")
    args = ap.parse_args()

    result = json.loads((args.clip_dir / "result.json").read_text())
    report = result["report"]
    cards = report["cards"]
    used = result["formation"]["used"]
    los, _ = _canonical(used["offense"], used["los"], None)
    hit = report["mechanisms"][0] if report["mechanisms"] else None
    key_id, hit_t = (hit["nfl_id"], hit["t"]) if hit else (None, None)
    post_t = 1.5  # the played middle is read at +1.5 s

    tracks = pd.read_parquet(args.clip_dir / "tracks.parquet")
    lo, hi = round(args.start / STEP), round(args.end / STEP)
    tracks = tracks[tracks.rel_frame.between(lo - TRAIL, hi)]
    frames = list(range(lo, hi + 1)) + [hi] * args.hold

    fig = plt.figure(figsize=(8, 5.6), dpi=90)
    fig.patch.set_facecolor("#111")
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.66))
    text_ax = fig.add_axes((0.02, 0.70, 0.96, 0.29))
    text_ax.axis("off")

    lines = [
        ("PRE-SNAP", cards["pre"]["call"], -99.0),
        ("POST-SNAP", cards["post"]["call"], post_t),
        ("DISGUISE", cards["disguise"]["call"], hit_t if hit_t is not None else post_t),
    ]

    def draw(rf: int) -> None:
        t = rf * STEP
        ax.clear()
        draw_field(ax, los - 12, los + 26)
        ax.axvline(los, color="#90caf9", lw=1.6, alpha=0.8)
        now = tracks[tracks.rel_frame.between(rf - TRAIL, rf)]
        if key_id is not None and rf >= 0:  # the key mover keeps his whole run since the snap
            key = tracks[(tracks.nfl_id == key_id) & tracks.rel_frame.between(0, rf)]
            now = pd.concat([now[now.nfl_id != key_id], key])
        for nid, p in now.groupby("nfl_id"):
            p = p.sort_values("rel_frame")
            is_key = nid == key_id and rf >= 0
            color = KEY if is_key else OFFENSE if p.side.iloc[0] == "Offense" else DEFENSE
            ax.plot(p.x, p.y, color=color, lw=2.4 if is_key else 1.2, alpha=0.6)
            last = p[p.rel_frame == rf]
            if len(last):
                ax.scatter(last.x, last.y, s=110 if is_key else 60, color=color, zorder=5,
                           edgecolor="white", lw=0.8)  # fmt: skip
        clock = "at the snap" if rf == 0 else f"{t:+.1f} s from the snap"
        ax.text(los - 11.5, 51.3, clock, color="white", fontsize=11, va="top", weight="bold")
        legend = [("offense", OFFENSE), ("defense", DEFENSE)]
        if hit:
            legend.append((MOVER.get(hit["mechanism"], "the disguise"), KEY))
        for i, (name, color) in enumerate(legend):
            ax.text(los + 25.5, 51.3 - i * 2.6, f"● {name}", color=color, fontsize=10, ha="right",
                    va="top", weight="bold")  # fmt: skip

        text_ax.clear()
        text_ax.axis("off")
        for i, (head, call, shown_from) in enumerate(lines):
            on = t >= shown_from
            y = 0.82 - i * 0.33
            text_ax.text(0.0, y, head, color=KEY if on else "#555", fontsize=11, weight="bold",
                         transform=text_ax.transAxes, va="center")  # fmt: skip
            text_ax.text(0.17, y, call if on else "...", color="white" if on else "#555",
                         fontsize=11, transform=text_ax.transAxes, va="center", wrap=True)  # fmt: skip

    anim = FuncAnimation(fig, draw, frames=frames, interval=STEP * 1000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=round(1 / STEP)))
    plt.close(fig)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e3:.0f} kB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
