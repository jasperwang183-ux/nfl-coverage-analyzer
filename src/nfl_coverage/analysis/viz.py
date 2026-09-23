"""Render one canonical play as a field diagram PNG (shared by viz_play and labeling)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nfl_coverage.config import FIELD_WIDTH


def draw_field(ax, x_min: float, x_max: float) -> None:
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


def render_play(
    play: pd.DataFrame, out_path: str, note: str = "", mark_frame: int | None = None
) -> str:
    """Blue = offense, red = defense. Dotted = pre-snap, solid = post-snap, dot = snap.

    `mark_frame` draws a hollow marker at that rel_frame (e.g. the run-read frame)
    so a reviewer sees where the shell was read.
    """
    label = play.coverage_type.iloc[0]
    mz = play.man_zone.iloc[0]
    pad = 8
    x_min, x_max = play.x.min() - pad, play.x.max() + pad

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax, x_min, x_max)

    for _, pdata in play.groupby("nfl_id"):
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
        if mark_frame is not None:
            m = pdata[pdata.rel_frame == mark_frame]
            if len(m):
                ax.scatter(m.x, m.y, facecolors="none", edgecolors=color, s=110, lw=1.5, zorder=5)

    gid, pid = play.game_id.iloc[0], play.play_id.iloc[0]
    title = f"{play.dataset.iloc[0]}  game {gid}  play {pid}   |   coverage: {label} ({mz})"
    if note:
        title += f"   |   {note}"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
