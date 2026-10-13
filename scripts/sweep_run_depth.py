"""Sweep shell_rotation's played deep line (RUN_DEEP_DEPTH) on the tuning weeks.

Scores the played middle against the coverage label (`harness.run_shell_accuracy`,
what `holdout.shell_mof` reports) at each depth, per season, and prints how often the
rotation verdict fires, so a depth that moves who counts as rotating shows up even
when accuracy holds. The pick is the middle of the depths within 2 standard errors
of each season's best, in both seasons. When no depth holds in both, the pick is the
depth with the smallest worst-case loss: the larger of the two seasons' gaps below
their own best.

Usage: python scripts/sweep_run_depth.py
"""

from __future__ import annotations

import math
import warnings

import pandas as pd

from nfl_coverage.analysis.disguise import shell_rotation
from nfl_coverage.analysis.harness import run_shell_accuracy
from nfl_coverage.analysis.holdout import FRAMES_DIR, tuning_weeks
from nfl_coverage.data.ball_spots import load_ball_spots
from nfl_coverage.features.alignment import KEYS

DEPTHS = [9.0 + 0.5 * i for i in range(17)]  # 9-17 yd
FIRED = ("rotate_to_mof_closed", "rotate_to_mof_open")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    spots = load_ball_spots()
    rows = []
    for ds in ("bdb2021", "bdb2026"):
        for w in tuning_weeks(ds):
            frames = pd.read_parquet(FRAMES_DIR / f"{ds}_w{w:02d}.parquet")
            labels = frames.groupby(KEYS)["coverage_type"].first().reset_index()
            for depth in DEPTHS:
                t = shell_rotation(frames, run_deep_depth=depth, ball_spots=spots)
                t = t.merge(labels, on=KEYS, how="left")
                acc = run_shell_accuracy(t).loc[ds]
                fired = t["shell_rotation_label"].isin(FIRED).sum()
                rows.append(
                    (ds, w, depth, acc["run_mof_accuracy"] * acc["n"], acc["n"], fired, len(t))
                )
        print(f"{ds}: weeks {tuning_weeks(ds)}")
    t = pd.DataFrame(rows, columns=["dataset", "week", "depth", "hits", "n", "fired", "plays"])
    by = t.groupby(["dataset", "depth"])[["hits", "n", "fired", "plays"]].sum()
    by["accuracy"] = by["hits"] / by["n"]
    by["rotation_rate"] = by["fired"] / by["plays"]
    print(by[["accuracy", "n", "rotation_rate"]].round(4).to_string())

    plateau = pd.Series(True, index=DEPTHS)
    loss = pd.Series(0.0, index=DEPTHS)
    for ds in ("bdb2021", "bdb2026"):
        acc = by.loc[ds, "accuracy"]
        best = acc.max()
        se = math.sqrt(best * (1 - best) / by.loc[(ds, acc.idxmax()), "n"])
        plateau &= best - acc < 2 * se
        loss = loss.combine(best - acc, max)
    inside = [d for d in DEPTHS if plateau[d]]
    if not inside:
        pick = loss.idxmin()
        print(
            f"\nno plateau: nothing holds in both seasons; smallest worst-case loss "
            f"{pick:g} yd ({loss[pick]:.3f} below the worse season's best)"
        )
        return
    edge = (
        " (runs to the edge of the range)"
        if inside[0] == DEPTHS[0] or inside[-1] == DEPTHS[-1]
        else ""
    )
    median = pd.Series(inside).median()
    pick = min(inside, key=lambda d: abs(d - median))
    print(f"\nplateau {inside[0]:g}-{inside[-1]:g} yd{edge}; pick {pick:g} yd")


if __name__ == "__main__":
    main()
