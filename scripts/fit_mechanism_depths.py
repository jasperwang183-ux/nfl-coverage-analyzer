"""Sweep the disguise mechanisms' depth thresholds on the tuning weeks (never the holdout).

Depths are yards past the ball. Swept: the deep-middle run's landmark depth
(`mechanisms.RUN_LANDMARK_MIN`, shared by the pole run and the late post, which fires
too rarely to fit and inherits the pole run's pick; the pole run keeps its 13 yd cap,
`POLE_LANDMARK_MAX`) and the corner roll-up's cushion at
the snap and depth at the landmark (`ROLL_UP_ALIGN_MIN`, `ROLL_UP_LANDMARK_MAX`). Speeds
and widths stay: a speed is a difference, so the zero cancels. The buzz sweeps its landing
width and the corners' cushion (`BUZZ_LAT_MIN`, `BUZZ_SOFT_MIN`).

A setting is scored by its lift for the coverages the mechanism implies: the implied share
of plays where it fires over the share of all plays. Settings firing on fewer than
--min-n plays across both seasons are dropped (a rule that fires on five plays is noise).
The plateau is the settings within 2 standard errors of each season's best lift, in both
seasons; the pick is its middle (each setting's median, then the nearest setting inside
it). With no plateau, the pick is the smallest worst-case loss: the larger of the two
seasons' gaps below their own best lift.

The buzz is picked by the analyst's walk instead, because its precision keeps rising as an
edge tightens and has no peak to plateau around. Start at the most precise setting whose
seasons agree within STABLE_GAP points; loosen one edge a step at a time while the step
buys at least --ratio right calls per wrong call it lets in and the row stays over the
level's precision. One short step is looked past: the walk stops only when two steps in a
row fall short. The stop is strong if it has the row's play floor, the strong precision
(no season more than SEASON_SLACK under it), seasons within STABLE_GAP, and a neighbouring
setting that passes too; else the walk reruns on the lean precision; else the row is
dropped.

Defender phases for every tuning play are cached in
data/processed/phases_tuning_ball.parquet; --rebuild re-reads the frames.

Usage: python scripts/fit_mechanism_depths.py [--min-n 40] [--ratio 2.5] [--rebuild]
"""

from __future__ import annotations

import argparse
import itertools
import math
import warnings

import numpy as np
import pandas as pd

from nfl_coverage.analysis.holdout import FRAMES_DIR, tuning_weeks
from nfl_coverage.analysis.mechanisms import (
    BUZZ_SOFT,
    CORNER_ROLL_UP,
    POLE_RUN,
    _buzz,
    _corner_roll_up,
    _pole_run,
)
from nfl_coverage.config import DATA_PROCESSED
from nfl_coverage.data.ball_spots import load_ball_spots
from nfl_coverage.features.alignment import KEYS
from nfl_coverage.features.phases import defender_phases

CACHE = DATA_PROCESSED / "phases_tuning_ball.parquet"
DATASETS = ("bdb2021", "bdb2026")
STABLE_GAP = 5.0  # points between the seasons' precision
SEASON_SLACK = 3.0  # points a season may sit under the pooled floor


def halves(lo: float, hi: float) -> list[float]:
    return [lo + 0.5 * i for i in range(round((hi - lo) / 0.5) + 1)]


SWEEPS = {
    "pole_run": {
        "grid": {"landmark_min": halves(7.0, 12.0)},
        "fires": lambda p, landmark_min: _pole_run(p, landmark_min),
        "implies": POLE_RUN.implies,
    },
    "corner_roll_up": {
        "grid": {"align_min": halves(4.0, 8.0), "landmark_max": halves(5.0, 10.0)},
        "fires": lambda p, align_min, landmark_max: _corner_roll_up(
            p, align_min=align_min, landmark_max=landmark_max
        ),
        "implies": CORNER_ROLL_UP.implies,
    },
    "buzz_soft": {
        "grid": {"lat_min": halves(6.0, 10.0), "soft_min": halves(2.0, 7.0)},
        "fires": lambda p, lat_min, soft_min: _buzz(p, lat_min, soft_min),
        "implies": BUZZ_SOFT.implies,
        "walk": {"strong": 85.0, "lean": 70.0, "min_plays": 200},
    },
}


def load(rebuild: bool) -> pd.DataFrame:
    """Defender phases of every labeled tuning play, with the play's coverage label."""
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    releases = pd.read_parquet(DATA_PROCESSED / "releases.parquet")
    spots = load_ball_spots()
    parts = []
    for ds in DATASETS:
        for w in tuning_weeks(ds):
            frames = pd.read_parquet(FRAMES_DIR / f"{ds}_w{w:02d}.parquet")
            labels = (
                frames.dropna(subset=["coverage_type"])
                .groupby(KEYS, as_index=False)["coverage_type"]
                .first()
            )
            phases = defender_phases(frames, releases, ball_spots=spots)
            parts.append(phases.merge(labels, on=KEYS))
            print(f"  {ds} w{w:02d}")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    return out


def score(phases: pd.DataFrame, name: str, min_n: int) -> pd.DataFrame:
    spec = SWEEPS[name]
    plays = phases.groupby(KEYS)["coverage_type"].first()
    implied = plays.isin(spec["implies"])
    rows = []
    for values in itertools.product(*spec["grid"].values()):
        kw = dict(zip(spec["grid"], values))
        hit = phases.loc[spec["fires"](phases, **kw).fillna(False), KEYS].drop_duplicates()
        fired = plays.index.isin(pd.MultiIndex.from_frame(hit))
        right = int((fired & implied).sum())
        row = {**kw, "n": int(fired.sum()), "right": right, "wrong": int(fired.sum()) - right}
        row["prec"] = 100 * right / row["n"] if row["n"] else np.nan
        for ds in DATASETS:
            in_ds = plays.index.get_level_values("dataset") == ds
            base = implied[in_ds].mean()
            n = int((fired & in_ds).sum())
            p = implied[fired & in_ds].mean() if n else np.nan
            row[f"{ds} n"] = n
            row[f"{ds} lift"] = p / base
            row[f"{ds} se"] = math.sqrt(p * (1 - p) / n) / base if n else np.nan
            row[f"{ds} prec"] = 100 * p
        rows.append(row)
    t = pd.DataFrame(rows)
    t["gap"] = (t[f"{DATASETS[0]} prec"] - t[f"{DATASETS[1]} prec"]).abs()
    return t[t["n"] >= min_n].reset_index(drop=True)


def pick(t: pd.DataFrame, settings: list[str]) -> tuple[pd.Series, str]:
    plateau = pd.Series(True, index=t.index)
    loss = pd.Series(0.0, index=t.index)
    for ds in DATASETS:
        lift = t[f"{ds} lift"]
        best = lift.idxmax()
        plateau &= lift[best] - lift < 2 * t.loc[best, f"{ds} se"]
        loss = loss.combine(lift[best] - lift, max)
    inside = t[plateau]
    if inside.empty:
        return t.loc[loss.idxmin()], "no plateau: smallest worst-case loss"
    median = inside[settings].median()
    spread = (inside[settings].max() - inside[settings].min()).replace(0, 1)
    chosen = inside.loc[((inside[settings] - median).abs() / spread).sum(axis=1).idxmin()]
    ranges = []
    for s in settings:
        lo, hi = inside[s].min(), inside[s].max()
        grid = t[s]
        edge = " (runs to the edge)" if lo == grid.min() or hi == grid.max() else ""
        ranges.append(f"{s} {lo:g}-{hi:g}{edge}")
    return chosen, f"plateau of {len(inside)}: " + ", ".join(ranges)


def walk(t: pd.DataFrame, settings: list[str], ratio: float, floor: float) -> int:
    """The analyst's walk: index of the setting where loosening stops paying `ratio`."""
    at = t[t["gap"] <= STABLE_GAP].sort_values(["prec", "n"], ascending=False).index[0]

    def pays(a: int, b: int) -> bool:
        gained = t.loc[b, "right"] - t.loc[a, "right"]
        invited = t.loc[b, "wrong"] - t.loc[a, "wrong"]
        return gained >= ratio * invited if invited > 0 else gained > 0

    moved = True
    while moved:
        moved = False
        for s in settings:
            others = [o for o in settings if o != s]
            on_line = (t[others] == t.loc[at, others]).all(axis=1)
            line = t[on_line].sort_values(s).index.tolist()
            i = line.index(at)
            step = (
                1
                if i + 1 < len(line) and (i == 0 or t.loc[line[i + 1], "n"] > t.loc[at, "n"])
                else -1
            )
            ahead = [line[j] for j in (i + step, i + 2 * step) if 0 <= j < len(line)]
            if ahead and pays(at, ahead[0]) and t.loc[ahead[0], "prec"] >= floor:
                at, moved = ahead[0], True
            elif len(ahead) == 2 and pays(ahead[0], ahead[1]) and t.loc[ahead[1], "prec"] >= floor:
                at, moved = ahead[1], True
    return at


def walk_pick(t: pd.DataFrame, settings: list[str], ratio: float, floors: dict) -> tuple[int, str]:
    """The walk's stop and its level, strong / lean / dropped, by the row's floors."""
    seasons = t[[f"{ds} prec" for ds in DATASETS]].min(axis=1)
    for level in ("strong", "lean"):
        at = walk(t, settings, ratio, floors[level])
        near = (t[settings] - t.loc[at, settings]).abs().sum(axis=1) == 0.5
        ok = (
            (t["n"] >= floors["min_plays"])
            & (t["prec"] >= floors[level])
            & (seasons >= floors[level] - SEASON_SLACK)
            & (t["gap"] <= STABLE_GAP)
        )
        if ok[at] and ok[near].any():
            return at, level
    return at, "dropped"


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--ratio", type=float, default=2.5, help="right calls per wrong, the walk")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    phases = load(args.rebuild)
    print(f"{phases.groupby(KEYS).ngroups} labeled tuning plays")
    for name, spec in SWEEPS.items():
        settings = list(spec["grid"])
        t = score(phases, name, args.min_n)
        print(f"\n=== {name} (implies {', '.join(sorted(spec['implies']))}) ===")
        if t.empty:
            print(f"no setting fires on {args.min_n} plays")
            continue
        print(t.round(3).to_string(index=False))
        if "walk" in spec:
            at, verdict = walk_pick(t, settings, args.ratio, spec["walk"])
            c = t.loc[at]
            seasons = " / ".join(f"{c[f'{ds} prec']:.1f}" for ds in DATASETS)
            print(
                f"walk at {args.ratio:g} right per wrong stops at "
                + ", ".join(f"{s} {c[s]:g}" for s in settings)
                + f": {c['n']:.0f} plays, {c['prec']:.1f}% ({seasons}), gap {c['gap']:.1f}"
                + f" -> {verdict}"
            )
            continue
        chosen, how = pick(t, settings)
        print(f"{how}\npick: " + ", ".join(f"{s} {chosen[s]:g}" for s in settings))


if __name__ == "__main__":
    main()
