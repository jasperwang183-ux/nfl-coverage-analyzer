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
dropped. Elite precision exception (analyst, 2026-10-20): a row with an "elite" entry
whose setting is at least ELITE_PREC precise on the tuning weeks needs only
ELITE_MIN_PLAYS plays instead of its floor; the other checks stay.

The pressure rows score against the 2018 rush count (`plays.numberOfPassRushers`: 4 or
fewer, or 5 or more) instead of the coverage label, on 2018 only, with weeks 1-7 and 8-13
standing in for the two seasons.

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
    FRONT_BLUFF,
    POLE_RUN,
    SECOND_LEVEL_RUSH,
    _buzz,
    _corner_roll_up,
    _front_bluff,
    _pole_run,
    _second_level_rush,
)
from nfl_coverage.config import DATA_PROCESSED
from nfl_coverage.data.ball_spots import load_ball_spots
from nfl_coverage.data.context import load_play_context
from nfl_coverage.features.alignment import KEYS
from nfl_coverage.features.phases import defender_phases

CACHE = DATA_PROCESSED / "phases_tuning_ball.parquet"
DATASETS = ("bdb2021", "bdb2026")
STABLE_GAP = 5.0  # points between the seasons' precision
SEASON_SLACK = 3.0  # points a season may sit under the pooled floor
# The elite precision exception: this precise on the tuning weeks, this many plays suffice
ELITE_PREC = 88.0
ELITE_MIN_PLAYS = 50
ELITE = {"elite": (ELITE_PREC, ELITE_MIN_PLAYS)}
# Pressure rows score on 2018 only, so the two halves of its tuning weeks stand in for the
# two seasons.
BLOCKS = ("w1-7", "w8-13")
RUSH = {"target": "rush_label", "by": ("block", BLOCKS)}


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
        "walk": {"strong": 85.0, "lean": 70.0, "min_plays": 200, "slack": SEASON_SLACK},
    },
    "front_bluff": {
        "grid": {"lat_max": halves(2.0, 4.0), "bail_min": halves(0.0, 3.0)},
        "fires": lambda p, lat_max, bail_min: _front_bluff(p, lat_max, bail_min),
        "implies": FRONT_BLUFF.implies,
        "walk": {"strong": 75.0, "lean": 65.0, "min_plays": 75, **ELITE},
        **RUSH,
    },
    "second_level_rush": {
        "grid": {"lat_max": halves(2.0, 4.0), "downhill_max": halves(-6.0, -2.0)},
        "fires": lambda p, lat_max, downhill_max: _second_level_rush(p, lat_max, downhill_max),
        "implies": SECOND_LEVEL_RUSH.implies,
        "walk": {"strong": 70.0, "lean": 60.0, "min_plays": 75, **ELITE},
        **RUSH,
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


def with_rush(phases: pd.DataFrame) -> pd.DataFrame:
    """Add each 2018 play's rush label (RUSH_LE4 / RUSH_GE5) and week block; NaN elsewhere."""
    ctx = load_play_context("bdb2021").dropna(subset=["number_of_pass_rushers"])
    ctx = ctx.assign(
        dataset="bdb2021",
        rush_label=np.where(ctx["number_of_pass_rushers"] <= 4, "RUSH_LE4", "RUSH_GE5"),
        block=np.where(ctx["week"] <= 7, BLOCKS[0], BLOCKS[1]),
    )
    return phases.merge(ctx[KEYS + ["rush_label", "block"]], on=KEYS, how="left")


def score(phases: pd.DataFrame, name: str, min_n: int) -> pd.DataFrame:
    spec = SWEEPS[name]
    target = spec.get("target", "coverage_type")
    by, groups = spec.get("by", ("dataset", DATASETS))
    phases = phases[phases[target].notna()]
    plays = phases.groupby(KEYS)[[target, by]].first()
    implied = plays[target].isin(spec["implies"])
    rows = []
    for values in itertools.product(*spec["grid"].values()):
        kw = dict(zip(spec["grid"], values))
        hit = phases.loc[spec["fires"](phases, **kw).fillna(False), KEYS].drop_duplicates()
        fired = plays.index.isin(pd.MultiIndex.from_frame(hit))
        right = int((fired & implied).sum())
        row = {**kw, "n": int(fired.sum()), "right": right, "wrong": int(fired.sum()) - right}
        row["prec"] = 100 * right / row["n"] if row["n"] else np.nan
        for ds in groups:
            in_ds = (plays[by] == ds).to_numpy()
            base = implied[in_ds].mean()
            n = int((fired & in_ds).sum())
            p = implied[fired & in_ds].mean() if n else np.nan
            row[f"{ds} n"] = n
            row[f"{ds} lift"] = p / base
            row[f"{ds} se"] = math.sqrt(p * (1 - p) / n) / base if n else np.nan
            row[f"{ds} prec"] = 100 * p
        rows.append(row)
    t = pd.DataFrame(rows)
    t["gap"] = (t[f"{groups[0]} prec"] - t[f"{groups[1]} prec"]).abs()
    return t[t["n"] >= min_n].reset_index(drop=True)


def pick(
    t: pd.DataFrame, settings: list[str], groups: tuple[str, ...] = DATASETS
) -> tuple[pd.Series, str]:
    plateau = pd.Series(True, index=t.index)
    loss = pd.Series(0.0, index=t.index)
    for ds in groups:
        lift = t[f"{ds} lift"]
        best = lift.idxmax()
        plateau &= lift[best] - lift < 2 * t.loc[best, f"{ds} se"]
        loss = np.maximum(loss, lift[best] - lift)  # NaN (a group with no plays) stays NaN
    inside = t[plateau]
    if inside.empty:
        return t.loc[loss.dropna().idxmin()], "no plateau: smallest worst-case loss"
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


def walk(t: pd.DataFrame, settings: list[str], ratio: float, floor: float) -> int | None:
    """The analyst's walk: index of the setting where loosening stops paying `ratio`.

    None when no setting's seasons agree within STABLE_GAP: there is nowhere to start.
    """
    stable = t[t["gap"] <= STABLE_GAP]
    if stable.empty:
        return None
    at = stable.sort_values(["prec", "n"], ascending=False).index[0]

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
            # loosening is the direction that fires on more plays; on a tie, the upper side
            up = t.loc[line[i + 1], "n"] if i + 1 < len(line) else -1
            down = t.loc[line[i - 1], "n"] if i > 0 else -1
            step = 1 if up >= down else -1
            ahead = [line[j] for j in (i + step, i + 2 * step) if 0 <= j < len(line)]
            if ahead and pays(at, ahead[0]) and t.loc[ahead[0], "prec"] >= floor:
                at, moved = ahead[0], True
            elif len(ahead) == 2 and pays(ahead[0], ahead[1]) and t.loc[ahead[1], "prec"] >= floor:
                at, moved = ahead[1], True
    return at


def walk_pick(
    t: pd.DataFrame, settings: list[str], ratio: float, floors: dict, groups: tuple[str, str]
) -> tuple[int | None, str]:
    """The walk's stop and its level, strong / lean / dropped, by the row's floors."""
    worst = t[[f"{g} prec" for g in groups]].min(axis=1)
    at = None
    for level in [lv for lv in ("strong", "lean") if lv in floors]:
        at = walk(t, settings, ratio, floors[level])
        if at is None:
            return None, "dropped"
        near = (t[settings] - t.loc[at, settings]).abs().sum(axis=1) == 0.5
        elite_prec, elite_plays = floors.get("elite", (np.inf, np.inf))
        enough = (t["n"] >= floors["min_plays"]) | (
            (t["n"] >= elite_plays) & (t["prec"] >= elite_prec)
        )
        ok = (
            enough
            & (t["prec"] >= floors[level])
            & (worst >= floors[level] - floors.get("slack", np.inf))
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
    phases = with_rush(load(args.rebuild))
    print(f"{phases.groupby(KEYS).ngroups} labeled tuning plays")
    for name, spec in SWEEPS.items():
        settings = list(spec["grid"])
        t = score(phases, name, args.min_n)
        print(f"\n=== {name} (implies {', '.join(sorted(spec['implies']))}) ===")
        if t.empty:
            print(f"no setting fires on {args.min_n} plays")
            continue
        print(t.round(3).to_string(index=False))
        groups = spec.get("by", ("dataset", DATASETS))[1]
        if spec.get("walk", False) is None:
            print("walk floors not set yet")
            continue
        if "walk" in spec:
            at, verdict = walk_pick(t, settings, args.ratio, spec["walk"], groups)
            if at is None:
                print(f"no setting's groups agree within {STABLE_GAP:g} pts -> {verdict}")
                continue
            c = t.loc[at]
            seasons = " / ".join(f"{c[f'{g} prec']:.1f}" for g in groups)
            print(
                f"walk at {args.ratio:g} right per wrong stops at "
                + ", ".join(f"{s} {c[s]:g}" for s in settings)
                + f": {c['n']:.0f} plays, {c['prec']:.1f}% ({seasons}), gap {c['gap']:.1f}"
                + f" -> {verdict}"
            )
            continue
        chosen, how = pick(t, settings, groups)
        print(f"{how}\npick: " + ", ".join(f"{s} {chosen[s]:g}" for s in settings))


if __name__ == "__main__":
    main()
