"""Fit the corner roll-up on the tuning weeks: corner phases by label, then a threshold sweep.

Part 1 prints quantiles of each corner's phases (align_depth, first_v_depth,
commit_v_depth, landmark_depth, |landmark_lat|) for Cover 2 zone, Cover 3 zone (Cloud
lives there), Cover 4 zone, Cover 1 man and the rest. Part 2 sweeps the thresholds of
`mechanisms._corner_roll_up` and prints the settings that lift Cover 2 zone most: fire
rate, label shares when it fires, and the lift for Cover 2 zone alone and for Cover 2 or
3 zone. Pick a setting, write it into mechanisms.py, then run
`check_mechanisms.py --candidates` for the lift by shown look.

Tuning weeks only. The holdout is scored once, by `check_mechanisms.py --holdout`, after
the row is in MECHANISMS.

Usage:
  python scripts/fit_corner_roll_up.py               # reads the weeks once, caches corners
  python scripts/fit_corner_roll_up.py --soft 6      # part 1 for corners aligned >= 6 yd off
  python scripts/fit_corner_roll_up.py --rebuild     # re-read the frames
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

from nfl_coverage.analysis.holdout import FRAMES_DIR, TUNING
from nfl_coverage.analysis.mechanisms import _corner_roll_up
from nfl_coverage.analysis.tendencies import CLASSES, shown_reads
from nfl_coverage.config import DATA_PROCESSED
from nfl_coverage.features.alignment import KEYS
from nfl_coverage.features.phases import defender_phases

CACHE = DATA_PROCESSED / "corner_phases_tuning.parquet"
GROUPS = ["COVER_2_ZONE", "COVER_3_ZONE", "COVER_4_ZONE", "COVER_1_MAN"]
COLUMNS = ["align_depth", "first_v_depth", "commit_v_depth", "landmark_depth", "abs_lat"]
GRID = {
    "align_min": [5.0, 6.0, 7.0, 8.0],
    "first_min": [None, -0.5, 0.0, 0.5],
    "commit_max": [0.0, -0.5, -1.0, -1.5, -2.0, -3.0],
    "landmark_max": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "lat_min": [None, 6.0, 10.0, 12.0, 14.0, 16.0, 18.0],
}


def load(rebuild: bool) -> pd.DataFrame:
    """One row per corner at the snap of a labeled tuning play, with its label and look."""
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    releases = pd.read_parquet(DATA_PROCESSED / "releases.parquet")
    parts = []
    for ds, ws in TUNING.items():
        for w in ws:
            frames = pd.read_parquet(FRAMES_DIR / f"{ds}_w{w:02d}.parquet")
            labels = frames.groupby(KEYS, as_index=False)["coverage_type"].first()
            plays = labels.merge(shown_reads(frames)[KEYS + ["shown"]], on=KEYS)
            phases = defender_phases(frames, releases)
            parts.append(phases[phases["role"] == "corner"].merge(plays, on=KEYS))
            print(f"  {ds} w{w:02d}")
    corners = pd.concat(parts, ignore_index=True)
    corners = corners[corners["coverage_type"].isin(CLASSES)].reset_index(drop=True)
    corners.to_parquet(CACHE, index=False)
    return corners


def distributions(corners: pd.DataFrame, soft: float | None) -> None:
    c = corners.assign(abs_lat=corners["landmark_lat"].abs())
    if soft is not None:
        c = c[c["align_depth"] >= soft]
    c = c.assign(group=c["coverage_type"].where(c["coverage_type"].isin(GROUPS), "other"))
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    title = f"corners aligned >= {soft:g} yd off" if soft is not None else "all corners"
    print(f"\n=== {title}: quantiles by label (n corners) ===")
    for col in COLUMNS:
        t = c.groupby("group")[col].quantile(q).unstack()
        t.columns = [f"p{int(x * 100)}" for x in q]
        t.insert(0, "n", c.groupby("group")[col].count())
        print(f"\n{col}\n{t.reindex(GROUPS + ['other']).round(2).to_string()}")


def sweep(corners: pd.DataFrame, min_n: int, top: int) -> None:
    play = corners.groupby(KEYS).ngroup().to_numpy()
    labels = corners.groupby(KEYS)["coverage_type"].first().to_numpy()
    n_plays = len(labels)
    c2, c3, c1 = (labels == "COVER_2_ZONE"), (labels == "COVER_3_ZONE"), (labels == "COVER_1_MAN")
    base2, base23 = c2.mean(), (c2 | c3).mean()
    rows = []
    for values in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, values))
        mask = _corner_roll_up(corners, **kw).fillna(False).to_numpy()
        fired = np.bincount(play[mask], minlength=n_plays) > 0
        n = int(fired.sum())
        if n < min_n:
            continue
        s2, s3 = c2[fired].mean(), c3[fired].mean()
        rows.append(
            {
                **kw,
                "n": n,
                "fires": n / n_plays,
                "cover_2_zone": s2,
                "cover_3_zone": s3,
                "cover_1_man": c1[fired].mean(),
                "lift_2": s2 / base2,
                "lift_2_or_3": (s2 + s3) / base23,
                "catches_2": fired[c2].mean(),
            }
        )
    print(
        f"\n=== sweep: {n_plays} plays, Cover 2 zone {base2:.3f}, Cover 2 or 3 zone "
        f"{base23:.3f}; settings firing on >= {min_n} plays, best Cover 2 zone lift first ==="
    )
    if not rows:
        print("no setting fires often enough; lower --min-n")
        return
    t = pd.DataFrame(rows).sort_values(["lift_2", "n"], ascending=False).head(top)
    print(t.round(3).to_string(index=False))


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-read the frames, not the cache")
    ap.add_argument("--soft", type=float, help="part 1 only for corners aligned this deep")
    ap.add_argument("--min-n", type=int, default=40, help="fewest plays a setting must fire on")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    corners = load(args.rebuild)
    distributions(corners, args.soft)
    sweep(corners, args.min_n, args.top)


if __name__ == "__main__":
    main()
