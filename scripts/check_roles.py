"""Check the geometric snap roles against roster positions on the tuning weeks.

Roster position is the validation target only: `snap_roles` never reads it. Prints,
per dataset, a role x position-group table, the share of CBs read as corners and of
safeties read as safeties, and how often the mike exists and is a linebacker.

Usage:
  python scripts/check_roles.py [--weeks 1 2 3 4]
"""

from __future__ import annotations

import argparse

import pandas as pd

from nfl_coverage.analysis.holdout import FRAMES_DIR, HOLDOUT
from nfl_coverage.analysis.roles import ROLES, snap_roles
from nfl_coverage.features.alignment import KEYS

GROUPS = {
    "CB": "CB",
    "FS": "S",
    "SS": "S",
    "S": "S",
    "ILB": "ILB",
    "MLB": "ILB",
    "LB": "ILB",
    "OLB": "OLB",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
}
GROUP_ORDER = ["CB", "S", "ILB", "OLB", "DL", "other"]


def check(dataset: str, weeks: list[int]) -> None:
    held = set(weeks) & set(HOLDOUT[dataset])
    if held:
        raise SystemExit(f"{dataset} weeks {sorted(held)} are the frozen holdout")
    frames = pd.concat(
        [pd.read_parquet(FRAMES_DIR / f"{dataset}_w{w:02d}.parquet") for w in weeks],
        ignore_index=True,
    )
    roles = snap_roles(frames)
    snap = frames[(frames["rel_frame"] == 0) & (frames["side"] == "Defense")]
    pos = snap.drop_duplicates(KEYS + ["nfl_id"])[KEYS + ["nfl_id", "position"]]
    r = roles.merge(pos, on=KEYS + ["nfl_id"], how="left")
    r["group"] = r["position"].map(GROUPS).fillna("other")

    n_plays = r.groupby(KEYS).ngroups
    print(f"\n=== {dataset} weeks {weeks}: {n_plays} plays, {len(r)} defenders")
    table = pd.crosstab(r["role"], r["group"], margins=True)
    table = table.reindex(index=[*ROLES, "All"], columns=[*GROUP_ORDER, "All"], fill_value=0)
    print(table.to_string())
    print("\ncolumn shares (what each position group is read as):")
    print(
        (pd.crosstab(r["role"], r["group"], normalize="columns") * 100)
        .round(1)
        .reindex(index=list(ROLES), columns=GROUP_ORDER)
        .to_string()
    )

    cb, s = r[r["group"] == "CB"], r[r["group"] == "S"]
    print(f"\nCB read as corner:     {(cb['role'] == 'corner').mean():.1%} of {len(cb)}")
    print(f"S read as safety:      {(s['role'] == 'safety').mean():.1%} of {len(s)}")
    corners = r[r["role"] == "corner"]
    print(f"corner reads that are CB: {(corners['group'] == 'CB').mean():.1%} of {len(corners)}")
    safeties = r[r["role"] == "safety"]
    print(f"safety reads that are S:  {(safeties['group'] == 'S').mean():.1%} of {len(safeties)}")

    mike = r[r["is_mike"]]
    per_play = mike.groupby(KEYS).size()
    print(f"\nplays with a mike:     {len(per_play) / n_plays:.1%} (max per play {per_play.max()})")
    lb = mike["position"].isin(["ILB", "MLB", "LB"]).mean()
    print(f"mike is ILB/MLB/LB:    {lb:.1%} of {len(mike)}")
    print(f"mike is ILB/MLB/LB/OLB: {(lb + (mike['position'] == 'OLB').mean()):.1%}")
    print("mike by position:", mike["position"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weeks", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    for ds in ("bdb2021", "bdb2026"):
        check(ds, args.weeks)


if __name__ == "__main__":
    main()
