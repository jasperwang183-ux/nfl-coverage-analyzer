"""Build data/processed/ball_spots.parquet: the ball's x at the snap of every cached play.

Usage: python scripts/build_ball_spots.py

Reads the raw tracking week behind every cached data/processed/frames/{dataset}_wNN.parquet
(the cached frames are read for their play keys and the stand-in LOS, never rewritten).
The report checks the table: how many plays have a spot, how far the stand-in LOS
(`features.alignment._los_estimate`) sits from the ball, and for 2018 how far the tracked
football is from the recorded spot (plays.csv `absoluteYardlineNumber`).
"""

from __future__ import annotations

import re
import time

import pandas as pd

from nfl_coverage.config import BDB2021, BDB2026, DATA_PROCESSED, FIELD_LENGTH
from nfl_coverage.data.ball_spots import ball_spots_bdb2021, ball_spots_bdb2026
from nfl_coverage.features.alignment import _los_estimate

FRAMES_DIR = DATA_PROCESSED / "frames"
DEST = DATA_PROCESSED / "ball_spots.parquet"
KEYS = ["game_id", "play_id"]


def _week_spots(dataset: str, week: int) -> pd.DataFrame:
    if dataset == "bdb2021":
        raw = pd.read_csv(
            BDB2021 / f"week{week}.csv",
            usecols=["gameId", "playId", "frameId", "event", "team", "playDirection", "x", "y"],
        )
        spots = ball_spots_bdb2021(raw)
        return spots.merge(_recorded_bdb2021(raw), on=KEYS, how="left")
    raw = pd.read_csv(
        BDB2026 / "train" / f"input_2023_w{week:02d}.csv",
        usecols=[*KEYS, "frame_id", "play_direction", "absolute_yardline_number"],
    )
    return ball_spots_bdb2026(raw)


def _recorded_bdb2021(raw: pd.DataFrame) -> pd.DataFrame:
    """plays.csv's recorded spot in canonical x, for the report only."""
    plays = pd.read_csv(
        BDB2021 / "plays.csv", usecols=["gameId", "playId", "absoluteYardlineNumber"]
    )
    direction = raw.drop_duplicates(["gameId", "playId"])[["gameId", "playId", "playDirection"]]
    rec = plays.merge(direction, on=["gameId", "playId"])
    spot = rec["absoluteYardlineNumber"]
    rec["recorded_x"] = spot.where(rec["playDirection"] != "left", FIELD_LENGTH - spot)
    return rec.rename(columns={"gameId": "game_id", "playId": "play_id"})[[*KEYS, "recorded_x"]]


def build() -> pd.DataFrame:
    parts = []
    for path in sorted(FRAMES_DIR.glob("*_w*.parquet")):
        dataset, week = re.fullmatch(r"(\w+)_w(\d+)", path.stem).groups()
        t0 = time.time()
        spots = _week_spots(dataset, int(week))
        cached = pd.read_parquet(path, columns=[*KEYS, "rel_frame", "side", "x"])
        snap_off = cached[(cached["rel_frame"] == 0) & (cached["side"] == "Offense")]
        stand_in = _los_estimate(snap_off.assign(dataset=dataset)).rename("stand_in").reset_index()
        # The cached frames define the play set (the loader drops plays it cannot join).
        kept = (
            cached[KEYS].drop_duplicates().merge(spots.drop(columns="dataset"), on=KEYS, how="left")
        )
        kept = kept.merge(stand_in[[*KEYS, "stand_in"]], on=KEYS, how="left", validate="1:1")
        kept["dataset"] = dataset
        print(
            f"{path.stem}: {len(kept):,} cached plays, {kept['ball_x'].isna().sum()} without a "
            f"spot, {time.time() - t0:.0f}s"
        )
        parts.append(kept)
    return pd.concat(parts, ignore_index=True)


def report(spots: pd.DataFrame) -> None:
    for dataset, grp in spots.groupby("dataset"):
        print(f"\n{dataset}: {len(grp):,} plays, {grp['ball_x'].isna().sum()} without a spot")
        gap = (grp["stand_in"] - grp["ball_x"]).dropna()
        q = gap.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(
            "  stand-in LOS minus ball (yd) q05/q25/q50/q75/q95: "
            + " / ".join(f"{v:.2f}" for v in q)
        )
        if grp["recorded_x"].notna().any():
            diff = (grp["ball_x"] - grp["recorded_x"]).dropna()
            q = diff.abs().quantile([0.5, 0.9, 0.99])
            print(
                "  tracked football minus recorded spot, |yd| q50/q90/q99: "
                + " / ".join(f"{v:.2f}" for v in q)
                + f"; over 2 yd: {(diff.abs() > 2).sum()}"
            )


if __name__ == "__main__":
    spots = build()
    report(spots)
    spots[["dataset", *KEYS, "ball_x"]].to_parquet(DEST, index=False)
    print(f"\nwrote {DEST} ({len(spots):,} plays)")
