"""Coverage labels from nflverse participation data.

The Big Data Bowl 2021 dataset (2018 season) ships pre- and post-snap tracking for
every pass play but no coverage labels. nflverse's play-by-play participation data
carries per-play `defense_coverage_type` and `defense_man_zone_type` for the same
seasons and shares the NFL's GSIS identifiers, so the two join on
(old_game_id, play_id) — which are exactly BDB's (gameId, playId).

Source: https://github.com/nflverse/nflverse-data/releases (pbp_participation asset,
public, no auth).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from nfl_coverage.config import DATA_RAW

NFLVERSE_DIR = DATA_RAW / "nflverse"
PARTICIPATION_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp_participation/pbp_participation_{season}.parquet"
)

# nflverse -> our canonical labels (matching the BDB 2026 team_coverage_type enums)
COVERAGE_MAP = {
    "COVER_0": "COVER_0_MAN",
    "COVER_1": "COVER_1_MAN",
    "COVER_2": "COVER_2_ZONE",
    "COVER_3": "COVER_3_ZONE",
    "COVER_4": "COVER_4_ZONE",
    "COVER_6": "COVER_6_ZONE",
    "2_MAN": "COVER_2_MAN",
    "TWO_MAN": "COVER_2_MAN",
    "PREVENT": "PREVENT",
}


def download_participation(season: int, force: bool = False) -> Path:
    """Download the participation parquet for one season; returns the local path."""
    NFLVERSE_DIR.mkdir(parents=True, exist_ok=True)
    dest = NFLVERSE_DIR / f"pbp_participation_{season}.parquet"
    if dest.exists() and not force:
        return dest
    url = PARTICIPATION_URL.format(season=season)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_coverage_labels(season: int) -> pd.DataFrame:
    """Per-play coverage labels for one season, keyed to join BDB tracking data.

    Returns columns: gameId, playId, week, coverage_type (canonical enum or NaN),
    man_zone, defenders_in_box, raw_coverage (untranslated nflverse value).
    """
    path = download_participation(season)
    part = pd.read_parquet(path)

    # nflverse_game_id looks like "2022_01_BUF_LA"; old_game_id is the 10-digit
    # GSIS id BDB uses as gameId.
    week = part["nflverse_game_id"].str.split("_").str[1].astype(int)

    out = pd.DataFrame(
        {
            "gameId": part["old_game_id"].astype(int),
            "playId": part["play_id"].astype(int),
            "week": week,
            "raw_coverage": part["defense_coverage_type"],
            "coverage_type": part["defense_coverage_type"].map(COVERAGE_MAP),
            "man_zone": part["defense_man_zone_type"],
            "defenders_in_box": part["defenders_in_box"],
        }
    )
    return out


def attach_labels(plays: pd.DataFrame, season: int) -> pd.DataFrame:
    """Left-join coverage labels onto a BDB plays/tracking frame with gameId/playId."""
    labels = load_coverage_labels(season)
    return plays.merge(
        labels.drop(columns=["week"]),
        on=["gameId", "playId"],
        how="left",
        validate="m:1",
    )
