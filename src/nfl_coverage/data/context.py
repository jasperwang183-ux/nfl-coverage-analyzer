"""Per-play context (defense, down, distance, ...) for both datasets, one schema.

The canonical frames carry positions and the coverage label only. Disguise rates
by team, down and distance need the play metadata that lives in
supplementary_data.csv (bdb2026) and plays.csv + games.csv (bdb2021).
"""

from __future__ import annotations

import pandas as pd

from nfl_coverage.config import BDB2021, BDB2026

CONTEXT_COLUMNS = [
    "game_id",
    "play_id",
    "week",
    "offense",
    "defense",
    "quarter",
    "down",
    "yards_to_go",
    "defenders_in_box",
    "number_of_pass_rushers",  # bdb2021 only: 2023 tracking has no rushers and no count
]


def _context_bdb2026(supplementary: pd.DataFrame) -> pd.DataFrame:
    df = supplementary.rename(
        columns={
            "possession_team": "offense",
            "defensive_team": "defense",
            "defenders_in_the_box": "defenders_in_box",
        }
    ).assign(number_of_pass_rushers=float("nan"))
    return df[CONTEXT_COLUMNS].drop_duplicates(["game_id", "play_id"]).reset_index(drop=True)


def _context_bdb2021(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    df = plays.rename(
        columns={
            "gameId": "game_id",
            "playId": "play_id",
            "possessionTeam": "offense",
            "yardsToGo": "yards_to_go",
            "defendersInTheBox": "defenders_in_box",
            "numberOfPassRushers": "number_of_pass_rushers",
        }
    )
    g = games.rename(
        columns={"gameId": "game_id", "homeTeamAbbr": "home", "visitorTeamAbbr": "away"}
    )
    df = df.merge(g[["game_id", "home", "away", "week"]], on="game_id", how="left", validate="m:1")
    df["defense"] = df["home"].where(df["offense"] == df["away"], df["away"])
    return df[CONTEXT_COLUMNS].drop_duplicates(["game_id", "play_id"]).reset_index(drop=True)


def load_play_context(dataset: str) -> pd.DataFrame:
    """Play context for one dataset ('bdb2021' or 'bdb2026'), keyed by game_id/play_id."""
    if dataset == "bdb2026":
        return _context_bdb2026(pd.read_csv(BDB2026 / "supplementary_data.csv", low_memory=False))
    if dataset == "bdb2021":
        return _context_bdb2021(
            pd.read_csv(BDB2021 / "plays.csv"), pd.read_csv(BDB2021 / "games.csv")
        )
    raise ValueError(f"unknown dataset {dataset!r}")


def distance_bucket(yards_to_go: pd.Series) -> pd.Series:
    """Short (<= 3), medium (4-7), long (>= 8) — the buckets a DC game-plans by."""
    return pd.cut(yards_to_go, [-1, 3, 7, 200], labels=["short", "medium", "long"])
