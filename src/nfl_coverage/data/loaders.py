"""Load BDB 2021 and BDB 2026 tracking weeks into the canonical frame schema.

Each loader is split into a pure `_canonicalize_*` transform (unit-testable on tiny
synthetic frames) and a thin IO wrapper that reads the raw CSVs. See
`standardize.CANONICAL_COLUMNS` for the output contract.
"""

from __future__ import annotations

import pandas as pd

from nfl_coverage.config import BDB2021, BDB2026
from nfl_coverage.data.nflverse import load_coverage_labels
from nfl_coverage.data.standardize import add_rel_frame, finalize, flip_left_plays

# ---------------------------------------------------------------- BDB 2026

_BDB2026_RENAMES = {
    "nfl_id": "nfl_id",
    "player_position": "position",
    "player_side": "side",
}


def _canonicalize_bdb2026(tracking: pd.DataFrame, supplementary: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize one week of BDB 2026 input tracking. Frame 1 is the snap (verified)."""
    df = tracking.rename(columns=_BDB2026_RENAMES)
    df = flip_left_plays(df, direction_col="play_direction")
    df["rel_frame"] = df["frame_id"] - 1

    labels = supplementary.rename(
        columns={"team_coverage_type": "coverage_type", "team_coverage_man_zone": "man_zone"}
    )[["game_id", "play_id", "coverage_type", "man_zone"]]
    df = df.merge(labels, on=["game_id", "play_id"], how="left", validate="m:1")
    return finalize(df, dataset="bdb2026")


def load_bdb2026_week(week: int) -> pd.DataFrame:
    tracking = pd.read_csv(BDB2026 / "train" / f"input_2023_w{week:02d}.csv")
    supplementary = pd.read_csv(BDB2026 / "supplementary_data.csv", low_memory=False)
    return _canonicalize_bdb2026(tracking, supplementary)


# ---------------------------------------------------------------- BDB 2021

_BDB2021_RENAMES = {
    "gameId": "game_id",
    "playId": "play_id",
    "nflId": "nfl_id",
    "frameId": "frame_id",
    "playDirection": "play_direction",
}


def _canonicalize_bdb2021(
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    games: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Canonicalize one week of BDB 2021 tracking.

    `labels` is the output of nflverse.load_coverage_labels(2018) (nullable — plays
    without labels are kept with NaN so unlabeled plays can still feed the vision work).
    """
    df = tracking.rename(columns=_BDB2021_RENAMES)
    df = df[df["team"] != "football"].copy()  # ball rows: no nfl_id, no side

    # side: map home/away -> club abbr, compare with possessionTeam
    game_teams = games.rename(columns={"gameId": "game_id"})[
        ["game_id", "homeTeamAbbr", "visitorTeamAbbr"]
    ]
    play_offense = plays.rename(columns={"gameId": "game_id", "playId": "play_id"})[
        ["game_id", "play_id", "possessionTeam"]
    ]
    df = df.merge(game_teams, on="game_id", validate="m:1")
    df = df.merge(play_offense, on=["game_id", "play_id"], validate="m:1")
    club = df["homeTeamAbbr"].where(df["team"] == "home", df["visitorTeamAbbr"])
    df["side"] = (club == df["possessionTeam"]).map({True: "Offense", False: "Defense"})
    df = df.drop(columns=["homeTeamAbbr", "visitorTeamAbbr", "possessionTeam", "team"])

    # snap alignment: ball_snap event (frame 11 in practice, but derive per play)
    snap_frames = (
        df.loc[df["event"] == "ball_snap"]
        .groupby(["game_id", "play_id"], as_index=False)["frame_id"]
        .min()
        .rename(columns={"frame_id": "snap_frame"})
    )
    df = add_rel_frame(df, snap_frames)
    df = flip_left_plays(df, direction_col="play_direction")

    label_cols = labels.rename(columns={"gameId": "game_id", "playId": "play_id"})[
        ["game_id", "play_id", "coverage_type", "man_zone"]
    ]
    df = df.merge(label_cols, on=["game_id", "play_id"], how="left", validate="m:1")
    return finalize(df, dataset="bdb2021")


def load_bdb2021_week(week: int, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    tracking = pd.read_csv(BDB2021 / f"week{week}.csv")
    plays = pd.read_csv(BDB2021 / "plays.csv")
    games = pd.read_csv(BDB2021 / "games.csv")
    if labels is None:
        labels = load_coverage_labels(2018)
    return _canonicalize_bdb2021(tracking, plays, games, labels)
