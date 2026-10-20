"""Each defender's alignment, trigger and landmark: the three phases a disguise plays out in.

  alignment  where he stands at the snap and the role that implies (`analysis.roles`);
  trigger    his velocity over the first steps and over the commit window, from
             finite differences of positions (film has no speed or direction columns);
  landmark   where he is at the earlier of LANDMARK_CAP_S and the pass release: after
             the throw defenders break on the ball, and after ~2 s routes drag man
             defenders deep, so neither says what the coverage was.

Depth is yards past the LOS, lateral is y - formation center, both from the snap.
One row per defender; a phase whose frames are missing (a play too short, a film
track dropped) is NaN rather than guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_coverage.analysis.roles import snap_roles
from nfl_coverage.analysis.shell import _formation_center
from nfl_coverage.features.alignment import KEYS, snap_los

FIRST_STEPS = (0.3, 0.7)  # s after the snap
COMMIT = (0.7, 1.2)  # s: a disguise is decided by here (a spin is done by ~0.5 s)
LANDMARK_CAP_S = 2.0
FIRST_MOVE_YD = 1.0  # displacement from the snap spot that counts as moving

PHASE_COLUMNS = [
    "position",  # roster position at the snap; None on film
    "role",
    "is_mike",
    "align_depth",
    "align_lat",
    "first_v_depth",
    "first_v_lat",
    "commit_v_depth",
    "commit_v_lat",
    "first_move_t",
    "landmark_t",
    "landmark_depth",
    "landmark_lat",
    "landmark_o",
]


def _snap_reference(
    frames: pd.DataFrame,
    los: float | None,
    center: float | None,
    ball_spots: pd.DataFrame | None = None,
):
    snap = frames[(frames["rel_frame"] == 0) & (frames["side"] == "Offense")]
    los_s = snap_los(snap, ball_spots)
    ref = pd.concat([los_s, _formation_center(snap).rename("center")], axis=1)
    if los is not None:
        ref["los"] = los
    if center is not None:
        ref["center"] = center
    return ref


def _at(d: pd.DataFrame, rel_frame: int, name: str) -> pd.DataFrame:
    rows = d[d["rel_frame"] == rel_frame].set_index(KEYS + ["nfl_id"])
    return rows[["depth", "lat"]].add_prefix(f"{name}_")


def defender_phases(
    frames: pd.DataFrame,
    releases: pd.DataFrame | None = None,
    los: float | None = None,
    center: float | None = None,
    step: float = 0.1,
    ball_spots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """KEYS + nfl_id + PHASE_COLUMNS for every defender tracked at the snap.

    `releases` holds KEYS + `release_rel_frame` (data/processed/releases.parquet);
    without it the landmark is read at LANDMARK_CAP_S or the play's last frame.
    `los`/`center` override the per-play estimates (film), `step` is s per rel_frame.
    `ball_spots` (KEYS + ball_x, `data.ball_spots`) measures depth from the ball instead
    of the stand-in LOS.
    """
    # one cached 2018 play (game 2018091605) carries every row twice
    frames = frames.drop_duplicates(KEYS + ["nfl_id", "rel_frame"])
    roles = snap_roles(frames, los, center, ball_spots).set_index(KEYS + ["nfl_id"])
    ref = _snap_reference(frames, los, center, ball_spots)
    d = frames[(frames["side"] == "Defense") & (frames["rel_frame"] >= 0)].join(ref, on=KEYS)
    d = d.assign(depth=d["x"] - d["los"], lat=d["y"] - d["center"])

    def f(t: float) -> int:
        return round(t / step)

    out = roles[["role", "is_mike", "depth", "lateral"]].rename(
        columns={"depth": "align_depth", "lateral": "align_lat"}
    )
    out["position"] = d[d["rel_frame"] == 0].set_index(KEYS + ["nfl_id"])["position"]
    for name, (a, b) in (("first", FIRST_STEPS), ("commit", COMMIT)):
        pa, pb = _at(d, f(a), "a"), _at(d, f(b), "b")
        both = pa.join(pb, how="inner")
        out[f"{name}_v_depth"] = (both["b_depth"] - both["a_depth"]) / (b - a)
        out[f"{name}_v_lat"] = (both["b_lat"] - both["a_lat"]) / (b - a)

    start = d[d["rel_frame"] == 0].set_index(KEYS + ["nfl_id"])[["x", "y"]]
    moved = d.join(start, on=KEYS + ["nfl_id"], rsuffix="_0")
    moved = moved[np.hypot(moved["x"] - moved["x_0"], moved["y"] - moved["y_0"]) >= FIRST_MOVE_YD]
    out["first_move_t"] = moved.groupby(KEYS + ["nfl_id"])["rel_frame"].min() * step

    last = d.groupby(KEYS)["rel_frame"].max().rename("last")
    cap = last.clip(upper=f(LANDMARK_CAP_S))
    if releases is not None:
        rel = releases.set_index(KEYS)["release_rel_frame"].reindex(cap.index)
        cap = cap.where(rel.isna() | (rel >= cap), rel).astype(int)
    land = d.join(cap.rename("landmark_rf"), on=KEYS)
    land = land[land["rel_frame"] == land["landmark_rf"]].set_index(KEYS + ["nfl_id"])
    out["landmark_t"] = land["rel_frame"] * step
    out["landmark_depth"] = land["depth"]
    out["landmark_lat"] = land["lat"]
    out["landmark_o"] = land.get("o", np.nan)
    return out[PHASE_COLUMNS].reset_index()
