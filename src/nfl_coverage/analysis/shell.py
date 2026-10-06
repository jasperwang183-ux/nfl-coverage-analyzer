"""Rule-based shell reader: what a quarterback sees in one frame.

The coverage shell is the count of deep defenders and whether the middle of the
field is open (nobody deep in the middle: Cover 2 / 4 / 6 looks) or closed (a
post safety: Cover 1 / 3 looks), plus corner depth on each side and the box
count. It is deterministic, interpretable, and needs only one frame, so the
same reader serves tracking data, a still, and (later) a clip.

Read at the snap it is the "shown" shell; read again ~1.5 s later (with a
deeper threshold, because every zone defender has dropped by then) it is the
shell actually played. A change between the two is a rotation disguise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_coverage.config import FIELD_WIDTH
from nfl_coverage.features.alignment import KEYS, snap_los

DEEP_DEPTH = 9.0  # yards beyond the LOS that reads as a deep (shell) defender at the snap
MOF_HALFWIDTH = 3.5  # a post player sits within this of the formation center
MOF_PARTNER_MIN = 8.0  # ...and has no deep partner closer than this to the center (split safeties)
MOF_SINGLE_HALFWIDTH = 8.0  # a lone deep safety shaded this far off the ball is still single-high
BOX_DEPTH = 5.0
PRESS_DIST = 1.5

SHELL_COLUMNS = [
    "n_deep",
    "mof_closed",
    "n_box",
    "cb_low_depth",
    "cb_high_depth",
    "cb_low_press",
    "cb_high_press",
    "shell",
]


def shell_label(n_deep: int, mof_closed: bool) -> str:
    """Human shell name, e.g. '2-high MOF-open' (Cover 2/4/6 look), '1-high MOF-closed'."""
    if n_deep == 0:
        return "0-high"
    if n_deep >= 4:
        return "4-deep"  # quarters: no post player to speak of, the middle is shared
    deep = f"{n_deep}-high" if n_deep <= 2 else "3-deep"
    return f"{deep} {'MOF-closed' if mof_closed else 'MOF-open'}"


def mof_is_closed(deep_offsets: np.ndarray) -> bool:
    """Middle of the field closed: one deep player over the middle with no split partner.

    `deep_offsets` are |y - center| for the deep defenders. Two-high safeties on
    the hashes both sit a few yards off the center, so "someone within the band"
    is not enough: the nearest must be inside MOF_HALFWIDTH and the next-nearest
    outside MOF_PARTNER_MIN (a corner in a deep third, or nobody).
    """
    if len(deep_offsets) == 0:
        return False
    o = np.sort(np.abs(deep_offsets))
    eps = 0.05  # a player measured exactly on the band edge is on the band, not outside it
    if len(o) == 1:  # a lone deep player is the post player even when shaded off the ball
        return bool(o[0] <= MOF_SINGLE_HALFWIDTH + eps)
    return bool(o[0] <= MOF_HALFWIDTH + eps and o[1] >= MOF_PARTNER_MIN - eps)


def _formation_center(offense_snap: pd.DataFrame) -> pd.Series:
    """y the defense is centered on: the QB at the snap, else the field middle."""
    qb = offense_snap[offense_snap["position"] == "QB"].groupby(KEYS)["y"].first()
    plays = offense_snap.groupby(KEYS).size().index
    return qb.reindex(plays).fillna(FIELD_WIDTH / 2)


def shell_state(
    frames: pd.DataFrame,
    rel_frame: int = 0,
    deep_depth: float = DEEP_DEPTH,
    los: float | None = None,
    center: float | None = None,
    ball_spots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per play: the shell as seen at `rel_frame`.

    The LOS and formation center are taken from the snap frame (rel_frame 0),
    so post-snap reads measure the defense against the same line the offense
    started on. A clip has no linked snap frame, so `los`/`center` can be given
    explicitly (canonical coords). Plays missing the requested frame are dropped.
    """
    defense = deep_defenders(frames, rel_frame, deep_depth, los, center, ball_spots)
    at = frames[frames["rel_frame"] == rel_frame]
    off_at = at[(at["side"] == "Offense") & (at["position"] != "QB")]
    defense["press"] = _near_offense(defense, off_at)

    rows = []
    for key, d in defense.groupby(KEYS):
        low = d.loc[d["y"].idxmin()]
        high = d.loc[d["y"].idxmax()]
        n_deep = int(d["deep"].sum())
        mof_closed = mof_is_closed(d.loc[d["deep"], "offset"].to_numpy())
        rows.append(
            {
                **dict(zip(KEYS, key)),
                "n_deep": n_deep,
                "mof_closed": mof_closed,
                "n_box": int((d["depth"] < BOX_DEPTH).sum()),
                "cb_low_depth": float(low["depth"]),
                "cb_high_depth": float(high["depth"]),
                "cb_low_press": bool(low["press"]),
                "cb_high_press": bool(high["press"]),
                "shell": shell_label(n_deep, mof_closed),
            }
        )
    return pd.DataFrame(rows, columns=KEYS + SHELL_COLUMNS)


def _near_offense(defense: pd.DataFrame, offense: pd.DataFrame) -> pd.Series:
    """Per defender: is any offensive player (same play) within PRESS_DIST yards?"""
    out = pd.Series(False, index=defense.index)
    off_groups = {k: g[["x", "y"]].to_numpy() for k, g in offense.groupby(KEYS)}
    for key, d in defense.groupby(KEYS):
        o = off_groups.get(key)
        if o is None or len(o) == 0:
            continue
        dxy = d[["x", "y"]].to_numpy()[:, None, :] - o[None, :, :]
        out.loc[d.index] = (np.hypot(dxy[..., 0], dxy[..., 1]).min(axis=1) < PRESS_DIST).tolist()
    return out


def deep_defenders(
    frames: pd.DataFrame,
    rel_frame: int = 0,
    deep_depth: float = DEEP_DEPTH,
    los: float | None = None,
    center: float | None = None,
    ball_spots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-defender geometry at `rel_frame` (LOS and center from the snap).

    Without `los`, the LOS is the ball spot from `ball_spots` when given, else the
    stand-in (`features.alignment.snap_los`).

    Columns: KEYS + nfl_id, x, y, depth, offset (|y - center|), deep, widest
    (the outermost defender on each side — the corners, whoever is playing
    them). Fresh integer index: concatenated weeks carry duplicate labels.
    """
    at = frames[(frames["rel_frame"] == rel_frame) & (frames["side"] == "Defense")]
    if los is None or center is None:
        snap = frames[frames["rel_frame"] == 0]
        offense_snap = snap[snap["side"] == "Offense"]
        los_s = snap_los(offense_snap, ball_spots)
        center_s = _formation_center(offense_snap).rename("center")
        d = at.join(los_s, on=KEYS).join(center_s, on=KEYS)
        if los is not None:
            d["los"] = los
        if center is not None:
            d["center"] = center
    else:
        d = at.assign(los=los, center=center)
    d = d.dropna(subset=["los"]).reset_index(drop=True)
    d["depth"] = d["x"] - d["los"]
    d["offset"] = (d["y"] - d["center"]).abs()
    d["deep"] = d["depth"] >= deep_depth
    g = d.groupby(KEYS)["y"]
    d["widest"] = (d["y"] == g.transform("min")) | (d["y"] == g.transform("max"))
    return d[KEYS + ["nfl_id", "x", "y", "depth", "offset", "deep", "widest"]]
