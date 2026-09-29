"""Defender roles at the snap from geometry alone.

Film tracks carry only `side` (no roster position), so every role here comes from
where a defender stands relative to the line of scrimmage and the ball: the same
code runs on tracking data and on film. Roster `position` is never read; it only
validates the rules (`scripts/check_roles.py`).

Rules, first match wins (depth = yards past the LOS, offset = |y - center|):

  box     in the tackle box: depth < BOX_ROLE_DEPTH and offset <= BOX_HALFWIDTH,
          or on the line (depth < LINE_DEPTH) out to EDGE_HALFWIDTH - an edge
          rusher is part of the front, not an overhang.
  safety  depth >= DEEP_DEPTH and either not a widest defender, or a widest one
          still within WIDE_SAFETY_OFFSET of the ball (a crop, or no receiver split
          out that side): no split puts a deep corner that tight.
  corner  the widest defender on each side (shell.deep_defenders' `widest`).
  apex    everyone else: overhangs, slot defenders, walked-down safeties.

The mike is the off-ball box defender (depth >= LINE_DEPTH) nearest the ball
laterally, at most one per play. Linemen are excluded so the nose tackle on film
(which tracking data mostly lacks) cannot take the job.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from nfl_coverage.analysis.shell import DEEP_DEPTH, _formation_center, deep_defenders
from nfl_coverage.features.alignment import KEYS

Role = Literal["corner", "safety", "box", "apex"]
ROLES: tuple[Role, ...] = ("corner", "safety", "box", "apex")

# Off-ball linebackers sit 5-7 yd deep on these pass plays (median ~5.5 yd): the 5 yd
# box-count line (shell.BOX_DEPTH) drops a third of them, 6 yd a quarter. Inside the
# box width, 7-8 yd is still ~2.5 linebackers per safety on tuning weeks; 8-9 yd is not.
BOX_ROLE_DEPTH = 8.0
BOX_HALFWIDTH = 5.0  # the tackle box, measured off the ball
LINE_DEPTH = 3.0  # linemen and edge rushers align within ~2.5 yd of the offense's LOS
EDGE_HALFWIDTH = 8.0  # a 9-technique / wide edge off a tight end
WIDE_SAFETY_OFFSET = 8.0

ROLE_COLUMNS = ["role", "depth", "lateral", "dist_to_ball_lat", "is_mike"]


def _center_by_play(frames: pd.DataFrame, center: float | None) -> pd.Series | float:
    if center is not None:
        return center
    snap = frames[(frames["rel_frame"] == 0) & (frames["side"] == "Offense")]
    return _formation_center(snap).rename("center")


def snap_roles(
    frames: pd.DataFrame, los: float | None = None, center: float | None = None
) -> pd.DataFrame:
    """One row per defender at the snap: KEYS + nfl_id + ROLE_COLUMNS.

    `lateral` is y - center (signed), `dist_to_ball_lat` its absolute value; the
    formation center stands in for the ball, which canonical frames do not carry.
    `los`/`center` override the per-play estimates (film), as in `shell_state`.
    """
    d = deep_defenders(frames, 0, DEEP_DEPTH, los=los, center=center)
    c = _center_by_play(frames, center)
    if isinstance(c, pd.Series):
        d = d.join(c, on=KEYS)
    else:
        d["center"] = c
    d["lateral"] = d["y"] - d["center"]
    d["dist_to_ball_lat"] = d["offset"]

    depth, off = d["depth"], d["offset"]
    in_box = ((depth < BOX_ROLE_DEPTH) & (off <= BOX_HALFWIDTH)) | (
        (depth < LINE_DEPTH) & (off <= EDGE_HALFWIDTH)
    )
    safety = d["deep"] & (~d["widest"] | (off <= WIDE_SAFETY_OFFSET))
    d["role"] = np.select(
        [in_box, safety, d["widest"]], ["box", "safety", "corner"], default="apex"
    )

    d["is_mike"] = False
    off_ball = d[(d["role"] == "box") & (depth >= LINE_DEPTH)]
    if len(off_ball):
        d.loc[off_ball.groupby(KEYS)["offset"].idxmin(), "is_mike"] = True
    return d[KEYS + ["nfl_id"] + ROLE_COLUMNS].reset_index(drop=True)
