"""The front and the rush: who is at the line at the snap, and who drops out of it.

The analyst's definitions (2026-10-26):
  - at the line: within 1 yd of the line of scrimmage before the snap, hand down or a
    standing edge player tight to it. Film places a player by his feet, which sit 0.7-2.4 yd
    off the ball for a down lineman; on six page clips every lineman sat at 2.4 yd or less
    and every linebacker at 4.2 yd or more, so film draws the line at FRONT_LINE_DEPTH.
  - a drop: a defender who backpedals or turns 3+ yd back within the first 0.5-1.5 s.
The front is named the football way, at-the-line count then second-level count ("4-2",
"6-2"). Rushers are the at-the-line players who did not drop plus second-level players who
reached the line; the rush class follows from them. Nothing checks the rush on film or 2023
tracking (2023 tracks no rushers, 2018 no linemen), so the card calls it unsure and asks for
a manual input; the box count is the one number the 2023 labels confirm.

Input is canonical tracks with the linemen kept (`vision.track.canonical_tracks(...,
drop_line=False)`): offense moving right, depth = x - LOS, lateral = y - center.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

FRONT_LINE_DEPTH = 2.5  # yd past the LOS (feet): at the line (the analyst picked 2.5 on film)
SECOND_LEVEL_DEPTH = 7.0  # yd: second-level defenders sit past the line up to here
BOX_HALFWIDTH = 8.0  # yd off the center: the box (film matched 2 of 5 labels, 3 off by one)
BOX_DEPTH = 8.0  # yd past the LOS: the box
DROP_YD = 3.0  # yd back from where he stood at the snap
DROP_BY_S = 1.5  # s after the snap a drop or a rush must show by
RUSH_REACH_DEPTH = FRONT_LINE_DEPTH  # yd: a second-level player this close has come to rush
TEAM_SIZES = (10, 11)  # defenders at the snap a front is read from; else the sides are off


@dataclass
class FrontRead:
    front: str  # "4-2"; "" when unreadable
    n_line: int
    n_second: int
    n_box: int
    rushers: int
    dropped: int  # at-the-line players who dropped
    rush_class: str  # "standard rush" / "blitz" / "simulated pressure" / ""
    unsure: str  # why nothing confirms it (always set on film)
    reasons: list[str] = field(default_factory=list)


def rush_class(n_line: int, rushers: int) -> str:
    """Five or more rushers is a blitz; five or more shown at the line with four or fewer
    rushing is simulated pressure; otherwise a standard rush."""
    if rushers >= 5:
        return "blitz"
    if n_line >= 5:
        return "simulated pressure"
    return "standard rush"


def front_read(tracks: pd.DataFrame, los: float, center: float, step: float = 0.1) -> FrontRead:
    """Read the front at rel_frame 0 and the rush over the first DROP_BY_S seconds."""
    d = tracks[tracks["side"] == "Defense"].copy()
    d["depth"] = d["x"] - los
    d["lat"] = d["y"] - center
    snap = d[d["rel_frame"] == 0].set_index("nfl_id")
    if len(snap) not in TEAM_SIZES:
        return FrontRead(
            "", 0, 0, 0, 0, 0, "",
            f"the defense counts {len(snap)} at the snap, so the sides are off",
        )  # fmt: skip
    inside = snap["lat"].abs() <= BOX_HALFWIDTH
    line = snap[inside & (snap["depth"] <= FRONT_LINE_DEPTH)]
    second = snap[
        inside & (snap["depth"] > FRONT_LINE_DEPTH) & (snap["depth"] <= SECOND_LEVEL_DEPTH)
    ]
    box = snap[inside & (snap["depth"] <= BOX_DEPTH)]

    early = d[(d["rel_frame"] > 0) & (d["rel_frame"] * step <= DROP_BY_S + 1e-9)]
    deepest = early.groupby("nfl_id")["depth"].max()
    shallowest = early.groupby("nfl_id")["depth"].min()
    gain = (deepest - snap["depth"]).reindex(line.index)
    dropped = int((gain >= DROP_YD).sum())
    came = shallowest.reindex(second.index) <= RUSH_REACH_DEPTH
    rushers = len(line) - dropped + int(came.sum())
    reasons = [
        (
            f"{len(line)} at the line (within {FRONT_LINE_DEPTH:g} yd, feet), {len(second)} at "
            f"the second level ({FRONT_LINE_DEPTH:g}-{SECOND_LEVEL_DEPTH:g} yd), {len(box)} in "
            "the box"
        ),
        (
            f"by +{DROP_BY_S:g} s {dropped} of the line dropped {DROP_YD:g}+ yd and "
            f"{int(came.sum())} from the second level came to the line"
        ),
    ]
    return FrontRead(
        front=f"{len(line)}-{len(second)}",
        n_line=len(line),
        n_second=len(second),
        n_box=len(box),
        rushers=rushers,
        dropped=dropped,
        rush_class=rush_class(len(line), rushers),
        unsure="nothing on film checks who rushed; the box count is the one number the "
        "2023 labels confirm",
        reasons=reasons,
    )
