"""How well film tracks keep each player's identity, scored against tracking-data truth.

For a film clip of a play that is also in the Big Data Bowl tracking data, both are
canonical frames on the same snap-aligned 10 Hz grid, but film's absolute yard lines are
arbitrary (auto naming fixes only their parity), so the film frame is first translated
onto the truth at the snap. Then, frame by frame, film defenders are matched to true
defenders (Hungarian, within MATCH_YD). A true player keeps his identity while the film
id matched to him is the one matched at the snap; each change is a switch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MATCH_YD = 3.0


@dataclass(frozen=True)
class PlayerIdentity:
    truth_id: int
    snap_film_id: int | None
    frames: int  # frames the true player was scored on
    kept: int  # frames matched to his snap film id
    switches: int  # changes of the matched film id (unmatched frames are not changes)
    mean_error_yd: float


def _match(film: pd.DataFrame, truth: pd.DataFrame) -> dict[int, int]:
    """truth nfl_id -> film nfl_id for one frame, pairs farther than MATCH_YD dropped."""
    if film.empty or truth.empty:
        return {}
    d = np.hypot(
        truth["x"].to_numpy()[:, None] - film["x"].to_numpy()[None, :],
        truth["y"].to_numpy()[:, None] - film["y"].to_numpy()[None, :],
    )
    rows, cols = linear_sum_assignment(d)
    return {
        int(truth["nfl_id"].iloc[r]): int(film["nfl_id"].iloc[c])
        for r, c in zip(rows, cols, strict=True)
        if d[r, c] <= MATCH_YD
    }


def align_at_snap(film: pd.DataFrame, truth: pd.DataFrame, iters: int = 3) -> pd.DataFrame:
    """Film shifted by the translation that best lays its snap frame on the truth's."""
    f0 = film[film["rel_frame"] == 0]
    t0 = truth[truth["rel_frame"] == 0]
    dx = t0["x"].median() - f0["x"].median()
    dy = t0["y"].median() - f0["y"].median()
    for _ in range(iters):
        moved = f0.assign(x=f0["x"] + dx, y=f0["y"] + dy)
        d = np.hypot(
            t0["x"].to_numpy()[:, None] - moved["x"].to_numpy()[None, :],
            t0["y"].to_numpy()[:, None] - moved["y"].to_numpy()[None, :],
        )
        rows, cols = linear_sum_assignment(d)
        dx += float(np.median(t0["x"].to_numpy()[rows] - moved["x"].to_numpy()[cols]))
        dy += float(np.median(t0["y"].to_numpy()[rows] - moved["y"].to_numpy()[cols]))
    return film.assign(x=film["x"] + dx, y=film["y"] + dy)


def identity_report(
    film: pd.DataFrame, truth: pd.DataFrame, last_rel_frame: int, side: str = "Defense"
) -> list[PlayerIdentity]:
    """Per true player on `side`, identity kept and switches from the snap to `last_rel_frame`.

    `film` and `truth` are canonical frames of one play (x, y, nfl_id, rel_frame, side).
    """
    film = align_at_snap(film, truth)
    f = film[film["side"] == side]
    t = truth[truth["side"] == side]
    snap = _match(f[f["rel_frame"] == 0], t[t["rel_frame"] == 0])
    out = []
    for tid in sorted(t["nfl_id"].unique()):
        frames = kept = switches = 0
        prev, errors = snap.get(tid), []
        for rf in range(last_rel_frame + 1):
            m = _match(f[f["rel_frame"] == rf], t[t["rel_frame"] == rf])
            if tid not in t.loc[t["rel_frame"] == rf, "nfl_id"].to_numpy():
                continue
            frames += 1
            fid = m.get(tid)
            if fid is None:
                continue
            kept += fid == snap.get(tid)
            switches += prev is not None and fid != prev
            prev = fid
            a = f[(f["rel_frame"] == rf) & (f["nfl_id"] == fid)].iloc[0]
            b = t[(t["rel_frame"] == rf) & (t["nfl_id"] == tid)].iloc[0]
            errors.append(float(np.hypot(a["x"] - b["x"], a["y"] - b["y"])))
        out.append(
            PlayerIdentity(
                int(tid),
                snap.get(tid),
                frames,
                int(kept),
                int(switches),
                float(np.mean(errors)) if errors else float("nan"),
            )
        )
    return out
