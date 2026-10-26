"""The coverage diagram: the called coverage's textbook zones, and who went to them.

The post-snap card names a coverage; this module lays that coverage's zones on the field
(measured from the line of scrimmage and the ball), matches each dropping defender to one
zone where he stands at +2.0 s (the landmark the mechanism rules read), and checks whether
he got there. Man coverages draw each underneath defender's receiver instead.

The zones are a template for the call, not measured from the play: a wrong call draws the
wrong picture. The fit line is the check. When the defenders do not land in the zones
drawn, the card says so.

Coordinates are canonical: x = field length (offense moving right), y = field width.
A zone is a rectangle in depth past the LOS (d0..d1) and field y (y0..y1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from nfl_coverage.config import FIELD_WIDTH

LANDMARK_S = 2.0  # s: where each defender is matched to a zone (mechanisms.py reads +2.0 s)
REACH_BY_S = 3.0  # s: a defender reached his zone if he is in it by then
REACH_YD = 1.5  # yd: within this of the zone counts as in it (track noise, zone edges)
RUSH_DEPTH = 1.0  # yd: a defender this close to the LOS at the landmark rushed; no zone
SAME_PLAYER_YD = 1.0  # yd: two defender tracks this close at the landmark are one player
NUMBERS_Y = 12.0  # yd from a sideline: the painted numbers, where flats end and curls begin
MAN_TRAIL_YD = 3.5  # yd: a man defender stays this close to his receiver on average
FIT_GOOD = 0.6  # share of zone defenders in their zones; under it the call is questioned
DEEP = (12.0, 30.0)  # yd past the LOS: the deep zones
UNDER = (0.0, 10.0)  # the flats
HOOK = (4.0, 12.0)  # the curl and hook zones


@dataclass
class Zone:
    name: str
    d0: float
    d1: float
    y0: float
    y1: float
    deep: bool

    def distance(self, depth: float, y: float) -> float:
        """Yards from (depth, y) to the zone; 0 inside it."""
        dd = max(self.d0 - depth, 0.0, depth - self.d1)
        dy = max(self.y0 - y, 0.0, y - self.y1)
        return float(np.hypot(dd, dy))

    def center(self) -> tuple[float, float]:
        return (self.d0 + self.d1) / 2, (self.y0 + self.y1) / 2


@dataclass
class Diagram:
    coverage: str  # the name drawn ("Cover 3 zone"); "" when the call names no coverage
    kind: str  # "zone" / "man" / ""
    zones: list[dict] = field(default_factory=list)  # Zone fields, x0 / x1 in field x
    assigned: list[dict] = field(default_factory=list)  # nfl_id, zone, reached
    matchups: list[dict] = field(default_factory=list)  # nfl_id (defender), receiver
    fit: str = ""  # the card's line: how many got where the call says they go


def _split(n: int, y0: float = 0.0, y1: float = FIELD_WIDTH) -> list[tuple[float, float]]:
    edges = np.linspace(y0, y1, n + 1)
    return [(float(a), float(b)) for a, b in pairwise(edges)]


def _deep(names: list[str], spans: list[tuple[float, float]]) -> list[Zone]:
    return [Zone(n, *DEEP, a, b, True) for n, (a, b) in zip(names, spans)]


def _flats() -> list[Zone]:
    far = FIELD_WIDTH - NUMBERS_Y
    return [Zone("flat", *UNDER, 0.0, NUMBERS_Y, False),
            Zone("flat", *UNDER, far, FIELD_WIDTH, False)]  # fmt: skip


def template(coverage: str, ball: float, quarters_low: bool = True) -> list[Zone]:
    """The zones a coverage drops into, with the underneath split around the ball.

    `quarters_low`: Cover 6 puts its quarters on the low-y side (else the high side);
    `diagram` tries both and keeps the one the defenders fit.
    """
    far = FIELD_WIDTH - NUMBERS_Y
    thirds = _split(3)
    if coverage == "cover3":
        return [*_deep(["deep third", "deep middle third", "deep third"], thirds), *_flats(),
                Zone("curl / hook", *HOOK, NUMBERS_Y, ball, False),
                Zone("hook / curl", *HOOK, ball, far, False)]  # fmt: skip
    if coverage in ("cover2", "tampa2"):
        under = [*_flats(), Zone("curl", *HOOK, NUMBERS_Y, ball - 5, False),
                 Zone("curl", *HOOK, ball + 5, far, False)]  # fmt: skip
        if coverage == "tampa2":  # the pole runner's lane down the middle
            under.append(Zone("deep middle (pole)", HOOK[0], 20.0, ball - 5, ball + 5, True))
        else:
            under.append(Zone("middle hook", *HOOK, ball - 5, ball + 5, False))
        return [*_deep(["deep half", "deep half"], _split(2)), *under]
    if coverage == "cover4":
        return [*_deep(["deep quarter"] * 4, _split(4)),
                Zone("curl / flat", *UNDER, 0.0, ball - 6, False),
                Zone("middle hook", *HOOK, ball - 6, ball + 6, False),
                Zone("curl / flat", *UNDER, ball + 6, FIELD_WIDTH, False)]  # fmt: skip
    if coverage == "cover6":
        half = FIELD_WIDTH / 2
        q_side, h_side = ((0.0, half), (half, FIELD_WIDTH))
        if not quarters_low:
            q_side, h_side = h_side, q_side
        deep = [*_deep(["deep quarter"] * 2, _split(2, *q_side)), *_deep(["deep half"], [h_side])]
        flat_y = (0.0, NUMBERS_Y) if not quarters_low else (far, FIELD_WIDTH)  # cloud corner
        return [*deep, Zone("flat (cloud)", *UNDER, *flat_y, False),
                Zone("curl", *HOOK, NUMBERS_Y, ball, False),
                Zone("hook", *HOOK, ball, far, False)]  # fmt: skip
    if coverage == "cover1":
        return _deep(["deep middle"], [thirds[1]])
    if coverage == "cover2man":
        return _deep(["deep half", "deep half"], _split(2))
    return []  # cover0: all man


KINDS = {"cover0": "man", "cover1": "man", "cover2man": "man"}
NAMES = {"cover0": "Cover 0 man", "cover1": "Cover 1 man", "cover2man": "Cover 2 man",
         "cover2": "Cover 2 zone", "tampa2": "Tampa 2", "cover3": "Cover 3 zone",
         "cover4": "Cover 4 zone", "cover6": "Cover 6 zone"}  # fmt: skip


def coverage_key(call: str) -> str:
    """The template for a post-snap card's call ("Cover 3 zone (...)." -> "cover3")."""
    if call.startswith("Tampa 2"):
        return "tampa2"
    if call.startswith("Cover 2 trap"):
        return "cover2"
    m = re.match(r"Cover (\d) (zone|man)\b", call)
    if not m:
        return ""  # a shell, not a coverage: nothing to draw
    n, mz = m.groups()
    if n == "2":
        return "cover2man" if mz == "man" else "cover2"
    return f"cover{n}" if n in "013468" else ""


def _at(tracks: pd.DataFrame, rel: int, side: str) -> pd.DataFrame:
    rows = tracks[(tracks.rel_frame == rel) & (tracks.side == side)]
    return rows.drop_duplicates("nfl_id").set_index("nfl_id")


def _assign(zones: list[Zone], depth: pd.Series, y: pd.Series) -> tuple[list, float]:
    """One defender per zone, nearest first (ties broken toward the zone's center)."""
    if not zones or depth.empty:
        return [], 0.0
    cost = np.array([[z.distance(d, yy) + 0.05 * np.hypot(d - z.center()[0], yy - z.center()[1])
                      for z in zones] for d, yy in zip(depth, y)])  # fmt: skip
    rows, cols = linear_sum_assignment(cost)
    pairs = [(depth.index[r], c) for r, c in zip(rows, cols)]
    return pairs, float(cost[rows, cols].sum())


def _reached(tracks: pd.DataFrame, nfl_id: int, zone: Zone, los: float, lo: int, hi: int) -> bool:
    p = tracks[(tracks.nfl_id == nfl_id) & tracks.rel_frame.between(lo, hi)]
    return any(zone.distance(x - los, y) <= REACH_YD for x, y in zip(p.x, p.y))


def _matchups(tracks: pd.DataFrame, ids: list, hi: int) -> list[dict]:
    """Each defender's closest receiver over the snap to the landmark, if he stayed on him."""
    window = tracks[tracks.rel_frame.between(0, hi)]
    if not (window.side == "Offense").any():
        return []
    off =window[window.side == "Offense"].pivot_table(index="rel_frame", columns="nfl_id",
                                                       values=["x", "y"])  # fmt: skip
    out = []
    for did in ids:
        d = window[(window.nfl_id == did)].drop_duplicates("rel_frame").set_index("rel_frame")
        best, best_gap = None, MAN_TRAIL_YD
        for oid in off["x"].columns:
            gap = np.hypot(off["x"][oid] - d.x, off["y"][oid] - d.y).dropna()
            if len(gap) >= 5 and gap.mean() < best_gap:
                best, best_gap = oid, float(gap.mean())
        if best is not None:
            out.append({"nfl_id": int(did), "receiver": int(best), "gap": round(best_gap, 1)})
    return out


def _one_per_player(landmark: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Film sometimes gives one player two track ids for a while: keep the longer track of
    any two within SAME_PLAYER_YD of each other at the landmark."""
    length = tracks.groupby("nfl_id").size()
    keep: list = []
    for i in length.reindex(landmark.index).sort_values(ascending=False).index:
        x, y = landmark.loc[i, "x"], landmark.loc[i, "y"]
        if all(np.hypot(x - landmark.loc[k, "x"], y - landmark.loc[k, "y"]) > SAME_PLAYER_YD
               for k in keep):  # fmt: skip
            keep.append(i)
    return landmark.loc[keep]


def diagram(tracks: pd.DataFrame, los: float, ball: float, call: str, step: float = 0.1) -> Diagram:
    """The coverage diagram for a post-snap call on canonical tracks."""
    key = coverage_key(call)
    if not key:
        return Diagram("", "", fit="No coverage was named after the snap, so none is drawn.")
    land, reach = round(LANDMARK_S / step), round(REACH_BY_S / step)
    landmark = _one_per_player(_at(tracks, land, "Defense"), tracks)
    if landmark.empty:
        return Diagram(NAMES[key], KINDS.get(key, "zone"),
                       fit=f"No defender is tracked at +{LANDMARK_S:.1f} s to check the drawing.")  # fmt: skip
    depth = landmark.x - los
    dropped = depth > RUSH_DEPTH
    depth, y = depth[dropped], landmark.y[dropped]

    options = [template(key, ball)] + ([template(key, ball, False)] if key == "cover6" else [])
    zones, pairs = min(((z, *_assign(z, depth, y)) for z in options), key=lambda o: o[2])[:2]

    assigned = [{"nfl_id": int(i), "zone": c,
                 "reached": _reached(tracks, i, zones[c], los, land, reach)} for i, c in pairs]  # fmt: skip
    in_zone = {a["nfl_id"] for a in assigned}
    kind = KINDS.get(key, "zone")
    out = Diagram(NAMES[key], kind, assigned=assigned)
    out.zones = [{"name": z.name, "x0": los + z.d0, "x1": los + z.d1, "y0": z.y0, "y1": z.y1,
                  "deep": z.deep} for z in zones]  # fmt: skip
    counts = {"rushed": int(dropped.size - dropped.sum()), "extra": 0, "under": 0, "routes": 0}
    if kind == "man":
        under = [i for i in depth.index if i not in in_zone]
        out.matchups = _matchups(tracks, under, land)
        receivers = _at(tracks, land, "Offense")
        counts.update(under=len(under), routes=int((receivers.x - los > RUSH_DEPTH).sum()))
    else:
        counts["extra"] = len(depth) - len(assigned)
    out.fit = _fit_line(out, zones, counts)
    return out


def _fit_line(d: Diagram, zones: list[Zone], counts: dict) -> str:
    n, ok = len(d.assigned), sum(a["reached"] for a in d.assigned)
    filled = {a["zone"] for a in d.assigned}
    empty = sorted({z.name for i, z in enumerate(zones) if i not in filled})
    # the deep zones decide the coverage: one a defender was matched to and never reached
    # argues against the call (an empty one is more often a safety out of frame)
    deep_missed = [zones[a["zone"]].name for a in d.assigned
                   if zones[a["zone"]].deep and not a["reached"]]  # fmt: skip
    by = f"by +{REACH_BY_S:.1f} s"
    parts, doubt = [], bool(deep_missed)
    if n == 1:
        z = zones[d.assigned[0]["zone"]].name
        parts.append(f"The {z} defender {'reached' if ok else 'never reached'} his zone {by}.")
    elif n:
        which = f"{d.coverage.removesuffix(' zone')} zones" if d.kind == "zone" else "deep zones"
        parts.append(f"{ok} of {n} defenders reached their {which} {by}.")
        doubt |= ok / n < FIT_GOOD
    if deep_missed and n > 1:
        parts.append(f"Matched but never got there: {', '.join(deep_missed)}.")
    if d.kind == "man" and counts["under"]:
        if counts["routes"] < 2:
            parts.append("Too few receivers are tracked downfield to check the man coverage.")
        else:
            m, u = len(d.matchups), counts["under"]
            parts.append(f"{m} of {u} underneath defenders stayed within {MAN_TRAIL_YD:.1f} yd of "
                         "one receiver.")  # fmt: skip
            doubt |= m / u < FIT_GOOD
    if empty:
        parts.append(f"Nobody tracked in: {', '.join(empty)}.")
    if counts["extra"]:
        parts.append(f"{counts['extra']} more dropped with no zone left to fill.")
    if counts["rushed"]:
        parts.append(f"{counts['rushed']} within a yard of the line at +{LANDMARK_S:.1f} s rushed.")
    if doubt:
        parts.append("The play does not fit the drawing well, so check the call.")
    return " ".join(parts)
