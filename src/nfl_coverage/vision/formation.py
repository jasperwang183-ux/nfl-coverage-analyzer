"""Line of scrimmage, offense side and ball position from the bodies at the snap (rung 3).

Film gives field positions of every detected body and two jersey-color clusters,
but not which team has the ball, where the ball is, or which painted yard it sits
on. Until now the analyst typed all three (`--los`, `--offense`, `--ball-y`).
This module infers them from the snap frame, each with a confidence and reasons,
so the page can show them for a one-glance confirm instead of asking.

- **Offense side: the defense is the deeper side.** Safeties align 10-15 yd off
  the ball; the offense's deepest player (shotgun QB, a back) sits ~4.5 yd back
  and never past ~9. Scored on Big Data Bowl week 1 with every other play
  mirrored: right on 99.3% (2018) and 98.4% (2023) of plays; requiring a 3 yd
  margin: 95.8% / 95.7% right, 0.0% / 0.4% wrong, the rest abstain. Officials
  (the referee lines up 10-15 yd behind the offense) must be excluded first -
  `is_player` carries the striped-shirt and color-outlier flags.
- **LOS: the gap between the two teams' front rows.** By rule the neutral zone
  separates the lines, so along the offense direction the defense's line
  bodies come first, then the offense's. The split with the fewest jersey-color
  violations (ties: the widest gap) is the neutral zone; the LOS is its middle
  and its edges are an honest range. The densest band of bodies - the rough
  estimate the snap detector uses - is biased toward whichever line has more
  bodies detected (1.4 yd on clip 2, where the lines merged into five boxes).
- **Ball y: the shotgun QB, else the center.** In the shotgun the QB lines up
  behind the center: |QB y - ball y| is 0.14 yd at the median and 0.41 at the
  95th percentile (Big Data Bowl 2018 week 1). Under center, the QB is inside
  the line cluster, and the center (the offensive line body nearest the
  middle of both lines) is the best left - ~1 yd off, flagged low.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nfl_coverage.vision.snap import LINE_DEPTH, LINE_HALFWIDTH, estimate_los

DEPTH_MARGIN_HIGH = 3.0  # yd between the two sides' deepest bodies: high confidence
DEPTH_MARGIN_MIN = 1.0  # below this the side is not called
BAND_DEPTH = LINE_DEPTH + 0.5  # line bodies searched this far either side of the rough LOS
MAX_NEUTRAL_ZONE = 3.0  # a wider gap between the fronts means a line was not detected
TEAM_MAJORITY = 0.7  # share of the deep defenders that must share one color cluster
DEEP_DEFENDER = 7.0  # yd off the ball on the defense side: surely a defender
QB_DEPTH = (3.0, 8.5)  # shotgun/pistol QB depth behind the LOS (yd)
QB_LATERAL = 2.0  # max |QB y - center y| (yd)


@dataclass
class Formation:
    los: float  # field x of the line of scrimmage (raw, unflipped coordinates)
    los_range: tuple[float, float]  # the neutral zone: defense front .. offense front
    los_confidence: str  # high / medium / low
    offense: str | None  # 'left' / 'right': the side of the LOS the offense occupies
    offense_confidence: str
    ball_y: float | None  # field y of the ball (raw)
    ball_y_confidence: str
    reasons: list[str] = field(default_factory=list)
    # index (into the input arrays) of a lone deep body the side rule took for an
    # official; the pipeline sets it aside before sides are assigned
    suspect_official: int | None = None


def _kth_deepest(depths: np.ndarray, k: int) -> float:
    d = np.sort(depths)[::-1]
    return float(d[k - 1]) if len(d) >= k else 0.0


def offense_from_depth(x: np.ndarray, los: float) -> tuple[str | None, str, str]:
    """(offense side, confidence, reason) from how deep each side goes.

    The defense is the deeper side. The deepest body per side decides when the
    second-deepest agrees (or is too close to say); when they disagree, one lone
    body is deep on a side - on film that is usually an official the striped-shirt
    check missed (the referee stands 10-15 yd behind the offense; on the
    Dolphins-Chargers clip one stood in the end zone) - and the second-deepest
    decides, at low confidence. On Big Data Bowl week 1 (every other play
    mirrored) this is 98.6% / 97.9% right on clean data (2018 / 2023) against
    99.1% / 98.2% for the deepest alone, and 76% / 70% right with a referee
    planted 14 yd behind the offense, where the deepest alone drops to 52% / 28%.
    """
    dep = x - los
    right, left = dep[dep > 0], -dep[dep < 0]
    m1 = _kth_deepest(right, 1) - _kth_deepest(left, 1)  # > 0: the right side is deeper
    m2 = _kth_deepest(right, 2) - _kth_deepest(left, 2)
    depths = (
        f"deepest {_kth_deepest(left, 1):.1f} / {_kth_deepest(right, 1):.1f} yd, "
        f"second-deepest {_kth_deepest(left, 2):.1f} / {_kth_deepest(right, 2):.1f} yd "
        "(left / right of the LOS)"
    )
    if np.sign(m1) == np.sign(m2) or abs(m2) < DEPTH_MARGIN_MIN:
        if abs(m1) < DEPTH_MARGIN_MIN:
            return None, "low", f"{depths} -> too close to call"
        defense = "right" if m1 > 0 else "left"
        conf = "high" if abs(m1) >= DEPTH_MARGIN_HIGH else "medium"
        return (
            "left" if defense == "right" else "right",
            conf,
            f"{depths} -> the {defense} side is the defense",
        )
    defense = "right" if m2 > 0 else "left"
    lone = "left" if defense == "right" else "right"
    return (
        lone,
        "low",
        (
            f"{depths} -> one body alone deep on the {lone} (an official?); the "
            f"second-deepest says the {defense} side is the defense - check it"
        ),
    )


def cluster_sides(
    x: np.ndarray, labels: np.ndarray, los: float, offense: str
) -> dict[int, str] | None:
    """Color cluster -> 'Offense'/'Defense', anchored on the deep defenders; None if unclear.

    The bodies at least DEEP_DEFENDER yd off the ball on the defense side are
    the surest defenders on the field; their majority cluster is the defense.
    Clusters are not mapped by where *all* their clear bodies stand: the offense
    has almost none clear of the line (receivers align on it), and in hard sun
    a shaded defender can join the white-jersey cluster (three did on clip 1).
    """
    depth = (los - x) if offense == "right" else (x - los)  # > 0 on the defense side
    deep = depth >= DEEP_DEFENDER
    if deep.sum() < 2:
        return None
    counts = np.bincount(labels[deep].astype(int), minlength=2)
    k_def = int(np.argmax(counts))
    if counts[k_def] / counts.sum() < TEAM_MAJORITY:
        return None
    return {k_def: "Defense", 1 - k_def: "Offense"}


def split_fronts(s: np.ndarray, is_off: np.ndarray) -> tuple[int, float, float]:
    """Neutral zone in offense-positive coordinates: (violations, low edge, high edge).

    `s` = signed distance along the offense direction (defense first, offense
    after); each candidate split between consecutive bodies counts offense bodies
    before it and defense bodies after it. Fewest violations wins; ties go to
    the widest gap.
    """
    order = np.argsort(s)
    s, is_off = s[order], is_off[order]
    best = None
    for i in range(len(s) - 1):
        viol = int(is_off[: i + 1].sum() + (~is_off[i + 1 :]).sum())
        gap = float(s[i + 1] - s[i])
        key = (viol, -gap)
        if best is None or key < best[0]:
            best = (key, float(s[i]), float(s[i + 1]))
    return best[0][0], best[1], best[2]


def infer_formation(
    xy: np.ndarray,
    labels: np.ndarray,
    is_player: np.ndarray,
    los_hint: float | None = None,
    offense_hint: str | None = None,
) -> Formation:
    """LOS, offense side and ball y from the bodies on the field at the snap.

    `xy` are field positions (raw coordinates), `labels` the 2-means jersey-color
    cluster of each, `is_player` False for officials and color outliers. Hints
    (a typed or confirmed value) are used as given and the rest is inferred
    around them.
    """
    xy = np.asarray(xy, dtype=float)
    labels = np.asarray(labels)
    keep = np.asarray(is_player, dtype=bool)
    kept_index = np.flatnonzero(keep)
    x, y, lab = xy[keep, 0], xy[keep, 1], labels[keep]
    suspect = None
    reasons: list[str] = []
    if len(x) < 6:
        raise ValueError(f"need at least 6 players at the snap to read a formation, got {len(x)}")

    rough = los_hint if los_hint is not None else estimate_los(x, LINE_DEPTH)

    # offense side
    if offense_hint is not None:
        offense, off_conf = offense_hint, "given"
        reasons.append(f"offense side given: {offense}")
    else:
        offense, off_conf, why = offense_from_depth(x, rough)
        reasons.append(why)
        if offense is not None and "alone deep" in why:
            # the lone deep body on the offense side: not a player
            dep = (x - rough) if offense == "right" else (rough - x)
            j = int(np.argmax(dep))
            suspect = int(kept_index[j])
            keep_j = np.ones(len(x), dtype=bool)
            keep_j[j] = False
            x, y, lab = x[keep_j], y[keep_j], lab[keep_j]

    # LOS from the two teams' fronts
    los, los_range, los_conf = rough, (rough - 1.0, rough + 1.0), "low"
    near = np.abs(x - rough) < LINE_DEPTH
    cy = float(np.median(y[near])) if near.any() else float(np.median(y))
    band = (np.abs(x - rough) < BAND_DEPTH) & (np.abs(y - cy) < LINE_HALFWIDTH)
    sides = cluster_sides(x, lab, rough, offense) if offense else None
    if los_hint is not None:
        los_conf = "given"
        los_range = (los_hint, los_hint)
        reasons.append(f"LOS given: x={los_hint:.1f}")
    elif sides is None or offense is None:
        reasons.append(
            f"LOS x={rough:.1f} from the densest band of bodies (no team split: "
            + ("offense side unknown)" if offense is None else "jersey colors do not separate)")
        )
    else:
        is_off = np.array([sides[int(k)] == "Offense" for k in lab[band]])
        sign = 1.0 if offense == "right" else -1.0
        s = (x[band] - rough) * sign
        if is_off.any() and (~is_off).any() and band.sum() >= 3:
            viol, lo, hi = split_fronts(s, is_off)
            los = rough + sign * (lo + hi) / 2
            edges = sorted([rough + sign * lo, rough + sign * hi])
            los_range = (edges[0], edges[1])
            width = hi - lo
            los_conf = (
                "high"
                if viol == 0 and width <= MAX_NEUTRAL_ZONE
                else "medium"
                if viol <= 1 and width <= MAX_NEUTRAL_ZONE
                else "low"
            )
            reasons.append(
                f"LOS x={los:.1f}: middle of the neutral zone {edges[0]:.1f}-{edges[1]:.1f} "
                f"between the two lines' front bodies ({int(band.sum())} line bodies, "
                f"{viol} on the wrong side by jersey color; densest band said {rough:.1f})"
            )
        else:
            reasons.append(
                f"LOS x={rough:.1f} from the densest band of bodies (one team missing "
                "from the line)"
            )

    # ball y: shotgun QB behind the center, else the center
    ball_y, ball_conf = None, "low"
    if offense is not None:
        sign = 1.0 if offense == "right" else -1.0
        depth = (x - los) * sign  # > 0 on the offense side
        line_off = band & (depth > -0.5) & (depth < BAND_DEPTH)
        if sides is not None:
            line_off &= np.array([sides[int(k)] == "Offense" for k in lab])
        if line_off.any():
            # the center: the offensive line body nearest the middle of both lines
            ci = np.flatnonzero(line_off)[np.argmin(np.abs(y[line_off] - cy))]
            center_y = float(y[ci])
            qb = (
                (depth >= QB_DEPTH[0])
                & (depth <= QB_DEPTH[1])
                & (np.abs(y - center_y) < QB_LATERAL)
            )
            if qb.any():
                qi = np.flatnonzero(qb)[np.argmin(np.abs(y[qb] - center_y))]
                ball_y, ball_conf = float(y[qi]), "medium"
                reasons.append(
                    f"ball y={ball_y:.1f}: shotgun QB {depth[qi]:.1f} yd behind the LOS, "
                    f"{abs(ball_y - center_y):.1f} yd from the center's y"
                )
            else:
                ball_y, ball_conf = center_y, "low"
                reasons.append(
                    f"ball y={ball_y:.1f}: the center (offensive line body nearest the line's middle); "
                    "no shotgun QB found - under center, ~1 yd uncertain"
                )
    return Formation(
        los=float(los),
        los_range=(float(los_range[0]), float(los_range[1])),
        los_confidence=los_conf,
        offense=offense,
        offense_confidence=off_conf,
        ball_y=ball_y,
        ball_y_confidence=ball_conf if ball_y is not None else "low",
        reasons=reasons,
        suspect_official=suspect,
    )
