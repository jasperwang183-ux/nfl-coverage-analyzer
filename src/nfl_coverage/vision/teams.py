"""Offense/defense assignment from jersey colors — no uniform database needed.

Principle (angle-invariant): every frame of a game shows exactly two team colors,
so cluster each detection's torso color into two groups per image; detections that
match neither cluster (referees, chain crew) are outliers. Geometry is used once,
at the cluster level: the color group with the majority of its members on the
offensive side of the LOS is the offense. This avoids the per-player LOS
knife-edge that homography error can flip.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

# Lighting (Lab L) swings ~100 units between a shadowed and a sunlit jersey of the
# same team, while team identity lives mostly in chroma (a/b) — so L is down-
# weighted before clustering. OUTLIER_DIST is in this scaled space: intra-team
# spread stays under ~20, a black/white-striped referee lands 25+ from either
# team center.
L_WEIGHT = 0.3
OUTLIER_DIST = 25.0


def torso_colors(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Median Lab color of each detection's jersey region, field pixels masked out.

    image is BGR (as from cv2.imread); boxes are (x1, y1, x2, y2) pixels. The crop
    is the central-upper part of the box (torso), skipping grass and legs.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    out = np.zeros((len(boxes), 3), dtype=float)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        w, h = x2 - x1, y2 - y1
        cx1, cx2 = int(x1 + 0.25 * w), int(x1 + 0.75 * w)
        cy1, cy2 = int(y1 + 0.15 * h), int(y1 + 0.55 * h)
        crop_hsv = hsv[cy1:cy2, cx1:cx2].reshape(-1, 3)
        crop_lab = lab[cy1:cy2, cx1:cx2].reshape(-1, 3)
        if len(crop_lab) == 0:  # degenerate box
            crop_hsv = hsv[int(y1) : int(y2), int(x1) : int(x2)].reshape(-1, 3)
            crop_lab = lab[int(y1) : int(y2), int(x1) : int(x2)].reshape(-1, 3)
        grass = (
            (crop_hsv[:, 0] >= 35)
            & (crop_hsv[:, 0] <= 90)
            & (crop_hsv[:, 1] > 50)
            & (crop_hsv[:, 2] > 40)
        )
        px = crop_lab[~grass] if (~grass).sum() >= 20 else crop_lab
        out[i] = np.median(px, axis=0)
    return out


def cluster_teams(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split detections into two team-color clusters and flag non-player outliers.

    Returns (labels, is_player): labels in {0, 1} (meaningless for outliers),
    is_player False where the color matches neither team (referee, etc.).
    """
    if len(colors) < 4:
        raise ValueError(f"need at least 4 detections to cluster teams, got {len(colors)}")
    scaled = np.asarray(colors, dtype=float) * np.array([L_WEIGHT, 1.0, 1.0])
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(scaled)
    labels = km.labels_
    dist = np.linalg.norm(scaled - km.cluster_centers_[labels], axis=1)
    is_player = dist <= OUTLIER_DIST
    # a cluster reduced to a single member is itself suspect, but with two real
    # teams on the field that cannot happen — keep the check simple for now
    return labels, is_player


def sides_from_clusters(
    labels: np.ndarray, field_x: np.ndarray, los: float, offense: str, band: float = 2.5
) -> np.ndarray:
    """Per-player 'Offense'/'Defense': geometry where it is certain, color in between.

    Pre-snap, formation rules make geometry decisive away from the ball — nobody
    lines up yards past the LOS on the wrong side — so players more than `band`
    yards from the LOS take their geometric side outright (this also overrides a
    color misread, e.g. a washed-out deep safety clustering with white jerseys).
    Inside the band, where homography error can flip the geometric split, a
    player takes the side of its color cluster, with each cluster's side voted
    by its geometrically-certain members.

    field_x is each detection's field x in the same (unflipped) coordinates as
    `los`; `offense` is 'left' or 'right' — which side of the LOS the offense
    occupies.
    """
    if offense not in ("left", "right"):
        raise ValueError(f"offense must be 'left' or 'right', got {offense!r}")
    on_off_side = field_x <= los if offense == "left" else field_x >= los
    sides = np.where(on_off_side, "Offense", "Defense").astype(object)
    certain = np.abs(field_x - los) >= band
    for k in (0, 1):
        members = labels == k
        voters = members & certain if (members & certain).any() else members
        cluster_side = "Offense" if on_off_side[voters].mean() > 0.5 else "Defense"
        sides[members & ~certain] = cluster_side
    return sides.astype(str)
