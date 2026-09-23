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


# Officials: black-and-white stripes blur into a mid-gray, chroma-neutral torso at
# broadcast resolution, which the L down-weighting then confuses with a white
# jersey. White jerseys are far brighter (L > ~190), colored jerseys carry chroma.
REF_MAX_L = 190.0
REF_MAX_CHROMA_A = 7.0
REF_MAX_CHROMA_B = 9.0
REF_MIN_DARK_FRAC = 0.08  # stripe remnants: some clearly dark pixels survive the blur
REF_DARK_L = 70.0


def referee_mask(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """True where a detection's torso is a chroma-neutral mid-gray: a striped official.

    Same torso crop and grass mask as `torso_colors`. A gray torso qualifies when
    it is neither white-bright nor colored, and either carries dark stripe pixels
    or is simply too dark to be a white jersey.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    out = np.zeros(len(boxes), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        w, h = x2 - x1, y2 - y1
        rows = slice(int(y1 + 0.15 * h), int(y1 + 0.55 * h))
        cols = slice(int(x1 + 0.25 * w), int(x1 + 0.75 * w))
        crop_hsv, crop_lab = hsv[rows, cols].reshape(-1, 3), lab[rows, cols].reshape(-1, 3)
        if len(crop_lab) < 20:
            continue
        grass = (
            (crop_hsv[:, 0] >= 35)
            & (crop_hsv[:, 0] <= 90)
            & (crop_hsv[:, 1] > 50)
            & (crop_hsv[:, 2] > 40)
        )
        px = crop_lab[~grass] if (~grass).sum() >= 20 else crop_lab
        L, a, b = np.median(px, axis=0)
        neutral = abs(a - 128) <= REF_MAX_CHROMA_A and abs(b - 128) <= REF_MAX_CHROMA_B
        dark_frac = (px[:, 0] < REF_DARK_L).mean()
        out[i] = neutral and L < REF_MAX_L and (dark_frac >= REF_MIN_DARK_FRAC or L < 150)
    return out


class TeamColors:
    """The two team-color centers of a play and which side each one plays.

    Learned on the pre-snap frame, where geometry decides sides reliably, and
    applied to later frames where it does not (receivers cross the LOS, corners
    trail them): post-snap, jersey color is the only honest side signal.
    """

    def __init__(self, centers: np.ndarray, sides: tuple[str, str]):
        self.centers = np.asarray(centers, dtype=float)  # (2, 3) scaled Lab
        self.sides = tuple(sides)

    @classmethod
    def from_frame(
        cls, colors: np.ndarray, labels: np.ndarray, is_player: np.ndarray, sides: np.ndarray
    ) -> TeamColors:
        """Cluster centers + majority side per cluster from one frame's assignment.

        `sides` is per *player* (length is_player.sum()), as `sides_from_clusters`
        returns it; `labels`/`colors` are per detection.
        """
        scaled = np.asarray(colors, dtype=float) * np.array([L_WEIGHT, 1.0, 1.0])
        player_labels = labels[is_player]
        player_colors = scaled[is_player]
        centers, cluster_sides = [], []
        for k in (0, 1):
            m = player_labels == k
            centers.append(player_colors[m].mean(axis=0) if m.any() else np.full(3, np.nan))
            votes = sides[m]
            is_off = m.any() and (votes == "Offense").mean() > 0.5
            cluster_sides.append("Offense" if is_off else "Defense")
        return cls(np.array(centers), (cluster_sides[0], cluster_sides[1]))

    def assign(self, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sides, is_player) for new detections by nearest team color."""
        scaled = np.asarray(colors, dtype=float) * np.array([L_WEIGHT, 1.0, 1.0])
        dist = np.linalg.norm(scaled[:, None, :] - self.centers[None, :, :], axis=2)
        dist = np.where(np.isnan(dist), np.inf, dist)
        nearest = dist.argmin(axis=1)
        is_player = dist.min(axis=1) <= OUTLIER_DIST
        sides = np.array([self.sides[k] for k in nearest], dtype=str)
        return sides, is_player
