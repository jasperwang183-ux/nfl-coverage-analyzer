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
# spread stays under ~20; officials are rejected by `referee_mask` (stripes),
# other bystanders as color outliers.
L_WEIGHT = 0.1
OUTLIER_DIST = 30.0


def _scale(desc: np.ndarray) -> np.ndarray:
    """Down-weight both L entries of a (N, 6) two-mode torso descriptor."""
    return np.asarray(desc, dtype=float) * np.array([L_WEIGHT, 1.0, 1.0, L_WEIGHT, 1.0, 1.0])


def _torso_crops(image: np.ndarray, boxes: list[tuple[float, float, float, float]]):
    """Per box: (Lab crop (h, w, 3), grass mask (h, w)) of the jersey region."""
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    out = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        rows = slice(int(y1 + 0.15 * h), int(y1 + 0.55 * h))
        cols = slice(int(x1 + 0.25 * w), int(x1 + 0.75 * w))
        crop_hsv, crop_lab = hsv[rows, cols], lab[rows, cols]
        if crop_lab.size == 0:  # degenerate box
            crop_hsv = hsv[int(y1) : int(y2), int(x1) : int(x2)]
            crop_lab = lab[int(y1) : int(y2), int(x1) : int(x2)]
        grass = (
            (crop_hsv[..., 0] >= 35)
            & (crop_hsv[..., 0] <= 90)
            & (crop_hsv[..., 1] > 50)
            & (crop_hsv[..., 2] > 40)
        )
        out.append((crop_lab, grass))
    return out


def _two_modes(px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k=2 clustering of Lab pixels: (centers (2, 3), fractions (2,), labels)."""
    import cv2

    data = px.astype(np.float32)
    if len(data) < 40:
        c = np.median(data, axis=0)
        return np.stack([c, c]), np.array([1.0, 0.0]), np.zeros(len(data), int)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    frac = np.bincount(labels, minlength=2) / len(labels)
    return centers, frac, labels


def torso_modes(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """(N, 2, 3) Lab: the dark and the bright mode of each torso crop, grass masked."""
    out = np.zeros((len(boxes), 2, 3), dtype=float)
    for i, (crop, grass) in enumerate(_torso_crops(image, boxes)):
        px = crop[~grass] if (~grass).sum() >= 20 else crop.reshape(-1, 3)
        centers, _, _ = _two_modes(px)
        out[i] = centers[np.argsort(centers[:, 0])]
    return out


def torso_colors(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """(N, 6) jersey descriptor: dark-mode Lab then bright-mode Lab of the torso crop.

    image is BGR (as from cv2.imread); boxes are (x1, y1, x2, y2) pixels. The crop
    is the central-upper part of the box (torso). In hard sunlight half a torso
    is deep shadow, so a single "jersey color" does not exist: a white jersey is
    (dark gray, white), a royal-blue one (dark blue, yellow numbers). Keeping
    both modes, with lightness almost ignored, is what separates the teams; a
    flat crop simply repeats its one mode.
    """
    return torso_modes(image, boxes).reshape(len(boxes), 6)


MIN_TEAM_SIZE = 3  # a color cluster with fewer members than this is bystanders, not a team


def cluster_teams(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split detections into two team-color clusters and flag non-player outliers.

    Returns (labels, is_player): labels in {0, 1} (meaningless for outliers),
    is_player False where the color matches neither team (chain crew, a
    bystander, a UI element). A 2-means split can be hijacked by one extreme
    outlier (it gets a cluster of its own and both teams share the other), so
    any cluster smaller than MIN_TEAM_SIZE is set aside and the split is redone
    on the rest.
    """
    if len(colors) < 4:
        raise ValueError(f"need at least 4 detections to cluster teams, got {len(colors)}")
    scaled = _scale(colors)
    active = np.ones(len(scaled), dtype=bool)
    for _ in range(3):
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(scaled[active])
        sizes = np.bincount(km.labels_, minlength=2)
        if sizes.min() >= MIN_TEAM_SIZE or active.sum() <= 2 * MIN_TEAM_SIZE:
            break
        small = km.labels_ == int(np.argmin(sizes))
        idx = np.flatnonzero(active)
        active[idx[small]] = False
    centers = km.cluster_centers_
    dist = np.linalg.norm(scaled[:, None, :] - centers[None, :, :], axis=2)
    labels = dist.argmin(axis=1)
    is_player = dist.min(axis=1) <= OUTLIER_DIST
    return labels, is_player


def sides_from_clusters(
    labels: np.ndarray,
    field_x: np.ndarray,
    los: float,
    offense: str,
    band: float = 2.5,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """Per-player 'Offense'/'Defense': geometry where it is certain, color in between.

    Pre-snap, formation rules make geometry decisive away from the ball — nobody
    lines up yards past the LOS on the wrong side — so players more than `band`
    yards from the LOS take their geometric side outright (this also overrides a
    color misread, e.g. a washed-out deep safety clustering with white jerseys).
    Inside the band, where homography error can flip the geometric split, a
    player takes the side whose *certain* players look most like him: with
    `colors` (the per-player descriptors) that is the nearest of the two sides'
    mean descriptors; without, the vote of his color cluster.

    field_x is each detection's field x in the same (unflipped) coordinates as
    `los`; `offense` is 'left' or 'right' — which side of the LOS the offense
    occupies.
    """
    if offense not in ("left", "right"):
        raise ValueError(f"offense must be 'left' or 'right', got {offense!r}")
    on_off_side = field_x <= los if offense == "left" else field_x >= los
    sides = np.where(on_off_side, "Offense", "Defense").astype(object)
    certain = np.abs(field_x - los) >= band
    contested = ~certain
    if colors is not None and (certain & on_off_side).any() and (certain & ~on_off_side).any():
        scaled = _scale(colors)
        off_c = scaled[certain & on_off_side].mean(axis=0)
        def_c = scaled[certain & ~on_off_side].mean(axis=0)
        d_off = np.linalg.norm(scaled[contested] - off_c, axis=1)
        d_def = np.linalg.norm(scaled[contested] - def_c, axis=1)
        sides[contested] = np.where(d_off <= d_def, "Offense", "Defense")
        return sides.astype(str)
    for k in (0, 1):
        members = labels == k
        voters = members & certain if (members & certain).any() else members
        cluster_side = "Offense" if on_off_side[voters].mean() > 0.5 else "Defense"
        sides[members & contested] = cluster_side
    return sides.astype(str)


# Officials: black-and-white vertical stripes. Both torso modes are chroma-neutral
# and the mean brightness of the torso columns alternates several times left to
# right (at broadcast resolution the stripes can blur to a flat gray, which is
# then indistinguishable from a shaded white jersey and is not flagged).
# A shaded white jersey also has two neutral modes but one lit/shaded boundary;
# numbers on a jersey give at most a couple of switches and usually a colored mode.
REF_NEUTRAL_A = 10.0
REF_NEUTRAL_B = 12.0
REF_MIN_CONTRAST = 40.0  # L between the two modes
REF_MIN_SWITCHES = 5  # a real official showed 6 on a 34-px crop; a two-tone jersey 4 on 49 px
REF_HYSTERESIS = 8.0


def _column_switches(crop_L: np.ndarray, grass: np.ndarray, lo: float, hi: float) -> int:
    """Times the per-column mean brightness crosses the mid-level (with hysteresis).

    An official's stripes run vertically, so brightness alternates left to right
    across the torso; a lit/shaded jersey crosses once, numbers a couple of times.
    """
    L = np.where(grass, np.nan, crop_L.astype(float))
    rows = np.nanmean(L, axis=0)
    rows = rows[~np.isnan(rows)]
    if len(rows) < 6:
        return 0
    mid, band = (lo + hi) / 2, REF_HYSTERESIS
    state, switches = None, 0
    for r in rows:
        new = True if r > mid + band else False if r < mid - band else state
        if state is not None and new is not None and new != state:
            switches += 1
        if new is not None:
            state = new
    return switches


def referee_mask(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """True where a detection's torso is black-and-white striped."""
    out = np.zeros(len(boxes), dtype=bool)
    for i, (crop, grass) in enumerate(_torso_crops(image, boxes)):
        if crop.size == 0 or crop.shape[0] < 6:
            continue
        px = crop[~grass] if (~grass).sum() >= 20 else crop.reshape(-1, 3)
        centers, _, _ = _two_modes(px)
        neutral = (np.abs(centers[:, 1] - 128) <= REF_NEUTRAL_A) & (
            np.abs(centers[:, 2] - 128) <= REF_NEUTRAL_B
        )
        l_dark, l_bright = centers[:, 0].min(), centers[:, 0].max()
        if not neutral.all() or l_bright - l_dark < REF_MIN_CONTRAST:
            continue
        g = grass if grass.shape == crop.shape[:2] else np.zeros(crop.shape[:2], bool)
        out[i] = _column_switches(crop[..., 0], g, l_dark, l_bright) >= REF_MIN_SWITCHES
    return out


class TeamColors:
    """The two team-color centers of a play and which side each one plays.

    Learned on the pre-snap frame, where geometry decides sides reliably, and
    applied to later frames where it does not (receivers cross the LOS, corners
    trail them): post-snap, jersey color is the only honest side signal.
    """

    def __init__(self, centers: np.ndarray, sides: tuple[str, str]):
        self.centers = np.asarray(centers, dtype=float)  # (2, 6) scaled two-mode Lab
        self.sides = tuple(sides)

    @classmethod
    def from_frame(
        cls, colors: np.ndarray, labels: np.ndarray, is_player: np.ndarray, sides: np.ndarray
    ) -> TeamColors:
        """Cluster centers + majority side per cluster from one frame's assignment.

        `sides` is per *player* (length is_player.sum()), as `sides_from_clusters`
        returns it; `labels`/`colors` are per detection.
        """
        scaled = _scale(colors)
        player_labels = labels[is_player]
        player_colors = scaled[is_player]
        centers, cluster_sides = [], []
        for k in (0, 1):
            m = player_labels == k
            centers.append(player_colors[m].mean(axis=0) if m.any() else np.full(6, np.nan))
            votes = sides[m]
            is_off = m.any() and (votes == "Offense").mean() > 0.5
            cluster_sides.append("Offense" if is_off else "Defense")
        return cls(np.array(centers), (cluster_sides[0], cluster_sides[1]))

    def assign(self, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sides, is_player) for new detections by nearest team color."""
        scaled = _scale(colors)
        dist = np.linalg.norm(scaled[:, None, :] - self.centers[None, :, :], axis=2)
        dist = np.where(np.isnan(dist), np.inf, dist)
        nearest = dist.argmin(axis=1)
        is_player = dist.min(axis=1) <= OUTLIER_DIST
        sides = np.array([self.sides[k] for k in nearest], dtype=str)
        return sides, is_player
