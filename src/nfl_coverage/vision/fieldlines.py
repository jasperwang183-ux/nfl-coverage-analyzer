"""Automatic field calibration from painted markings (all-22 / coaches film).

Clicking landmarks is slow and imprecise; the markings are not. On a sideline
all-22 frame the yard lines are long white near-vertical lines, the hash marks
two rows of short ticks, the sidelines long near-horizontal bands, and the yard
numbers two rows of digits at known distances from the sidelines. Everything
lives on the ground plane at known field coordinates, so their intersections are
calibration landmarks:

    yard line i  x  {far hash 29.7, near hash 23.6, far sideline 53.3, near 0,
                     far number edges 41.3 / 39.3, near number edges 14 / 12}

The analyst supplies only what a machine cannot know from one frame: which
painted yard each detected line is (left to right) and which sideline is at
the top. Sub-yard accuracy follows from fitting each line on hundreds of pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np

from nfl_coverage.config import FIELD_WIDTH
from nfl_coverage.vision.homography import FieldCalibration

HASH_FAR = 29.7  # y of the far hash row (23.6 yd from the far sideline)
HASH_NEAR = 23.6
NUMBER_ROWS_FAR = (41.3, 39.3)  # top / bottom edge of the far numbers
NUMBER_ROWS_NEAR = (14.0, 12.0)


def white_mask(
    image: np.ndarray, roi: tuple[int, int, int, int] | None = None, sat_max=60, val_min=170
) -> np.ndarray:
    """Binary mask of white paint, restricted to `roi` = (x1, y1, x2, y2) if given."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < sat_max) & (hsv[..., 2] > val_min)).astype(np.uint8) * 255
    if roi is not None:
        x1, y1, x2, y2 = roi
        keep = np.zeros_like(mask)
        keep[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        mask = keep
    return mask


def grass_mask(image: np.ndarray) -> np.ndarray:
    """Green turf (including turf in shadow): hue 28-95 on OpenCV's 0-180 scale."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    g = (hsv[..., 0] >= 28) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 35)
    return g.astype(np.uint8) * 255


PAINT_TOPHAT = 40  # paint is this much brighter (V) than its 31-px neighborhood
PAINT_SAT_MAX = 70  # white paint in shade picks up the sky's blue; saturated overlays do not pass
PAINT_NEAR_GRASS = 15  # px: paint lies on turf, player UI and letterbox text do not


def paint_mask(image: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """White field paint, robust to shadow and bounded to the field.

    `white_mask` (a fixed brightness cutoff) loses paint in stadium-roof shadow and
    keeps anything bright and gray: the video player's title, controls and cursor,
    letterbox text. Here paint is (a) locally bright - a white top-hat on V, so a
    line in shade still stands out from the shaded turf around it; (b) unsaturated;
    (c) on the field - inside the largest grass region (closed over its paint and
    players) and within a few pixels of grass. The telestrator's red circles, the
    broadcast's yellow and blue lines and the orange down marker are saturated and
    never pass.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, kernel)
    # locally bright (paint in shade) or plainly bright (a sideline band wider than
    # the top-hat kernel); the field bounds below keep UI text out either way
    bright = ((tophat > PAINT_TOPHAT) | (hsv[..., 2] > 170)) & (hsv[..., 1] < PAINT_SAT_MAX)
    grass = grass_mask(image)
    field = cv2.morphologyEx(
        grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 61))
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(field)
    if n > 1:
        field = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    near = cv2.dilate(
        grass,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PAINT_NEAR_GRASS, PAINT_NEAR_GRASS)),
    )
    mask = (bright & (near > 0) & (field > 0)).astype(np.uint8) * 255
    if roi is not None:
        x1, y1, x2, y2 = roi
        keep = np.zeros_like(mask)
        keep[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        mask = keep
    return mask


MIN_LINE_ANGLE = 20  # deg from horizontal: flatter long lines are sidelines / hash rows


@dataclass
class YardLine:
    """x = a * y + b in pixels, fitted on `n` white pixels."""

    a: float
    b: float
    n: int

    def x_at(self, y: float) -> float:
        return self.a * y + self.b


def detect_yard_lines(
    mask: np.ndarray,
    min_length: int = 250,
    ref_y: float | None = None,
    merge_px: int = 60,
    votes: int = 120,
) -> list[YardLine]:
    """Long white lines crossing the frame top to bottom, left to right, each refit on its pixels.

    Any slant but near-horizontal is a yard-line candidate (sidelines and hash
    rows are the horizontal family): a sideline camera near midfield shows yard
    lines within 30 deg of vertical, but one near a goal line or zoomed in tight
    shows them 35-55 deg off it (all the 2026-09-25 test clips). `min_length`
    (px) and `votes` (Hough accumulator threshold) scale with how zoomed-in the
    frame is: a zoomed-out full-width view has thinner, shorter lines.
    """
    h = mask.shape[0]
    ref_y = h / 2 if ref_y is None else ref_y
    edges = cv2.Canny(mask, 50, 150)
    found = cv2.HoughLinesP(edges, 1, np.pi / 360, votes, minLineLength=min_length, maxLineGap=40)
    if found is None:
        return []
    segs = []
    for x1, y1, x2, y2 in found.reshape(-1, 4).astype(float):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if MIN_LINE_ANGLE < ang < 180 - MIN_LINE_ANGLE and y2 != y1:
            segs.append((x1, y1, x2, y2))

    def x_at(seg, y):
        x1, y1, x2, y2 = seg
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

    segs.sort(key=lambda s: x_at(s, ref_y))
    clusters: list[list] = []
    for s in segs:
        if clusters and abs(x_at(s, ref_y) - x_at(clusters[-1][-1], ref_y)) < merge_px:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    ys_all = np.flatnonzero(mask.any(axis=1))
    lines = []
    for cl in clusters:
        ys = np.arange(ys_all.min(), ys_all.max() + 1)
        guess = np.mean([[x_at(s, y) for y in ys] for s in cl], axis=0)
        slope = abs(np.mean([(s[2] - s[0]) / (s[3] - s[1]) for s in cl]))
        half = int(8 * np.sqrt(1 + slope**2))  # a slanted line is wider along an image row
        pts = []
        for y, xg in zip(ys, guess):
            lo, hi = int(xg) - half, int(xg) + half + 1
            if lo < 0 or hi >= mask.shape[1]:
                continue
            row = mask[y, lo:hi]
            if row.any():
                pts.append((y, lo + np.flatnonzero(row).mean()))
        if len(pts) < 50:
            continue
        pts = np.array(pts)
        a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
        lines.append(YardLine(float(a), float(b), len(pts)))
    return lines


@dataclass
class Row:
    """y = m * x + c in pixels (a hash row or a sideline edge)."""

    m: float
    c: float
    n: int

    def y_at(self, x: float) -> float:
        return self.m * x + self.c


def fit_hash_rows(
    mask: np.ndarray, y_range: tuple[int, int], n_rows: int = 2, seed: int = 0
) -> list[Row]:
    """The hash-mark rows: sequential RANSAC lines through tick-sized white blobs."""
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    ticks = np.array(
        [
            cent[i]
            for i in range(1, n)
            if 8 <= stats[i, 2] <= 45
            and 3 <= stats[i, 3] <= 14
            and y_range[0] < cent[i][1] < y_range[1]
        ]
    )
    rng = np.random.default_rng(seed)
    rows: list[Row] = []
    remaining = ticks
    for _ in range(n_rows):
        if len(remaining) < 4:
            break
        best = None
        for _ in range(3000):
            i, j = rng.choice(len(remaining), 2, replace=False)
            (xa, ya), (xb, yb) = remaining[i], remaining[j]
            if abs(xb - xa) < 300:
                continue
            m = (yb - ya) / (xb - xa)
            if abs(m) > 0.15:
                continue
            c = ya - m * xa
            inl = np.abs(remaining[:, 1] - (m * remaining[:, 0] + c)) < 5
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < 4:
            break
        m, c = np.polyfit(remaining[best, 0], remaining[best, 1], 1)
        rows.append(Row(float(m), float(c), int(best.sum())))
        remaining = remaining[~best]
    rows.sort(key=lambda r: r.c)
    return rows


def fit_sideline(mask: np.ndarray, y_range: tuple[int, int], inner: str) -> Row | None:
    """Field-side edge of a sideline band found within `y_range`.

    `inner` is 'below' for the far sideline (field is below the band in the
    image) or 'above' for the near one.
    """
    y1, y2 = y_range
    pts = []
    for x in range(0, mask.shape[1], 8):
        ys = np.flatnonzero(mask[y1:y2, x])
        if len(ys) >= 3:
            pts.append((x, y1 + (ys.max() if inner == "below" else ys.min())))
    if len(pts) < 30:
        return None
    pts = np.array(pts, float)
    m, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    resid = np.abs(pts[:, 1] - (m * pts[:, 0] + c))
    keep = resid < 6
    if keep.sum() < 30:
        return None
    m, c = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
    return Row(float(m), float(c), int(keep.sum()))


def intersect(line: YardLine, row: Row) -> tuple[float, float]:
    y = (row.m * line.b + row.c) / (1 - row.m * line.a)
    return line.x_at(y), y


@dataclass
class AutoCalibration:
    calibration: FieldCalibration
    pixel: np.ndarray
    field: np.ndarray
    errors: np.ndarray
    lines: list[YardLine]
    notes: list[str] = field(default_factory=list)


def number_edges(
    mask: np.ndarray,
    line: YardLine,
    calib: FieldCalibration,
    rows: tuple[float, float],
    x_field: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
    """Top/bottom edge of the painted number straddling `line`, from a first-pass calibration.

    The first pass predicts where the digit box is; its white rows (with the
    yard line itself blanked) give the true edges, which are then landmarks at
    known field y. Returns [(pixel, field), (pixel, field)] or None.
    """
    inv = np.linalg.inv(calib.matrix)
    probe = np.array([[x_field, rows[0] + 1.0], [x_field, rows[1] - 1.0]], np.float32)
    px = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), inv).reshape(-1, 2)
    y_lo, y_hi = int(min(px[:, 1])), int(max(px[:, 1]))
    if y_lo < 0 or y_hi >= mask.shape[0] or y_hi - y_lo < 20:
        return None
    half_w = int(1.4 * (y_hi - y_lo))  # a two-digit number is ~2.5x wider than tall
    out = []
    ys_white = []
    left_rows = right_rows = 0
    for y in range(y_lo, y_hi):
        cx = int(line.x_at(y))
        lo = max(0, cx - half_w)
        seg = mask[y, lo : cx + half_w].copy()
        c0 = min(len(seg), max(0, (cx - lo) - 14))
        seg[c0 : c0 + 29] = 0  # blank the yard line
        if (seg > 0).sum() > 12:
            ys_white.append(y)
            # a painted number straddles its line, a digit on each side: a box on a
            # 5-yd line can catch the neighboring number, but only on one side
            left_rows += (seg[:c0] > 0).sum() > 4
            right_rows += (seg[c0 + 29 :] > 0).sum() > 4
    if min(left_rows, right_rows) < 5:
        return None
    if len(ys_white) < 10:
        return None
    top, bottom = ys_white[0], ys_white[-1]
    # a digit is 2 yd tall: reject a box that grabbed a player, logo or stray paint
    expected = abs(px[1, 1] - px[0, 1]) * (2.0 / 4.0)  # probe spans 4 yd around the digit
    if not 0.7 * expected <= (bottom - top) <= 1.4 * expected:
        return None
    for y_px, y_field in ((top, rows[0]), (bottom, rows[1])):
        out.append(((line.x_at(y_px), float(y_px)), (x_field, y_field)))
    return out


# ------------------------------------------------------------ line indexing

INDEX_TOL = 0.2  # a spacing within 20% of a whole number of local 5-yd units


def _spacing_units(d: np.ndarray, i: int) -> float:
    """Spacing d[i] in local 5-yd units.

    The unit is the median of up to four neighboring spacings, each first divided
    by its own whole number of units, so a missed line next door (a double
    spacing) does not inflate it, and perspective drift across the frame does
    not either.
    """
    m0 = np.median(d)
    per = d / np.maximum(1, np.round(d / m0))
    nb = np.concatenate([per[max(0, i - 2) : i], per[i + 1 : i + 3]])
    unit = np.median(nb) if len(nb) else per[i]
    return float(d[i] / unit)


def index_lines(lines: list[YardLine], ref_y: float, max_drop: int = 2) -> list[int | None]:
    """5-yd index of each detected line (None = not a yard line), from the spacing pattern.

    Yard lines are 5 yd apart, so their crossings with any image row are evenly
    spaced up to a slow perspective drift. A spacing of about two units is a
    missed line (the index skips one); a spacing that is not near a whole number
    of units means one of its two lines is not a yard line (the edge of the
    end-zone paint on the Dolphins-Chargers wide shot, an overlay), and the one
    whose removal leaves the others closest to whole units is dropped.
    """
    u = np.array([ln.x_at(ref_y) for ln in lines])
    keep = list(range(len(u)))
    for _ in range(max_drop):
        if len(keep) < 3:
            break
        d = np.diff(u[keep])
        bad = []
        for i in range(len(d)):
            r = _spacing_units(d, i)
            if r < 1 - INDEX_TOL or abs(r - round(r)) > INDEX_TOL:
                bad.append((abs(r - round(r)), i))
        if not bad:
            break
        _, i = max(bad)
        best = None
        for cand in (keep[i], keep[i + 1]):
            k2 = [j for j in keep if j != cand]
            d2 = np.diff(u[k2])
            err = sum(
                abs(_spacing_units(d2, t) - max(1, round(_spacing_units(d2, t))))
                for t in range(len(d2))
            )
            if best is None or err < best[0]:
                best = (err, cand)
        keep.remove(best[1])
    offsets: list[int | None] = [None] * len(u)
    if not keep:
        return offsets
    offsets[keep[0]] = 0
    d = np.diff(u[keep])
    k = 0
    for t in range(len(d)):
        k += max(1, round(_spacing_units(d, t)))
        offsets[keep[t + 1]] = k
    return offsets


def field_x_for_lines(
    offsets: list[int | None], named: int, painted: int, going_up: bool
) -> list[float | None]:
    """Field x of every indexed line, from one line's painted yard.

    `named` is the index (0-based, left to right) of the line the person read,
    `painted` its painted yard (0 = goal line, 50 = midfield), `going_up` whether
    the painted numbers increase to its right. Field x increases left to right
    in the image (the calibrations' convention); lines that are not yard lines
    (offset None) stay None.
    """
    if not 0 <= named < len(offsets) or offsets[named] is None:
        raise ValueError(f"line {named + 1} is not one of the indexed yard lines")
    if not 0 <= painted <= 50 or painted % 5:
        raise ValueError(f"painted yard must be 0-50 in steps of 5, got {painted}")
    x_named = 60.0 if painted == 50 else (painted + 10.0 if going_up else 110.0 - painted)
    xs = [None if o is None else x_named + 5.0 * (o - offsets[named]) for o in offsets]
    known = [x for x in xs if x is not None]
    if min(known) < 10 or max(known) > 110:
        raise ValueError(
            "those yard numbers run past a goal line: check the painted number and its direction"
        )
    return xs


# ------------------------------------------------------------ tick rows (1-yd marks)


def tick_blobs(mask: np.ndarray) -> np.ndarray:
    """Centroids of small paint blobs: 1-yd ticks, and plenty of noise the row fit rejects."""
    n, _, st, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros((0, 2))
    area = st[1:, cv2.CC_STAT_AREA]
    big = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT])
    ok = (area >= 10) & (area <= 900) & (big >= 5) & (big <= 70)
    return cent[1:][ok]


def _fit_p1(k: np.ndarray, u: np.ndarray) -> np.ndarray:
    """1-D projective map u = (a k + b) / (c k + 1) through (k, u) by least squares."""
    a = np.column_stack([k, np.ones_like(k), -k * u])
    sol, *_ = np.linalg.lstsq(a, u, rcond=None)
    return sol


def _eval_p1(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    return (p[0] * k + p[1]) / (p[2] * k + 1)


@dataclass
class TickRow:
    row: Row
    matched: list[tuple[tuple[float, float], float]]  # (tick pixel, field x) per matched yard

    @property
    def n(self) -> int:
        return len(self.matched)


TICK_MIN = 4  # matched 1-yd slots for a row to count
TICK_BEYOND = 4  # yd past the outermost indexed lines to look for ticks


def _match_row(
    m: float, c: float, lines: list[YardLine], xs_field: list[float], ticks: np.ndarray
) -> list[tuple[tuple[float, float], float]]:
    """Ticks on the predicted 1-yd slots of the image row y = m x + c."""
    pts = np.array([intersect(ln, Row(m, c, 0)) for ln in lines])
    k = np.asarray(xs_field, float)
    p = _fit_p1(k, pts[:, 0])
    grid = np.arange(np.floor(k.min()) - TICK_BEYOND, np.ceil(k.max()) + TICK_BEYOND + 1)
    x_grid = _eval_p1(p, grid)
    step = np.abs(np.gradient(x_grid))
    slot = np.round(grid) % 5 != 0  # the yard lines themselves are not ticks
    ks, xs, step = grid[slot], x_grid[slot], step[slot]
    ys = m * xs + c
    if len(ticks) == 0 or len(ks) == 0:
        return []
    dx = np.abs(ticks[None, :, 0] - xs[:, None])
    dy = np.abs(ticks[None, :, 1] - ys[:, None])
    ok = (dx < 0.3 * step[:, None]) & (dy < np.maximum(5.0, 0.2 * step[:, None]))
    out = []
    for i in range(len(ks)):
        j = np.flatnonzero(ok[i])
        if len(j):
            jj = j[np.argmin(dx[i, j] + dy[i, j])]
            out.append(((float(ticks[jj, 0]), float(ticks[jj, 1])), float(ks[i])))
    return out


def find_tick_rows(
    mask: np.ndarray,
    lines: list[YardLine],
    xs_field: list[float],
    max_rows: int = 6,
    n_hyp: int = 2500,
    seed: int = 0,
) -> list[TickRow]:
    """Rows of 1-yd ticks, scored by structure, top to bottom.

    A candidate row through two tick blobs crosses the yard lines at known field
    x; a 1-D projective fit along the row then predicts where every 1-yd tick
    between and beyond them must sit. The row's score is how many predicted
    slots hold a blob - random paint (jersey numbers, digits, shadows) almost
    never lines up on that grid. Tick rows run along both hashes and inside both
    sidelines; `hash_pair` picks the hashes. Replaces blob-shape rules, which
    broke on views where the ticks are slanted dashes rather than dots.
    """
    if len(lines) < 2:
        return []
    ticks = tick_blobs(mask)
    h, w = mask.shape[:2]
    ticks = ticks[(ticks[:, 1] > 0.05 * h) & (ticks[:, 1] < 0.97 * h)]
    if len(ticks) < TICK_MIN:
        return []
    rng = np.random.default_rng(seed)
    seen, cands = set(), []
    for _ in range(n_hyp):
        i, j = rng.choice(len(ticks), 2, replace=False)
        (xa, ya), (xb, yb) = ticks[i], ticks[j]
        if abs(xb - xa) < 150:
            continue
        m = (yb - ya) / (xb - xa)
        if abs(m) > 0.4:
            continue
        c = ya - m * xa
        key = (round(m, 2), round(c / 4))
        if key in seen:
            continue
        seen.add(key)
        matched = _match_row(m, c, lines, xs_field, ticks)
        if len(matched) >= TICK_MIN:
            cands.append((len(matched), m, c, matched))
    cands.sort(key=lambda z: -z[0])
    rows: list[TickRow] = []
    for _, m, c, matched in cands:
        xy = np.array([t for t, _ in matched])
        m2, c2 = np.polyfit(xy[:, 0], xy[:, 1], 1)
        matched2 = _match_row(m2, c2, lines, xs_field, ticks)
        if len(matched2) < len(matched):
            m2, c2, matched2 = m, c, matched
        mine = {t for t, _ in matched2}
        if any(
            abs((m2 - r.row.m) * w / 2 + (c2 - r.row.c)) < 12
            or len(mine & {t for t, _ in r.matched}) >= 0.5 * min(len(mine), r.n)
            for r in rows
        ):
            continue  # the same physical row, matched again at a slightly different angle
        rows.append(TickRow(Row(float(m2), float(c2), len(matched2)), matched2))
        if len(rows) == max_rows:
            break
    rows.sort(key=lambda r: r.row.c + r.row.m * w / 2)
    return rows


def hash_pair_candidates(rows: list[TickRow]) -> list[tuple[TickRow, TickRow]]:
    """Adjacent pairs of tick rows (far above near), each a hypothesis for the two hashes.

    Ticks run along both hashes and inside both sidelines, and stray blobs in a
    bench area can line up too, so which adjacent pair is the hashes is decided
    by the fit: the right pair agrees with the sidelines and painted numbers, a
    wrong one does not (`search_calibration`).
    """
    return [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]


# ------------------------------------------------------------ the fit


@dataclass
class FieldMarks:
    """What a frame's paint shows, computed once and shared by every fit attempt."""

    mask: np.ndarray
    lines: list[YardLine]
    offsets: list[int | None]
    ref_y: float


def find_marks(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    min_length: int = 250,
    votes: int = 120,
) -> FieldMarks:
    h, w = image.shape[:2]
    roi = roi or (0, 0, w, h)
    mask = paint_mask(image, roi)
    ref_y = (roi[1] + roi[3]) / 2
    lines = detect_yard_lines(mask, min_length, ref_y=ref_y, votes=votes)
    return FieldMarks(mask, lines, index_lines(lines, ref_y) if lines else [], ref_y)


MIN_FIT_LANDMARKS = 8
MIN_FIT_ROWS = 3  # distinct field y wanted: hashes alone constrain depth only weakly
HASH_ONLY_WARNING = (
    "calibrated on the hash rows alone (no sideline or painted numbers in view): distances "
    "along the field are solid, but depth toward the sidelines is extrapolated and can be "
    "off by a yard or more near the edges of the frame"
)


def auto_calibrate(
    image: np.ndarray,
    yards_left_to_right: list[float | None],
    roi: tuple[int, int, int, int] | None = None,
    far_sideline_y: tuple[int, int] | None = None,
    near_sideline_y: tuple[int, int] | None = None,
    hash_y: tuple[int, int] | None = None,
    numbers: bool = True,
    max_error: float = 1.0,
    min_length: int = 250,
    votes: int = 120,
    marks: FieldMarks | None = None,
    hash_rows: tuple[TickRow, TickRow] | None = None,
) -> AutoCalibration:
    """Fit a FieldCalibration from the painted markings of one all-22 frame.

    yards_left_to_right : field x (0-120, end zones included) of each yard line
        that `detect_yard_lines` finds, in image order; None for a detected line
        that is not a yard line (`index_lines` flags those). Field x increases
        to the right in the image.
    roi : (x1, y1, x2, y2) of the video area (default: the whole frame; the paint
        mask already ignores the player UI and letterbox).
    far_sideline_y / near_sideline_y : image y ranges holding each sideline band
        (None if off-frame). The far sideline (top of the image) is y=53.3.
    hash_y : image y range with both hash rows, used only if the tick-row search
        finds no hash pair.
    marks : precomputed paint analysis (`search_calibration` reuses it).
    hash_rows : the (far, near) hash tick rows; None fits hash rows on tick blobs
        inside `hash_y` instead.

    Landmarks: every yard line x both hash rows; every 1-yd hash tick matched on
    the grid (x at the whole yard, y on its hash) - a landmark per yard instead
    of per 5 yd; the sidelines where asked; the edges of the painted numbers.
    Landmarks off the joint fit by more than `max_error` yd are dropped and the
    fit redone.
    """
    h, w = image.shape[:2]
    roi = roi or (0, 0, w, h)
    marks = marks or find_marks(image, roi, min_length, votes)
    mask, lines = marks.mask, marks.lines
    notes = [f"{len(lines)} yard lines detected"]
    if len(lines) != len(yards_left_to_right):
        raise ValueError(
            f"detected {len(lines)} yard lines but {len(yards_left_to_right)} yard values given; "
            f"lines at x(mid)={[round(ln.x_at(marks.ref_y)) for ln in lines]}"
        )
    used = [(ln, float(x)) for ln, x in zip(lines, yards_left_to_right) if x is not None]
    if len(used) < 2:
        raise ValueError(f"need at least 2 named yard lines, got {len(used)}")
    rows_far_near: list[Row] = []
    tick_px, tick_f = [], []
    if hash_rows is not None:
        for tr, fy in zip(hash_rows, (HASH_FAR, HASH_NEAR)):
            rows_far_near.append(tr.row)
            for p, fx in tr.matched:
                tick_px.append(p)
                tick_f.append((fx, fy))
        notes.append(f"hash rows from 1-yd ticks: {hash_rows[0].n} + {hash_rows[1].n} matched")
    else:
        y1, y2 = roi[1], roi[3]
        hy = hash_y or (y1 + (y2 - y1) // 4, y1 + 3 * (y2 - y1) // 4)
        rows_far_near = fit_hash_rows(mask, hy)
        if len(rows_far_near) != 2:
            raise ValueError(f"no hash rows found (blob fit found {len(rows_far_near)})")
        notes.append(
            f"hash rows from tick blobs in y {hy}: {rows_far_near[0].n} + {rows_far_near[1].n}"
        )

    pixel, fld = [], []
    for ln, fx in used:
        for row, fy in zip(rows_far_near, (HASH_FAR, HASH_NEAR)):
            pixel.append(intersect(ln, row))
            fld.append((fx, fy))
    pixel += tick_px
    fld += tick_f
    for rng_, inner, fy, name in (
        (far_sideline_y, "below", FIELD_WIDTH, "far"),
        (near_sideline_y, "above", 0.0, "near"),
    ):
        if rng_ is None:
            continue
        side = fit_sideline(mask, rng_, inner)
        if side is None:
            notes.append(f"{name} sideline not found")
            continue
        notes.append(f"{name} sideline on {side.n} columns")
        for ln, fx in used:
            pixel.append(intersect(ln, side))
            fld.append((fx, fy))
    pixel, fld = np.array(pixel, float), np.array(fld, float)
    calib = FieldCalibration.from_points(pixel, fld)

    if numbers:
        extra_px, extra_f = [], []
        for ln, fx in used:
            if fx % 10:  # numbers sit on the 10s only
                continue
            for rows_ in (NUMBER_ROWS_FAR, NUMBER_ROWS_NEAR):
                got = number_edges(mask, ln, calib, rows_, fx)
                if got:
                    for p, f in got:
                        extra_px.append(p)
                        extra_f.append(f)
        if extra_px:
            pixel = np.vstack([pixel, extra_px])
            fld = np.vstack([fld, extra_f])
            notes.append(f"{len(extra_px)} number edges")
    # joint fit, drop what disagrees with it, refit
    for _ in range(2):
        trial = FieldCalibration.from_points(pixel, fld)
        err = np.hypot(*(trial.to_field(pixel) - fld).T)
        keep = err < max_error
        if keep.all() or keep.sum() < 4:
            break
        notes.append(f"{int((~keep).sum())} landmarks off the joint fit dropped")
        pixel, fld = pixel[keep], fld[keep]
    calib = FieldCalibration.from_points(pixel, fld)
    errors = np.hypot(*(calib.to_field(pixel) - fld).T)
    return AutoCalibration(calib, pixel, fld, errors, lines, notes)


def draw_landmarks(image: np.ndarray, auto: AutoCalibration) -> np.ndarray:
    out = image.copy()
    for (x, y), (fx, fy), e in zip(auto.pixel, auto.field, auto.errors):
        color = (0, 255, 255) if e < 0.5 else (0, 0, 255)
        cv2.circle(out, (int(x), int(y)), 8, color, 2)
        cv2.putText(
            out,
            f"{fx:.0f},{fy:.1f}",
            (int(x) + 8, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out


# ------------------------------------------------------------ search and confirm

SEARCH_MAX_MEAN = 0.3  # yd: a candidate fit must reproduce its landmarks this well...
SEARCH_MAX_ERROR = 0.8  # ...with no landmark worse than this


def field_x_from_painted(first_painted: int, going_up: bool, n: int) -> list[float]:
    """Field x (0-120, x increasing to the right in the image) of `n` yard lines 5 yd apart.

    `first_painted` is the painted yard of the leftmost detected line (0 = goal
    line, 50 = midfield) and `going_up` whether the painted numbers increase to
    its right. Painted numbers rise to the 50 and fall after it; the field x
    keeps rising, since the calibrations define x as increasing left to right
    in the image (the offense side is inferred separately, per play).
    """
    if not 0 <= first_painted <= 50 or first_painted % 5:
        raise ValueError(f"painted yard must be 0-50 in steps of 5, got {first_painted}")
    x0 = first_painted + 10 if going_up else 110 - first_painted
    if n < 1:
        raise ValueError("no yard lines were found on this frame")
    if first_painted == 50:
        x0 = 60
    xs = [float(x0 + 5 * k) for k in range(n)]
    if xs[-1] > 110:
        raise ValueError(f"{n} lines from the painted {first_painted} run past the goal line")
    return xs


def painted_from_field_x(x: float) -> int:
    """The painted yard number of a yard line at field x."""
    return round(x - 10 if x <= 60 else 110 - x)


@dataclass
class CalibrationSearch:
    best: AutoCalibration
    settings: dict  # far_sideline_y / near_sideline_y / hash_y / min_length / votes used
    n_tried: int
    n_ok: int
    n_rows: int  # distinct field-y rows among the landmarks (sidelines, hashes, numbers)
    warning: str | None = None  # set when the best fit is anchored on the hash rows alone


def search_calibration(
    image: np.ndarray,
    yards_left_to_right: list[float | None],
    roi: tuple[int, int, int, int] | None = None,
    min_length: int = 250,
    votes: int = 120,
    marks: FieldMarks | None = None,
) -> CalibrationSearch:
    """Try the sideline bands a person would tune by hand; keep the best-anchored fit.

    The paint, the yard lines and the 1-yd tick rows are found once; candidates
    vary where (and whether) the far and near sideline bands are looked for, and
    the hash band only when no hash pair came out of the tick rows. A fit counts
    only if it reproduces its landmarks (mean < SEARCH_MAX_MEAN, max <
    SEARCH_MAX_ERROR yd) on at least MIN_FIT_LANDMARKS landmarks (four always fit
    exactly). The most distinct field rows wins, then the most landmarks. A fit
    on the two hash rows alone is accepted when nothing better exists - on a
    tight all-22 zoom it is often all the paint there is - but it carries a
    warning: depth toward the sidelines is extrapolated from 6 yd of rows.
    """
    h, w = image.shape[:2]
    roi = roi or (0, 0, w, h)
    marks = marks or find_marks(image, roi, min_length, votes)
    used = [(ln, x) for ln, x in zip(marks.lines, yards_left_to_right) if x is not None]
    if len(used) < 2:
        raise ValueError(f"need at least 2 named yard lines, got {len(used)}")
    tick_rows = find_tick_rows(marks.mask, [ln for ln, _ in used], [x for _, x in used])
    _, y1, _, y2 = roi
    span = y2 - y1
    # hash hypotheses: each adjacent pair of tick rows, then tick blobs in a band
    hashes: list = [("ticks", pair) for pair in hash_pair_candidates(tick_rows)]
    hashes += [
        ("band", (y1 + int(span * a), y1 + int(span * b)))
        for a, b in ((0.25, 0.75), (0.30, 0.80), (0.20, 0.70), (0.35, 0.85))
    ]
    fars = [None] + [
        (y1 + int(span * a), y1 + int(span * (a + 0.15))) for a in (0.0, 0.08, 0.16, 0.24)
    ]
    nears = [None, (y1 + int(span * 0.85), y2)]
    fits, tried, last_error, underdetermined = [], 0, None, None
    for kind, hy in hashes:
        for fs in fars:
            for ns in nears:
                tried += 1
                try:
                    a = auto_calibrate(
                        image,
                        yards_left_to_right,
                        roi,
                        far_sideline_y=fs,
                        near_sideline_y=ns,
                        hash_y=hy if kind == "band" else None,
                        max_error=SEARCH_MAX_ERROR,
                        marks=marks,
                        hash_rows=hy if kind == "ticks" else None,
                    )
                except ValueError as e:
                    last_error = e
                    continue
                rows = len(set(np.round(a.field[:, 1], 1)))
                if a.errors.mean() >= SEARCH_MAX_MEAN or a.errors.max() >= SEARCH_MAX_ERROR:
                    continue
                if rows < 2 or len(a.pixel) < MIN_FIT_LANDMARKS:
                    underdetermined = (rows, len(a.pixel))
                    continue
                settings = {
                    "hash": kind,
                    "hash_y": hy
                    if kind == "band"
                    else [round(r.row.c + r.row.m * w / 2) for r in hy],
                    "far_sideline_y": fs,
                    "near_sideline_y": ns,
                    "min_length": min_length,
                    "votes": votes,
                }
                fits.append((rows, len(a.pixel), -float(a.errors.mean()), a, settings))
    if not fits:
        if underdetermined is not None:
            raise ValueError(
                f"the markings fit, but only on {underdetermined[1]} landmarks across "
                f"{underdetermined[0]} field row(s) - too few to pin down the camera. Pick a "
                "frame with more of the field in view, or search harder for lines."
            )
        raise ValueError(
            f"no calibration reproduced its landmarks within {SEARCH_MAX_MEAN} yd "
            f"({tried} settings tried; last error: {last_error}). Check the yard numbers."
        )
    rows, _, _, best, settings = max(fits, key=lambda f: f[:3])
    warning = HASH_ONLY_WARNING if rows < MIN_FIT_ROWS else None
    return CalibrationSearch(best, settings, tried, len(fits), rows, warning)


def n_number_edges(a: AutoCalibration) -> int:
    """How many painted-number edges a fit used as landmarks."""
    return sum(int(n.split()[0]) for n in a.notes if n.endswith("number edges"))


SHIFT_EDGE_MARGIN = 4  # number edges the 5-yd-shifted naming must win by to warn


def five_yard_shift_warning(
    image: np.ndarray,
    yards: list[float | None],
    search: CalibrationSearch,
    min_length: int = 250,
    votes: int = 120,
    marks: FieldMarks | None = None,
) -> str | None:
    """Warn when the painted numbers prefer the naming shifted by 5 yd.

    The fit cannot tell a 5-yd mislabel (every line shifted one place still fits);
    the painted numbers can - they sit on the 10s only. Refit with the labels
    shifted by 5 yd and warn when that fit finds clearly more number edges.
    """
    shifted = [None if y is None else y + 5 for y in yards]
    if max(y for y in shifted if y is not None) > 110:
        shifted = [None if y is None else y - 10 for y in shifted]
    try:
        alt = search_calibration(image, shifted, None, min_length, votes, marks)
    except ValueError:
        return None
    mine, theirs = n_number_edges(search.best), n_number_edges(alt.best)
    if theirs < mine + SHIFT_EDGE_MARGIN:
        return None
    tens = [k + 1 for k, y in enumerate(shifted) if y is not None and round(y) % 10 == 0]
    return (
        f"The painted numbers sit on line{'s' if len(tens) > 1 else ''} "
        f"{', '.join(map(str, tens))} (the 10-yard lines), not on the lines your "
        f"naming makes the 10s ({theirs} number edges found that way, {mine} "
        "yours). Check the painted number you entered."
    )


@dataclass
class AutoNaming:
    yards: list[float | None]
    search: CalibrationSearch
    edge_margin: int  # number edges the chosen parity won by (0 = the numbers could not tell)


def auto_name(
    image: np.ndarray,
    marks: FieldMarks,
    min_length: int = 250,
    votes: int = 120,
) -> AutoNaming:
    """Name the yard lines without a person: which lines carry the painted numbers.

    A read is the same wherever the field is shifted along its length (depths are
    measured from the LOS), so the absolute yard is arbitrary: the middle line is
    put near midfield. Only the parity matters - the fit anchors the painted
    numbers on the 10s - so both parities are fitted and the one whose 10s find
    more number edges wins, then more field rows, then the smaller error.
    """
    known = [o for o in marks.offsets if o is not None]
    if len(known) < 2:
        raise ValueError(f"only {len(known)} yard lines found")
    mid = known[len(known) // 2]
    tries = []
    for shift in (0.0, 5.0):
        yards = [None if o is None else 60.0 + shift + 5.0 * (o - mid) for o in marks.offsets]
        try:
            search = search_calibration(image, yards, None, min_length, votes, marks)
        except ValueError:
            continue
        b = search.best
        tries.append(((n_number_edges(b), search.n_rows, -float(b.errors.mean())), yards, search))
    if not tries:
        raise ValueError("no fit on this frame's yard lines")
    tries.sort(key=lambda t: t[0], reverse=True)
    margin = tries[0][0][0] - tries[1][0][0] if len(tries) > 1 else 0
    return AutoNaming(tries[0][1], tries[0][2], margin)


def grid_residual(xs: np.ndarray) -> float:
    """Median distance (yd) of projected yard-line x's from the nearest whole 5 yd."""
    xs = np.asarray(xs, float)
    if not len(xs):
        return float("nan")
    return float(np.median(np.abs(xs - 5.0 * np.round(xs / 5.0))))


def grid_drift(
    image: np.ndarray, calib: FieldCalibration, marks: FieldMarks | None = None
) -> float:
    """How far (yd) the frame's own yard lines sit off the 5-yd grid under `calib`.

    Each line is sampled at three heights so a wrong tilt shows as well as a
    sideways slide. A whole 5-yd slip is invisible here: it still lands on the grid.
    """
    m = marks if marks is not None else find_marks(image)
    h = image.shape[0]
    ys = [m.ref_y - 0.1 * h, m.ref_y, m.ref_y + 0.1 * h]
    pts = np.float32([[ln.x_at(y), y] for ln in m.lines for y in ys]).reshape(-1, 2)
    return grid_residual(calib.to_field(pts)[:, 0]) if len(pts) else float("nan")


def refit_from_paint(
    image: np.ndarray, approx: FieldCalibration, marks: FieldMarks | None = None
) -> CalibrationSearch:
    """Refit a frame on its own paint, its yard lines named by an approximate calibration.

    The approximation only has to be within 2.5 yd: each line takes the nearest
    whole 5 yd consistent with the spacing. Raises ValueError when fewer than two
    lines can be named or the naming runs past a goal line.
    """
    m = marks if marks is not None else find_marks(image)
    named = [(ln, o) for ln, o in zip(m.lines, m.offsets) if o is not None]
    if len(named) < 2:
        raise ValueError(f"{len(named)} yard lines to name")
    xs = [
        float(approx.to_field(np.float32([[ln.x_at(m.ref_y), m.ref_y]]))[0, 0]) for ln, _ in named
    ]
    base = 5.0 * round(float(np.median([x - 5.0 * o for x, (_, o) in zip(xs, named)])) / 5)
    yards = [None if o is None else base + 5.0 * o for o in m.offsets]
    known = [y for y in yards if y is not None]
    if min(known) < 10 or max(known) > 110:
        raise ValueError("the named lines run past a goal line")
    return search_calibration(image, yards, marks=m)


def draw_field_grid(
    image: np.ndarray, calib: FieldCalibration, color=(0, 255, 255), thickness: int = 2
) -> np.ndarray:
    """The fitted field drawn back onto the frame: yard lines every 5 yd, hash rows,
    number rows and sidelines. It should sit on the painted markings everywhere
    players stand; where it drifts, the calibration is wrong there."""
    out = image.copy()
    hinv = np.linalg.inv(calib.matrix)
    h, w = image.shape[:2]

    def px(x: float, y: float):
        p = hinv @ np.array([x, y, 1.0])
        if p[2] <= 0:
            return None
        u, v = p[:2] / p[2]
        if not (-2 * w < u < 3 * w and -2 * h < v < 3 * h):
            return None
        return int(u), int(v)

    def polyline(pts):
        pts = [p for p in pts if p is not None]
        for a, b in pairwise(pts):
            cv2.line(out, a, b, color, thickness, cv2.LINE_AA)

    for x in range(10, 115, 5):
        polyline([px(x, y) for y in np.linspace(0, FIELD_WIDTH, 12)])
        p = px(x, (HASH_FAR + HASH_NEAR) / 2)
        if p is not None and 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.putText(
                out,
                str(painted_from_field_x(x)),
                (p[0] + 6, p[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
    for y in (
        0.0,
        NUMBER_ROWS_NEAR[1],
        NUMBER_ROWS_NEAR[0],
        HASH_NEAR,
        HASH_FAR,
        NUMBER_ROWS_FAR[1],
        NUMBER_ROWS_FAR[0],
        FIELD_WIDTH,
    ):
        polyline([px(x, y) for x in np.linspace(0, 120, 49)])
    return out
