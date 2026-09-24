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
    """Long near-vertical white lines, left to right, each refit on its pixels.

    `min_length` (px) and `votes` (Hough accumulator threshold) scale with how
    zoomed-in the frame is: a zoomed-out full-width view has thinner, shorter
    lines and needs both lowered.
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
        if 60 < ang < 120 and y2 != y1:
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
        pts = []
        for y, xg in zip(ys, guess):
            lo, hi = int(xg) - 8, int(xg) + 9
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
    for y in range(y_lo, y_hi):
        cx = int(line.x_at(y))
        seg = mask[y, max(0, cx - half_w) : cx + half_w].copy()
        c0 = min(len(seg), max(0, half_w - 14))
        seg[c0 : c0 + 29] = 0  # blank the yard line
        if (seg > 0).sum() > 12:
            ys_white.append(y)
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


def auto_calibrate(
    image: np.ndarray,
    yards_left_to_right: list[float],
    roi: tuple[int, int, int, int],
    far_sideline_y: tuple[int, int] | None = None,
    near_sideline_y: tuple[int, int] | None = None,
    hash_y: tuple[int, int] | None = None,
    numbers: bool = True,
    max_error: float = 1.0,
    min_length: int = 250,
    votes: int = 120,
) -> AutoCalibration:
    """Fit a FieldCalibration from the painted markings of one all-22 frame.

    yards_left_to_right : field x (0-120, end zones included) of each yard line
        that `detect_yard_lines` will find, in image order. Numbers increasing
        to the right means the low-x half of the field is in view.
    roi : (x1, y1, x2, y2) of the video area (exclude player UI, letterbox).
    far_sideline_y / near_sideline_y : image y ranges holding each sideline
        band (None if off-frame). The far sideline (top of the image) is y=53.3.
    hash_y : image y range containing both hash rows (default: middle half).
    """
    _, y1, _, y2 = roi
    mask = white_mask(image, roi)
    lines = detect_yard_lines(mask, min_length, ref_y=(y1 + y2) / 2, votes=votes)
    notes = [f"{len(lines)} yard lines detected"]
    if len(lines) != len(yards_left_to_right):
        raise ValueError(
            f"detected {len(lines)} yard lines but {len(yards_left_to_right)} yard values given; "
            f"lines at x(mid)={[round(ln.x_at((y1 + y2) / 2)) for ln in lines]}"
        )
    hash_y = hash_y or (y1 + (y2 - y1) // 4, y1 + 3 * (y2 - y1) // 4)
    rows = fit_hash_rows(mask, hash_y)
    if len(rows) != 2:
        raise ValueError(f"expected 2 hash rows, fitted {len(rows)}")
    notes.append(f"hash rows on {rows[0].n} + {rows[1].n} ticks")
    pixel, fld = [], []
    for ln, fx in zip(lines, yards_left_to_right):
        for row, fy in zip(rows, (HASH_FAR, HASH_NEAR)):
            pixel.append(intersect(ln, row))
            fld.append((fx, fy))
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
        for ln, fx in zip(lines, yards_left_to_right):
            pixel.append(intersect(ln, side))
            fld.append((fx, fy))
    pixel, fld = np.array(pixel, float), np.array(fld, float)
    calib = FieldCalibration.from_points(pixel, fld)

    if numbers:
        extra_px, extra_f = [], []
        for ln, fx in zip(lines, yards_left_to_right):
            if fx % 10:  # numbers sit on the 10s only
                continue
            for rows_ in (NUMBER_ROWS_FAR, NUMBER_ROWS_NEAR):
                got = number_edges(mask, ln, calib, rows_, fx)
                if got:
                    for p, f in got:
                        extra_px.append(p)
                        extra_f.append(f)
        if extra_px:
            cand_px = np.vstack([pixel, extra_px])
            cand_f = np.vstack([fld, extra_f])
            trial = FieldCalibration.from_points(cand_px, cand_f)
            err = np.hypot(*(trial.to_field(cand_px) - cand_f).T)
            keep = err < max_error
            notes.append(f"{keep[len(pixel) :].sum()} of {len(extra_px)} number edges kept")
            pixel, fld = cand_px[keep], cand_f[keep]
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
