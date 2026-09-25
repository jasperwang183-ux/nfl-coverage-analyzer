"""Field calibration from painted markings, on a synthetic field seen through a known camera."""

import cv2
import numpy as np
import pytest

from nfl_coverage.vision.fieldlines import (
    HASH_FAR,
    HASH_NEAR,
    NUMBER_ROWS_FAR,
    NUMBER_ROWS_NEAR,
    YardLine,
    field_x_for_lines,
    find_marks,
    index_lines,
    paint_mask,
    search_calibration,
)
from nfl_coverage.vision.homography import FieldCalibration

PPY = 24  # top-down render: pixels per yard
FIELD_GREEN = (40, 125, 45)  # BGR
PAINT = (235, 235, 235)


def _topdown(numbers: bool = True) -> np.ndarray:
    """A top-down field: x 0-120 (columns), y 0-53.3 (rows, y=53.3 at the top)."""
    h, w = int(56 * PPY), 120 * PPY
    img = np.full((h, w, 3), FIELD_GREEN, np.uint8)

    def row(y):  # field y -> image row (far sideline on top)
        return round((54.8 - y) * PPY)

    def col(x):
        return round(x * PPY)

    for y in (0.0, 53.3):  # sideline bands (outside the field of play)
        y0, y1 = (row(y), row(y - 2)) if y == 0 else (row(y + 2), row(y))
        cv2.rectangle(img, (0, y0), (w, y1), PAINT, -1)
    for x in range(10, 111, 5):  # yard lines
        cv2.line(img, (col(x), row(53.3)), (col(x), row(0)), PAINT, 3)
    for x in range(10, 111):  # 1-yd ticks on both hashes
        if x % 5:
            for y in (HASH_FAR, HASH_NEAR):
                cv2.line(img, (col(x), row(y + 0.3)), (col(x), row(y - 0.3)), PAINT, 3)
    if numbers:  # two digit blocks straddling each 10-yd line, 2 yd tall
        for x in range(20, 101, 10):
            for lo, hi in (NUMBER_ROWS_NEAR[::-1], NUMBER_ROWS_FAR[::-1]):
                for side in (-1, 1):
                    xa, xb = x + side * 0.6, x + side * 1.8
                    cv2.rectangle(
                        img, (col(min(xa, xb)), row(hi)), (col(max(xa, xb)), row(lo)), PAINT, 4
                    )
    return img


def _camera(img_top: np.ndarray, x_view=(38.0, 72.0), size=(1600, 900)):
    """Warp the top-down field into an elevated sideline camera view.

    Returns (frame, true FieldCalibration pixel -> field).
    """
    w, h = size
    x0, x1 = x_view
    # image corners -> field points: a trapezoid (far side narrower) = perspective
    field_quad = np.float32([[x0 + 4, 53.3], [x1 - 4, 53.3], [x1 + 3, 0.0], [x0 - 3, 0.0]])
    img_quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    truth = FieldCalibration.from_points(img_quad, field_quad)

    def to_top(pts):
        return np.column_stack([pts[:, 0] * PPY, (54.8 - pts[:, 1]) * PPY]).astype(np.float32)

    m = cv2.getPerspectiveTransform(to_top(field_quad), img_quad)
    frame = cv2.warpPerspective(img_top, m, (w, h), borderValue=FIELD_GREEN)
    return frame, truth


def _grid_error(calib, truth, size):
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 9), np.linspace(0.15 * h, 0.9 * h, 7))
    g = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    return np.hypot(*(calib.to_field(g) - truth.to_field(g)).T)


# ------------------------------------------------------------- indexing


def _lines_at(xs):
    return [YardLine(0.0, float(x), 100) for x in xs]


def test_index_lines_even_spacing():
    assert index_lines(_lines_at([100, 400, 700, 1000]), 0) == [0, 1, 2, 3]


def test_index_lines_skips_a_missed_line_and_drops_a_stray_one():
    base = np.cumsum([0] + [300 * (1 - 0.03 * i) for i in range(8)]) + 50
    missed = np.delete(base, 3)
    assert index_lines(_lines_at(missed), 0) == [0, 1, 2, 4, 5, 6, 7, 8]
    stray = np.sort(np.append(base, base[4] + 0.45 * (base[5] - base[4])))
    offs = index_lines(_lines_at(stray), 0)
    assert offs[5] is None and offs[:5] == [0, 1, 2, 3, 4] and offs[6:] == [5, 6, 7, 8]


def test_field_x_for_lines_from_any_named_line():
    offs = [0, 1, None, 2, 3]
    # line 4 (index 3) is the painted 45 with numbers going up to its right
    assert field_x_for_lines(offs, 3, 45, True) == [45.0, 50.0, None, 55.0, 60.0]
    # the same lines on the other half: line 1 is the painted 30, numbers going down
    assert field_x_for_lines(offs, 0, 30, False) == [80.0, 85.0, None, 90.0, 95.0]
    with pytest.raises(ValueError):
        field_x_for_lines(offs, 2, 40, True)  # not a yard line
    with pytest.raises(ValueError):
        field_x_for_lines(offs, 0, 5, False)  # runs past the goal line


# ------------------------------------------------------------- the fit


def test_paint_mask_ignores_player_ui_off_the_field():
    frame, _ = _camera(_topdown())
    ui = np.zeros((200, frame.shape[1], 3), np.uint8)  # a black control bar with white text
    cv2.putText(
        ui, "0:08 / 19:15   in this video", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, PAINT, 5
    )
    both = np.vstack([frame, ui])
    m = paint_mask(both)
    assert m[frame.shape[0] :].sum() == 0
    assert m[: frame.shape[0]].sum() > 0


def test_search_recovers_the_camera_on_a_synthetic_field():
    size = (1600, 900)
    frame, truth = _camera(_topdown(), size=size)
    marks = find_marks(frame)
    yard = [o for o in marks.offsets if o is not None]
    assert len(yard) >= 5
    # name the lines from the truth (the person's step), then fit
    xs = [
        None
        if o is None
        else round(truth.to_field(np.float32([[ln.x_at(marks.ref_y), marks.ref_y]]))[0, 0] / 5) * 5
        for ln, o in zip(marks.lines, marks.offsets)
    ]
    r = search_calibration(frame, xs, marks=marks)
    assert r.n_rows >= 3
    assert len(r.best.pixel) >= 20  # yard-line crossings plus 1-yd ticks
    assert any("1-yd ticks" in n for n in r.best.notes)
    err = _grid_error(r.best.calibration, truth, size)
    assert np.median(err) < 0.3 and err.max() < 1.0


def test_numbers_prefer_the_true_labeling_over_a_5_yard_shift():
    frame, truth = _camera(_topdown())
    marks = find_marks(frame)
    xs = [
        None
        if o is None
        else round(truth.to_field(np.float32([[ln.x_at(marks.ref_y), marks.ref_y]]))[0, 0] / 5) * 5
        for ln, o in zip(marks.lines, marks.offsets)
    ]
    shifted = [None if x is None else x + 5 for x in xs]

    def edges(labels):
        r = search_calibration(frame, labels, marks=marks)
        return sum(int(n.split()[0]) for n in r.best.notes if n.endswith("number edges"))

    assert edges(xs) >= edges(shifted) + 4


def test_a_fit_on_the_hash_rows_alone_is_kept_with_a_depth_warning():
    # no sidelines, no numbers: only the hashes - often all a tight all-22 zoom shows
    frame, truth = _camera(_topdown(numbers=False))
    crop = frame[250:650]  # both sidelines out of view
    marks = find_marks(crop)
    xs = [
        None
        if o is None
        else round(
            truth.to_field(np.float32([[ln.x_at(marks.ref_y), marks.ref_y + 250]]))[0, 0] / 5
        )
        * 5
        for ln, o in zip(marks.lines, marks.offsets)
    ]
    r = search_calibration(crop, xs, marks=marks)
    assert r.n_rows == 2
    assert r.warning is not None and "hash rows alone" in r.warning
    # along the field it is still right where the hashes are
    mid = np.float32([[crop.shape[1] / 2, crop.shape[0] / 2]])
    got = r.best.calibration.to_field(mid)[0]
    want = truth.to_field(mid + np.float32([[0, 250]]))[0]
    assert abs(got[0] - want[0]) < 0.5
