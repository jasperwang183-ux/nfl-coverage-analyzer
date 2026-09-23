"""Field-visibility guard: cropped analysis regions are caught and described."""

import numpy as np
import pytest

from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.visibility import (
    check_visibility,
    defender_count_warnings,
    field_to_pixel,
    required_region,
)

# Perspective camera from test_homography (side-on, mild keystone).
PERSPECTIVE = np.array(
    [
        [14.0, -3.0, 120.0],
        [1.5, 6.0, 80.0],
        [0.0004, 0.004, 1.0],
    ]
)


def _project(matrix: np.ndarray, field_pts: np.ndarray) -> np.ndarray:
    homog = np.hstack([field_pts, np.ones((len(field_pts), 1))]) @ matrix.T
    return homog[:, :2] / homog[:, 2:]


def _calib(matrix: np.ndarray, field: np.ndarray) -> FieldCalibration:
    return FieldCalibration.from_points(_project(matrix, field), field)


def _topdown(x_offset_px: float = 0.0) -> FieldCalibration:
    """Axis-aligned 10 px/yd camera: pixel = 10 * field - offset. Crops are exact."""
    matrix = np.array([[10.0, 0.0, -x_offset_px], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
    field = np.array([[0, 0], [120, 0], [0, 53.3], [120, 53.3]], float)
    return _calib(matrix, field)


class TestRequiredRegion:
    def test_extends_toward_the_defense(self):
        assert required_region(50.0, "left") == (47.0, 68.0, 0.0, 53.3)
        assert required_region(50.0, "right") == (32.0, 53.0, 0.0, 53.3)

    def test_clamped_to_field(self):
        assert required_region(110.0, "left")[1] == 120.0
        assert required_region(5.0, "right")[0] == 0.0

    def test_bad_offense_rejected(self):
        with pytest.raises(ValueError, match="offense"):
            required_region(50.0, "up")


class TestFieldToPixel:
    def test_inverts_calibration(self):
        field = np.array([[30, 0], [30, 53.3], [50, 0], [50, 53.3], [40, 26.6], [60, 10]], float)
        calib = _calib(PERSPECTIVE, field)
        pts = np.array([[35.0, 12.0], [55.0, 48.0]])
        px, front = field_to_pixel(calib, pts)
        assert front.all()
        assert np.allclose(px, _project(PERSPECTIVE, pts), atol=0.5)

    def test_points_behind_the_camera_are_flagged(self):
        # homogeneous w = 0.0004x - 0.03y + 1 goes negative past y ~ 33: a horizon
        horizon = PERSPECTIVE.copy()
        horizon[2, 1] = -0.03
        field = np.array([[30, 0], [30, 20], [50, 0], [50, 20], [40, 10]], float)
        calib = _calib(horizon, field)
        _, front = field_to_pixel(calib, [[40.0, 10.0], [40.0, 50.0]])
        assert front.tolist() == [True, False]
        # and the guard refuses the far-sideline corners regardless of where the
        # meaningless pixels land
        report = check_visibility(calib, (5000, 5000), los=40.0, offense="left")
        assert not report.ok
        assert not report.corner_visible["deep, far sideline"]


class TestCheckVisibility:
    def test_full_view_passes(self):
        # region x in [47, 68] -> px [470, 680]; y -> py [0, 533]
        report = check_visibility(_topdown(), (540, 700), los=50.0, offense="left")
        assert report.ok
        assert report.warnings == []
        assert report.fraction_visible == 1.0
        assert report.band == (0.0, 53.3)
        assert report.depth_visible == 18.0
        assert "OK" in report.summary()

    def test_perspective_full_view_passes(self):
        field = np.array([[30, 0], [30, 53.3], [50, 0], [50, 53.3], [40, 26.6], [60, 10]], float)
        calib = _calib(PERSPECTIVE, field)
        report = check_visibility(calib, (500, 1100), los=50.0, offense="left")
        assert report.ok, report.warnings
        assert report.fraction_visible == 1.0

    def test_deep_crop_is_named(self):
        # width 621 px -> x <= 62 visible -> 12 yd beyond a LOS at 50
        report = check_visibility(_topdown(), (540, 621), los=50.0, offense="left")
        assert not report.ok
        assert report.depth_visible == 12.0
        assert report.band == (0.0, 53.3)
        assert any("deep zone beyond 12 yd" in w for w in report.warnings)
        assert not any("sideline strip" in w for w in report.warnings)
        assert report.summary().startswith("PARTIAL VIEW")

    def test_sideline_crop_is_named(self):
        # height 401 px -> y <= 40 visible; depth untouched
        report = check_visibility(_topdown(), (401, 700), los=50.0, offense="left")
        assert not report.ok
        assert report.depth_visible == 18.0
        assert report.band == (0.0, 40.0)
        assert any("sideline strip y > 40 yd" in w for w in report.warnings)
        assert not any("deep zone" in w for w in report.warnings)

    def test_backfield_crop_is_named(self):
        # pixel x = 10 * field_x - 490: x < 49 is left of the image
        report = check_visibility(_topdown(490.0), (540, 300), los=50.0, offense="left")
        assert not report.ok
        assert any("backfield beyond 1 yd" in w for w in report.warnings)

    def test_offense_right_mirrors_depth(self):
        # region x in [32, 53]; cropping x < 40 (offset 400 px) cuts 10 yd of depth
        report = check_visibility(_topdown(400.0), (540, 200), los=50.0, offense="right")
        assert not report.ok
        assert report.depth_visible == 10.0
        assert any("deep zone beyond 10 yd" in w for w in report.warnings)

    def test_slack_tolerates_the_paint(self):
        # 1 yd of far sideline and 0.5 yd of depth missing: within calibration slack
        report = check_visibility(_topdown(), (524, 676), los=50.0, offense="left")
        assert report.ok
        assert report.fraction_visible < 1.0

    def test_los_off_frame(self):
        report = check_visibility(_topdown(700.0), (540, 100), los=50.0, offense="left")
        assert not report.ok
        assert any("line of scrimmage" in w for w in report.warnings)


class TestDefenderCount:
    def test_band(self):
        assert defender_count_warnings(7) == []
        assert defender_count_warnings(5) == []
        assert defender_count_warnings(11) == []
        assert "only 4" in defender_count_warnings(4)[0]
        assert "12" in defender_count_warnings(12)[0]
