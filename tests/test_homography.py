"""Homography calibration round-trip on synthetic camera geometry."""

import numpy as np
import pytest

from nfl_coverage.vision.homography import FieldCalibration, on_field

# A plausible perspective matrix mapping field yards -> pixels (synthetic camera).
FIELD_TO_PIXEL = np.array(
    [
        [14.0, -3.0, 120.0],
        [1.5, 6.0, 80.0],
        [0.0004, 0.004, 1.0],
    ]
)


def _project(field_pts: np.ndarray) -> np.ndarray:
    homog = np.hstack([field_pts, np.ones((len(field_pts), 1))]) @ FIELD_TO_PIXEL.T
    return homog[:, :2] / homog[:, 2:]


def test_round_trip_recovers_field_points():
    field = np.array([[30, 0], [30, 53.3], [50, 0], [50, 53.3], [40, 26.6], [60, 10]], float)
    calib = FieldCalibration.from_points(_project(field), field)
    players = np.array([[35.0, 12.0], [42.5, 30.1], [55.0, 48.0]])
    recovered = calib.to_field(_project(players))
    assert np.allclose(recovered, players, atol=0.05)
    assert calib.reprojection_error() < 0.05


def test_save_load_round_trip(tmp_path):
    field = np.array([[20, 0], [20, 53.3], [60, 0], [60, 53.3]], float)
    calib = FieldCalibration.from_points(_project(field), field)
    path = tmp_path / "calib.json"
    calib.save(path)
    loaded = FieldCalibration.load(path)
    pts = _project(np.array([[33.0, 15.0]]))
    assert np.allclose(loaded.to_field(pts), calib.to_field(pts), atol=1e-3)


def test_too_few_points_rejected():
    with pytest.raises(ValueError, match=">= 4"):
        FieldCalibration.from_points([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])


def test_on_field_mask():
    pts = np.array([[60.0, 26.0], [-5.0, 26.0], [60.0, 60.0], [119.5, 0.5]])
    assert on_field(pts).tolist() == [True, False, False, True]
