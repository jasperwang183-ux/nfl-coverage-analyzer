"""Pixel -> field-coordinate mapping via a plane homography.

A still or clip frame is calibrated once with >= 4 point correspondences between
image pixels and known field locations (yard-line / hash / sideline intersections),
expressed in the canonical field frame: x in [0, 120] yards including end zones,
y in [0, 53.3] with y = 0 at the sideline nearest x-axis origin conventions used
by the tracking data. The resulting homography projects any ground-plane pixel
(e.g. a detected player's feet) into field coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from nfl_coverage.config import FIELD_LENGTH, FIELD_WIDTH


@dataclass
class FieldCalibration:
    """Homography from image pixels to field yards, built from point pairs."""

    pixel_points: np.ndarray  # (N, 2) float32
    field_points: np.ndarray  # (N, 2) float32, yards
    matrix: np.ndarray  # (3, 3) homography

    @classmethod
    def from_points(cls, pixel_points, field_points) -> "FieldCalibration":
        px = np.asarray(pixel_points, dtype=np.float32)
        fp = np.asarray(field_points, dtype=np.float32)
        if px.shape != fp.shape or px.ndim != 2 or px.shape[1] != 2:
            raise ValueError(f"point arrays must both be (N, 2); got {px.shape} vs {fp.shape}")
        if len(px) < 4:
            raise ValueError(f"need >= 4 correspondences, got {len(px)}")
        # threshold is in destination units — field *yards*, so keep it tight
        method = cv2.RANSAC if len(px) > 4 else 0
        # fit around the points' own center so the result does not depend on where the
        # field origin is: a naming shifted along the field must give the same camera
        center = fp.mean(axis=0)
        matrix, _ = cv2.findHomography(px, fp - center, method, ransacReprojThreshold=1.0)
        if matrix is None:
            raise ValueError("homography estimation failed (degenerate points?)")
        shift = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float64)
        matrix = shift @ matrix
        return cls(px, fp, matrix)

    def to_field(self, pixel_xy) -> np.ndarray:
        """Project (N, 2) pixel points onto the field plane; returns (N, 2) yards."""
        pts = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def reprojection_error(self) -> float:
        """Mean distance (yards) between projected calibration pixels and field points."""
        proj = self.to_field(self.pixel_points)
        return float(np.hypot(*(proj - self.field_points).T).mean())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(
                {
                    "pixel_points": self.pixel_points.tolist(),
                    "field_points": self.field_points.tolist(),
                    # persisted so a reload is bit-identical to the session that
                    # was validated (RANSAC refits are not deterministic)
                    "matrix": self.matrix.tolist(),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "FieldCalibration":
        raw = json.loads(Path(path).read_text())
        if "matrix" in raw:
            return cls(
                np.asarray(raw["pixel_points"], dtype=np.float32),
                np.asarray(raw["field_points"], dtype=np.float32),
                np.asarray(raw["matrix"], dtype=float),
            )
        return cls.from_points(raw["pixel_points"], raw["field_points"])


def on_field(
    field_xy: np.ndarray, margin: float = 1.0, y_margin: float | None = None
) -> np.ndarray:
    """Boolean mask of points inside the field boundary (plus a small margin).

    `y_margin` overrides the sideline tolerance: bench personnel stand with their
    feet on the sideline paint (y within +-1 of 0 or 53.3), so a frame that shows
    a sideline wants a *negative* y margin, i.e. feet at least half a yard inside.
    """
    x, y = field_xy[:, 0], field_xy[:, 1]
    ym = margin if y_margin is None else y_margin
    return (x >= -margin) & (x <= FIELD_LENGTH + margin) & (y >= -ym) & (y <= FIELD_WIDTH + ym)
