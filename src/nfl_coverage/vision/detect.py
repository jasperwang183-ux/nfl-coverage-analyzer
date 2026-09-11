"""YOLO person detection for football frames.

Returns each detection's *foot point* (bottom-center of the bounding box) — the
pixel where the player meets the ground plane, which is what the homography can
legitimately project into field coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PERSON_CLASS = 0  # COCO class id


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 pixels
    conf: float

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)


def detect_people(
    image_path: Path, model_name: str = "yolo11s.pt", conf: float = 0.3
) -> list[Detection]:
    """Run pretrained YOLO on an image; returns person detections."""
    from ultralytics import YOLO  # deferred: heavy import, and tests don't need it

    result = YOLO(model_name).predict(
        str(image_path), conf=conf, classes=[PERSON_CLASS], verbose=False
    )[0]
    boxes = result.boxes
    return [Detection(tuple(xyxy), c) for xyxy, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist())]


def foot_points(detections: list[Detection]) -> np.ndarray:
    """(N, 2) array of foot-point pixels."""
    return np.array([d.foot_point for d in detections], dtype=np.float32).reshape(-1, 2)
