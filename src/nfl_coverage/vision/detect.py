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


DEFAULT_IMGSZ = 1280  # the YOLO default of 640 loses far-side players on a 1080p+ all-22 frame


def detect_people(
    image_path: Path, model_name: str = "yolo11s.pt", conf: float = 0.3, imgsz: int = DEFAULT_IMGSZ
) -> list[Detection]:
    """Run pretrained YOLO on an image; returns person detections.

    `imgsz` is the inference resolution: 640 found 17 of 22 players on a 2560-wide
    coaches-film frame, 1280 found 29 boxes, 1920 all of them (plus the crowd).
    """
    from ultralytics import YOLO  # deferred: heavy import, and tests don't need it

    result = YOLO(model_name).predict(
        str(image_path), conf=conf, classes=[PERSON_CLASS], imgsz=imgsz, verbose=False
    )[0]
    boxes = result.boxes
    return [Detection(tuple(xyxy), c) for xyxy, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist())]


def foot_points(detections: list[Detection]) -> np.ndarray:
    """(N, 2) array of foot-point pixels."""
    return np.array([d.foot_point for d in detections], dtype=np.float32).reshape(-1, 2)


def dedupe_mask(field_xy: np.ndarray, confs: list[float], min_dist: float = 0.8) -> np.ndarray:
    """Drop overlapping duplicate detections of the same body.

    Greedy in confidence order: keep a detection only if no already-kept one
    projects within `min_dist` yards of it on the field. Returns a boolean mask.
    """
    keep = np.zeros(len(field_xy), dtype=bool)
    for i in np.argsort(confs)[::-1]:
        d = np.hypot(*(field_xy[keep] - field_xy[i]).T) if keep.any() else np.array([np.inf])
        if (d >= min_dist).all():
            keep[i] = True
    return keep
