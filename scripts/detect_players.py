"""Phase 2 first step: detect players in a football image and draw bounding boxes.

Runs a pretrained YOLO model (class 'person') on a screenshot/frame and saves an
annotated copy. This validates detection quality on real footage before we build
the homography (pixels -> field coordinates) and tracking stages.

Usage:
  python scripts/detect_players.py path\to\screenshot.png
  python scripts/detect_players.py clip.png --model yolo11m.pt --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class id


def detect(image_path: Path, model_name: str, conf: float, out_path: Path) -> int:
    model = YOLO(model_name)  # auto-downloads official weights on first use
    result = model.predict(str(image_path), conf=conf, classes=[PERSON_CLASS], verbose=False)[0]

    img = cv2.imread(str(image_path))
    boxes = result.boxes
    for xyxy, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            img,
            f"{c:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    n = len(boxes)
    cv2.putText(
        img,
        f"{n} players detected ({model_name}, conf>={conf})",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), img)
    print(f"{n} detections -> {out_path}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.image.with_name(args.image.stem + "_detected.png")
    detect(args.image, args.model, args.conf, out)
