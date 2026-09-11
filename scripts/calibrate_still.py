"""Interactively calibrate a still image: click field landmarks, enter their yards.

Click a recognizable ground-plane point (yard line x sideline / hash intersection),
then type its field coordinates in the console as "x y" (yards; x 0-120 including
end zones, y 0-53.3 with y=0 at the near sideline). Repeat for >= 5 points spread
across the image (4 is the minimum but leaves no redundancy to check the fit),
then press q in the image window to fit and save.

Usage: python scripts/calibrate_still.py screenshot.png --out calibration.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from nfl_coverage.vision.homography import FieldCalibration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.image.with_suffix(".calibration.json")

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    clicks: list[tuple[int, int]] = []
    pixel_points: list[tuple[float, float]] = []
    field_points: list[tuple[float, float]] = []

    # The callback only records clicks; the console prompt runs in the main loop
    # (input() inside a cv2 callback freezes the window on Windows).
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    win = "calibrate (click points, q to finish)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    while True:
        cv2.imshow(win, img)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        while clicks:
            px, py = clicks.pop(0)
            raw = input(f"pixel ({px}, {py}) -> field 'x y' in yards (blank to discard): ")
            if not raw.strip():
                continue
            try:
                fx, fy = (float(v) for v in raw.split())
            except ValueError:
                print(f"  could not parse {raw!r} — expected two numbers like '45 26.6', skipped")
                continue
            pixel_points.append((float(px), float(py)))
            field_points.append((fx, fy))
            cv2.circle(img, (px, py), 5, (0, 220, 255), -1)
            cv2.putText(
                img,
                f"({fx:g},{fy:g})",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.destroyAllWindows()

    calib = FieldCalibration.from_points(pixel_points, field_points)
    calib.save(out)
    err = calib.reprojection_error()
    print(f"{len(pixel_points)} points -> {out} (reprojection error {err:.2f} yd)")
    if len(pixel_points) == 4:
        print("note: with exactly 4 points the error is trivially ~0 — add a 5th to verify")


if __name__ == "__main__":
    main()
