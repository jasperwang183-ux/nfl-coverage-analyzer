"""Calibrate an all-22 frame from its painted markings, no clicking.

Detects the yard lines, hash rows, sidelines and yard numbers, intersects them,
and fits the homography (see vision/fieldlines.py). You supply what a single
frame cannot tell a machine: the painted yard of each detected line, left to
right, as field x (0-120, end zones included: the painted 30 on the low side of
the field is x=40, on the high side x=80).

Run it once without --yards to see how many lines were found and where; then:

  python scripts/calibrate_auto.py frame.png --yards 40 45 50 55 \\
      --roi 300 100 2260 1020 --far-sideline 100 240 --hash-y 400 750

Writes <frame>.calibration.json and <frame>_landmarks.png (yellow = landmark
within half a yard of the fit, red = worse).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from nfl_coverage.vision.fieldlines import (
    auto_calibrate,
    detect_yard_lines,
    draw_landmarks,
    white_mask,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--yards", type=float, nargs="*", default=None, help="field x per line, L->R")
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="x1 y1 x2 y2 of the video area")
    ap.add_argument(
        "--far-sideline", type=int, nargs=2, default=None, help="y range of the top band"
    )
    ap.add_argument(
        "--near-sideline", type=int, nargs=2, default=None, help="y range of the bottom band"
    )
    ap.add_argument(
        "--hash-y", type=int, nargs=2, default=None, help="y range holding both hash rows"
    )
    ap.add_argument("--no-numbers", action="store_true")
    ap.add_argument("--min-length", type=int, default=250, help="Hough min line length (px)")
    ap.add_argument("--votes", type=int, default=120, help="Hough accumulator threshold")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    h, w = img.shape[:2]
    roi = tuple(args.roi) if args.roi else (0, 0, w, h)
    if args.yards is None:
        mask = white_mask(img, roi)
        lines = detect_yard_lines(
            mask, args.min_length, ref_y=(roi[1] + roi[3]) / 2, votes=args.votes
        )
        mid = (roi[1] + roi[3]) / 2
        print(f"{len(lines)} yard lines; x at y={mid:.0f}: {[round(ln.x_at(mid)) for ln in lines]}")
        print("re-run with --yards <field x of each, left to right>")
        return
    auto = auto_calibrate(
        img,
        list(args.yards),
        roi,
        far_sideline_y=tuple(args.far_sideline) if args.far_sideline else None,
        near_sideline_y=tuple(args.near_sideline) if args.near_sideline else None,
        hash_y=tuple(args.hash_y) if args.hash_y else None,
        numbers=not args.no_numbers,
        min_length=args.min_length,
        votes=args.votes,
    )
    for n in auto.notes:
        print(f"  {n}")
    print(
        f"{len(auto.pixel)} landmarks; reprojection mean {auto.errors.mean():.2f} yd, "
        f"max {auto.errors.max():.2f}"
    )
    out = args.out or args.image.with_name(args.image.stem + ".calibration.json")
    auto.calibration.save(out)
    overlay = args.image.with_name(args.image.stem + "_landmarks.png")
    cv2.imwrite(str(overlay), draw_landmarks(img, auto))
    print(f"saved {out} and {overlay}")


if __name__ == "__main__":
    main()
