"""Find the snap in a clip from the line firing, and show the per-frame evidence.

Tracks people over [--start, --end], chains the anchor calibration to every
kept frame, converts foot points to field coordinates and runs
`vision.snap.detect_snap`. Prints the feature series (the input a learned
detector would take) and the call with its reasons; `--expect` prints the
offset from a known snap time.

Usage:
  python scripts/detect_snap.py clip.mp4 --calib frame_010.calibration.json --anchor 5.0 \\
      --start 4.0 --end 7.0 --los 49.9 --imgsz 1920 --expect 5.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.snap import detect_snap, field_positions
from nfl_coverage.vision.track import chain_calibrations, track_players


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="calibration of the anchor frame")
    ap.add_argument("--anchor", type=float, required=True, help="clip time (s) it was fitted at")
    ap.add_argument("--keyframe", action="append", default=[], help="T=calib.json, extra anchors")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument(
        "--los", type=float, default=None, help="field x of the LOS; estimated if absent"
    )
    ap.add_argument("--expect", type=float, default=None, help="known snap time (s) to compare")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--yolo", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--series", type=Path, default=None, help="write the feature series (csv)")
    args = ap.parse_args()

    frames = track_players(
        args.video, args.start, args.end, args.step, args.yolo, args.conf, args.imgsz
    )
    times = np.array([f.t for f in frames])
    anchor_i = int(np.argmin(np.abs(times - args.anchor)))
    keyframes = {}
    for kf in args.keyframe:
        t_s, path = kf.split("=", 1)
        keyframes[int(np.argmin(np.abs(times - float(t_s))))] = FieldCalibration.load(path)
    chain = chain_calibrations(frames, FieldCalibration.load(args.calib), anchor_i, keyframes)
    for n in chain.notes:
        print("  !", n)

    pos = field_positions(frames, chain.calibrations)
    det = detect_snap(pos, los=args.los, step_s=args.step)
    with pd.option_context("display.width", 140, "display.max_rows", 500):
        print(det.series.round(2).to_string(index=False))
    if args.series is not None:
        det.series.to_csv(args.series, index=False)
        print(f"series -> {args.series}")
    print()
    if det.frame is None:
        print(f"SNAP: not found [{det.confidence}]")
    else:
        print(
            f"SNAP: frame {det.frame} at {det.t:.1f} s [{det.confidence} confidence, "
            f"score {det.score:.2f}]; line fired at frame {det.fire_frame}"
        )
    for r in det.reasons:
        print("    -", r)
    if args.expect is not None and det.t is not None:
        print(
            f"expected {args.expect:.1f} s -> detected {det.t:.1f} s ({det.t - args.expect:+.1f} s)"
        )


if __name__ == "__main__":
    main()
