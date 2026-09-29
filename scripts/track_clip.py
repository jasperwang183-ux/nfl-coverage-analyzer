"""Track a play through a clip and read it like tracking data: shown vs played, with evidence.

Runs YOLO + ByteTrack over [--start, --end] and chains the anchor calibration to
every kept frame (saved under <out-dir>/tracked, so `--from-tracked` re-reads
without re-tracking), then `vision.pipeline.read_tracked`: the snap from the
line firing (or --snap), the formation at the snap (LOS, offense side, ball y -
each overridable by --los / --offense / --ball-y), sides, canonical 10 Hz
tracks, the shown and played structure, shell_rotation with the rotator named,
the variant model and the coverage report. Writes tracks.parquet, the top-down
render, the annotated snap frame and result.json (the web page's contract).

Usage:
  python scripts/track_clip.py clip.mp4 --calib frame_010.calibration.json --anchor 5.0 \\
      --start 4.5 --end 9.0 --imgsz 1920
  python scripts/track_clip.py clip.mp4 --from-tracked data/clips/clip_tracks/tracked --snap 5.6

--start must precede the snap by at least the detector's still window (0.5 s).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from nfl_coverage.analysis.viz import render_play
from nfl_coverage.config import MODELS_DIR, ROOT
from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.pipeline import (
    ReadError,
    TrackedClip,
    annotate_snap,
    read_tracked,
    track_frames,
)

CLIPS_DIR = ROOT / "data" / "clips"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", type=Path, default=None, help="calibration of the anchor frame")
    ap.add_argument("--anchor", type=float, default=None, help="clip time (s) it was fitted at")
    ap.add_argument("--keyframe", action="append", default=[], help="T=calib.json, extra anchors")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument(
        "--from-tracked", type=Path, default=None, help="re-read a saved tracked clip (no YOLO)"
    )
    ap.add_argument("--snap", type=float, default=None, help="snap time (s); detected if absent")
    ap.add_argument("--los", type=float, default=None, help="LOS field x; inferred if absent")
    ap.add_argument("--offense", choices=["left", "right"], default=None, help="inferred if absent")
    ap.add_argument("--ball-y", type=float, default=None, help="ball field y; inferred if absent")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "still_shown.joblib")
    ap.add_argument("--yolo", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or CLIPS_DIR / f"{args.video.stem}_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_tracked is not None:
        tracked_dir = args.from_tracked
    else:
        if None in (args.calib, args.anchor, args.start, args.end):
            raise SystemExit("--calib, --anchor, --start and --end are needed to track")
        keyframes = {}
        for kf in args.keyframe:
            t_s, path = kf.split("=", 1)
            keyframes[float(t_s)] = FieldCalibration.load(path)
        clip = track_frames(
            args.video,
            args.start,
            args.end,
            FieldCalibration.load(args.calib),
            args.anchor,
            keyframes,
            args.step,
            args.yolo,
            args.conf,
            args.imgsz,
        )
        tracked_dir = out_dir / "tracked"
        clip.save(tracked_dir)
        print(f"{len(clip.frames)} frames at {args.step}s tracked -> {tracked_dir}")
        for n in clip.notes:
            print("  !", n)
    clip = TrackedClip.load(tracked_dir)  # every read runs from the saved frames

    bundle = None
    if args.model.exists():
        from nfl_coverage.models.persist import load_bundle

        bundle = load_bundle(args.model)
    try:
        read = read_tracked(clip, args.snap, args.los, args.offense, args.ball_y, bundle=bundle)
    except ReadError as e:
        raise SystemExit(str(e)) from e

    det = read.snap
    if det.frame is None:
        print(f"snap: not detected [{det.confidence}]")
    else:
        print(
            f"snap: detected at {det.t:.1f} s [{det.confidence} confidence, score {det.score:.2f}]"
        )
    for r in det.reasons:
        print("    -", r)
    print(f"using the {read.snap_source} snap at {read.snap_t:.1f} s (frame {read.snap_index})")
    f = read.formation
    print(
        f"formation at the snap: offense {f.offense} [{f.offense_confidence}], "
        f"LOS x={f.los:.1f} (range {f.los_range[0]:.1f}-{f.los_range[1]:.1f}) "
        f"[{f.los_confidence}], ball y="
        + (f"{f.ball_y:.1f} [{f.ball_y_confidence}]" if f.ball_y is not None else "?")
    )
    for r in f.reasons:
        print("    -", r)
    for w in read.warnings:
        print("    !", w)
    print(
        f"using offense {read.offense}, LOS x={read.los:.1f}, ball y="
        + (f"{read.ball_y:.1f}" if read.ball_y is not None else "none")
    )
    d = read.to_dict()
    print(
        f"snap frame: {d['sides']['offense']} offense, {d['sides']['defense']} defense tracks; "
        f"{d['sides']['line']} on the line, {d['sides']['non_players']} non-players"
    )
    tracks = read.tracks
    tracks.to_parquet(out_dir / "tracks.parquet", index=False)
    print(
        f"canonical tracks: {tracks['nfl_id'].nunique()} players x rel_frame "
        f"{tracks['rel_frame'].min()}..{tracks['rel_frame'].max()}"
    )
    rot = read.rotation
    print(
        f"\nplayed shell read at rel_frame {read.played_frame} "
        f"(+{read.played_frame * clip.step:.1f} s, deep line {read.run_deep:.1f} yd)"
    )
    print(
        f"shell_rotation: {rot['shown_shell']} -> {rot['run_shell']}: "
        f"{rot['shell_rotation_label'] or 'no rotation'} (score {rot['shell_rotation_score']:.2f}); "
        f"biggest mover track {rot['rotator_id']:.0f} moved {rot['rotator_dd']:+.1f} yd in depth; "
        f"{rot['n_came_down']:.0f} came down, {rot['n_bailed_inside']:.0f} bailed inside, "
        f"{rot['n_corner_bail']:.0f} corner bails"
    )
    print("\n" + read.report.summary())
    print("\npost-snap movement features:")
    for k, v in read.features.items():
        print(f"    {k:<24} {v:.2f}")

    png = out_dir / "tracks.png"
    render_play(
        tracks.assign(dataset="clip", coverage_type="?", man_zone="?"),
        str(png),
        note="tracked",
        mark_frame=read.played_frame,
    )
    cv2.imwrite(str(out_dir / "snap_annotated.jpg"), annotate_snap(clip, read))
    (out_dir / "result.json").write_text(json.dumps(d, indent=2, allow_nan=False))
    print(f"\ntop-down render, annotated snap frame, result.json -> {out_dir}")


if __name__ == "__main__":
    main()
