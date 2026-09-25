"""Track a play through a clip and read it like tracking data: shown vs played, with evidence.

Runs YOLO + ByteTrack over [--start, --end], chains the anchor calibration to
every kept frame, finds the snap from the line firing (or takes --snap and
reports how far the detector lands from it), reads the formation at the snap
(LOS, offense side, ball y - each overridable by --los / --offense / --ball-y),
assigns sides on the snap frame, writes canonical 10 Hz tracks, and then applies the disguise components exactly as on Big Data Bowl
data: shell_rotation with the rotator named, structural reads at the snap and
at +2 s, and the post-snap movement features.

Usage:
  python scripts/track_clip.py clip.mp4 --calib frame_010.calibration.json --anchor 5.0 \\
      --start 4.5 --end 9.0 --imgsz 1920

--start must precede the snap by at least the detector's still window (0.5 s).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_coverage.analysis.disguise import RUN_FRAME, declared_frame, shell_rotation
from nfl_coverage.analysis.report import structural_read
from nfl_coverage.analysis.viz import render_play
from nfl_coverage.config import FIELD_WIDTH, ROOT
from nfl_coverage.features.movement import post_snap_features
from nfl_coverage.vision.formation import infer_formation
from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.snap import detect_snap, field_positions
from nfl_coverage.vision.track import (
    assign_sides,
    canonical_tracks,
    chain_calibrations,
    snap_bodies,
    track_players,
)

CLIPS_DIR = ROOT / "data" / "clips"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="calibration of the anchor frame")
    ap.add_argument(
        "--anchor", type=float, required=True, help="clip time (s) the calibration was fitted at"
    )
    ap.add_argument("--keyframe", action="append", default=[], help="T=calib.json, extra anchors")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument(
        "--snap", type=float, default=None, help="clip time (s) of the snap; detected if absent"
    )
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--los", type=float, default=None, help="LOS field x; inferred if absent")
    ap.add_argument("--offense", choices=["left", "right"], default=None, help="inferred if absent")
    ap.add_argument("--ball-y", type=float, default=None, help="ball field y; inferred if absent")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument(
        "--run-deep", type=float, default=12.0, help="post-snap deep threshold (yd) on film"
    )
    ap.add_argument("--yolo", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or CLIPS_DIR / f"{args.video.stem}_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = track_players(
        args.video, args.start, args.end, args.step, args.yolo, args.conf, args.imgsz
    )
    times = np.array([f.t for f in frames])
    anchor_i = int(np.argmin(np.abs(times - args.anchor)))
    print(f"{len(frames)} frames at {args.step}s; anchor frame {anchor_i}")
    keyframes = {}
    for kf in args.keyframe:
        t_s, path = kf.split("=", 1)
        keyframes[int(np.argmin(np.abs(times - float(t_s))))] = FieldCalibration.load(path)
    chain = chain_calibrations(frames, FieldCalibration.load(args.calib), anchor_i, keyframes)
    for n in chain.notes:
        print("  !", n)

    det = detect_snap(field_positions(frames, chain.calibrations), los=args.los, step_s=args.step)
    if det.frame is None:
        print(f"snap: not detected [{det.confidence}]: {'; '.join(det.reasons)}")
    else:
        print(
            f"snap: detected at frame {det.frame} ({det.t:.1f} s) [{det.confidence} confidence, "
            f"score {det.score:.2f}]"
        )
        for r in det.reasons:
            print("    -", r)
    if args.snap is not None:
        snap_i = int(np.argmin(np.abs(times - args.snap)))
        if det.t is not None:
            print(
                f"    manual --snap {args.snap:.1f} s (frame {snap_i}): detector {det.t - args.snap:+.1f} s"
            )
        print(f"using the manual snap frame {snap_i}")
    elif det.frame is not None:
        snap_i = det.frame
    else:
        raise SystemExit("no snap: pass --snap, or start the window earlier than the snap")

    bodies = snap_bodies(frames[snap_i], chain.calibrations[snap_i])
    form = infer_formation(bodies.xy, bodies.labels, bodies.is_player)
    print(
        f"formation at the snap: offense {form.offense} [{form.offense_confidence}], "
        f"LOS x={form.los:.1f} (range {form.los_range[0]:.1f}-{form.los_range[1]:.1f}) "
        f"[{form.los_confidence}], ball y="
        + (f"{form.ball_y:.1f} [{form.ball_y_confidence}]" if form.ball_y is not None else "?")
    )
    for r in form.reasons:
        print("    -", r)
    for name, given, inferred in [
        ("--offense", args.offense, form.offense),
        ("--los", args.los, form.los),
        ("--ball-y", args.ball_y, form.ball_y),
    ]:
        if given is None or inferred is None:
            continue
        if isinstance(given, str):
            differs, shown = given != inferred, inferred
        else:
            differs, shown = abs(given - inferred) >= 0.5, f"{inferred:.1f}"
        if differs:
            print(f"    ! {name} {given} given; the formation read says {shown}")
    offense = args.offense or form.offense
    if offense is None:
        raise SystemExit("offense side not readable from the formation: pass --offense")
    los = args.los if args.los is not None else form.los
    ball_y = args.ball_y if args.ball_y is not None else form.ball_y
    print(
        f"using offense {offense}, LOS x={los:.1f}, ball y="
        + (f"{ball_y:.1f}" if ball_y is not None else "none")
    )

    sides = assign_sides(frames[snap_i], chain.calibrations[snap_i], los, offense, bodies)
    n_off = sum(s == "Offense" for s in sides.side_by_id.values())
    n_def = sum(s == "Defense" for s in sides.side_by_id.values())
    print(
        f"snap frame: {n_off} offense, {n_def} defense tracks; {len(sides.line_ids)} on the line, "
        f"{sides.n_referee} non-players, {sides.n_offfield} off-field"
    )
    tracks = canonical_tracks(frames, chain.calibrations, sides, snap_i, offense, args.step)
    tracks.to_parquet(out_dir / "tracks.parquet", index=False)
    print(
        f"canonical tracks: {tracks['nfl_id'].nunique()} players x rel_frame "
        f"{tracks['rel_frame'].min()}..{tracks['rel_frame'].max()} -> {out_dir / 'tracks.parquet'}"
    )

    center = None
    if ball_y is not None:
        ball_c = FIELD_WIDTH - ball_y if offense == "right" else ball_y
        # a post safety on a hash-mark ball sits between the ball and the field's middle
        center = (ball_c + FIELD_WIDTH / 2) / 2
    los_c = (120 - los) if offense == "right" else los

    declared = declared_frame(tracks, RUN_FRAME, args.run_deep, center=center, los=los_c)
    print(f"\nplayed shell declares at rel_frame {declared} (+{declared * args.step:.1f} s)")
    rot = shell_rotation(
        tracks, run_frame=declared, run_deep_depth=args.run_deep, center=center, los=los_c
    ).iloc[0]
    print(
        f"\nshell_rotation: {rot['shown_shell']} -> {rot['run_shell']} at rel_frame {rot['run_frame']:.0f}: "
        f"{rot['shell_rotation_label'] or 'no rotation'} (score {rot['shell_rotation_score']:.2f}); "
        f"biggest mover track {rot['rotator_id']:.0f} moved {rot['rotator_dd']:+.1f} yd in depth; "
        f"{rot['n_came_down']:.0f} came down, {rot['n_bailed_inside']:.0f} bailed inside, "
        f"{rot['n_corner_bail']:.0f} corner bails"
    )
    snap = tracks[tracks["rel_frame"] == 0]
    read0 = structural_read(
        snap,
        los_c,
        center if center is not None else snap.loc[snap["side"] == "Offense", "y"].mean(),
        9.0,
        "snap",
        [],
    )
    print(f"SHOWN [{read0.confidence}]: {read0.shell} -> {read0.family}")
    for r in read0.reasons:
        print("    -", r)
    rf = int(rot["run_frame"])
    later = tracks[tracks["rel_frame"] == rf]
    if len(later):
        read1 = structural_read(
            later,
            los_c,
            center if center is not None else snap.loc[snap["side"] == "Offense", "y"].mean(),
            args.run_deep,
            f"+{rf * args.step:.1f} s",
            [],
            rel_frame=rf,
        )
        print(f"PLAYED [{read1.confidence}]: {read1.shell} -> {read1.family}")
        for r in read1.reasons:
            print("    -", r)
    feats = post_snap_features(tracks, horizon=rf).iloc[0]
    print("\npost-snap movement features:")
    for k, v in feats.items():
        if k.startswith("post_"):
            print(f"    {k:<24} {v:.2f}")
    png = out_dir / "tracks.png"
    tracks["dataset"] = "clip"
    render_play(
        tracks.assign(coverage_type="?", man_zone="?"), str(png), note="tracked", mark_frame=rf
    )
    print(f"\ntop-down render -> {png}")


if __name__ == "__main__":
    main()
