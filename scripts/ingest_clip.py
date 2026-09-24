"""Rung 3 thin slice: mp4 clip -> sampled frames -> confirmed shown-vs-run shell read.

Flow (each step asks the analyst unless the answer is given on the command line):
  1. sample one frame every --every seconds, save them, and write a contact
     sheet with index / time / motion score so the snap is easy to spot;
  2. the analyst names the pre-snap frame (--pre) and the post-snap read frame
     (--run; about 2 s after the snap is the calibrated read point);
  3. both frames go through the still pipeline with one calibration (fixed
     camera assumed; the field-visibility guard caveats every partial view);
  4. the shell reader compares them and proposes a rotation label; the analyst
     confirms or corrects (disguised? mechanism? note), and the verdict is
     appended to data/processed/disguise_labels.jsonl with the shell evidence.

Usage:
  python scripts/ingest_clip.py clip.mp4 --calib clip.calibration.json --los 45.2 --offense left
  python scripts/ingest_clip.py clip.mp4 --calib c.json --los 45.2 --offense left --pre 0 --run 4 --yes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from nfl_coverage.analysis.disguise import RUN_DEEP_DEPTH, shell_change_label
from nfl_coverage.analysis.harness import LABELS_PATH, MECHANISMS, record_label
from nfl_coverage.config import FIELD_WIDTH, MODELS_DIR, ROOT
from nfl_coverage.features.alignment import snapshot_features
from nfl_coverage.models.persist import load_bundle, predict_proba
from nfl_coverage.vision.clip import (
    analyze_image,
    camera_motion,
    clip_report,
    clip_shells,
    contact_sheet,
    motion_scores,
    sample_frames,
    save_frames,
    transfer_calibration,
)
from nfl_coverage.vision.homography import FieldCalibration

STATUS_COLORS = {  # BGR
    "Offense": (60, 180, 60),
    "Defense": (60, 60, 230),
    "Line": (150, 150, 150),
    "Referee": (0, 215, 255),
}
CLIPS_DIR = ROOT / "data" / "clips"


def ask(prompt: str, default: str | None = None) -> str:
    raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
    return raw or (default or "")


def ask_index(prompt: str, n: int, given: int | None) -> int:
    if given is not None:
        if not 0 <= given < n:
            raise SystemExit(f"{prompt}: {given} out of range 0..{n - 1}")
        return given
    while True:
        raw = ask(prompt)
        if raw.isdigit() and 0 <= int(raw) < n:
            return int(raw)
        print(f"  enter a frame index 0..{n - 1}")


def annotate(image_path: Path, fa, title: str, out: Path) -> None:
    img = cv2.imread(str(image_path))
    for det, st in zip(fa.detections, fa.status):
        x1, y1, x2, y2 = map(int, det.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), STATUS_COLORS[st], 2)
    lines = [(title, 0.9, (255, 255, 255))] + [(f"! {c}", 0.55, (0, 215, 255)) for c in fa.caveats]
    y = 30
    for text, scale, color in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(40 * scale)
    cv2.imwrite(str(out), img)


def open_file(path: Path) -> None:
    try:
        os.startfile(path)  # Windows default viewer; harmless no-op elsewhere
    except (AttributeError, OSError):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="calibration JSON (pre-snap frame)")
    ap.add_argument(
        "--run-calib",
        type=Path,
        default=None,
        help="separate calibration for the post-snap frame (needed when the camera moved)",
    )
    ap.add_argument(
        "--transfer-calib",
        action="store_true",
        help="when the camera moved, derive the post-snap calibration from the pre-snap one "
        "through the frame-to-frame homography (saved next to the frames)",
    )
    ap.add_argument(
        "--force", action="store_true", help="compare the frames even if the camera moved"
    )
    ap.add_argument("--los", type=float, required=True, help="line of scrimmage x (yards, 0-120)")
    ap.add_argument("--offense", choices=["left", "right"], required=True)
    ap.add_argument(
        "--ball-y",
        type=float,
        default=None,
        help="field y of the ball (yards, unflipped); the middle-of-field test is centered on it",
    )
    ap.add_argument("--every", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--pre", type=int, default=None, help="index of the pre-snap frame")
    ap.add_argument("--run", type=int, default=None, help="index of the post-snap read frame")
    ap.add_argument("--yes", action="store_true", help="accept the proposed label without asking")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--yolo", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument(
        "--imgsz", type=int, default=1280, help="YOLO inference size; 1920 for 2560-wide frames"
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "still_shown.joblib")
    ap.add_argument(
        "--run-deep",
        type=float,
        default=RUN_DEEP_DEPTH,
        help="deep threshold (yd) for the post-snap read; 14 at +2 s, ~12 at the throw",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or CLIPS_DIR / args.video.stem
    samples = sample_frames(args.video, args.every, args.start, args.end)
    if not samples:
        raise SystemExit(f"no frames sampled from {args.video}")
    paths = save_frames(samples, out_dir)
    scores = motion_scores(samples)
    sheet_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), contact_sheet(samples, scores))
    print(f"{len(samples)} frames every {args.every}s -> {out_dir}")
    for s, sc in zip(samples, scores):
        print(f"  #{s.index:<3} {s.t:5.1f}s  motion {sc:5.1f}")
    print(f"contact sheet: {sheet_path}")
    if args.pre is None or args.run is None:
        open_file(sheet_path)

    pre_i = ask_index("pre-snap frame index", len(samples), args.pre)
    run_i = ask_index("post-snap read frame index (~2 s after the snap)", len(samples), args.run)
    calib = FieldCalibration.load(args.calib)
    run_calib = FieldCalibration.load(args.run_calib) if args.run_calib else calib

    # one calibration only holds while the camera holds: broadcast cameras pan and
    # zoom after the snap, and a moved camera makes the post-snap read meaningless
    cam = camera_motion(samples[pre_i].image, samples[run_i].image)
    camera_caveats = []
    if cam.moved and args.run_calib is None and args.transfer_calib:
        t = transfer_calibration(calib, samples[pre_i].image, samples[run_i].image)
        run_calib = t.calibration
        run_calib.save(out_dir / f"frame_{run_i:03d}.calibration.json")
        print(
            f"camera moved ({cam.reason}); post-snap calibration transferred through the "
            f"frame homography: {t.n_inliers} inliers, reprojection {t.reprojection_error:.2f} yd "
            f"-> {out_dir / f'frame_{run_i:03d}.calibration.json'}"
        )
    elif cam.moved and args.run_calib is None:
        msg = f"camera moved between frames #{pre_i} and #{run_i} ({cam.reason})"
        if not args.force:
            raise SystemExit(
                f"{msg}.\nCalibrate the post-snap frame on its own "
                f"(scripts/calibrate_still.py {paths[run_i]}) and pass --run-calib, "
                "pass --transfer-calib to derive it from the camera motion, "
                "or --force to compare anyway (the read will be wrong)."
            )
        camera_caveats.append(msg + " — post-snap positions are unreliable")
    else:
        held = "separate calibration" if args.run_calib else "held"
        print(
            f"camera check: {held} ({cam.n_inliers} matched features, shift {cam.shift_px:.0f} px)"
        )

    print("\nanalyzing pre-snap frame...")
    pre = analyze_image(
        paths[pre_i], calib, args.los, args.offense, args.yolo, args.conf, imgsz=args.imgsz
    )
    print("analyzing post-snap frame...")
    run = analyze_image(
        paths[run_i],
        run_calib,
        args.los,
        args.offense,
        args.yolo,
        args.conf,
        reference=pre.teams,
        imgsz=args.imgsz,
    )
    run.caveats = camera_caveats + run.caveats
    center = None
    if args.ball_y is not None:  # canonical frame mirrors y for offense=right
        center = FIELD_WIDTH - args.ball_y if args.offense == "right" else args.ball_y
    shells = clip_shells(pre, run, args.run_deep, center=center)
    label, score = shell_change_label(
        shells["shown"]["n_deep"],
        shells["shown"]["mof_closed"],
        shells["run"]["n_deep"],
        shells["run"]["mof_closed"],
    )

    print(f"\nshown (frame #{pre_i}, {samples[pre_i].t:.1f}s): {shells['shown']['shell']}")
    print(f"run   (frame #{run_i}, {samples[run_i].t:.1f}s): {shells['run']['shell']}")
    for fa, name in ((pre, "pre"), (run, "run")):
        for c in fa.caveats:
            print(f"  ! {name}: {c}")
    proposed = f"{label or 'no rotation'} (score {score:.2f})"
    print(f"\nproposed: {proposed}")

    # the model's variant call on the snap frame, kept apart from the structural reads
    variant: list[tuple[str, float]] = []
    if args.model.exists():
        ranked = predict_proba(load_bundle(args.model), snapshot_features(pre.analysis)).iloc[0]
        variant = [(str(k), float(v)) for k, v in ranked.sort_values(ascending=False).items()]
    report = clip_report(
        pre,
        run,
        args.run_deep,
        variant,
        label,
        score,
        f"frame #{run_i} ({samples[run_i].t:.1f}s of clip)",
        center=center,
    )
    print("\n" + report.summary())

    annotate(paths[pre_i], pre, f"shown: {shells['shown']['shell']}", out_dir / "pre_analyzed.png")
    annotate(
        paths[run_i],
        run,
        f"run: {shells['run']['shell']}  |  {proposed}",
        out_dir / "run_analyzed.png",
    )
    print(f"annotated -> {out_dir / 'pre_analyzed.png'}, {out_dir / 'run_analyzed.png'}")

    if args.yes:
        disguised, mechanism, note = score >= 0.5, ("rotation" if score >= 0.5 else ""), ""
    else:
        open_file(out_dir / "run_analyzed.png")
        disguised = ask("disguised? y/n", "y" if score >= 0.5 else "n").lower().startswith("y")
        mechanism = ""
        if disguised:
            while True:
                mechanism = ask(f"mechanism {MECHANISMS}", "rotation")
                if mechanism in MECHANISMS:
                    break
        note = ask("note (optional)", "")

    result = {
        "video": str(args.video),
        "pre_frame": {"index": pre_i, "t": samples[pre_i].t, "path": str(paths[pre_i])},
        "run_frame": {"index": run_i, "t": samples[run_i].t, "path": str(paths[run_i])},
        "shells": shells,
        "proposed": {"label": label, "score": score},
        "verdict": {"disguised": disguised, "mechanism": mechanism, "note": note},
        "report": report.to_dict(),
        "caveats": {"pre": pre.caveats, "run": run.caveats},
        "camera": {"moved": cam.moved, "shift_px": cam.shift_px, "scale": cam.scale},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    if not args.no_log:
        record_label(
            game_id=0,
            play_id=pre_i,
            dataset=f"clip:{args.video.stem}",
            disguised=disguised,
            mechanism=mechanism,
            note=note,
            scores={"shell_rotation": score},
        )
        print(f"logged to {LABELS_PATH.name}")


if __name__ == "__main__":
    main()
