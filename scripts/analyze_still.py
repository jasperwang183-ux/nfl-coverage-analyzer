"""Rung 2 end-to-end: pre-snap screenshot -> "shown" coverage prediction, with QA.

Pipeline: YOLO person detection -> foot points -> homography (calibration JSON from
scripts/calibrate_still.py) -> jersey-color team clustering (referees rejected as
color outliers, offense/defense decided per cluster, not per player) -> line-of-
scrimmage box filter (drops OL/DL so the feature distribution matches the
coverage-only tracking data) -> canonical rel_frame-0 frame -> snap_* features ->
persisted still_shown model. The analyst then accepts or corrects the call (skip
with --no-review); every review is appended to data/processed/still_reviews.jsonl
with the play's features so corrections become labeled examples for retraining.

Usage:
  python scripts/analyze_still.py shot.png --calib shot.calibration.json \
      --los 45 --offense left
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nfl_coverage.analysis.review import REVIEWS_PATH, record_review, resolve_coverage
from nfl_coverage.config import COVERAGE_CLASSES, MODELS_DIR
from nfl_coverage.features.alignment import KEYS, snapshot_features
from nfl_coverage.models.persist import load_bundle, predict_proba
from nfl_coverage.vision.detect import dedupe_mask, detect_people, foot_points
from nfl_coverage.vision.homography import FieldCalibration, on_field
from nfl_coverage.vision.still import frame_from_points, line_player_mask
from nfl_coverage.vision.teams import cluster_teams, sides_from_clusters, torso_colors

STATUS_COLORS = {  # BGR
    "Offense": (60, 180, 60),
    "Defense": (60, 60, 230),
    "Line": (150, 150, 150),
    "Referee": (0, 215, 255),
}


def review_prompt(ranked: pd.Series) -> str:
    """Ask the analyst to accept or correct the model's call; returns the final class."""
    print("\nQA: press Enter to accept, or type the correct coverage")
    print(f"    (e.g. 'cover 2 man'; options: {' '.join(COVERAGE_CLASSES)})")
    while True:
        raw = input(f"shown = {ranked.index[0]} ({ranked.iloc[0]:.0%}) -> ").strip()
        if not raw:
            return ranked.index[0]
        try:
            return resolve_coverage(raw)
        except ValueError as e:
            print(f"  {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="calibration JSON for this camera")
    ap.add_argument("--los", type=float, required=True, help="line of scrimmage x (yards, 0-120)")
    ap.add_argument(
        "--offense",
        choices=["left", "right"],
        required=True,
        help="side of the LOS the offense occupies, in field coordinates",
    )
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "still_shown.joblib")
    ap.add_argument("--yolo", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--no-review", action="store_true", help="skip the QA prompt (accepts the model's call)"
    )
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="do not append to the review log (for experiments/re-runs of a reviewed play)",
    )
    args = ap.parse_args()
    out = args.out or args.image.with_name(args.image.stem + "_analyzed.png")

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    calib = FieldCalibration.load(args.calib)
    detections = detect_people(args.image, args.yolo, args.conf)
    keep = on_field(calib.to_field(foot_points(detections)))
    detections = [d for d, k in zip(detections, keep) if k]
    field_xy = calib.to_field(foot_points(detections))
    unique = dedupe_mask(field_xy, [d.conf for d in detections])
    detections = [d for d, k in zip(detections, unique) if k]
    field_xy = field_xy[unique]
    print(
        f"{len(detections)} detections on the field "
        f"(of {len(keep)} total, {int((~unique).sum())} duplicates dropped)"
    )

    # jersey-color team split; referees fall out as color outliers
    labels, is_player = cluster_teams(torso_colors(img, [d.box for d in detections]))
    sides = sides_from_clusters(labels[is_player], field_xy[is_player, 0], args.los, args.offense)
    status = np.where(is_player, "", "Referee").astype(object)
    status[is_player] = sides

    # canonical frame for players only; drop both lines to match the coverage-only
    # player population the model trained on
    frame = frame_from_points(field_xy[is_player], args.los, args.offense, side=sides)
    line = line_player_mask(frame)
    status[np.flatnonzero(is_player)[line.to_numpy()]] = "Line"
    analysis = frame[~line.to_numpy()]
    n_off = (analysis["side"] == "Offense").sum()
    n_def = (analysis["side"] == "Defense").sum()
    print(
        f"{n_off} offense + {n_def} coverage defenders analyzed | dropped "
        f"{int(line.sum())} line players, {int((~is_player).sum())} non-players"
    )

    feats = snapshot_features(analysis)
    ranked = predict_proba(load_bundle(args.model), feats).iloc[0].sort_values(ascending=False)
    print("\nshown coverage (top 3):")
    for cls, p in ranked.head(3).items():
        print(f"  {cls:<14} {p:.1%}")

    final = ranked.index[0] if args.no_review else review_prompt(ranked)
    if not args.no_log:
        record_review(
            image=str(args.image),
            predicted=ranked.index[0],
            final=final,
            probabilities=ranked.to_dict(),
            features=feats.iloc[0].drop(KEYS).to_dict(),
        )
        if final != ranked.index[0]:
            print(f"corrected to {final} (logged to {REVIEWS_PATH.name} for retraining)")

    for det, st in zip(detections, status):
        x1, y1, x2, y2 = map(int, det.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), STATUS_COLORS[st], 2)
    if final == ranked.index[0]:
        label = f"shown: {final} ({ranked.iloc[0]:.0%})"
    else:
        label = f"shown: {final} (corrected; model said {ranked.index[0]})"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.imwrite(str(out), img)
    print(f"\nannotated -> {out}")


if __name__ == "__main__":
    main()
