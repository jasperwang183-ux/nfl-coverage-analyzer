# Session Handoff — 2026-09-11

## What landed

- **Rung-2 still-image thin slice** (PR #1, merged): screenshot → YOLO foot points →
  `FieldCalibration` homography → canonical rel_frame-0 frame → persisted snap-feature
  model (`models/still_shown.joblib`, 54.8% holdout vs 33.5% majority). Synthetic
  validation: 100 real plays round-tripped through a fake camera give 100% top-1
  agreement with the direct tracking path.
- **QA loop** (`3ffcc45`): `analyze_still.py` prompts accept/correct after each call;
  every review is logged to `data/processed/still_reviews.jsonl` with the play's snap
  features, so corrections accumulate as retraining data (`nfl_coverage.analysis.review`).
- **First real-screenshot run** (KC @ CIN, elevated angle, `data/stills/` — gitignored):
  calibration 0.39 yd reprojection error on 13 painted-marking points; model said
  COVER_1_MAN 31.9%, analyst corrected to COVER_3_ZONE (27.6% — model's #2). Logged.
  Annotated output: `data/stills/kc-cin_presnap_analyzed.png`.

## Key findings from the first real run (next session's priorities)

1. **Domain gap — the big one.** The still feeds *all 22 players + the referee* into
   features (`snap_n_defenders = 16`), but the tracking datasets carry **coverage players
   only** (training mean 7.2 defenders, no linemen on either side). The model has never
   seen a 10-man box; this out-of-distribution shift likely drove the wrong top-1 call.
   Fix directions: filter detections to coverage-relevant defenders before feature
   extraction (drop bodies within ~1.5 yd of the LOS inside the tackle box ≈ both lines),
   and/or add still-robust features less sensitive to defender count.
2. **Side assignment is fragile near the LOS.** Pure `x ≤ LOS` split flipped several
   Chiefs (TE/wing cluster, a slot WR) to defense — homography error near the line is
   enough to flip anyone on the ball. Fix: cluster jersey colors from the detection crops
   (two dominant team colors) and use LOS only as a tiebreaker/orientation check.
3. **Referee counts as a player.** He's on the field and person-class; needs a filter
   (jersey color = black/white stripes, or the team-color clustering handles it as an
   outlier cluster).
4. **Calibration lessons** (baked into workflow, not code):
   - Users click hash/yard-line intersections well (<0.6 yd); guessed points on
     off-frame sidelines are poison (first attempt: 4.25 yd error from 3 guessed points).
   - A calibration whose points all sit in the hash band extrapolates badly to
     deep-safety depth. The far-side yard numbers (12–14 yd off the sideline, i.e.
     y ≈ 41.3/39.3 top row) are the right upper anchor — added 4 such points by
     measuring digit-edge pixels.
   - Don't eyeball the LOS from the image: project the ball pixel through the fitted
     calibration (eyeball was off by 2+ yd; measured x = 45.2).
   - Candidate automation: `calibrate_still.py` could suggest number-row anchors and
     validate point spread before saving.

## State of the repo

- `main` @ `3ffcc45`, clean, pushed. All 20 tests pass (`pytest tests/`).
- Trained artifact: `models/still_shown.joblib` (regenerate with
  `scripts/train_still_model.py`; needs `data/processed/features.parquet`).
- Real-play artifacts (all gitignored, on this machine only): `data/stills/*` (4 frames
  of one play + calibration JSON + `.bak` of the raw click set),
  `data/processed/still_reviews.jsonl` (1 record).
- The 3 post-snap frames of the play are unused by code so far — they're rung-3 input
  and the evidence base for QA verdicts.

## Suggested next-session plan (in order)

1. Detection filtering: team-color clustering for side assignment + ref rejection +
   lineman exclusion, so the feature distribution matches training. Re-run the KC-CIN
   still and see if the top-1 call flips to the corrected label.
2. More reviews: run the flow on several more screenshots to grow
   `still_reviews.jsonl` and see error patterns (the QA loop makes each play ~2 min).
3. Then the deferred candidates: disguise analysis module productization, or the
   PyTorch sequence model (roadmap #3).

## Environment notes

- Windows + OneDrive venv: native-extension imports (scipy/sklearn) can transiently fail
  with an "Application Control policy" DLL error — retry once before debugging.
- Ruff PostToolUse hook auto-formats `.py` on save; line length 100.
