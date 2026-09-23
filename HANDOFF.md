# Session Handoff — updated 2026-09-16

## Update 2026-09-16: domain-gap fixes landed (`1f7a57a`)

Priority 1 from the plan below is done: `vision/teams.py` (jersey-color clustering,
L down-weighted 0.3x; referees = color outliers; sides by geometry where certain
±2.5 yd of LOS, by cluster color in the contested band), `line_player_mask` (drops
both lines to match coverage-only training data), and field-space detection dedupe.
28 tests pass.

KC-CIN re-run (`--no-log`): 16 mixed "defenders" → **9 clean coverage defenders,
zero cross-team side errors**. Call moved from COVER_1_MAN 32% to **COVER_4_ZONE
31% / COVER_3_ZONE 25%** — a two-high quarters read on a play that *ran* Cover 3
after rotating post-snap. That shown-vs-run disagreement is the disguise signal the
product exists to surface, so this is arguably correct behavior for a shown-model,
not a miss. `data/stills/kc-cin_presnap_analyzed_v2.png` has the annotation.

Next: disguise-analysis module, planned 2026-09-16 as follows. First, on main:
scaffolding (shown/run predictions + play context joins) and the **evaluation
harness** — there are no ground-truth disguise labels, so fix the success
criteria up front (face validity on disguise-heavy defenses, cross-component
agreement, viz_play.py spot-checks of top flags; the KC-CIN play must flag).
Then **branch one worktree session per component**, each implementing the same
interface (canonical frames in → per-play score + evidence out), decomposed by
how real DCs disguise coverages:

1. **Shell rotation** — n_deep / MOF open-vs-closed change, snap → +2.5 s (the
   most common disguise; the KC-CIN play is one).
2. **Corner technique lies** — widest defenders' depth trajectory vs. snap
   alignment (press-then-bail = Cover 3 bail; soft-then-squat = Cover 2 trap).
3. **Pressure bluffs / sim pressures** — box bodies dropping out post-snap
   paired with non-box bodies inserting as rushers.
4. **Late rotation & motion response** — pre-snap-window movement; declares a
   bdb2021-only data dependency (2026 has no pre-snap frames).
5. **Man/zone masking** — man-looking leverage playing zone; composition of
   2 + 4 + man_mirror_frac dynamics.

Merge components, then aggregate into a play-level disguise score **plus a
taxonomy label** (rotation disguise / pressure bluff / technique lie). The A/B
comparison lives at the aggregation layer, evaluated by the shared harness. Remaining vision niggles: 1-2 DL wide of the box y-band survive
the line filter; merged O-line blobs undercount offense (harmless — they'd be
dropped anyway).

## Update 2026-09-23: field-visibility guard landed

`vision/visibility.py` — the model cannot tell a *cropped* player from an
*absent* one (a camera that cuts off a deep safety makes two-high read as
single-high, silently), and the defender count cannot catch it because training
carries coverage players only (mean 7.2). So the guard runs the homography
backwards: the analysis-critical region (full width, 3 yd behind the LOS to 18
yd beyond) is projected into pixel space and must sit inside the image (corners
of a 1 yd depth / 2 yd sideline slack-shrunk region — a ground-plane rectangle
projects to a convex quad, so corners decide). The report describes what *is*
usable as the largest fully visible rectangle containing the LOS (`band`,
`depth_visible`), so slanted image edges give honest caveats ("sideline strip
y < 16 yd off-frame", "deep zone beyond 16 yd off-frame") rather than a
pessimistic per-column minimum. Points behind the camera (negative homogeneous
w, sign taken from the calibration points) count as invisible.
`defender_count_warnings` adds the 5–11 plausibility band.

`analyze_still.py` prints the caveats, stamps `[PARTIAL VIEW]` + the reasons on
the annotated image, draws the required region as a yellow quad, and logs
`caveats` in each review record (`record_review(..., caveats=)`) so caveated
corrections can be down-weighted at retraining. `--strict` refuses instead.

**Finding:** the KC-CIN still is a partial view — the bottom of the frame cuts
the near sideline at y ≈ 16 (a boundary WR/CB pair on that side would be
invisible) and the deep edge reaches ~16 of 18 yd. Both safeties happen to be in
frame, so the Cover 4 / Cover 3 read stands, but every elevated-broadcast still
will trip the near-sideline check; expect the caveat to be the norm, not the
exception, until footage comes from a true all-22 angle.

Longer term: player-dropout augmentation when training the sequence model so
partial views degrade gracefully.

---

# Original handoff — 2026-09-11

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
