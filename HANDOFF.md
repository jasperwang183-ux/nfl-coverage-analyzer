# Session Handoff — updated 2026-09-24

## Next session (agreed 2026-09-24), in order

1. **The web page.** FastAPI + one page: upload an mp4 -> sampled frames and
   contact sheet -> the user picks the pre-snap frame and the snap time ->
   auto-calibration on that frame with a confirm step for which painted yard
   each detected line is (`calibrate_auto.py` logic; show the landmark
   overlay) -> track -> `CoverageReport` rendered under the playing video
   (structure + confidence + reasons, family, variant, disguise) with the
   annotated frames and the top-down render as the "why". `ingest_clip.py`
   and `track_clip.py` are the server-side sequence; `result.json` is the
   contract for the page.
2. **Snap detection.** The contact-sheet motion score already spikes at the
   snap on both sample clips (0 -> 6-16); a first cut is "first sustained
   motion after a still stretch", refined by the line players' first movement
   once tracks exist. Replaces the manual `--snap`.
3. **Tracker tuning.** Track loss at zooms (clip 2's deep 49er at rel 15);
   try BoT-SORT with appearance re-id, lower confidence for continuation, and
   feed the re-identified ids the same team reference. Then the chained
   calibration can be re-anchored automatically by re-running
   `auto_calibrate` on a frame whenever the camera check trips.
4. **Sequence model.** Two heads on one spatio-temporal transformer (set
   attention across players per frame, temporal transformer over frames),
   trained on the 33K tracking plays; evaluated on the film-confirmed clips in
   `disguise_labels.jsonl`. This is what gets past the 50% variant ceiling.

Sample clips with calibrations are tracked in `samples/sf-lar/` (see its README
for the exact commands that reproduce every number below).

## Product path (confirmed 2026-09-24): upload an mp4, watch it, read the coverage under it

The end product is a site: upload an mp4 -> the clip plays -> under it, the
shown coverage, the played coverage, confidence, why, and what it was
disguised as. Everything under the video is `analysis/report.CoverageReport`
(`ingest_clip.py` already prints it and writes it to `result.json`):

- **structure** (deep count, middle open/closed, corner technique) with its
  own confidence from threshold margins + pipeline caveats — this is the
  reliable read and the "why" (reasons in words);
- **family** the structure implies (Cover 2/4/6 vs 1/3 vs 0);
- **variant** = the model's probabilities, with their honest lower confidence;
- **disguise** verdict in words (shown vs played, mechanism).

Server-side pipeline for one upload = what `scripts/` does today, in order:
sample frames + contact sheet (`vision/clip.sample_frames`) -> the user picks
the pre-snap frame and the snap time (or we detect the snap from motion; the
contact-sheet motion score is the seed) -> auto-calibrate the pre-snap frame
(`vision/fieldlines.auto_calibrate`; the user confirms which painted yard each
line is — a two-click UI) -> track players from the snap (`vision/track.py`)
-> canonical 10 Hz tracks -> disguise components + report. The video player
shows the clip; the report renders below; the annotated frames / top-down
render are the "why" visuals. Not built: the web layer itself (FastAPI +
a page is enough: upload, a frame picker, a line-identity confirm, the
report), snap detection, and the sequence model.

## Player tracking (built 2026-09-24, `vision/track.py`, `scripts/track_clip.py`)

YOLO + ByteTrack over the clip's native frames (identities persist), one frame
kept per 0.1 s; the anchor calibration is chained frame to frame through
consecutive homographies (small motion each step, keyframes re-anchor);
sides are assigned once on the snap frame (color + geometry) and carried by
track id; line players dropped; canonical schema at 10 Hz with speeds from
finite differences. `shell_rotation` then runs on film exactly as on tracking
data, naming the rotator, and `post_snap_features` (man-mirror fraction etc.)
become available on film.

**Declared read (added after the first runs):** the played shell is taken at
the frame from +2 s on with the most deep defenders (min 5 tracked), not at a
fixed instant — a corner bailing to a third crosses the deep line later than a
safety already there. Both reads now use the known LOS (a 0.3 yd LOS estimate
difference flipped a corner at 11.9 vs 12.0 yd), a 4-deep label/family exists
(quarters: no MOF verdict), and on film the middle-of-field center is the
midpoint of the ball and the field middle (a post safety on a hash-mark ball
sits between them). With that, **clip 1 tracked = shown 2-high open -> played
4-deep, Cover 4, corner bails only (score 0.25): matches the analyst.** Clip 2
tracked declares 3 deep at rel_frame 21 (`rotation_up`, one non-corner bailed
deep) — the Cover 3 structure — with the middle verdict still sensitive to the
center; the throw-frame still remains the clean Cover 3 read.

**First runs vs analyst ground truth (clip 1 = Cover 4, clip 2 = man look
disguising Cover 3) — before the declared read the tracked reads did NOT match:**
- clip 1: 22 tracks sided at the snap (10 off / 11 def), both safeties and
  both corners tracked through +3.5 s; +2 s depths 14.6 / 13.4 / 12.6 with the
  fourth corner at 11.9 and the near safety 3.4 yd off the ball -> the reader
  says "3-deep, middle closed" (Cover 3) on a play that is quarters. Knife edge
  on both thresholds; report confidence is low, correctly. The 2 corner bails
  and man-mirror 0.31 are the quarters evidence a better rule would use.
- clip 2 (tracks start at the +1 s freeze): the deep 49er (track 1, 14.9 yd)
  is lost at rel_frame 15 when the camera zooms; re-identification by jersey
  color against the snap-frame team reference (added) did not recover him, so
  +3 s reads 2 deep instead of the 3 the throw-frame still finds.
- The still path (ingest_clip.py, separately calibrated frames) reproduces both
  calls; the tracking path is committed as a first slice for the movement
  features and rotator evidence, not yet as the source of the played call.
  Next: tracker tuning (BoT-SORT with appearance, lower conf on re-id), and a
  quarters-vs-Cover-3 rule that uses the corners' bail and man-mirror, not
  just the 3.5-yd middle band.



## Update 2026-09-23 (latest): first real clips — 49ers @ Rams (Melbourne, 2026 wk 1)

Two screen-captured all-22 clips from a film breakdown video (`data/clips/sf-lar/`,
gitignored). Both are paused-then-played sequences: clip 1 holds the pre-snap
look for 5.5 s then plays (49ers offense, moving left, LOS x=49.9); clip 2
shows the motion first, freezes ~1 s post-snap with telestrator circles, then
resumes at 16.5 s (Rams offense, LOS x=69.9). The frame the analyst calls
"pre-snap" must be chosen with that in mind (`--pre 10 --run 16` for clip 1,
`--pre 16 --run 35` for clip 2).

What the film forced, all committed:
- **Detection size**: YOLO at 640 found 17 of 22 players on a 2560-wide frame;
  `--imgsz 1920` finds all (default now 1280).
- **Auto-calibration from the markings** (`vision/fieldlines.py`,
  `scripts/calibrate_auto.py`): yard lines refit on their pixels, hash rows by
  RANSAC on tick blobs, sideline band edges, number-glyph edges predicted by a
  first pass. 0.14–0.23 yd on 14–20 landmarks per frame, no clicking; the
  analyst passes the painted yard of each detected line (`--yards`). Probe
  first without `--yards`; lower `--min-length/--votes` on zoomed-out frames.
  Number edges self-validate when nothing else constrains the near side
  (clip 2 frame 16 had a bad near fit until a glyph-height check was added) —
  prefer a visible sideline. Calibration *transfer* through the frame
  homography works for small pans but not a 0.73x zoom-out (33 inliers,
  wrong): fit the second frame directly.
- **Jerseys in hard sun**: every torso splits into a lit and a shaded mode;
  the per-channel median of a blue-and-yellow Rams torso is a gray that is
  nobody's color. Descriptor is now both modes (dark, bright) with L weighted
  0.1; officials by *vertical* stripe alternation across torso columns
  (≥ 5 switches); a lone outlier can no longer hijack the 2-means split; the
  contested ±2.5 yd band decides by nearest side-centroid of the geometrically
  certain players, not by cluster vote. Bench feet sit on the sideline paint
  (on-field now needs y ≥ 0.5) and bodies clipped by the frame bottom have no
  feet (dropped).
- **Middle-of-field center on stills** = median y of the line players (the
  deepest offensive player on film is often an offset back).

Results (pre → +2 s, both frames calibrated separately):
- clip 1: **2-high MOF-open → 2-high MOF-open, no rotation**; the post-snap
  frame has only 4 coverage defenders in view (camera follows the play).
- clip 2: **2-high MOF-closed → 2-high MOF-open, `rotate_to_mof_open`** — a
  deep safety over the ball at the snap spins out to a half. Pre-snap frame
  still counts 13 "defenders": two Rams receivers on the far side inside the
  LOS band read as defense (gray-dominant crops) — the residual color error.

**Analyst ground truth (2026-09-24):** clip 1 = Cover 4 (model shown COVER_4
48%, correct); clip 2 = man look disguising Cover 3 (model shown COVER_1_MAN
44%, correct as the *shown* call). At +2 s clip 2 still read two-high open —
the corners were at 6–8 yd, still bailing. At the throw frame (#39, 19.5 s,
auto-calibrated; note the line identities are the 40-45-50-45-40, i.e. x =
50…70 — mislabeling them by one line shifted every depth 5 yd and made the
pocket look like defenders) the read is **3 deep, one 3.5 yd off the ball**:
Cover 3, all three shaded to the field side. So the 14-yd post-snap threshold
calibrated on tracking at 2 s is too deep for thirds at the throw on film;
12 yd reads it. Both verdicts are in `disguise_labels.jsonl`.

Known weak spot: color on small/shaded crops. Next step if it keeps biting: a
chroma-histogram descriptor per torso (mass in blue / yellow / red / neutral
bins) matched to team histograms learned on the pre-snap frame, instead of two
modes. The analyst-confirmation prompts (`ingest_clip.py` without `--yes`) are
untested on these clips; every run so far used `--yes --no-log`.

## Update 2026-09-23 (later): disguise harness, shell reader, clip ingester

Methodology and calibration table: `docs/DISGUISE_PLAN.md`. Decisions that
changed from the 09-16 plan: **no hand-labeling of rendered tracking plays**;
ground truth is (a) the coverage label's implied middle-of-field state, which
scores the post-snap read on every labeled play, and (b) film confirmations
through the clip ingester. Disguise metric = pre-snap read ≠ post-snap read.

What landed:
- `analysis/shell.py` — rule-based shell reader (deep count, MOF open/closed,
  corner depth, box), one frame in, works on tracking frames and stills alike;
  explicit `los`/`center` for unlinked post-snap stills.
- `analysis/disguise.py` — `shell_rotation` (snap vs +2.0 s, per-defender
  evidence, labels `rotate_to_mof_closed/open`, `rotation_down/up` score 1;
  `corner_bail`, `threshold_flicker` score 0.25), `model_disagreement`,
  `shell_change_label` for two unlinked reads.
- `analysis/harness.py` + `scripts/run_disguise.py` — scores every cached week,
  joins play context (`data/context.py`), prints run-shell MOF accuracy vs
  label, flag rates by defense/down/distance, component agreement, audit
  queue; `disguise_labels.jsonl` log + precision/recall when labels exist.
- Week-1 numbers (both seasons, 1,682 plays): run-shell MOF accuracy **0.82**;
  a third of dropbacks flag; shown two-high that flag ran Cover 1/3 82% of the
  time, unflagged two-high ran Cover 2/4/6 76%. Rates by defense/down look
  sane (3rd-and-long > 1st-and-10; MIN/ARI top, NO/LAC bottom in 2018 wk 1).
- `vision/clip.py` + `scripts/ingest_clip.py` — mp4 -> sampled frames +
  contact sheet -> analyst picks pre-snap and +2 s frames -> still pipeline on
  each -> shell comparison -> confirm -> log. **Camera-motion guard** (ORB +
  RANSAC similarity between the two frames) refuses when the camera panned or
  zoomed unless `--run-calib` (a second calibration for the post-snap frame)
  or `--force` is given.

**Post-snap side assignment (fixed after the analyst review of `run_analyzed.png`):**
`sides_from_clusters` decides by position relative to the LOS, which is only
true before the snap — once receivers cross it, geometry labels them defenders.
Post-snap frames now take sides from `TeamColors` (cluster centers + side per
cluster, learned on the pre-snap frame, `vision/teams.py`) by nearest jersey
color only. Officials: black-and-white stripes blur to a chroma-neutral
mid-gray torso (L ≈ 123 on film) that the L down-weighting confused with white
jerseys; `referee_mask` rejects neutral gray torsos below L 190 with dark
stripe pixels. Both applied in `vision/clip.analyze_image`; the still script
uses the referee mask too.

**Calibration transfer (`--transfer-calib`):** a broadcast camera on a fixed
mount pans/tilts/zooms about its center, so frame A -> frame B is an exact
homography; `vision/clip.transfer_calibration` estimates it from ORB matches
and pushes frame A's clicked landmarks through it, refitting frame B's
calibration with no clicks (KC-CIN: 182 inliers, reprojection unchanged at
0.39 yd; landmarks verified on hash rows, the 50 through the logo, the far
"30"). The line filter now takes the snap LOS on post-snap frames (the offense
has moved, so its alignment no longer marks the line).

**Finding on the KC-CIN stills stitched as a clip:** the broadcast camera moved
106 px and zoomed 0.85x between the pre-snap and +1 s frames, so one
calibration cannot serve both; the guard refuses, correctly. With `--force`
the read came out 2-high open -> 2-high closed (a rotation to a post safety),
but that number is not trustworthy. With the transferred calibration the +2 s
frame reads **2-high MOF-open -> no rotation** (two deep Bengals on the far
side, nobody in the deep middle, near-sideline strip y < 14 cropped); the throw
frame is zoomed so tight that only 3 coverage defenders are in frame (read
0-high, caveated). So on this footage the Cover 3 rotation the analyst saw is
not visible in the two later frames the camera gives us — either it happened
off-frame on the cropped near side or after +2 s. A held all-22 camera is
still the honest test.

Next steps, in order: (1) a real mp4 from the reference videos, calibrated on
both frames, through `ingest_clip.py`; (2) mechanism components in worktrees
(`corner_technique` first — off-corner cushion collapse and press-bail are
readable from tracking now); (3) sequence model with two heads once film
confirmations exist to evaluate against. `analyze_still.py` still has its own
copy of the detection pipeline; `vision/clip.analyze_image` is the function
form — fold the script onto it when next touched.

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
