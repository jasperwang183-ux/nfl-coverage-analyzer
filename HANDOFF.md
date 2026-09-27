# Session Handoff — updated 2026-09-27

## Sequence model on the cards, and the play window (2026-09-27, later)

**Cards.** Each card carries its own sequence-model head (`CoverageReport.pre_model` /
`post_model`, a typed `ModelRead`): the pre-snap card the look up to the snap, the
post-snap card snap to +3 s, each with its top 3, a level and man/zone. The snap-frame
gradient boosting model stays in `report.variant` for disguise, still and verify, and is
the card's fallback when `models/sequence_v1.pt` is missing. Film reads rebuild the tracks
with the line kept for the model (it trained on all 22; its LOS estimate needs the offense)
and skip it off a 0.1 s grid. Corrections log `model_variant`; verify film records
`seq_pre_top` / `seq_post_top`. Clip 1 reads Cover 4 on both heads (88%, 96%), the user
marked both cards correct. Open: the structural level (crop, margins) and the model's level
(top vs second) can disagree on one card; capping the model's level at the structural one
is an option the user has not picked.

**Play window.** `first_motion` took the first two moving samples as the play. On the
Dolphins wide shot that is a camera pan at 5.5 s (peak 8), so the window (4-11 s) missed
the 15.6 s snap and the page stopped at needs_input. It is now the first run peaking at
`PLAY_PEAK` (10), else the first run; the window is built around it (`play_window`) and
the calibration is carried when it sits outside. Probe over the manifest: every clip keeps
its old window except the Dolphins wide shot, which now reads (snap 15.56 s). The strongest
run is not the rule: 1532's replay outmoves its play. Clip 2 still misses (two full plays
in one clip; motion cannot tell which) and is not on the page.

**The page's clips.** Results are written once and never recomputed, so clips from before
the rework showed old reads or waited on manual naming. Those 6 and 2 failed runs moved to
`data/web_archive/` (nothing deleted). The page now holds clip 1, 1326, 1529, 1533 and the
Dolphins wide shot, all run with this code. 1326 and 1533 are there for the user to check
against the film (snap, offense side, shells) with Correct / Wrong. A result version stamp
would flag stale clips automatically; not built.

## Sequence model scored on the frozen holdout (2026-09-27)

Branch `worktree-agent-a30822969fc9406b3` (remote `sequence-model`), not merged. Leakage
review found none: `train_sequence.splits` drops the holdout weeks before choosing
validation games, early stopping sees validation only; the pre-snap head is causal
(cutting every post-snap frame changed its probabilities by exactly 0.0 on week 2);
labels match `features.parquet` (8 classes). Trained on the RTX 5070: 40 epochs, best
epoch 33, 296 s -> `models/sequence_v1.pt`. `verify.py bdb --checks sequence` (tag
`sequence-v1`), scored ONCE, n = 8,234 holdout plays:

| model | top-1 | top-2 | top-3 |
|---|---|---|---|
| majority class | 0.328 | | |
| variant model (snap features) | 0.518 | 0.746 | 0.866 |
| pre-snap head | 0.592 | 0.807 | 0.909 |
| full-play head | 0.865 | 0.970 | 0.992 |

Pre-snap 0.63 (2018) / 0.55 (2023: inferred snap, single pre-snap frame); full-play 0.87
both. Biggest gains on Cover 6 (0.07 -> 0.80 full) and Cover 2 zone (0.26 -> 0.89);
Cover 0 is the one class the variant model still wins pre-snap (0.68 vs 0.45).
The heads disagree on 39.7% of plays; then the full-play head is right 80%, the pre-snap
head 11%; when they agree the call is right 91%. Of 3,695 plays with the pre-snap head
>= 60% confident, 21.7% change call and the full-play head is right on 79% of those:
the disguise signal. Do not re-score on the holdout.

Next: merge, wire `predict()` into `read_tracked` (pre-snap head -> pre-snap card, full-play
head -> post-snap card, probabilities replace the structural reads' coarse levels), and
run `verify.py film` before and after: the model has never seen real film tracks, and a
clip starting after the snap gets a pre-snap read from its first frame.

## Automatic ingest: naming without a person, and the page plan (2026-09-27)

**Naming QA.** Claude's 7 manifest namings were checked against the painted numbers
(a number straddles its own line). 1326 was 5 yd off: its red line is the 5, not the 10
(the leftmost line borders the end zone, so it is the goal line). Fixed in the manifest.
The other six are right. 1533's calibration frame carries telestrator circles.
The 5-yd shift warning missed 1326 (4 number edges vs 2, the margin is 4).

**Naming is automatic now** (`fieldlines.auto_name`). A read is translation-invariant
along the field: depth, shells, MOF, disguise and the variant all measure from the LOS.
So the absolute yard is arbitrary (the middle line goes near midfield), and only the
parity matters (which lines are the 10s, where the numbers are). Both parities are fitted
and the one whose 10s find more number edges wins, then more field rows, then smaller
error. It picks the confirmed parity on all 11 manifest clips (a tie on 1530 and the Dolphins
wide shot is settled by rows or makes no difference). `verify.py film --auto-naming`
runs the manifest that way and translates the manifest's LOS facts into its frame.

**Two fits depended on where the field origin sits**, found because auto naming moved
it. (1) `FieldCalibration.from_points`: the hash-only Dolphins wide fit moved ~0.4 yd
with a naming slid 60 yd, enough to flip band settings across the search's error cutoff.
A spurious 4-row fit then won, and the read found no snap. It is now fitted around the
points' own center (test: `tests/fixtures/hash_only_landmarks.json`, the real landmarks).
(2) The 1-yd tick fit `_fit_p1` (`u = (ak+b)/(ck+1)`): the `+1` is not translation-invariant
in k, so 1531b's tick hashes failed at some placements. k is now centered. After both, each
clip gives one identical fit at every placement from -100 to +100 yd.

**Verify.** Before the fixes, auto naming read 8/11: the Dolphins wide shot regressed, and
1531a (no snap) and 1531b (5 offense) fail as they did before. With the fixes,
`origin-fix` (manual) and `origin-fix-auto` (auto) differ ONLY in the arbitrary LOS, and
every LOS shift is a whole multiple of 10 yd (a wrong parity would shift by 5), so auto
naming agrees with the analyst on all 10 clips that reach a LOS; 9/11 read, the Dolphins wide shot is back, and 1531a/1531b
fail as before. Against `reanchor-x`, two reads changed and neither has a known truth:
1326 (corrected naming): offense left, 10 offense, snap 6.26 s, 2-high closed shown,
1-high closed played. 1533 (tick fit): 10/11 bodies (was 11/9), snap 29.25 s, played
1-high closed -> 2-high open. Check both against the film.

**The page (merged to main 2026-09-27 from `page-auto-ingest`).**
1. Automatic ingest. Upload an mp4 and the page picks the frame, names the lines and
   calibrates on its own. It falls back to the manual naming only when that fails
   (up to 3 still frames tried, nearest the suggested one first). The cap was 6; a probe
   over all candidates showed every manifest clip names on the first frame at ~5 s a try,
   and neighbouring still frames fail together, so 3 asks the person after ~15 s.
2. Pre-snap card. Call, confidence, then why the confidence is not high
   (`StructuralRead.confidence_why`: serious caveats plus any margin under 2 yd), reasons.
3. Post-snap card. The frame +2 to +5 s after the snap with the most defenders deep (the
   user agreed to 2-5 s: coverage is read before the throw). Call, confidence, reasons, the
   model's top 3 variants, the rotator's move.
4. Correct / Wrong on each card. Wrong requires a call (dropdown + note). Every save is
   appended to `data/processed/coverage_corrections.jsonl`, the film-confirmation loop.
5. The manual controls (frame, naming, window, overrides) sit in a collapsed Adjust section
   that opens on needs_input.
6. The absolute LOS is no longer shown as a yard line (auto naming does not know it;
   reading the digits would).

Stage table in `app.py` (`STAGES`, `ClipStore.advance` rejects illegal moves): uploaded ->
calibrating -> tracking -> reading -> done | needs_input{calibration|snap|offense} | error.
Clip 1 through the real server: done in ~1 min, 2-high open shown, Cover 4 played at +2.2 s,
Cover 4 50%. Both structural reads are "low" there. Decided: the card leads with the call
and the why follows the meter. Clip 1's pre-snap read is low from a 16% crop and the deep
zone cut at 14 yd; the post-snap read from a defender 0.4 yd off the deep line.

Open: a 2% crop or a 3-yd sideline strip counts as a serious caveat, the same as a 16%
crop. Harmless on clip 1 (the margin forces low anyway) but it can cost a clean read a
level. A minimum crop before a caveat counts needs a verify run before it lands.


## Verify harness, GPU, re-anchoring, the overlay (2026-09-26)

**Verify harness committed** (`scripts/verify.py`, `vision/verify.py`, `analysis/holdout.py`,
`.claude/skills/verify`). `verify.py film` runs the 11 manifest clips stage by stage;
`verify.py bdb` scores the frozen holdout; `compare A B` diffs runs. Use it before any
vision/analysis change (the skill says how). `data/verify/` and `.claude/worktrees/` are
gitignored now.

**CUDA torch.** The venv had `torch 2.14.0+cpu`. Installed `2.14.0+cu130` +
`torchvision 0.29.0+cu130` (the cu128 index stops at 2.11; the RTX 5070 needs CUDA 12.8+).
YOLO at imgsz 1920: 0.22 s -> 0.017 s per frame, same boxes. Tracking a clip is ~30 s now.

**Calibration re-anchoring** (`pipeline.reanchored_chain`, `fieldlines.grid_drift`,
`fieldlines.refit_from_paint`). Frame-to-frame chaining drifted up to 2.2 yd over a clip
(1531a), putting real players off the field. About once a second the chain is checked
against the frame's own yard lines; past 0.4 yd it is refit on that frame's paint. What
the session learned the hard way:
- A correction applied in one frame moves every body at once and the snap detector
  reads it as the line firing (Dolphins wide 15.6 -> 14.9 s, 1326 -> 4.96 s). The
  correction is now spread linearly over the frames since the previous anchor.
- A full refit on a hash-only frame also moves depth (y) by up to a yard with nothing to
  check it; blended, that still made every body drift and fired a false snap at 14.2 s.
  The correction is now along the field only (`along_field_correction`: x -> ax+by+c, y
  kept) because that is all the yard lines measure. Verified (`reanchor-x` run): the
  Dolphins snap is back to 15.63 s on both clips (the truth, read off the frames), clip 1
  and clip 2 unchanged, 1531a grid 2.18 -> 0.06 yd, 1326 now reads (3 -> 7 offense) with
  its snap moved 8.05 -> 4.26 s (not checked against the frames: do that first).
  9/11 clips produce a read (1531a: no snap; 1531b: 5 offense).
- Subtracting the off-line bodies' median motion in the snap detector was tried and
  reverted: at a real snap the QB and receivers move too, and it made the snap 2 frames late.

**Snap: the line must be seen set** (`snap.SET_FRAMES`). A fire with no still line
frames before it is skipped (a line already moving when the window starts is not a snap).

The subtraction run (`snap-common`) found a snap on 1531a (4.33 s) that the final code
does not; it came from the reverted change, so 1531a still needs its snap given or a
better detector.

**The page.**
- The read is drawn on the playing video: `vision/overlay.py` writes `overlay.json`
  (per-frame boxes as [x1,y1,x2,y2,kind], kind O/D/S(deep)/L/X, and the LOS polyline);
  a canvas over the `<video>` draws the nearest tracked frame and a SHOWN/SNAP/PLAYED label.
  Checked on clip 1 by rendering overlay.json on the snap/played frames: 2 deep at the
  snap, 4 deep at +2.2 s. **Not yet watched playing in a visible browser tab** (the
  automation tab was hidden, so Chrome never loaded the media).
- Timeline under the video: window, snap, played read; click to seek. "here" buttons set
  the window ends from the video time.
- A read that stops asks for what it needs (`ReadError.needs`: snap / offense; a carry that
  fails: calibration) and re-reads from the tracking cache.
- Tests: `tests/test_overlay.py`, needs-snap in `test_web.py`, and `tests/test_page_e2e.py`
  (slow: YOLO on samples/sf-lar/clip1.mp4 through the API, checks the analyst's facts;
  `pytest -m "not slow"` skips it).

**Sequence model (roadmap 4), in progress on branch `worktree-agent-a30822969fc9406b3`**
(commits 296e8f2 model, 4c8f358 training script; worktree `.claude/worktrees/agent-...`).
`models/sequence_try1.pt` trained 224 s on the GPU. Validation (non-holdout weeks):
full-play variant top-1 0.86 / top-2 0.97, man/zone 0.92; pre-snap-only top-1 0.59.
The existing snap-feature model is 0.52 top-1 on the holdout, so this looks like a big win
but it has NOT been scored on the holdout yet. Next: review the branch, score once on the
holdout, merge, wire `predict()` into `read_tracked` (pre-snap vs full-play = disguise).

**Open.** 1531b reads 5 offense (the Rams' line merges into a few boxes: detection, not
tracking); 6 manifest clips have yard-line namings Claude read off the numbers, awaiting
the user's confirmation (`scratchpad/naming_check.html` was shown); Dolphins wide grid
drift is perspective, not a slide, so the x-only correction leaves it at 0.82 yd.

## Calibration and formation hardened on new clips (2026-09-25, later)

The page failed on the user's other clips (Dolphins @ Chargers 2023 wk 1 all-22, and four
more 49ers-Rams captures): an internal error (0 yard lines found -> IndexError), fits
"that can't be fit", 5-yd naming that broke on a missed line, and reads confused by
non-players. Causes and fixes, all in `vision/fieldlines.py` unless noted:

- **Yard lines were thrown away for their slant.** Only lines within 30 deg of vertical
  were kept; near a goal line or zoomed in, yard lines run 35-55 deg off vertical. Any
  long line that is not near-horizontal is a candidate now (`MIN_LINE_ANGLE`). Lines
  found per frame went from 0-3 to 5-12 on the new clips.
- **Paint mask** (`paint_mask`): white top-hat (paint in roof shadow still stands out)
  or plain brightness, unsaturated, inside the largest turf region and next to turf.
  The video player's title/controls/cursor and letterbox text are gone; telestrator
  circles, broadcast yellow/blue lines and the orange down marker never passed.
- **Gap-aware line indexing** (`index_lines`): spacing in local 5-yd units; a double
  spacing is a missed line, an off-integer spacing drops the stray line (the edge of
  the end-zone paint on the Dolphins wide shot). The person names ANY line (painted
  number + direction), not "line 1"; `field_x_for_lines`.
- **Hash rows from 1-yd ticks** (`find_tick_rows`): a candidate row is scored by how
  many ticks sit on the 1-yd slots predicted from its crossings with the indexed yard
  lines. Every matched tick is a landmark (field x at the whole yard) - 20-60
  landmarks per frame instead of 2 per yard line. Tick rows also run inside both
  sidelines and junk rows appear in bench areas, so the search tries each adjacent pair
  as the hashes and lets the sidelines/numbers decide.
- **Painted numbers straddle their line** (a digit each side): `number_edges` requires
  paint on both sides, so a 5-yd line no longer borrows its neighbor's digits. That
  makes a 5-yd mislabel visible: the page refits with the labels shifted 5 yd and warns
  when the numbers clearly prefer the shift (5 of 5 Melbourne frames pick the true
  labeling this way).
- **Hash-only fits are accepted** (user decision: tight all-22 zooms often show nothing
  else) with a depth warning that follows the read as a caveat; >=3 field rows are
  still preferred when the frame has them. Under 8 landmarks is refused.
- **Window no longer has to contain the calibrated frame** (`pipeline.carry_calibration`):
  hop 0.25 s at a time through frame-to-frame homographies, then refit on the window's
  first frame from its own paint, labeled by the carried calibration.
- **Formation: a lone deep official flipped the offense side** (`formation.
  offense_from_depth`): the Dolphins clip's referee stood in the end zone 14 yd behind
  the offense and the striped-shirt check missed him. Deepest + second-deepest per
  side; when they disagree the second-deepest decides at low confidence and the lone
  body is set aside as an official (`Formation.suspect_official`). Week 1 BDB: 98.6% /
  97.9% right clean (vs 99.1 / 98.2 deepest-only), 76% / 70% with a planted referee
  (vs 52% / 28%).
- **At most 11 per side** (`track.cap_sides`): similar kits (powder blue vs white/aqua)
  put 13 on defense; the surplus moves back across from the color-decided bodies.

Validation: 17 frames (11 new, 6 Melbourne) all calibrate; the 5 good Melbourne
calibrations agree within 0.1-0.25 yd median with the true labels; grids checked by eye
on the Dolphins close-up and wide shot and the tight 49ers zooms. A synthetic field seen
through a known camera is recovered within 0.3 yd median (`tests/test_fieldlines.py`).
Dolphins clip end to end on the page's API, calibrated at 5.0 s, window 12-17 s: carried
in 28 hops, refit on 34 landmarks (hash-only, caveated), snap 15.6 s (high), offense left
(low: the official), LOS 19.6 (high), 10 offense / 11 defense, 2-high open shown and
played, no disguise, variant Cover 1 man 38%. No analyst call for that play yet.
Clip 1 unchanged (Cover 4, corner bail). 125 tests.

Still weak: jersey color between light kits; the striped-shirt check on small officials.

## The page (first slice built 2026-09-25)

`python scripts/serve.py` -> http://127.0.0.1:8000. Upload -> 0.5 s thumbnails with a
motion bar, the last still sample before sustained motion suggested -> yard lines found
and numbered on the chosen frame -> the user names line 1 (painted number + which way the
numbers run; per-line field x as a fallback) -> `fieldlines.search_calibration` (~40 band
settings in ~5 s: among fits reproducing their landmarks to < 0.3 yd mean / 0.8 max, the
one on the most distinct field rows wins) -> the fitted field grid drawn back on the
frame for the confirm -> Analyze (background thread, progress polled) ->
`vision/pipeline.py`: `track_frames` (cached under `tracked/` as JPEG + JSON) then
`read_tracked` (snap, formation, sides, tracks, shells, rotation, variant model,
`CoverageReport`) -> result under the video: shown/played cards, variant bars, disguise,
caveats, snap & formation table with overrides (re-read in seconds from the cache),
annotated snap frame, top-down render. Manual snap times are logged next to the
detector's call (`data/processed/snap_labels.jsonl`). `track_clip.py` is now a thin CLI
over the same pipeline (`--from-tracked` re-reads a cache). 117 tests pass (13 new for the
page on synthetic video).

**Clip 1 through the page, driven in the browser:** 3 lines found at 5.5 s, named 35 up;
fit on 10 landmarks / 6 rows, 0.20 yd; grid checked by eye on the painted lines, hashes,
far sideline and near numbers; tracking 2 min 20 s; snap 5.5 s detected (high), offense
right, LOS 49.0, ball y 26.1 -> 2-high open shown, 4-deep Cover 4 played, corner bail
only (0.25): the analyst's read. Snap override to 5.6 s re-read and logged; "back to
inferred" restored it.

**Fixed on the way:**
- The first page run called a 1.0 `rotation_up`: the line judge stood 0.6 yd inside the
  far sideline under the searched calibration, was sided as the widest defender, took
  the corner's role and made the corner's bail a non-corner bail. Snap-frame bodies within
  2 yd of a sideline are now not players (`track.SNAP_SIDELINE_CLEAR`; of ~24K snap
  positions in Big Data Bowl week 1 none is within 0.5 yd, one within 3).
- Cached calibrations kept only the matrix; the visibility check takes the camera side
  from the point pairs and called everything off-frame. The points are saved now.
- `corner_bail` / `threshold_flicker` got plain-language disguise sentences.

**Found, not fixed: `samples/sf-lar/calibrations/clip2_frame_035_17.50s.calibration.json`
is several yards wrong below the hashes** (its landmarks are all on the far half: rows
29.7, 39.3, 41.3, 53.3). The search refits that frame on 18 landmarks / 7 rows at 0.15
yd, and its grid sits on the painted lines where the saved one drifts 4-14 yd. That file
is a keyframe in clip 2's tracked read, so that read is suspect until it is refitted.
The other five sample calibrations agree with the search within 0.3 yd.

**Not in this slice:** overlay drawn on the playing video itself; the still path
(`ingest_clip.py`) behind the page; a window picker on the video (start/end are typed,
defaulted from the suggested frame -1 s / +4.5 s); multi-user job isolation (one YOLO job
at a time, local use only).

## Next session (agreed 2026-09-24), in order

1. **The web page - first slice done 2026-09-25 (see above).** Original plan: FastAPI + one page: upload an mp4 -> sampled frames and
   contact sheet -> the user picks the pre-snap frame and the snap time ->
   auto-calibration on that frame with a confirm step for which painted yard
   each detected line is (`calibrate_auto.py` logic; show the landmark
   overlay) -> track -> `CoverageReport` rendered under the playing video
   (structure + confidence + reasons, family, variant, disguise) with the
   annotated frames and the top-down render as the "why". `ingest_clip.py`
   and `track_clip.py` are the server-side sequence; `result.json` is the
   contract for the page.
2. **Snap detection and formation inference - done 2026-09-25.** Clip 1's tracked
   read now needs no manual input but the calibration (formation: see its
   section, including a 0.7 yd correction to clip 1's recorded LOS).
   **Snap detection** (`vision/snap.py`, `scripts/detect_snap.py`):
   Within one frame on both sample clips, and the clip 1 tracked read with no
   `--snap` reproduces the analyst's Cover 4. See the section below. Left:
   wire it into the web page as the default with the user's pick as override,
   and log each pick against the detector's call (the labeling loop for the
   1D CNN).
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

## Formation inference (built 2026-09-25, `vision/formation.py`)

`track_clip.py` no longer needs `--los`, `--offense` or `--ball-y`: at the
detected snap frame, `infer_formation` reads them from the bodies' field
positions and jersey-color clusters (via `track.snap_bodies`, shared with
`assign_sides`), each with a confidence and reasons; given values override, and
a disagreement of 0.5 yd or a different side is printed. Clip 1 end to end with
nothing given but the calibration: snap 5.5 s, offense right, LOS 49.0, ball y
26.1 -> shown 2-high open, played 4-deep Cover 4 (the analyst's call; the
fourth deep defender is at 12.1 yd, so the played read is low confidence,
correctly).

- **Offense side = the shallower side.** Deepest body per side of the rough
  LOS; the defense is the deeper side. Scored on Big Data Bowl week 1 with every
  other play mirrored: 99.3% (2018) / 98.4% (2023) right; with a 3 yd margin
  95.8% / 95.7% right, 0.0% / 0.4% wrong. Under a 1 yd margin it is not called
  (goal line). Officials must be flagged first: the referee stands 10-15 yd
  behind the offense (`is_player` = not striped, not a color outlier).
  Post-snap crossing (bodies ending 5+ yd past the LOS within 2 s) was measured
  too and is much weaker (74% right, 5% wrong): not used.
- **LOS = middle of the neutral zone** between the two teams' front line
  bodies: the split with the fewest jersey-color violations (ties: the widest
  gap). Color clusters are mapped to teams by the deep defenders (7+ yd): in
  clip 1's sun three defenders clustered with the white jerseys, and the
  offense has almost no bodies clear of the line to vote with. The densest band
  (the snap detector's rough LOS) was 1.1 yd off on clip 2 because more
  offensive than defensive line bodies were detected.
- **Ball y = the shotgun QB** (offense body 3-8.5 yd back within 2 yd of the
  center's y; 2018 tracking: |QB y - ball y| median 0.14, 95th pct 0.41 yd),
  **else the center** (the offensive line body nearest the lines' middle), low.

**Checked against the ball itself - and a correction to the recorded LOS.**
Clip 1's ball is visible under the center at 5.3 s and projects to x=49.2
(front tip 49.1), y=27.3. **The recorded 49.9 used by every earlier clip 1 read
is 0.7 yd too far toward the offense**: it matches the painted 40 (x=50) next to
the ball. So every clip 1 defender depth before 2026-09-25 was 0.7-0.9 yd too
deep. The tracked Cover 4 survives (12.1 yd vs the 12 yd threshold), but the
"12 yd reads it on film" threshold was set on reads that used the wrong LOS and
deserves a second look. Clip 2's ball is hidden at the snap; the center's hand
one frame earlier puts it at 70.0-70.5 (the camera pans there, ~0.5 yd
uncertain).

| clip | inferred offense | inferred LOS | ball-seen LOS | inferred ball y | reference ball y |
|---|---|---|---|---|---|
| 1 | right (high, margin 6.6 yd) | 49.0 (high), range 48.3-49.7 | 49.2 | 26.1 (low, under center) | 27.3 (seen) |
| 2 | right (high, margin 11.2 yd) | 70.2 (medium), range 69.3-71.0 | 70.0-70.5 | 23.5 (medium, shotgun QB) | 23.9 (recorded) |

Not yet wired: `ingest_clip.py` (the still path) still requires `--los` and
`--offense`; it can call `infer_formation` on its pre-snap frame the same way.

## Snap detection (built 2026-09-24, tuned on film 2026-09-25, `vision/snap.py`)

Decisions, agreed before building:
- **Placement: after the homography, in field space.** Yards are camera-invariant
  (a zoom changes pixel motion, a pan moves every pixel and no player). The
  order is track -> chain calibration (anchored on *any* still frame of the
  dwell, not the snap frame) -> `detect_snap` on unsided field positions ->
  `assign_sides` at the detected frame -> canonical tracks. Sides and line ids
  are assigned *at* the snap, so the detector cannot use either.
- **Primary signal: the line firing, not the ball.** Five bodies still by rule
  for >= 1 s then all moving in the same frame; a man in motion or a shift is
  one or two bodies; detection jitter is independent per body. The defense is
  excluded on purpose (their pre-snap movement is the disguise). The "line" is
  every body in the LOS band (both lines straddle it: more bodies, same event).
  The ball is a confirmation input (`ball_xy`) for when a detector exists: the
  person-only YOLO does not see it and it is occluded under center anyway.
- **Rule now, 1D CNN later, same interface.** No training set exists (2 clips;
  2026 tracking starts at the snap; 2021 has the snap at frame 11 on every
  play with 1 s of dwell and no film jitter). `snap_series` is the per-frame
  feature table a learned model would consume; the web page's snap pick is the
  labeling loop (log it against the detector's call, like `still_reviews`).

How it works: per frame, the set line = bodies in the LOS band (given `--los`,
or the densest 5-yd band of bodies) over the trailing 0.5 s. It *fires* when
the line's mean displacement from its window mean passes 0.3 yd (or its own
pre-fire noise, median + 4 robust sd from a window lagged 3 frames) for 2
frames, with 40% of the bodies individually past 0.3 yd. At least 4 bodies
must be set the frame before; only 3 need to stay tracked while it fires. The
snap frame (rel_frame 0, the last still frame) is the last frame before the
fire whose mean displacement is below half the fire threshold. Confidence from
bodies set / share fired / stillness up to the snap. 10 synthetic tests
(`tests/test_snap.py`), with the synthetic line at film values (10 yd/s^2,
0.05 yd jitter). `track_clip.py --snap` is optional: absent, the detected frame
is used; present, the offset is printed. `detect_snap.py --positions` saves
field positions and `--from-positions` re-runs the detector without
re-tracking (tuning takes seconds, not minutes).

**Film results (2026-09-25):**

| clip | reference snap | detected | fire confirmed | confidence |
|---|---|---|---|---|
| 1 | 5.5 s (playback resumes) | 5.5 s | 5.7 s | medium 0.68 |
| 2 | ~8.0 s (ball leaves the center 8.03-8.07 s, seen frame by frame) | 7.9 s | 8.1 s | medium 0.63 |

Clip 2's snap is also the frame the analyst used as "pre-snap" (8.0 s), so
that read sits right at the snap. Clip 1 tracked end to end with no `--snap`
(`--start 4.5`): shown 2-high MOF-open -> played 4-deep Cover 4, corner bails
only (0.25) - same as with the manual snap, and the analyst's call.

What the film changed, from the first cut:
- A fixed per-body rule (60% of bodies past 0.5 yd) confirmed the fire 0.5-0.7 s
  late: pass-setting linemen move a few tenths of a yard in the first 0.2 s,
  and O/D-line boxes merge once engaged (tracked count drops to 3). The line
  *mean* displacement jumps in the first frame after the snap on both clips
  (0.05 -> 0.49 yd; 0.22 -> 0.53 yd).
- A noise threshold estimated on the frames right before a candidate climbs
  with the ramp and never fires: the noise window is lagged 3 frames.
- A per-body flat-then-quadratic-ramp fit placed clip 2 0.4 s early (one
  jittery merged box, 3-4 points per body): replaced by the walk-back.
- Known limit, pinned by a test: a slow ramp in noisy tracks (half the film's
  acceleration, 0.08 yd jitter) lands up to 3 frames late.
- Only 5-6 of ~9 line bodies are found (merged boxes); the count is a
  confidence term, so better detection raises confidence, not accuracy.

Known limits to check on film: a breakdown video that resumes from a freeze
1 s post-snap (clip 2 at 16.5 s) shows a line already in pass sets moving all
at once - it is not the *first* fire if the window starts before the real
snap, so start the window early. The pixel motion score cannot place clip 2's
snap (the whole 5-9.4 s segment moves: man in motion + camera pan); that is
the case the line signal exists for.

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
- **2026-09-25: resolved - the user turned Smart App Control off; torch loads.**
  History: **2026-09-24 evening it was not transient any more.** `import torch` fails every time with
  `WinError 4551 ... torch\lib\torch.dll` and the Code Integrity log (event 3077) says the
  DLL "did not meet the Enterprise signing level requirements"; the registry shows Smart
  App Control enforcing (`HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy`,
  `VerifiedAndReputablePolicyState = 1`). Torch worked earlier the same day, so the
  policy most likely flipped from evaluation to on. Anything that loads YOLO is dead until
  Smart App Control is turned off (Windows Security > App & browser control) - a user
  decision, and it cannot be re-enabled without reinstalling Windows.
- Ruff PostToolUse hook auto-formats `.py` on save; line length 100.
