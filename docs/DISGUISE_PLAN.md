# Disguise detection — methodology (agreed 2026-09-23)

A disguise is the defense making the pre-snap picture look like one coverage while
running another. There are no charted disguise labels in either Big Data Bowl set;
the label is always the coverage *run*. So "shown" has to be constructed, and that
choice is the methodology.

## Decisions

1. **Disguise metric = pre-snap read ≠ post-snap read**, from two reads of the
   same play. Today both reads are the rule-based shell reader
   (`analysis/shell.py`); the planned sequence model gives the same two reads
   as two heads on one network (see *Sequence model* below).
2. **Shell reader is the primary "shown".** A deterministic reader for one
   frame: deep-defender count, middle of the field open or closed, corner depth
   per side, box count. Read at the snap it is the shown shell; read again at
   +2.0 s (deeper threshold) it is the shell played. It is what a quarterback
   reads, it is interpretable, and it works on two unlinked stills, so the same
   code serves tracking data, the still ladder, and clips.
3. **Model disagreement is a second component, not the definition.** The shown
   model (snap features -> run label) confident and wrong is the statistical
   "unusual alignment" signal. Where it disagrees with the shell reader is the
   audit set to look at by eye.
4. **No hand-labeling of rendered tracking plays.** Ground truth comes from two
   places instead: (a) the coverage label implies the *played* shell (Cover 1/3
   close the middle, Cover 2/4/6 keep it open), which makes the post-snap half
   of the reader measurable on every labeled play; (b) film, through the clip
   ingester (`scripts/ingest_clip.py`), where the analyst confirms what the
   pipeline sees on real frames. Both feed `data/processed/disguise_labels.jsonl`.
5. **Harness first, then one worktree per mechanism component**, all behind the
   same interface: canonical frames in, one row per play out with
   `<name>_score` in [0, 1], `<name>_label`, and evidence columns.

## Shell reader calibration (week 1 of both seasons, 1,682 labeled plays)

Post-snap read frame, post-snap deep threshold and the single-high band were
swept against the label-implied middle-of-field state. Precision/recall below
are for the headline disguise: shown two-high MOF-open, flagged
`rotate_to_mof_closed`, truth = ran Cover 1 or Cover 3.

| read frame | deep ≥ | single-high band | MOF acc | flag rate | precision | recall |
|---|---|---|---|---|---|---|
| 1.5 s | 12 | 3.5 | 0.64 | 15% | 0.78 | 0.20 |
| 1.5 s | 13 | 8 | 0.82 | 21% | 0.83 | 0.59 |
| **2.0 s** | **14** | **8** | **0.82** | **33%** | **0.82** | **0.82** |
| 2.0 s | 16 | 10 | 0.78 | 38% | 0.70 | 0.85 |

Chosen: read at 2.0 s (or the throw if earlier), deep ≥ 14 yd post-snap, ≥ 9 yd
at the snap, MOF closed = nearest deep player within 3.5 yd of the QB's y with no
partner inside 8 yd, or a lone deep player within 8 yd. A third of dropbacks
flag; unflagged two-high looks run Cover 2/4/6 three times out of four, flagged
ones run Cover 1/3 four times out of five. `scripts/run_disguise.py` prints the
MOF accuracy so any threshold change is measured the same way.

Labels and scores (`analysis/disguise.py`): `rotate_to_mof_closed`,
`rotate_to_mof_open`, `rotation_down` (a deep defender comes down),
`rotation_up` (a non-corner bails deep) score 1; `corner_bail` (resolves Cover 3
vs Cover 1, no lie about the shell) and `threshold_flicker` (label changed,
nobody moved 3 yd) score 0.25 and do not flag.

## Mechanism components

| Component | Observable in tracking? | Data | Status |
|---|---|---|---|
| `shell_rotation` — snap shell vs +2 s shell | yes | both | **built, calibrated** |
| `model_disagreement` — shown model confident and wrong | yes | both | **built** |
| `corner_technique` — press-then-bail (Cover 3 bail), off-then-squat (Cover 2 trap), cushion collapse (8 yd off, then trails a receiver = off-man) | yes | both | planned |
| `pressure_bluff` — box body at snap drops 5+ yd | **half**: rushers are not in the tracking data | both | planned |
| `man_zone_mask` — man leverage at snap (press, `o` toward receiver) with low man-mirror fraction after | yes; `o` only in 2026 | both | planned |
| `late_rotation` — pre-snap-second movement, motion response | needs pre-snap frames | **bdb2021 only** | planned |

Zone-match vs spot-drop is invisible to the labels and stays out of scope.

## Clip loop (rung 3 thin slice, built)

`scripts/ingest_clip.py`: mp4 -> one frame every 0.5 s + contact sheet with
motion scores -> analyst names the pre-snap frame and the +2 s frame -> both
through the still pipeline with one calibration (fixed camera assumed) -> shell
reader on each, `shell_change_label` proposes the rotation -> analyst confirms
mechanism -> `disguise_labels.jsonl` + `data/clips/<clip>/result.json`. No
tracking yet, so the rotator cannot be named and deep-count-only changes score
0.5. The three reference videos are 2026-season footage: usable here once
downloaded as mp4 and calibrated on one frame.

## Sequence model (next after film-confirmed data exists)

Spatio-temporal transformer, two heads: per frame, set attention across the
players (masked; 5-11 defenders, no natural order, which is why attention and
not a 1-D convolution), then a small temporal transformer over frame embeddings.
Head A reads frames ≤ snap (2018: the pre-snap second; 2023: the snap frame
only), head B reads snap..throw. Disguise = argmax A ≠ argmax B with confidence
gates, the shell reader as the interpretable cross-check. A bidirectional LSTM
is a fine cheap baseline for the temporal part. ~33K plays trains a 64-dim,
2-layer model in minutes.
