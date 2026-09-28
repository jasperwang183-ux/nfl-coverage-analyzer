# NFL Coverage Analyzer

A computer vision + machine learning system that analyzes NFL footage and tracking data to determine what coverage a defense is **showing pre-snap** versus what it **actually runs post-snap** — including detection of disguised coverages.

## How It Works

```
                 ┌──────────────────────────────────────────────────────────┐
  mp4 / still ─► │ frame sampling ─► auto-calibration from the painted        │
                 │ markings (yard lines, hashes, sidelines, numbers)         │
                 │ ─► YOLO + ByteTrack ─► foot points through the homography │ ─► field-space (x, y)
                 │ (calibration chained frame to frame as the camera pans)   │     tracks at 10 Hz
                 └──────────────────────────────────────────────────────────┘
                                       │
  NGS tracking data (Big Data Bowl) ───┤   (same canonical schema;
   = the TRAINING data                 │    film is inference-time input)
                                       ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │ Player filtering (film only): two-mode jersey descriptor  │
                 │ (lit + shaded), officials by stripes, sides by color      │
                 │ voted with formation geometry, line players dropped,      │
                 │ field-visibility guard (cropped ≠ absent)                 │
                 └──────────────────────────────────────────────────────────┘
                                       ▼
          ┌────────────────────────────┴───────────────────────────────┐
          ▼                                                            ▼
 ┌────────────────────────────────┐              ┌────────────────────────────────────┐
 │ Shell reader (rule-based):     │              │ Coverage classifier: Cover 0/1/2/  │
 │ deep count, middle open/closed,│              │ 3/4/6, man vs. zone — shown vs. run│
 │ corner technique, at the snap  │              │ (two-head sequence model; snap-    │
 │ and again post-snap            │              │ feature gradient boosting fallback)│
 └────────────────────────────────┘              └────────────────────────────────────┘
          │                                                            │
          └────────────────────────────┬───────────────────────────────┘
                                       ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │ Disguise components (shell rotation, model disagreement; │
                 │ corner technique / pressure bluff / man-zone mask planned)│
                 │ + evaluation harness: label-implied shell accuracy,      │
                 │ rates by defense/down/distance, film confirmations       │
                 └──────────────────────────────────────────────────────────┘
                                       ▼
                 CoverageReport: structure + its confidence, family,
                 model variant + its confidence, disguise verdict, reasons
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how each stage is computed
and what is built vs. planned.

### Key design insight

The Big Data Bowl datasets are **not video** — they are Next Gen Stats player-tracking data: (x, y) coordinates for every skill player at 10 Hz. Coverage labels come with the 2026 data (`team_coverage_type`, `team_coverage_man_zone`) and, for the 2018 season, from nflverse participation data joined by play ID. That's a feature, not a bug:

1. **Phase 1 — Coverage model on tracking data.** Train and validate the coverage classifier on clean, labeled tracking data first. This is where all the football logic lives (shells, rotations, man/zone cues, disguise).
2. **Phase 2 — Vision pipeline.** Build the YOLO + OpenCV pipeline that converts broadcast/All-22 footage into the *same* field-space (x, y) representation: player detection → tracking → field homography (mapping pixel coordinates to yard-line coordinates).
3. **Phase 3 — Integration.** Feed vision-derived tracks into the Phase 1 model, so the system works on raw film where no chip data exists.

This ordering means the hard ML problem (coverage recognition) is solved on ground-truth data before the noisy vision problem is layered on top.

## Tech Stack

- **Python / Pandas** — data pipeline over Big Data Bowl tracking CSVs
- **PyTorch** — sequence model (Transformer/LSTM over per-frame player states) for coverage classification
- **YOLO (Ultralytics)** — player detection in footage
- **OpenCV** — footage processing, field-line detection, homography to map pixels → field coordinates

## Datasets

| Dataset | What it provides |
|---|---|
| [NFL Big Data Bowl 2021](https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data) | 2018-season tracking for every pass play, weeks 1–17, **including 1s of pre-snap frames** — coverage labels joined from [nflverse participation data](https://github.com/nflverse/nflverse-data) (~17K labeled plays) |
| [NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data) | 2023-season tracking (snap to pass release) + built-in coverage labels (`team_coverage_type`, `team_coverage_man_zone`) (~14K labeled plays) |

See [data/README.md](data/README.md) for download instructions (Kaggle API — no MCP needed).

## Project Structure

```
nfl-coverage-analyzer/
├── data/                  # datasets (gitignored) + download instructions
├── notebooks/             # exploration & prototyping
├── scripts/               # download_data.py, train_*.py, calibrate_auto.py, serve.py,
│                          # analyze_still.py, ingest_clip.py, track_clip.py, detect_snap.py,
│                          # run_disguise.py, verify.py
└── src/nfl_coverage/
    ├── data/              # loading & cleaning tracking CSVs; play context (team/down/distance)
    ├── features/          # pre/post-snap feature engineering
    ├── models/            # coverage classifiers, the two-head sequence model, persistence
    ├── vision/            # detect, teams, homography, fieldlines (auto-calibration),
    │                      # visibility guard, still bridge, clip ingest, track (ByteTrack),
    │                      # snap detection, formation (LOS / offense / ball), overlay,
    │                      # verify (film health checks), pipeline
    ├── analysis/          # shell reader, disguise components, harness, report, viz
    └── web/               # the page: FastAPI app + one static page
```

### The page

```bash
.venv\Scripts\python scripts\serve.py      # then open http://127.0.0.1:8000
```

Upload an mp4. The page picks a still pre-snap frame, names the yard lines from the painted
numbers and calibrates on its own; it asks for the frame and naming only when that fails
(the manual controls sit in a collapsed Adjust section). Tracking runs in the background
(about a minute on a GPU). The read is drawn on the playing video (boxes, deep defenders,
the LOS) with a timeline for the window, snap and played read, and two cards sit under it:
the pre-snap card (the shown structure, its confidence and why it is not higher, and the
sequence model's pre-snap head) and the post-snap card (the played structure from +2 to +5 s,
the full-play head's top 3, the rotator's move). The disguise verdict compares the two.
The snap, offense side, line of scrimmage and ball position are inferred from the film and
can be typed over; a re-read takes seconds (tracked frames are cached). Correct / Wrong on
each card appends to `data/processed/coverage_corrections.jsonl`. Clips and results live
in `data/web/` (gitignored); snap times set by hand are logged against the detector's call
in `data/processed/snap_labels.jsonl`.

### Running film through it from the command line

```bash
# 1. sample frames + contact sheet, then pick the pre-snap and post-snap frames
.venv\Scripts\python scripts\ingest_clip.py clip.mp4 --calib pre.calibration.json --los 49.9 --offense right
# 2. calibrate a frame from its painted markings (probe first without --yards)
.venv\Scripts\python scripts\calibrate_auto.py frame.png --roi 300 100 2260 1020 --min-length 150 --votes 80
.venv\Scripts\python scripts\calibrate_auto.py frame.png --yards 40 45 50 55 --roi ... --far-sideline 100 240 --hash-y 400 750
# 3. track the play and read it like tracking data (rotator named, movement features)
#    (snap, LOS, offense side and ball position are inferred; --snap/--los/--offense/--ball-y override)
.venv\Scripts\python scripts\track_clip.py clip.mp4 --calib pre.calibration.json --anchor 5.0 \
    --start 4.5 --end 9.0 --imgsz 1920
```

Every run prints a `CoverageReport`: the structural read (deep count, middle open/closed,
corner technique) with its own confidence and reasons, the coverage family it implies, the
model's variant probabilities with their honest lower confidence, and the disguise verdict.
See `docs/DISGUISE_PLAN.md` for the methodology and calibration numbers, `HANDOFF.md` for
the state of play and known weak spots.

## Roadmap

- [x] **0. Setup** — download BDB 2021/2026 data, verify schemas and label joins (see docs/DATA_NOTES.md)
- [x] **1. Data pipeline** — parse tracking CSVs, standardize field coordinates, snap-align frames (`rel_frame`), cache to parquet
- [x] **2. Feature engineering** — snap alignment (shell, depth, press), post-snap movement (zone drops, man-mirroring), pre-snap motion (2018 data)
- [x] **3. Coverage classifier** — ✅ baseline (gradient boosting: RUN 58.1% / SHOWN 54.3% on 8 classes vs. 33.5% majority); ✅ PyTorch spatio-temporal sequence model on raw per-frame player states (`models/sequence.py`, `scripts/train_sequence.py`): 86.5% top-1 / 97.0% top-2 on the frozen holdout (n = 8,234) vs. 51.8% for the snap-feature model
- [x] **4. Pre-snap vs. post-snap heads** — ✅ one model, two heads: a causal pre-snap head ("what are they showing", 59.2% top-1 on the holdout) and a full-play head ("what did they run", 86.5%); they disagree on 39.7% of plays and the full-play head is right on 80% of those. Both heads are on the page's cards; next: telling two-high man from two-high zone on film
- [ ] **5. Disguise detection** — ✅ rule-based shell reader (deep count, middle open/closed, corner technique) as the primary "shown"; ✅ `shell_rotation` (snap vs +2 s, rotator named) and `model_disagreement` components behind one interface; ✅ harness with no hand labels: the coverage label implies the played middle-of-field state (0.82 accuracy on week 1 of both seasons), rates by defense/down/distance, film confirmations logged to `data/processed/disguise_labels.jsonl`; next: corner-technique, pressure-bluff and man/zone-mask components
- [ ] **6. Vision pipeline** — ✅ YOLO (input size 1280–1920 for all-22 frames), two-mode jersey descriptor with officials by stripes and sides voted with formation geometry, ✅ automatic field calibration from the painted markings (`vision/fieldlines.py`, ~0.2 yd, no clicking), ✅ field-visibility guard, ✅ ByteTrack player tracking with the calibration chained frame to frame (`vision/track.py`); ✅ snap detection from motion (the line must be seen set), ✅ calibration re-anchored on each frame's own yard lines, ✅ automatic yard-line naming (`fieldlines.auto_name`); next: fine-tuned detector (dense offensive lines merge into a few boxes)
- [x] **7. Still-image mode** — screenshot → detection → calibration → canonical frame → snap-feature model (54.8% holdout / 33.5% majority) via `scripts/analyze_still.py`, with the analyst QA loop (`data/processed/still_reviews.jsonl`)
- [ ] **8. Clip mode & overlay** — ✅ `scripts/ingest_clip.py`: mp4 → sampled frames → pre-snap and post-snap reads → `CoverageReport` (structure + confidence, family, variant, disguise, reasons) with annotated frames; ✅ `scripts/track_clip.py`: canonical 10 Hz tracks from film so the tracking-data components run unchanged (clip 1's tracked read matches the analyst end to end with the snap, LOS and offense side inferred; clip 2's tracked read uses a keyframe calibration found to be yards off below the hashes and is suspect — see HANDOFF.md); the still path is verified on two 2026 all-22 clips (Cover 4 held; Cover 1 look → Cover 3 at the throw); ✅ the page (`scripts/serve.py`): upload, calibrate one frame, the read under the video, overrides re-read in seconds; ✅ automatic ingest (no clicking), the read drawn on the playing video (`vision/overlay.py`), pre- and post-snap cards with the sequence model's calls and Correct / Wrong; ✅ `scripts/verify.py`: film health checks over the 11 manifest clips (9 read) and frozen-holdout scoring; next: film clips with a known disguise, a result version stamp so stale clips are flagged
- [ ] **9. Live feed overlay** — real-time(ish) per-play coverage overlay on a live video source (GPU + elevated camera angle required)
- [ ] **10. Team disguise tendencies** — ✅ `analysis/tendencies.py` + `scripts/build_tendencies.py`: per season and defense, `P(played coverage | shown MOF look, situation)` from weeks 1-13 of 2018 and 2023 (the holdout never read), the shown look from the page's own `structural_read`, each team cell shrunk toward the league with Dirichlet pseudo-counts (k = 80, chosen on weeks 1-9 → 10-13, where team × situation tables cut log loss from 1.477 to 1.383 against the league table by look); ✅ on the page's Disguise card: pick the defense and situation, see how often it keeps the look vs. the league. Face validity: the defenses least likely to play their two-high look are Vic Fangio's (2018 CHI, 2023 MIA). ✅ keyed on the defensive play-caller (`src/nfl_coverage/data/defensive_play_callers.csv`: every 2018 and 2023 stint with its source and a confidence; 13 of the 2018 rows name the title DC without a confirming source). Next: update the tables with film the page ingests

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then follow [data/README.md](data/README.md) to pull the Kaggle datasets.
