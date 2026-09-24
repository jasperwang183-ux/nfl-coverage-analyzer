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
 │ corner technique, at the snap  │              │ (gradient boosting now → sequence  │
 │ and again post-snap            │              │ model with two heads)              │
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
├── scripts/               # download_data.py, train_*.py, calibrate_auto.py,
│                          # analyze_still.py, ingest_clip.py, track_clip.py, run_disguise.py
└── src/nfl_coverage/
    ├── data/              # loading & cleaning tracking CSVs; play context (team/down/distance)
    ├── features/          # pre/post-snap feature engineering
    ├── models/            # coverage classifiers + persistence
    ├── vision/            # detect, teams, homography, fieldlines (auto-calibration),
    │                      # visibility guard, still bridge, clip ingest, track (ByteTrack)
    └── analysis/          # shell reader, disguise components, harness, report, viz
```

### Running film through it

```bash
# 1. sample frames + contact sheet, then pick the pre-snap and post-snap frames
.venv\Scripts\python scripts\ingest_clip.py clip.mp4 --calib pre.calibration.json --los 49.9 --offense right
# 2. calibrate a frame from its painted markings (probe first without --yards)
.venv\Scripts\python scripts\calibrate_auto.py frame.png --roi 300 100 2260 1020 --min-length 150 --votes 80
.venv\Scripts\python scripts\calibrate_auto.py frame.png --yards 40 45 50 55 --roi ... --far-sideline 100 240 --hash-y 400 750
# 3. track the play and read it like tracking data (rotator named, movement features)
.venv\Scripts\python scripts\track_clip.py clip.mp4 --calib pre.calibration.json --anchor 5.0 --start 5.0 --snap 5.5 --end 9.0 \
    --los 49.9 --offense right --ball-y 27.5 --imgsz 1920
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
- [ ] **3. Coverage classifier** — ✅ baseline (gradient boosting: RUN 58.1% / SHOWN 54.3% on 8 classes vs. 33.5% majority); next: PyTorch sequence model on raw per-frame player states
- [ ] **4. Pre-snap vs. post-snap heads** — one model on pre-snap frames only ("what are they showing"), one on post-snap frames ("what did they run")
- [ ] **5. Disguise detection** — ✅ rule-based shell reader (deep count, middle open/closed, corner technique) as the primary "shown"; ✅ `shell_rotation` (snap vs +2 s, rotator named) and `model_disagreement` components behind one interface; ✅ harness with no hand labels: the coverage label implies the played middle-of-field state (0.82 accuracy on week 1 of both seasons), rates by defense/down/distance, film confirmations logged to `data/processed/disguise_labels.jsonl`; next: corner-technique, pressure-bluff and man/zone-mask components
- [ ] **6. Vision pipeline** — ✅ YOLO (input size 1280–1920 for all-22 frames), two-mode jersey descriptor with officials by stripes and sides voted with formation geometry, ✅ automatic field calibration from the painted markings (`vision/fieldlines.py`, ~0.2 yd, no clicking), ✅ field-visibility guard, ✅ ByteTrack player tracking with the calibration chained frame to frame (`vision/track.py`); next: fine-tuned detector, snap detection from motion
- [x] **7. Still-image mode** — screenshot → detection → calibration → canonical frame → snap-feature model (54.8% holdout / 33.5% majority) via `scripts/analyze_still.py`, with the analyst QA loop (`data/processed/still_reviews.jsonl`)
- [ ] **8. Clip mode & overlay** — ✅ `scripts/ingest_clip.py`: mp4 → sampled frames → pre-snap and post-snap reads → `CoverageReport` (structure + confidence, family, variant, disguise, reasons) with annotated frames; ✅ `scripts/track_clip.py`: canonical 10 Hz tracks from film so the tracking-data components run unchanged (first slice: movement features and rotator evidence work; its played-coverage reads do not yet match the analyst on the two test clips — see HANDOFF.md); the still path is verified on two 2026 all-22 clips (Cover 4 held; Cover 1 look → Cover 3 at the throw); next: the upload site (video + report under it), annotated overlay on the clip, sequence model with two heads
- [ ] **9. Live feed overlay** — real-time(ish) per-play coverage overlay on a live video source (GPU + elevated camera angle required)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then follow [data/README.md](data/README.md) to pull the Kaggle datasets.
