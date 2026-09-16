# NFL Coverage Analyzer

A computer vision + machine learning system that analyzes NFL footage and tracking data to determine what coverage a defense is **showing pre-snap** versus what it **actually runs post-snap** — including detection of disguised coverages.

## How It Works

```
                 ┌─────────────────────────────────────────────┐
  Game film ──►  │ YOLO player detection ──► OpenCV homography  │ ──► field-space (x, y) tracks
                 └─────────────────────────────────────────────┘
                                       │
  NGS tracking data (Big Data Bowl) ───┤   (same representation;
   = the TRAINING data                 │    film is inference-time input)
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │ Player filtering (film only): jersey-color   │
                 │ team split, referee rejection, line drop     │
                 └─────────────────────────────────────────────┘
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │ Feature engineering: alignment/shell, depth, │
                 │ rotation, man/zone movement cues             │
                 └─────────────────────────────────────────────┘
                                       │
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │ Coverage classifier: Cover 0/1/2/3/4/6,      │
                 │ man vs. zone — shown (pre-snap) vs. run      │
                 │ (gradient boosting now → PyTorch sequence)   │
                 └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    Disguise detection: shown ≠ run
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
├── scripts/               # download_data.py, train.py, etc.
└── src/nfl_coverage/
    ├── data/              # loading & cleaning tracking CSVs
    ├── features/          # pre/post-snap feature engineering
    ├── models/            # PyTorch coverage classifiers
    ├── vision/            # YOLO detection + OpenCV homography (Phase 2)
    └── analysis/          # disguise detection & reporting
```

## Roadmap

- [x] **0. Setup** — download BDB 2021/2026 data, verify schemas and label joins (see docs/DATA_NOTES.md)
- [x] **1. Data pipeline** — parse tracking CSVs, standardize field coordinates, snap-align frames (`rel_frame`), cache to parquet
- [x] **2. Feature engineering** — snap alignment (shell, depth, press), post-snap movement (zone drops, man-mirroring), pre-snap motion (2018 data)
- [ ] **3. Coverage classifier** — ✅ baseline (gradient boosting: RUN 58.1% / SHOWN 54.3% on 8 classes vs. 33.5% majority); next: PyTorch sequence model on raw per-frame player states
- [ ] **4. Pre-snap vs. post-snap heads** — one model on pre-snap frames only ("what are they showing"), one on post-snap frames ("what did they run")
- [ ] **5. Disguise detection** — flag plays where the pre-snap prediction differs from post-snap truth with high confidence; quantify disguise rates by team/down/distance
- [ ] **6. Vision pipeline** — YOLO fine-tuned for player detection, jersey/team classification, field homography via line detection, multi-object tracking (e.g., ByteTrack/DeepSORT)
- [ ] **7. Still-image mode** — ✅ thin slice: screenshot → YOLO foot points → homography (`scripts/calibrate_still.py`, click-based) → canonical frame → persisted snap-feature model (54.8% holdout / 33.5% majority) via `scripts/analyze_still.py`; ✅ QA loop: analyst accepts/corrects each call, reviews logged with features to `data/processed/still_reviews.jsonl` as future training data; next: automatic field-line calibration, LOS/offense-side inference
- [ ] **8. Clip mode & overlay** — few-second clips → full shown-vs-run + disguise, annotated back onto the video
- [ ] **9. Live feed overlay** — real-time(ish) per-play coverage overlay on a live video source (GPU + elevated camera angle required)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then follow [data/README.md](data/README.md) to pull the Kaggle datasets.
