# NFL Coverage Analyzer

A computer vision + machine learning system that analyzes NFL footage and tracking data to determine what coverage a defense is **showing pre-snap** versus what it **actually runs post-snap** — including detection of disguised coverages.

## How It Works

```
                 ┌─────────────────────────────────────────────┐
  Game film ──►  │ YOLO player detection ──► OpenCV homography  │ ──► field-space (x, y) tracks
                 └─────────────────────────────────────────────┘
                                       │
  NGS tracking data (Big Data Bowl) ───┤   (same representation)
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │ Feature engineering: alignment, depth,       │
                 │ leverage, rotation, man/zone movement cues   │
                 └─────────────────────────────────────────────┘
                                       │
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │ PyTorch sequence model: Cover 0/1/2/3/4/6,   │
                 │ man vs. zone — pre-snap shell vs. post-snap  │
                 └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    Disguise detection: pre-snap ≠ post-snap
```

### Key design insight

The Big Data Bowl datasets are **not video** — they are Next Gen Stats player-tracking data: (x, y) coordinates for all 22 players at 10 Hz, with play-level labels (including PFF coverage labels such as `pff_passCoverage` and `pff_manZone` in the 2025 data). That's a feature, not a bug:

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
| [NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data) | 2023-season tracking data (10 Hz, all 22 players up to the pass) + coverage labels (`team_coverage_type`, `team_coverage_man_zone`) — the core training set |
| ~~NFL Big Data Bowl 2025~~ | **No longer available** — removed from Kaggle at the host's request; its PFF coverage labels are replaced by the 2026 labels above |

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

- [ ] **0. Setup** — download BDB 2025/2026 data, explore schema in a notebook
- [ ] **1. Data pipeline** — parse tracking CSVs, standardize field coordinates (flip plays to one direction), split frames into pre-snap / post-snap windows
- [ ] **2. Feature engineering** — safety depth/width, CB leverage & cushion, defender-to-receiver assignment distances, post-snap rotation vectors, man-indicator (defender mirrors motion) vs. zone-indicator (defender stays spatial)
- [ ] **3. Coverage classifier** — baseline (logistic regression / XGBoost on engineered features), then PyTorch sequence model on raw per-frame player states; report per-class accuracy vs. PFF labels
- [ ] **4. Pre-snap vs. post-snap heads** — one model on pre-snap frames only ("what are they showing"), one on post-snap frames ("what did they run")
- [ ] **5. Disguise detection** — flag plays where the pre-snap prediction differs from post-snap truth with high confidence; quantify disguise rates by team/down/distance
- [ ] **6. Vision pipeline** — YOLO fine-tuned for player detection, jersey/team classification, field homography via line detection, multi-object tracking (e.g., ByteTrack/DeepSORT)
- [ ] **7. Integration & demo** — run the coverage model on vision-derived tracks from real footage; overlay predictions on video

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then follow [data/README.md](data/README.md) to pull the Kaggle datasets.
