# NFL Coverage Analyzer

Computer vision + ML system: detect what coverage an NFL defense shows pre-snap vs. what it runs post-snap, including disguised coverages. See README.md for full architecture and roadmap.

## Product ladder (each rung builds on the last — don't skip)

1. **Tracking-data analyzer (v0.1, done)** — coverage + disguise from Big Data Bowl tracking data.
2. **Still images** — ingest pre-snap (+ optional post-snap) screenshots → "shown" coverage read from alignment. Stills cannot carry movement, so the "run" call stays weak here by design.
3. **Clips** — ingest a few seconds of video → detect, track, homography → full shown-vs-run + disguise, rendered back as an annotated overlay on the clip.
4. **Live feed overlay (end goal)** — same pipeline on a live video source with per-play overlay (~seconds of latency is acceptable). Requires a GPU and an elevated/all-22-style camera angle; broadcast angles crop the deep safeties.

Every vision stage must output the same canonical frame schema the tracking models consume (`src/nfl_coverage/data/standardize.py`) — that is what makes rungs 2-4 reuse the same trained models.

## Ground rules

- **Python 3.11+, Windows.** Use the project venv at `.venv\` for everything (`.venv\Scripts\python`, `.venv\Scripts\pip`). Never install packages globally.
- **Data is sacred and gitignored.** Never commit anything under `data/` (Kaggle competition data cannot be redistributed). Never edit raw CSVs in place — derived data goes to `data/processed/` as parquet.
- **Phase order matters.** Phase 1 trains the coverage model on Big Data Bowl *tracking data* (not video). Phase 2 builds the YOLO/OpenCV pipeline that converts footage into the same field-coordinate representation. Don't mix vision code into the data/model pipeline.
- Formatting is ruff (a PostToolUse hook auto-formats `.py` files). Line length 100.
- Tests live in `tests/`, run with `pytest`. Data-pipeline functions get unit tests with tiny synthetic tracking frames — never test against the real dataset.

## Architecture map

- `src/nfl_coverage/data/` — load/standardize Big Data Bowl 2021 (2018 season, pre+post-snap, labels joined from nflverse) and 2026 (2023 season, labels in supplementary_data.csv) into one canonical schema
- `src/nfl_coverage/features/` — pre-snap alignment + post-snap movement features
- `src/nfl_coverage/models/` — baseline sklearn → PyTorch sequence models
- `src/nfl_coverage/vision/` — Phase 2 only: YOLO detection, OpenCV homography, tracking
- `src/nfl_coverage/analysis/` — disguise detection, reporting, play visualization
- `src/nfl_coverage/web/` — the page (FastAPI + one static page) over `vision/pipeline.py`; run with `scripts/serve.py`
- `scripts/download_data.py` — Kaggle API download (rules must be accepted on kaggle.com first)

## Football domain conventions

- Field coords: x ∈ [0, 120] yards (end zones included), y ∈ [0, 53.3]. Standardize all plays to offense moving left→right before feature extraction.
- Tracking data is 10 Hz. The 2026 input files have no snap-event column — the snap frame must be inferred (see the bdb-data skill); post-snap window for coverage classification runs from snap to pass release.
- Coverage classes come from `team_coverage_type` (Cover-1/2/3/4/6 variants); man/zone is the separate `team_coverage_man_zone` label. Normalize the raw values before training.
- A "disguise" = high-confidence pre-snap prediction ≠ post-snap ground truth (or post-snap prediction).

## Workflow

- Small, focused commits; imperative mood subject lines.
- Notebooks are for exploration only — promote anything reusable into `src/nfl_coverage/` before it gets imported twice.
