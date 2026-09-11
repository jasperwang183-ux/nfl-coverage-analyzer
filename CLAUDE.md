# NFL Coverage Analyzer

Computer vision + ML system: detect what coverage an NFL defense shows pre-snap vs. what it runs post-snap, including disguised coverages. See README.md for full architecture and roadmap.

## Ground rules

- **Python 3.11+, Windows.** Use the project venv at `.venv\` for everything (`.venv\Scripts\python`, `.venv\Scripts\pip`). Never install packages globally.
- **Data is sacred and gitignored.** Never commit anything under `data/` (Kaggle competition data cannot be redistributed). Never edit raw CSVs in place — derived data goes to `data/processed/` as parquet.
- **Phase order matters.** Phase 1 trains the coverage model on Big Data Bowl *tracking data* (not video). Phase 2 builds the YOLO/OpenCV pipeline that converts footage into the same field-coordinate representation. Don't mix vision code into the data/model pipeline.
- Formatting is ruff (a PostToolUse hook auto-formats `.py` files). Line length 100.
- Tests live in `tests/`, run with `pytest`. Data-pipeline functions get unit tests with tiny synthetic tracking frames — never test against the real dataset.

## Architecture map

- `src/nfl_coverage/data/` — load/standardize Big Data Bowl 2026 CSVs (labels: `team_coverage_type`, `team_coverage_man_zone` in supplementary_data.csv; the 2025 dataset was removed from Kaggle — see data/README.md)
- `src/nfl_coverage/features/` — pre-snap alignment + post-snap movement features
- `src/nfl_coverage/models/` — baseline sklearn → PyTorch sequence models
- `src/nfl_coverage/vision/` — Phase 2 only: YOLO detection, OpenCV homography, tracking
- `src/nfl_coverage/analysis/` — disguise detection, reporting, play visualization
- `scripts/download_data.py` — Kaggle API download (rules must be accepted on kaggle.com first)

## Football domain conventions

- Field coords: x ∈ [0, 120] yards (end zones included), y ∈ [0, 53.3]. Standardize all plays to offense moving left→right before feature extraction.
- Tracking data is 10 Hz. The 2026 input files have no snap-event column — the snap frame must be inferred (see the bdb-data skill); post-snap window for coverage classification runs from snap to pass release.
- Coverage classes come from `team_coverage_type` (Cover-1/2/3/4/6 variants); man/zone is the separate `team_coverage_man_zone` label. Normalize the raw values before training.
- A "disguise" = high-confidence pre-snap prediction ≠ post-snap ground truth (or post-snap prediction).

## Workflow

- Small, focused commits; imperative mood subject lines.
- Notebooks are for exploration only — promote anything reusable into `src/nfl_coverage/` before it gets imported twice.
