# Data Setup

## Datasets

- **BDB 2021** (`data/raw/bdb2021/`) — 2018 season, every pass play, weeks 1–17, 2.17 GB.
  Includes **1s of pre-snap frames** per play plus events (`ball_snap`, `man_in_motion`, ...).
  Coverage labels come from nflverse participation data (below): ~17.3K labeled plays.
- **BDB 2026** (`data/raw/bdb2026/`) — 2023 season, weeks 1–18, snap-to-throw tracking with
  built-in coverage labels (`team_coverage_type`, `team_coverage_man_zone`): ~14.1K labeled plays.
- **nflverse participation** (`data/raw/nflverse/`) — free per-play coverage labels
  (2.9 MB parquet per season), auto-downloaded by `src/nfl_coverage/data/nflverse.py`
  and joined to tracking on (gameId, playId).

See `.claude/skills/bdb-data/SKILL.md` for schemas and `docs/DATA_NOTES.md` for verified
facts (frame windows, label distributions, join hit-rates).

## Downloading

The Kaggle CLI is installed in the project venv. Authenticate with an API token
(kaggle.com → Settings → API → Create New Token → save as `%USERPROFILE%\.kaggle\kaggle.json`),
accept the competition rules on each competition page, then:

```bash
python scripts/download_data.py
```

Then build the processed cache and features:

```bash
python scripts/build_dataset.py
python scripts/train_baseline.py
```

## Layout

```
data/
├── raw/
│   ├── bdb2021/                 # week[1-17].csv, plays.csv, games.csv, players.csv
│   ├── bdb2026/114239_.../      # supplementary_data.csv, train/input+output CSVs
│   └── nflverse/                # pbp_participation_<season>.parquet
└── processed/
    ├── frames/                  # canonical per-week parquet (build_dataset.py)
    └── features.parquet         # per-play feature table (features/build.py)
```

> Competition data is for personal/educational use — it is gitignored and must never be
> committed or redistributed. Keep credentials out of the repo (`kaggle.json` is gitignored).
