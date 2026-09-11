# Data Setup

## Status

- ✅ **BDB 2026 analytics** — downloaded and extracted to `data/raw/bdb2026/` (2023 season, 18 weeks). This is the primary dataset: tracking data + coverage labels (`team_coverage_type`, `team_coverage_man_zone` in `supplementary_data.csv`).
- ❌ **BDB 2025** — *no longer available*: the dataset was removed from Kaggle at the host's request (its download zip contains only a README saying so). The project plan originally targeted its `pff_passCoverage` labels; the 2026 labels replace them.

See `.claude/skills/bdb-data/SKILL.md` for the full schema reference and gotchas.

## Layout

```
data/
├── raw/
│   └── bdb2026/114239_nfl_competition_files_published_analytics_final/
│       ├── supplementary_data.csv          # play-level metadata + coverage labels
│       └── train/
│           ├── input_2023_w01..w18.csv     # tracking frames up to the pass (~45 MB/wk)
│           └── output_2023_w01..w18.csv    # ball-in-air positions, predicted players only
└── processed/                              # cleaned parquet produced by the pipeline
```

## Re-downloading (if ever needed)

The Kaggle CLI is installed in the project venv. Authenticate with an API token
(kaggle.com → Settings → API → Create New Token → save as `%USERPROFILE%\.kaggle\kaggle.json`,
or set the `KAGGLE_API_TOKEN` environment variable yourself), accept the competition rules on
the website, then:

```bash
.venv/Scripts/kaggle competitions download -c nfl-big-data-bowl-2026-analytics -p data/raw/bdb2026
```

> Competition data is for personal/educational use — it is gitignored and must never be
> committed or redistributed. Keep credentials out of the repo (`kaggle.json` is gitignored too).

## Possible supplementary datasets (future)

- Older Big Data Bowls (2021–2024) have full 22-player post-snap tracking; BDB 2021's bonus
  files included coverage labels for 2018 week 1. Worth revisiting if the 2026 post-snap
  window proves too short.
- Vision phase (Phase 2) needs actual footage — that never comes from Kaggle.
