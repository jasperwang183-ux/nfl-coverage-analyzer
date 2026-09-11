---
name: bdb-data
description: Schema and gotchas for the NFL Big Data Bowl tracking datasets (2021 and 2026) under data/raw/. Use whenever reading, joining, or engineering features from tracking CSVs, supplementary_data.csv, or nflverse coverage labels.
---

# Big Data Bowl Data Reference

Two tracking datasets feed the project, loaded into one canonical schema by
`src/nfl_coverage/data/loaders.py` (see `standardize.CANONICAL_COLUMNS`):

- **BDB 2021** (2018 season) under `data/raw/bdb2021/` — every pass play, weeks 1–17,
  **with 1s of pre-snap frames**; coverage labels joined from nflverse.
- **BDB 2026 analytics** (2023 season) under
  `data/raw/bdb2026/114239_nfl_competition_files_published_analytics_final/` — snap-to-throw
  frames with built-in labels.

## BDB 2021 (2018) essentials

- Files: `week[1-17].csv` + `plays.csv`, `games.csv`, `players.csv`. **camelCase** keys
  (`gameId`, `playId`, `nflId`, `frameId`).
- `event` column: `ball_snap` (frame 11 on virtually every play → frames 1–10 are the
  pre-snap second), `line_set`, `man_in_motion`, `shift`, `pass_forward`, `tackle`, ...
- `team` is home/away (not a club code) — resolve Offense/Defense via games.csv +
  plays.csv `possessionTeam`. Ball rows have `team == "football"`, NaN nflId — drop.
- No linemen; 11–17 players/play; receivers carry a `route` string.
- Labels: `nfl_coverage.data.nflverse.attach_labels(plays, season=2018)` — 89.8% hit-rate,
  ~17.3K labeled plays. See docs/DATA_NOTES.md for distributions.

## BDB 2026 files

| File | Grain | Size |
|---|---|---|
| `supplementary_data.csv` | one row per play (all weeks) | 7 MB |
| `train/input_2023_w[01-18].csv` | one row per player per frame — tracking up to the pass | ~45 MB each |
| `train/output_2023_w[01-18].csv` | one row per predicted player per frame — positions while ball is in the air | ~1 MB each |

Join keys: (`game_id`, `play_id`) everywhere; add `nfl_id` for player-level joins.
Note: snake_case here (`game_id`), unlike older BDB years' camelCase (`gameId`).

## Labels (in supplementary_data.csv)

- **`team_coverage_type`** — the coverage classes (Cover-1/2/3/4/6 etc.). Inspect actual value set before training; normalize messy variants.
- **`team_coverage_man_zone`** — man vs. zone.
- Rich context: `offense_formation`, `receiver_alignment`, `defenders_in_the_box`, `dropback_type`, `route_of_targeted_receiver`, `pass_result`, `expected_points_added`.
- Only **pass plays** are included (this was a pass-prediction competition), and `play_nullified_by_penalty` should be filtered.

## input_*.csv columns

`game_id, play_id, player_to_predict, nfl_id, frame_id, play_direction, absolute_yardline_number, player_name, player_height, player_weight, player_birth_date, player_position, player_side, player_role, x, y, s, a, dir, o, num_frames_output, ball_land_x, ball_land_y`

- `player_side`: Offense/Defense — no need to join to find defenders.
- `player_role`: e.g. Targeted Receiver / Passer / Defensive Coverage / Other Route Runner.
- `ball_land_x/y` + `num_frames_output`: where and when the pass arrives (constant per play).
- **No `event` or `frame_type` column.** The frame window must be established empirically (see gotchas) — check `docs/DATA_NOTES.md` for what we've verified.

## Gotchas

1. **Direction**: `play_direction` is "left" or "right". Standardize to left→right: when left, `x = 120 - x`, `y = 53.3 - y`, rotate `o`/`dir` by 180°. Field: x ∈ [0,120] (end zones included), y ∈ [0,53.3]; 10 Hz frames.
2. **Frame window (VERIFIED)**: **frame 1 = the snap** (speeds ~0, ramping immediately after); the last frame is the pass release; median play is 27 frames (~2.7s). There are no pre-snap frames — "pre-snap look" features come from the frame-1 static alignment. See `docs/DATA_NOTES.md`.
3. **Size**: each input week is ~45 MB (fine to load one week whole; loading all 18 at once is ~800 MB — prefer per-week processing, cache to parquet in `data/processed/`).
4. **Orientation vs direction**: `o` = facing, `dir` = movement direction; degrees, 0 = +y, clockwise.
5. **`output_*.csv` only contains players with `player_to_predict = True`** — it is NOT full 22-player post-snap tracking. Post-snap coverage inference must work from the post-snap portion of `input_*.csv` (frames between snap and throw).
6. Roster `player_position` (CB/FS/SS/...) is not alignment — compute alignment (deep, slot, box) from coordinates.
7. **Only ~9 players per play** — route runners, the passer, and coverage defenders. No O-line, no pass rushers. Player count varies per play (pad/mask in models); use `defenders_in_the_box` from supplementary for box info.
