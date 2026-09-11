# Verified Data Notes — BDB 2026

Empirical facts checked against the actual files (2026-09-10, week 1 + supplementary). Update this file as more is verified.

## Scale

- **18,009 labeled pass plays**, 349 games, weeks 1–18 of the 2023 season
- Week 1 tracking: 285,714 rows, 819 plays
- Frames per play: min 9, median 27 (~2.7s), max 74

## Coverage label distribution (`team_coverage_type`)

| Label | Plays |
|---|---|
| COVER_3_ZONE | 5,664 |
| COVER_1_MAN | 4,108 |
| COVER_4_ZONE | 2,860 |
| COVER_2_ZONE | 2,518 |
| COVER_6_ZONE | 1,693 |
| COVER_0_MAN | 781 |
| COVER_2_MAN | 332 |
| PREVENT | 48 |
| NaN | 5 |

`team_coverage_man_zone`: ZONE_COVERAGE 12,783 / MAN_COVERAGE 5,221 / NaN 5.
Values are clean enums (no messy free text). Class imbalance is real: weight losses or stratify.

## Frame window (verified on sample plays)

- **Frame 1 is the snap.** Mean offensive speed at frame 1–3 is ~0 and ramps immediately after — there is no pre-snap dwell period, and the last frame is the pass release.
- Therefore in this project: **"pre-snap look" = the frame-1 static alignment** (positions, depths, leverage at the moment of the snap). Pre-snap motion history is *not* available.
- Disguise detection framing stays intact: shell shown at the snap (frame 1) vs. coverage revealed by movement (frames 2..throw).

## Player subset (important!)

- Plays carry **~9 players, not 22**. `player_role` includes only: `Passer`, `Targeted Receiver`, `Other Route Runner`, `Defensive Coverage`.
- No offensive line, no pass rushers. Consequences:
  - Coverage classification uses **coverage defenders only** (which is arguably cleaner).
  - Box-count / pressure features are not derivable from tracking (use `defenders_in_the_box` from supplementary_data.csv instead).
  - The number of tracked players varies per play — models need padding/masking, not a fixed 22-slot tensor.

## BDB 2021 + nflverse labels (verified 2026-09-10) — the pre-snap dataset

**BDB 2021** (2018 season, all pass plays, weeks 1–17, extracted to `data/raw/bdb2021/`,
2.17 GB) is the dataset with pre-snap frames. Verified on week 1 + plays.csv:

- **Pre-snap depth: exactly 1.0 s** — `ball_snap` is at frameId 11 on every play
  (min = max = 11 except one stray). Frames 1–10 are pre-snap; `line_set`,
  `man_in_motion`, and `shift` events appear in the window. Post-snap runs through
  the tackle (pass_forward, pass_arrived, tackle, ... all present).
- **nflverse join hit-rate: 89.8%** of the 19,239 plays → **17,279 labeled plays**:
  COVER_3 5,848 · COVER_1 4,953 · COVER_2 2,206 · COVER_4 1,953 · COVER_6 1,135 ·
  COVER_0 658 · 2_MAN 434 · PREVENT 92 (labels via
  `nfl_coverage.data.nflverse.attach_labels(plays, season=2018)`; 2.9 MB parquet
  cached in `data/raw/nflverse/`).
- 11–17 players per play (median 13); **no linemen** (same as 2026). Ball rows have
  `team == "football"` and NaN nflId — drop them. `team` is home/away (not club
  abbr): resolve side via games.csv + plays.csv possessionTeam.
- Week 1 tracking: 986,022 rows / 1,034 plays; schema is camelCase (`gameId`,
  `frameId`), files are `week[1-17].csv`, and rows include `route` per receiver.

Combined with BDB 2026: **~35K labeled plays across two seasons (2018 + 2023)** in one
canonical label space — cross-season train/test comes free.

## Loading notes

- `supplementary_data.csv` triggers a mixed-type warning on `play_action` — pass explicit dtypes or `low_memory=False`.
- pandas 3.0 returns Arrow-backed string columns (`ArrowStringArray`) — fine, but don't assume numpy object dtype.
