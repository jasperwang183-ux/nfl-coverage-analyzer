# Architecture — how the pieces actually work

Status-honest companion to the README diagram (updated 2026-09-28). **Built** means
in `src/` with tests; **planned** means roadmap.

```
Game film ──► YOLO detection ──► homography ──► field-space (x, y) tracks   [built: stills + clips]
                                                        │
NGS tracking data (Big Data Bowl) ──────────────────────┤  (same canonical schema)
                                                        ▼
                                Player filtering  [built]
                                jersey-color team split · referee rejection
                                line-player drop · duplicate dedupe
                                                        ▼
                                Feature engineering  [built, see below]
                                                        ▼
                                Coverage classifier
                                two-head sequence model on raw frames  [built]
                                gradient boosting on engineered features (fallback)
                                                        ▼
                                Disguise detection: shown ≠ run  [built]
```

Key asymmetry: **the Big Data Bowl tracking data is the training data, not a
benchmark.** Models never train on film; film is inference-time input converted
into the same canonical frame schema (`data/standardize.py`). The player-filtering
stage exists because film shows all 22 + officials while the tracking datasets
carry coverage-relevant players only (~7 defenders/play) — without it the model
sees out-of-distribution inputs (proved on the first real screenshot, HANDOFF.md).

## Feature engineering (`src/nfl_coverage/features/`)

Everything anchors on the **LOS estimate**: 75th percentile of offensive x
(`alignment._los_estimate`) — receivers align on the line, QB/backs sit behind it,
so a high quantile tracks the LOS without ball tracking (bdb2026 has none).
**Depth = defender x − LOS**, in yards past the line (offense always standardized
to move left → right, so +x is downfield for the defense).

### Alignment / shell (`snap_*`, one frame, both datasets, stills-compatible)

| Feature | What it reads |
|---|---|
| `n_deep_8`, `n_deep_12` | safeties ≥ 8 / ≥ 12 yd deep — the one-high vs two-high shell count |
| `depth_2nd` | depth of the 2nd-deepest defender — lone FS (shallow 2nd) vs split safeties (deep 2nd) |
| `deep_y_spread` | y std-dev of the ≥ 8 yd group — split-wide two-shell vs stacked middle |
| `max_depth`, `mean_depth` | overall depth structure |
| `n_box_5` | bodies within 5 yd of the LOS (post-filtering ≈ walked-up LBs/overhangs) |
| `def_y_std` | defense condensed vs spread |
| `n_press` | defenders within 1.5 yd of a non-QB offensive player — press vs off looks |

### Man/zone movement cues (`post_*`, rel_frame 0..25 ≈ snap → 2.5 s)

Per-defender first→last displacement over the window, then aggregated:
`post_n_dropping_5/_10` (zone drops gain depth), `post_max_depth_gain`,
`post_mean_disp/_speed`, `post_lateral_std`. The strongest cue is
`post_man_mirror_frac` (`movement._man_mirror_frac`): for each defender, take the
per-frame distance to the *nearest* non-QB offensive player, average it over the
window; the feature is the fraction of defenders with mean < 4 yd. Man defenders
stay attached to an assignment; zone defenders separate to landmarks.

### Rotation (`pre_*`, rel_frame −10..0, bdb2021 only — the 1 s of pre-snap data)

Displacement in the last second before the snap: `pre_max_disp`,
`pre_max/min_depth_change`, `pre_n_moving_2`, `pre_max_speed`. Late safety
rotation, motion response, and creeping blitzers live here. bdb2026 starts at the
snap, so these are 2018-season-only features.

### Not yet implemented (diagram debt)

- **Leverage** — inside/outside shade of each defender vs. his nearest receiver
  (sign of the y-offset, receiver-relative). A strong man-coverage/help tell;
  planned alongside the sequence model or as an alignment feature.

## Coverage classifier

**Today (built):** `HistGradientBoostingClassifier` on the engineered features.
Two variants: RUN (all features, 58.1% on 2018 holdout) / SHOWN (no post-snap,
54.3%), plus the persisted stills model (snap-only, 54.8%, 33.5% majority
baseline; `scripts/train_still_model.py` → `models/still_shown.joblib`).

**Built (roadmaps #3 and #4): two-head sequence model** (`models/sequence.py`,
`scripts/train_sequence.py`, weights in `models/sequence_v1.pt`):

1. **Spatial encoder** - each frame is an unordered set of up to 24 players,
   positions only (x from the LOS, y from the formation center, side); velocities
   are finite differences, as on film, so tracking's `s`/`a`/`dir` are ignored.
   Attention across players gives one embedding per frame.
2. **Causal temporal encoder** - a transformer over 1 s before the snap to 3 s
   after, at 10 Hz. The output at frame t sees frames up to t only.
3. **Heads** - 8-way coverage variant and man/zone on every output. The pre-snap
   read is the output at the snap ("shown"), the full-play read the output at the
   last frame ("run"): one model, one pass.

Frozen holdout (n = 8,234, scored once): full-play head 86.5% top-1 / 97.0% top-2,
pre-snap head 59.2%, snap-feature GBM 51.8%, majority 32.8%. On film the pre-snap
head fills the page's pre-snap card and the full-play head the post-snap card; the
GBM stays the fallback when the weights are missing. It cannot yet tell two-high
man from two-high zone on film (one analyst-labeled 2-Man clip read Cover 2 zone).

## Disguise detection (`src/nfl_coverage/analysis/`)

A disguise is the pre-snap read differing from the post-snap read. `shell.py` reads
the structure (deep count, middle open or closed) at the snap and again after it;
`disguise.py` puts `shell_rotation` (snap vs. +2 s, rotator named) and
`model_disagreement` behind one interface; `harness.py` scores them with no hand
labels (the coverage label implies the played middle-of-field state) and rates
them by defense, down and distance. On film, the page's two cards carry the same
comparison and analyst verdicts land in `data/processed/coverage_corrections.jsonl`.
Methodology and calibration: [DISGUISE_PLAN.md](DISGUISE_PLAN.md).

