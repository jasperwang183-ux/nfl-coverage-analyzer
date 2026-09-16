# Architecture — how the pieces actually work

Status-honest companion to the README diagram (updated 2026-09-16). **Built** means
in `src/` with tests; **planned** means roadmap.

```
Game film ──► YOLO detection ──► homography ──► field-space (x, y) tracks   [built: stills]
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
                                today: gradient boosting on engineered features
                                planned: PyTorch sequence model on raw frames
                                                        ▼
                                Disguise detection: shown ≠ run  [prototype]
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

**Planned (roadmap #3): PyTorch sequence model** on raw canonical frames instead
of hand features:

1. **Per-frame set encoder** — a frame is an unordered, variable-size set of
   players, each `(x, y, s, a, dir, side)`; shared per-player MLP + attention
   pooling (padding + mask for the varying count) → one embedding per frame.
2. **Temporal encoder** — Transformer encoder or GRU over the 10 Hz embedding
   sequence.
3. **Heads** — 8-way coverage class (class-weighted; COVER_3 alone is ~33%) and a
   separate man/zone head. Pre-snap-frames-only input → "shown"; post-snap window
   → "run".

Why it should beat the GBM: it can learn cues never hand-coded (leverage, pattern
matching, rotation trajectories), and its per-frame input is exactly what the
vision pipeline emits — so it transfers unchanged to clips (rung 3). ~31K labeled
plays across 2018+2023 supports a small (1–5M param) model. Note: stills can only
feed one frame, so the sequence model pays off at clips and beyond, not rung 2.

## Disguise detection (prototype)

Current definition: SHOWN-model confident (p ≥ 0.5) and wrong vs. the played
coverage — ~15 lines in `train_baseline.py`. Next step is promoting it into
`src/nfl_coverage/analysis/` with per-team/down/distance rates; two algorithm
variants are planned for an A/B comparison on separate branches.
