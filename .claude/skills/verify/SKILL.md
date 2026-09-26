---
name: verify
description: Verify the NFL coverage analyzer beyond its unit tests - does the film pipeline actually run on clips it was not built on, and are its predictions right on labeled plays it was not tuned on. Use whenever a refactor, algorithm swap, threshold change or new feature touches vision/, analysis/, features/ or models/; whenever the user asks "does it still work", "is this better", "why can't this clip be analyzed", or reports a clip the page cannot ingest; and before committing any change to calibration, tracking, formation, shells, disguise or the variant model - even if they only ask to "run the tests", because the unit tests only cover synthetic frames and the two Melbourne fixtures.
---

# Verify the pipeline

`pytest` proves the functions do what their authors meant on synthetic frames. It
does not prove the pipeline survives film it has never seen, or that a change made
the coverage call more right. That gap is why clips that "pass the tests" still fail
on the page. This skill closes it with two independent checks, both driven by
`scripts/verify.py`:

| part | question | command | truth used |
|---|---|---|---|
| 1. health | does every stage run and give a sane answer on unseen film? | `verify.py film` | none needed: self-consistency checks |
| 2. accuracy | are the predictions right? | `verify.py bdb` (+ the film calibration checks) | real coverage labels on frozen Big Data Bowl holdout weeks |

Both save a run under `data/verify/runs/<tag>/` (default tag = the commit, `-dirty`
when `src/` or `scripts/` have changes), and `verify.py compare A B` diffs two runs.
**A change is judged by the diff between the run before it and the run after it,
never by one run in isolation.**

## The before/after loop (the normal use)

1. **Baseline first, on unchanged code.** If the tree is clean, run the checks now
   with `--tag before-<change>`. A run already saved for the same commit (a tag
   without `-dirty`) can be reused. If the change is already made, don't stash
   the user's work. Run HEAD from a worktree that shares this checkout's `data/`
   and `models/` (they are gitignored, so a bare worktree has neither). The venv
   imports `src/` from the main checkout, so point `PYTHONPATH` at the worktree:
   ```bash
   git worktree add ../nfl-verify-base HEAD
   cmd //c "mklink /J ..\nfl-verify-base\data data" && cmd //c "mklink /J ..\nfl-verify-base\models models"
   PYTHONPATH=../nfl-verify-base/src .venv/Scripts/python ../nfl-verify-base/scripts/verify.py film --tag before-<change>
   cmd //c "rmdir ..\nfl-verify-base\data ..\nfl-verify-base\models" && git worktree remove ../nfl-verify-base
   ```
   Remove the junctions with `rmdir` *before* removing the worktree. Deleting
   through a junction deletes the real data.
2. **Make the change.**
3. **Run the same checks again** with `--tag after-<change>`.
4. **`verify.py compare before-<change> after-<change>`**, then report.

Pick the checks from what the change touches. Running more than needed is fine;
skipping one the change can affect is how regressions ship.

| change touches | film | bdb checks | notes |
|---|---|---|---|
| `vision/fieldlines.py`, `homography.py`, `clip.py` (calibration, carry) | yes | - | grid + calibration stages are the signal |
| `vision/detect.py`, `track.py` detection/tracking, `teams.py` | yes | - | the cache is keyed on these files' contents, so it re-tracks by itself |
| `vision/formation.py` | yes | `formation` | holdout scores clean and film-like (noisy) snaps |
| `vision/snap.py` | yes | - | only film has snap truth (`expect.snap_t`) |
| `analysis/shell.py`, `report.py`, `disguise.py` thresholds | yes | `shell film_read disguise` | `film_read` runs the film path's exact calls |
| `features/`, `models/`, retraining | yes | `variant` | retrains without the holdout weeks |
| `web/app.py` only | - | - | `pytest tests/test_web.py`, then drive the page |

Film reads after tracking take seconds per clip; tracking takes ~40 s per clip on
this machine and is cached per clip, window, naming and tracking-code hash
(`data/verify/cache/`; `--retrack` forces it, e.g. after a YOLO weights swap). `bdb`
takes a few minutes (`film_read` loops per play; `--sample` sets plays per dataset).

## Reading the results

**Film verdicts.** Each clip gets `ok`, `warn`, or `fail@<stage>`: the first stage
that could not produce what the next one needs. Later stages are then absent, not
passed. What each stage checks and where to look when it fails:

| stage | fails when | first place to look |
|---|---|---|
| probe | video missing/unreadable, window outside the clip | the manifest path (Captures clips live outside the repo) |
| lines | < 2 yard lines on the calibration frame | `fieldlines.paint_mask`, `detect_yard_lines`; try another frame |
| calibration | no fit reproduces its landmarks; warns on hash-only fits or when the painted numbers prefer the naming shifted 5 yd | the naming first (a 5-yd warning usually means the manifest is wrong), then `search_calibration` |
| carry | a hop can't match the next frame (a cut, a jump) | `clip.transfer_calibration`; or calibrate inside the window |
| tracking | YOLO/ByteTrack crash, or < 18 bodies per frame (warn) | `track.track_players`, `imgsz` |
| grid | yard lines found independently on tracked frames sit off the chained calibration's 5-yd grid (warn >= 0.75 yd, fail >= 1.5 yd); `worst_at_s` says when | `track.chain_calibrations` drift; a whole 5-yd slip is invisible here |
| read | `ReadError` (no snap, no offense side), or sides far from 11 v 11 | `snap.detect_snap`, `formation.infer_formation`, `track.assign_sides` |
| expect | a fact a person recorded disagrees (offense, snap +-0.25 s, LOS +-1 yd, shell counts) | the stage that produces that value |

A `warn` is a finding: report it; don't let it pass as ok. Low offense confidence,
9 or 12 bodies on a side, and hash-only calibrations are the conditions that
produced the wrong reads so far.

**Holdout accuracy.** `compare` marks each rate change `+`, `-` or `~`; `~` means
it moved less than two standard errors, so it is probably noise. Say so rather than
claiming an improvement. Reference points: formation offense side ~98-99% right on
clean snaps (week 1), played-shell MOF accuracy ~0.82 (week 1, tracking reader),
variant top-1 around 50% (snap features alone cap it). A drop in `formation/*_filmlike`
with a flat `*_clean` means the change is brittle to what film does to positions.

## Keep the holdout honest

`analysis/holdout.HOLDOUT` (2018 weeks 14-17, 2023 weeks 14-18) is frozen. Its
value is that nothing was tuned on it. If you sweep a threshold, do it on the
other weeks (`scripts/run_disguise.py --weeks ...`), then score the chosen value
on the holdout **once**. Re-scoring on the holdout after every tweak turns it into
training data, and it stops telling the truth. If a user asks you to tune against
it, say this plainly and offer the tuning weeks instead.

The same applies to the film manifest: when a clip fails, fixing the code until
that one clip passes is fine. But re-run every clip afterwards, because a
fix aimed at one clip often breaks another, and that shows up only in `compare`.

## Adding a clip (the manifest)

`samples/verify/clips.json` is committed; the videos are not (Captures clips stay
in `C:\Users\jaspe\Videos\Captures`, samples in `samples/`). Each entry stores what
a person does once on the page, so every later run is automatic:

1. `verify.py add <clip.mp4> --id <name>` picks the page's calibration frame,
   draws the numbered yard lines to `data/verify/add/<name>_lines_<t>s.jpg`, prints
   each line's pixel x and an entry stub.
2. Look at the image. Painted numbers sit on the 10-yd lines with a digit on
   each side of the line, so the line a number straddles is that number. Pick a
   line, set `named_x` to its x, `painted` to its number (0-50), and `going_up`
   to whether the numbers rise to its right. Naming by pixel x, not index,
   keeps the entry valid when detection finds one more or fewer line.
3. `window` defaults to the page's (-1 s / +4.5 s around the frame). Adjust it if
   the snap isn't inside. `given` holds values a person would type on the page
   (a snap time when tracking starts after the snap). `expect` holds facts a person
   actually knows (the analyst's call). Leave it empty rather than guessing:
   a wrong `expect` produces false alarms.
4. Record who named it in `source`. If you read the numbers yourself, say so and ask
   the user to confirm. Then run `verify.py film --only <id>`: a
   5-yd-shift warning at calibration means the naming is probably wrong.

## Reporting

Lead with the verdict, then the evidence:

```
Verdict: <safe to keep | regression in X | inconclusive because Y>

Film (before -> after): <n>/<N> clips produce a read (was <m>/<N>)
  <clip>: <old verdict> -> <new verdict>   <the metric that moved and why it matters>
Holdout: <check/part/metric>: <old> -> <new> (<+|-|~>, n=<n>)
Not covered: <what this run could not see, e.g. tracker change without --retrack,
              whole 5-yd grid slips, clips without `expect`>
```

Include the "Not covered" line whenever a check was skipped or a limit applies.
Overstating coverage is worse than reporting a gap. Keep unit tests in the loop
too: new helpers still get synthetic-frame tests in `tests/`.
