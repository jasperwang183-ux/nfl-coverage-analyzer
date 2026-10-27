# Sample clips

Two screen-captured all-22 clips from a film breakdown of 49ers at Rams, Melbourne,
2026 week 1, with the calibrations fitted on the frames used. They are the test
cases for the film pipeline; the analyst's ground truth is:

| clip | offense | shown | played | notes |
|---|---|---|---|---|
| `sf-lar/clip1.mp4` (14 s) | 49ers, LOS x=49.2, ball y=27.3 (ball visible under center; 49.9 was used before 2026-09-25) | two-high (Cover 4 look) | Cover 4 | paused pre-snap until 5.5 s, then plays; camera pans |
| `sf-lar/clip2.mp4` (26 s) | Rams, LOS x=70.0-70.5 (center's hand; the ball is hidden), ball y=23.9 | single-high man look (Cover 1) | Cover 3 | motion plays 6-9 s, freeze with telestration ~1 s post-snap, playback resumes at 16.5 s |

The still-path commands below keep the values the recorded reads were made with
(clip 1 LOS 49.9, which is 0.7 yd too far toward the offense: it matches the
painted 40 next to the ball, not the ball). The tracked clip 1 command infers
the LOS, offense side and ball y from the snap frame instead
(`vision/formation.py`).

Both were captured from a YouTube film-study video, so the videos are not in the
repository (`.gitignore` keeps `samples/**/*.mp4` out); only their calibrations and the
analyst's facts are. To run the commands below or the slow tests, capture the two plays
yourself and save them at `samples/sf-lar/clip1.mp4` and `clip2.mp4`. Without them,
`tests/test_page_e2e.py` skips.

## Reproduce the reads

```bash
# clip 1: pre-snap frame 10 (5.0 s), post-snap frame 16 (8.0 s), each with its own calibration
.venv\Scripts\python scripts\ingest_clip.py samples\sf-lar\clip1.mp4 \
    --calib samples\sf-lar\calibrations\clip1_frame_010_05.00s.calibration.json \
    --run-calib samples\sf-lar\calibrations\clip1_frame_016_08.00s.calibration.json \
    --imgsz 1920 --los 49.9 --ball-y 27.5 --offense right --pre 10 --run 16 --yes --no-log

# clip 2: pre-snap frame 16 (8.0 s), throw frame 39 (19.5 s), deep threshold 12 at the throw
.venv\Scripts\python scripts\ingest_clip.py samples\sf-lar\clip2.mp4 \
    --calib samples\sf-lar\calibrations\clip2_frame_016_08.00s.calibration.json \
    --run-calib samples\sf-lar\calibrations\clip2_frame_039_19.50s.calibration.json \
    --imgsz 1920 --los 69.9 --ball-y 23.9 --offense right --pre 16 --run 39 --run-deep 12 --yes --no-log

# snap detection from the line firing (prints the per-frame series and the call)
.venv\Scripts\python scripts\detect_snap.py samples\sf-lar\clip1.mp4 \
    --calib samples\sf-lar\calibrations\clip1_frame_010_05.00s.calibration.json --anchor 5.0 \
    --start 4.0 --end 7.5 --los 49.9 --imgsz 1920 --expect 5.5    # detected 5.5 s
# clip 2: the ball leaves the center at 8.03-8.07 s (checked frame by frame); detected 7.9 s
.venv\Scripts\python scripts\detect_snap.py samples\sf-lar\clip2.mp4 \
    --calib samples\sf-lar\calibrations\clip2_frame_016_08.00s.calibration.json --anchor 8.0 \
    --start 6.0 --end 9.6 --los 69.9 --imgsz 1920

# tracked reads (10 Hz canonical tracks, rotator named, movement features).
# clip 1 is fully automatic: snap detected (5.5 s); LOS 49.0, offense right, ball y 26.1
# inferred -> shown 2-high open, played 4-deep Cover 4 (fourth deep defender at 12.1 yd)
.venv\Scripts\python scripts\track_clip.py samples\sf-lar\clip1.mp4 \
    --calib samples\sf-lar\calibrations\clip1_frame_010_05.00s.calibration.json --anchor 5.0 \
    --keyframe 8.0=samples\sf-lar\calibrations\clip1_frame_016_08.00s.calibration.json \
    --start 4.5 --end 9.0 --imgsz 1920
# clip 2's tracking starts after the freeze, past the snap: snap and formation are given
.venv\Scripts\python scripts\track_clip.py samples\sf-lar\clip2.mp4 \
    --calib samples\sf-lar\calibrations\clip2_frame_032_16.00s.calibration.json --anchor 16.0 \
    --keyframe 17.5=samples\sf-lar\calibrations\clip2_frame_035_17.50s.calibration.json \
    --keyframe 19.5=samples\sf-lar\calibrations\clip2_frame_039_19.50s.calibration.json \
    --start 16.0 --snap 16.4 --end 20.0 --los 69.9 --offense right --ball-y 23.9 --imgsz 1920
```

The calibrations were fitted with `scripts/calibrate_auto.py` from the painted markings
(yard lines, hash rows, sideline, numbers); the line identities used are in docs/HANDOFF.md.
