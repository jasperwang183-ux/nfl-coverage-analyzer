"""Snap detection on synthetic field-space tracks: line fire, not first motion."""

import numpy as np
import pandas as pd
import pytest

from nfl_coverage.vision.snap import detect_snap, estimate_los, fit_onset, snap_series

LOS = 50.0
STEP = 0.1


def _play(
    snap_frame: int,
    n_frames: int = 30,
    jitter: float = 0.08,
    accel: float = 5.0,
    motion_id: int | None = None,
    motion_from: int = 0,
    seed: int = 0,
    n_line: int = 7,
) -> pd.DataFrame:
    """Seven line bodies straddling the LOS, two deep safeties, two receivers, a QB.

    Everyone is still until `snap_frame`, then the line and the skill players
    accelerate at `accel` yd/s^2. `motion_id` walks laterally at 4 yd/s from
    `motion_from` (a man in motion: one body moving before the snap).
    """
    rng = np.random.default_rng(seed)
    bodies = {}
    ys = np.linspace(23.0, 30.0, n_line)
    for k in range(n_line):
        bodies[k] = (LOS - 1.0 if k % 2 else LOS + 1.0, ys[k], 1.0)  # OL/DL: fires forward/back
    bodies[20] = (LOS - 6.0, 26.5, -1.0)  # QB in the gun, drops
    bodies[21] = (LOS + 13.0, 15.0, 0.0)  # safeties: still
    bodies[22] = (LOS + 13.0, 38.0, 0.0)
    bodies[23] = (LOS - 0.5, 5.0, 1.0)  # receivers release
    bodies[24] = (LOS - 0.5, 48.0, 1.0)
    rows = []
    for i in range(n_frames):
        t = i * STEP
        for tid, (x0, y0, direction) in bodies.items():
            tau = max(0.0, (i - snap_frame) * STEP)
            x = x0 + direction * 0.5 * accel * tau**2
            y = y0
            if tid == motion_id and i >= motion_from:
                y = y0 + 4.0 * (i - motion_from) * STEP
            rows.append(
                {
                    "frame": i,
                    "t": t,
                    "nfl_id": tid,
                    "x": x + rng.normal(0, jitter),
                    "y": y + rng.normal(0, jitter),
                }
            )
    return pd.DataFrame(rows)


def test_estimate_los_finds_the_dense_band():
    x = np.array([49.0, 51.0, 49.2, 50.8, 49.1, 51.2, 44.0, 63.0, 63.5, 30.0])
    assert abs(estimate_los(x) - 50.0) < 0.6


def test_fit_onset_places_the_ramp_start():
    tau = np.arange(12) - 5.0  # flat for indices 0..5, ramp after
    disp = np.where(tau > 0, 0.03 * tau**2, 0.0) + np.array(
        [0.02, -0.01, 0.0, 0.01, -0.02, 0.0] + [0] * 6
    )
    assert fit_onset(disp) == 5


def test_detects_the_snap_from_the_line_fire():
    pos = _play(snap_frame=12)
    det = detect_snap(pos, los=LOS)
    assert det.frame is not None
    assert abs(det.frame - 12) <= 1
    assert det.fire_frame > det.frame
    assert det.n_line >= 6
    assert det.confidence in ("high", "medium")


def test_snap_found_without_a_given_los():
    pos = _play(snap_frame=10)
    det = detect_snap(pos)
    assert det.frame is not None and abs(det.frame - 10) <= 1
    assert abs(det.series["los"].dropna().iloc[0] - LOS) < 1.0


def test_man_in_motion_is_not_the_snap():
    # a receiver walks across the formation from frame 3; the snap is frame 15
    pos = _play(snap_frame=15, motion_id=23, motion_from=3)
    det = detect_snap(pos, los=LOS)
    assert det.frame is not None and abs(det.frame - 15) <= 1


def test_one_line_body_shifting_is_not_the_snap():
    # a tight end (line body 0) trades sides from frame 4; the snap is frame 16
    pos = _play(snap_frame=16, motion_id=0, motion_from=4)
    det = detect_snap(pos, los=LOS)
    assert det.frame is not None and abs(det.frame - 16) <= 1


def test_no_fire_means_no_snap():
    pos = _play(snap_frame=100, n_frames=20)  # never snaps inside the window
    det = detect_snap(pos, los=LOS)
    assert det.frame is None
    assert det.confidence == "low"
    assert "never fired" in det.reasons[0]


def test_too_few_line_bodies_means_no_snap():
    pos = _play(snap_frame=10, n_line=2)
    det = detect_snap(pos, los=LOS)
    assert det.frame is None
    assert "set line bodies" in det.reasons[0]


def test_ball_confirms_and_contradicts():
    pos = _play(snap_frame=12)
    still_ball = {i: (LOS, 26.5) for i in range(30)}
    moving_ball = {
        i: (LOS - (0.0 if i <= 12 else 5.0 * min(1.0, (i - 12) * STEP / 0.4)), 26.5)
        for i in range(30)
    }
    yes = detect_snap(pos, los=LOS, ball_xy=moving_ball)
    no = detect_snap(pos, los=LOS, ball_xy=still_ball)
    assert yes.score > no.score
    assert any("confirms" in r for r in yes.reasons)
    assert any("contradicts" in r for r in no.reasons)


def test_series_has_one_row_per_frame_and_the_expected_columns():
    pos = _play(snap_frame=12, n_frames=20)
    s = snap_series(pos, los=LOS)
    assert len(s) == 20
    for c in ("n_line", "line_disp", "line_moving", "all_disp", "ball_disp", "los"):
        assert c in s.columns
    assert s["ball_disp"].isna().all()
    # before the snap the line is still; well after it, it has all moved
    assert s.loc[s["frame"] == 11, "line_moving"].iloc[0] == 0.0
    assert s.loc[s["frame"] == 19, "line_moving"].iloc[0] == pytest.approx(1.0)
