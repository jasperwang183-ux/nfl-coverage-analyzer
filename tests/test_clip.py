"""Clip ingestion: frame sampling, motion hints, unlinked shell comparison."""

import cv2
import numpy as np
import pytest

from nfl_coverage.analysis.disguise import shell_change_label
from nfl_coverage.analysis.shell import shell_state
from nfl_coverage.vision.clip import (
    camera_motion,
    contact_sheet,
    motion_scores,
    sample_frames,
    transfer_calibration,
)
from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.still import frame_from_points


def _write_video(path, n_frames=6, fps=2.0, size=(64, 48)):
    """Tiny mp4: each frame a different flat gray level, plus a bright jump at frame 3."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        pytest.skip("mp4v VideoWriter not available in this OpenCV build")
    for i in range(n_frames):
        level = 200 if i == 3 else 40 + 5 * i
        w.write(np.full((size[1], size[0], 3), level, dtype=np.uint8))
    w.release()


def test_sample_every_half_second(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video)  # 6 frames at 2 fps = 3 s
    samples = sample_frames(video, every_s=0.5)
    assert [s.index for s in samples] == list(range(6))
    assert [s.t for s in samples] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert samples[0].image.shape == (48, 64, 3)


def test_window_and_motion_hint(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video)
    samples = sample_frames(video, every_s=0.5, start_s=1.0, end_s=2.0)
    assert [s.t for s in samples] == [1.0, 1.5, 2.0]
    scores = motion_scores(sample_frames(video, every_s=0.5))
    assert scores[0] == 0.0
    assert np.argmax(scores) == 3  # the bright frame is the biggest jump
    sheet = contact_sheet(sample_frames(video, every_s=0.5), scores, cols=3, width=96)
    assert sheet.shape[1] == 3 * 96 and sheet.shape[0] == 2 * 72


def test_missing_video_raises(tmp_path):
    with pytest.raises(ValueError, match="could not open"):
        sample_frames(tmp_path / "nope.mp4")


class TestShellChangeLabel:
    def test_mof_flips_score_full(self):
        assert shell_change_label(2, False, 1, True) == ("rotate_to_mof_closed", 1.0)
        assert shell_change_label(1, True, 2, False) == ("rotate_to_mof_open", 1.0)

    def test_count_only_changes_score_half(self):
        assert shell_change_label(2, False, 3, False) == ("rotation_up", 0.5)
        assert shell_change_label(2, False, 1, False) == ("rotation_down", 0.5)

    def test_no_change(self):
        assert shell_change_label(2, False, 2, False) == ("", 0.0)


def test_shell_state_with_imposed_los():
    # post-snap still: the offense has moved downfield, so the LOS estimate from
    # these players would be wrong; the imposed snap LOS keeps depths honest
    offense = [(52.0, 26.6), (58.0, 5.0), (60.0, 48.0)]  # QB scrambling, WRs 8-10 yd out
    defense = [(53.0, 5.0), (66.0, 26.0), (55.0, 40.0)]
    sides = np.array(["Offense"] * 3 + ["Defense"] * 3)  # everyone is past the LOS now
    frame = frame_from_points(np.array(offense + defense), los=50.0, offense="left", side=sides)
    imposed = shell_state(frame, 0, deep_depth=14.0, los=50.0, center=26.6).iloc[0]
    assert imposed["n_deep"] == 1 and imposed["mof_closed"]
    naive = shell_state(frame, 0, deep_depth=14.0).iloc[0]
    assert naive["n_deep"] == 0  # LOS read off the moved offense (~59) hides the safety


class TestCameraMotion:
    @staticmethod
    def _field():
        # textured "field": random blobs plus yard lines, so ORB has corners to match
        rng = np.random.default_rng(0)
        img = np.full((360, 640, 3), 40, dtype=np.uint8)
        for _ in range(150):
            x, y = int(rng.integers(0, 600)), int(rng.integers(0, 340))
            color = tuple(int(v) for v in rng.integers(80, 255, 3))
            cv2.rectangle(img, (x, y), (x + 12, y + 8), color, -1)
        for x in range(40, 640, 60):
            cv2.line(img, (x, 0), (x, 359), (230, 230, 230), 2)
        return img

    def test_held_camera(self):
        img = self._field()
        cam = camera_motion(img, cv2.GaussianBlur(img, (3, 3), 0))
        assert not cam.moved and cam.shift_px < 3 and abs(cam.scale - 1) < 0.01

    def test_pan_is_detected(self):
        img = self._field()
        panned = cv2.warpAffine(img, np.float32([[1, 0, 40], [0, 1, 6]]), (640, 360))
        cam = camera_motion(img, panned)
        assert cam.moved and 30 < cam.shift_px < 50

    def test_cut_to_another_camera_is_moved(self):
        img = self._field()
        other = np.random.default_rng(1).integers(0, 255, img.shape, dtype=np.uint8)
        assert camera_motion(img, other).moved


def test_transfer_calibration_follows_a_pan_and_zoom():
    img = TestCameraMotion._field()
    # frame A calibration: four image corners of a 60 x 40 yd field patch
    px = np.float32([[40, 30], [600, 30], [600, 330], [40, 330]])
    field = np.float32([[20, 5], [80, 5], [80, 45], [20, 45]])
    calib_a = FieldCalibration.from_points(px, field)
    # the camera pans right and zooms in 20%: a homography between frames
    H = np.array([[1.2, 0.02, -60.0], [0.01, 1.2, -20.0], [0.0, 0.00005, 1.0]])
    img_b = cv2.warpPerspective(img, H, (640, 360))
    t = transfer_calibration(calib_a, img, img_b)
    assert t.n_inliers >= 30
    assert t.reprojection_error < 0.2
    # a point seen in frame B maps to the same field spot as its frame A original
    pa = np.float32([[300.0, 200.0]])
    pb = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.allclose(t.calibration.to_field(pb), calib_a.to_field(pa), atol=0.5)
