"""The per-frame overlay the page draws on the video, on synthetic reads."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.overlay import deep_ids, field_view, los_pixels


def _read(offense="left", los=40.0):
    # canonical: offense moving right, so depth = x - canonical LOS
    rows = [
        (1, 0, 52.0, "Defense"),  # 12 yd deep at the snap
        (2, 0, 46.0, "Defense"),  # 6 yd: not deep
        (3, 0, 30.0, "Offense"),
        (1, 20, 50.0, "Defense"),  # 10 yd after the snap: under the played depth
        (2, 20, 55.0, "Defense"),  # 15 yd after the snap
    ]
    tracks = pd.DataFrame(rows, columns=["nfl_id", "rel_frame", "x", "side"])
    return SimpleNamespace(offense=offense, los=los, tracks=tracks, step=0.1)


def test_deep_uses_the_shown_depth_up_to_the_snap_and_the_count_test_after():
    r = _read()
    assert deep_ids(r, 0) == {1}
    assert deep_ids(r, 20) == {2}


def test_deep_after_the_snap_counts_where_a_defender_is_headed():
    r = _read()
    # 1 more frame for defender 1, 3 frames before frame 20: 8 yd -> 10 yd is 6.7 yd/s,
    # headed past the 10.5 yd count line
    r.tracks = pd.concat(
        [r.tracks, pd.DataFrame([(1, 17, 48.0, "Defense")], columns=r.tracks.columns)]
    )
    assert deep_ids(r, 20) == {1, 2}


def test_deep_measures_from_the_canonical_los_when_the_offense_is_on_the_right():
    r = _read(offense="right", los=80.0)  # canonical LOS = 120 - 80 = 40
    assert deep_ids(r, 0) == {1}


def test_los_pixels_run_along_the_field_line_under_a_known_camera():
    # 10 px per yard, far sideline at the top
    px = np.float32([[0, 0], [1200, 0], [1200, 533], [0, 533]])
    field = np.float32([[0, 53.3], [120, 53.3], [120, 0], [0, 0]])
    calib = FieldCalibration.from_points(px, field)
    pts = np.array(los_pixels(calib, 40.0, 1200, 533))
    assert len(pts) >= 10
    assert np.allclose(pts[:, 0], 400, atol=1)
    assert pts[:, 1].min() <= 1 and pts[:, 1].max() >= 532


def test_field_view_keys_players_by_frame_in_canonical_yards():
    r = _read(offense="right", los=80.0)
    r.tracks = r.tracks.assign(y=20.0)
    r.centers = SimpleNamespace(mechanisms=26.0)
    view = field_view(r)
    assert (view["los"], view["ball"], view["step"]) == (40.0, 26.0, 0.1)
    assert view["frames"][0] == [[1, "D", 52.0, 20.0], [2, "D", 46.0, 20.0], [3, "O", 30.0, 20.0]]
    assert sorted(view["frames"]) == [0, 20]
