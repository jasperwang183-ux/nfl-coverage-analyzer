"""Identity scoring of film tracks against tracking truth, on synthetic plays."""

import pandas as pd
import pytest

from nfl_coverage.vision.identity import align_at_snap, identity_report


def _truth(n_frames=11):
    rows = []
    for rf in range(n_frames):
        rows += [
            {"nfl_id": 1, "rel_frame": rf, "x": 60.0 + rf * 0.5, "y": 20.0, "side": "Defense"},
            {"nfl_id": 2, "rel_frame": rf, "x": 60.0 + rf * 0.5, "y": 30.0, "side": "Defense"},
            {"nfl_id": 3, "rel_frame": rf, "x": 70.0, "y": 26.0, "side": "Defense"},
            {"nfl_id": 9, "rel_frame": rf, "x": 50.0, "y": 26.0, "side": "Offense"},
        ]
    return pd.DataFrame(rows)


def _film(truth, swap_from=None, shift=(-30.0, 4.0)):
    film = truth.assign(x=truth["x"] + shift[0], y=truth["y"] + shift[1])
    film["nfl_id"] = film["nfl_id"] + 100
    if swap_from is not None:  # the tracker trades the two ids from this frame on
        late = film["rel_frame"] >= swap_from
        film.loc[late & (film["nfl_id"] == 101), "nfl_id"] = -1
        film.loc[late & (film["nfl_id"] == 102), "nfl_id"] = 101
        film.loc[late & (film["nfl_id"] == -1), "nfl_id"] = 102
    return film


def test_alignment_undoes_the_arbitrary_film_origin():
    truth = _truth()
    aligned = align_at_snap(_film(truth), truth)
    assert aligned["x"].iloc[0] == pytest.approx(truth["x"].iloc[0])
    assert aligned["y"].iloc[0] == pytest.approx(truth["y"].iloc[0])


def test_a_clean_track_keeps_every_identity():
    truth = _truth()
    report = {p.truth_id: p for p in identity_report(_film(truth), truth, 10)}
    assert all(p.switches == 0 and p.kept == p.frames == 11 for p in report.values())
    assert report[1].snap_film_id == 101
    assert report[1].mean_error_yd == pytest.approx(0.0)


def test_a_swap_counts_once_per_player_and_stops_the_kept_frames():
    truth = _truth()
    report = {p.truth_id: p for p in identity_report(_film(truth, swap_from=6), truth, 10)}
    assert report[1].switches == 1 and report[1].kept == 6
    assert report[2].switches == 1 and report[2].kept == 6
    assert report[3].switches == 0 and report[3].kept == 11
