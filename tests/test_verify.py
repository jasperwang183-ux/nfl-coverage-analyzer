import numpy as np
import pandas as pd
import pytest

from nfl_coverage.analysis.holdout import formation_side, perturb_snap
from nfl_coverage.vision.fieldlines import grid_residual
from nfl_coverage.vision.verify import (
    _rollup,
    compare_expect,
    naming_yards,
    side_status,
)


def test_grid_residual_is_distance_to_the_nearest_five():
    assert grid_residual(np.array([35.0, 40.0, 45.0])) == 0.0
    assert grid_residual(np.array([35.4, 40.4, 45.4])) == pytest.approx(0.4)
    assert grid_residual(np.array([37.5])) == pytest.approx(2.5)  # half a line: worst case
    assert np.isnan(grid_residual(np.array([])))


def test_side_status_is_decided_by_the_defense():
    assert side_status(11, 11) == "ok"
    assert side_status(10, 10) == "ok"
    assert side_status(7, 11) == "warn"  # merged linemen: common, not fatal
    assert side_status(11, 9) == "warn"
    assert side_status(11, 7) == "fail"
    assert side_status(11, 14) == "fail"
    assert side_status(3, 9) == "fail"  # no formation to read


def test_compare_expect_tolerances_and_missing_values():
    result = {
        "snap": {"t": 5.53},
        "formation": {"used": {"offense": "right", "los": 48.9}},
        "report": {"shown": {"mof_closed": False, "n_deep": 2}, "played": None},
    }
    out = compare_expect(
        {
            "offense": "right",
            "snap_t": 5.5,
            "los": 49.2,
            "shown_n_deep": 2,
            "played_n_deep": 4,
            "unknown_key": 1,
        },
        result,
    )
    assert out["offense"]["ok"] and out["snap_t"]["ok"] and out["los"]["ok"]
    assert out["shown_n_deep"]["ok"]
    assert not out["played_n_deep"]["ok"]  # no played read: a miss, not a crash
    assert "unknown_key" not in out


def test_naming_by_pixel_x_survives_an_extra_detected_line():
    offsets = [0, 1, 2, 3]
    calib = {"named_x": 700, "painted": 30, "going_up": True}
    # the named line is the second detected one; an extra line on the left must not shift it
    assert naming_yards(offsets, calib, [300, 700, 1100, 1500]) == [35.0, 40.0, 45.0, 50.0]


def test_naming_refuses_a_missing_painted_yard_and_a_far_named_x():
    with pytest.raises(ValueError, match="painted"):
        naming_yards([0, 1], {"named_x": 300, "painted": None}, [300, 700])
    with pytest.raises(ValueError, match="no detected line"):
        naming_yards([0, 1], {"named_x": 500, "painted": 30}, [300, 700])


def test_rollup_names_the_first_failing_stage():
    st = [
        {"name": "lines", "status": "ok"},
        {"name": "grid", "status": "fail"},
        {"name": "read", "status": "fail"},
    ]
    assert _rollup(st) == "fail@grid"
    assert _rollup([{"name": "a", "status": "warn"}]) == "warn"
    assert _rollup([{"name": "a", "status": "ok"}]) == "ok"


def _snap_frame(game: int, play: int) -> pd.DataFrame:
    """A canonical rel_frame-0 snap: offense on the low-x side of a LOS at x=50."""
    off_x = [49.0] * 5 + [45.0, 44.0, 49.0, 48.0, 49.0, 48.5]
    off_y = [22.0, 24.0, 26.0, 28.0, 30.0, 26.0, 28.0, 10.0, 40.0, 5.0, 48.0]
    def_x = [51.0] * 4 + [54.0, 55.0, 54.0, 57.0, 63.0, 63.0, 57.0]
    def_y = [22.0, 25.0, 28.0, 31.0, 20.0, 26.0, 33.0, 8.0, 18.0, 36.0, 45.0]
    n = len(off_x) + len(def_x)
    return pd.DataFrame(
        {
            "dataset": "bdb2021",
            "game_id": game,
            "play_id": play,
            "nfl_id": np.arange(n),
            "rel_frame": 0,
            "x": off_x + def_x,
            "y": off_y + def_y,
            "side": ["Offense"] * len(off_x) + ["Defense"] * len(def_x),
        }
    )


def test_formation_side_scores_clean_mirrored_plays():
    frames = pd.concat([_snap_frame(1, p) for p in range(6)], ignore_index=True)
    out = formation_side(frames, noisy=False)
    assert out["n"] == 6
    assert out["right"] == 1.0 and out["wrong"] == 0.0


def test_perturb_snap_drops_two_and_may_plant_an_official():
    snap = _snap_frame(1, 1)
    xy = snap[["x", "y"]].to_numpy(float)
    is_off = (snap["side"] == "Offense").to_numpy()
    rng = np.random.default_rng(3)
    sizes = set()
    for _ in range(20):
        pxy, labels, is_player = perturb_snap(xy, is_off, False, rng)
        assert len(pxy) == len(labels) == len(is_player)
        sizes.add(len(pxy))
    assert sizes <= {len(xy) - 2, len(xy) - 1}  # two missed, plus the official half the time
    assert len(sizes) == 2


def _late_deep_play() -> pd.DataFrame:
    """Two deep safeties until 2.8 s, then everyone runs past 12 yd (a play gone downfield)."""
    rows = []
    for rf in range(0, 41):
        for i in range(5):
            rows.append(("Offense", i, 48.0, 20.0 + 3 * i, rf))
        for i in range(7):
            deep = i < 2 or rf >= 28
            x = 50.0 + (14.0 if deep else 5.0)
            rows.append(("Defense", 10 + i, x, 5.0 + 7 * i, rf))
    df = pd.DataFrame(rows, columns=["side", "nfl_id", "x", "y", "rel_frame"])
    return df.assign(dataset="bdb2021", game_id=1, play_id=1, position="WR")


def test_film_read_rule_reads_the_middle_early_and_the_count_late():
    from nfl_coverage.analysis.holdout import FilmReadRule

    # two safeties 14 yd deep and split wide throughout; five more defenders run past
    # 12 yd from 2.8 s: a read at 2.2 s still sees only the two safeties
    read = FilmReadRule().read(_late_deep_play(), 50.0, 26.0)
    assert read.n_deep == 2 and not read.mof_closed
    late = FilmReadRule(count_read=(3.0, 12.0)).read(_late_deep_play(), 50.0, 26.0)
    assert late.n_deep >= 4


def test_deep_count_map_covers_every_mof_label():
    from nfl_coverage.analysis.harness import DEEP_COUNT_BY_COVERAGE, MOF_CLOSED_BY_COVERAGE

    assert set(DEEP_COUNT_BY_COVERAGE) == set(MOF_CLOSED_BY_COVERAGE)
    for label, closed in MOF_CLOSED_BY_COVERAGE.items():
        if closed is not None:
            assert DEEP_COUNT_BY_COVERAGE[label] is not None
