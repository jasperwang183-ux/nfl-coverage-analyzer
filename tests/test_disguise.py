"""Shell reader, shell-rotation component, and harness on tiny synthetic plays."""

import numpy as np
import pandas as pd
import pytest

from nfl_coverage.analysis.disguise import shell_rotation
from nfl_coverage.analysis.harness import (
    audit_queue,
    component_agreement,
    evaluate,
    flag_rates,
    load_labels,
    record_label,
)
from nfl_coverage.analysis.shell import shell_label, shell_state
from nfl_coverage.data.context import _context_bdb2021, _context_bdb2026
from nfl_coverage.data.standardize import finalize

# Two-high shell, offense -> right, LOS at 50 (matches tests/test_still.py).
OFFENSE = {"QB": (43.0, 26.6), "WR1": (49.5, 5.0), "WR2": (49.5, 48.0), "TE": (49.5, 35.0)}
DEFENSE = {
    "CB1": (50.5, 5.2),  # press on WR1
    "CB2": (57.0, 47.0),  # off on WR2
    "LB1": (54.0, 20.0),
    "LB2": (54.0, 32.0),
    "NB": (56.0, 12.0),
    "FS": (63.0, 18.0),
    "SS": (63.0, 35.0),
}

HOLD = {"FS": (66.0, 18.0), "SS": (66.0, 35.0)}  # two-high safeties backpedal, shell unchanged


def _play(game_id, play_id, moves=None, n_frames=16, base=None):
    """Frames 0..n_frames-1; `moves` gives the end position for players that move."""
    moves = moves or {}
    rows = []
    for i, (name, (x, y)) in enumerate({**OFFENSE, **(base or DEFENSE)}.items()):
        side = "Offense" if name in OFFENSE else "Defense"
        end = moves.get(name, (x, y))
        for f in range(n_frames):
            t = f / (n_frames - 1)
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": play_id,
                    "nfl_id": i + 1,
                    "frame_id": f + 1,
                    "rel_frame": f,
                    "x": x + (end[0] - x) * t,
                    "y": y + (end[1] - y) * t,
                    "s": 0.0,
                    "a": 0.0,
                    "o": np.nan,
                    "dir": np.nan,
                    "side": side,
                    "position": "QB" if name == "QB" else None,
                    "coverage_type": "COVER_3_ZONE",
                    "man_zone": "ZONE_COVERAGE",
                }
            )
    return finalize(pd.DataFrame(rows), dataset="test")


class TestShellReader:
    def test_two_high_open_at_snap(self):
        s = shell_state(_play(1, 1), 0).iloc[0]
        assert s["n_deep"] == 2
        assert not s["mof_closed"]
        assert s["shell"] == "2-high MOF-open"
        assert s["cb_low_press"] and not s["cb_high_press"]
        assert s["cb_high_depth"] == pytest.approx(7.5, abs=0.1)

    def test_labels(self):
        assert shell_label(0, False) == "0-high"
        assert shell_label(1, True) == "1-high MOF-closed"
        assert shell_label(3, True) == "3-deep MOF-closed"

    def test_post_snap_read_uses_snap_los(self):
        # SS rotates down to 5 yd; FS slides to the middle: Cover 3 sky look
        play = _play(1, 1, {"SS": (55.0, 30.0), "FS": (65.0, 26.0)})
        s = shell_state(play, 15).iloc[0]
        assert s["n_deep"] == 1
        assert s["mof_closed"]
        assert s["shell"] == "1-high MOF-closed"


class TestShellRotation:
    def test_rotation_down_flags(self):
        play = _play(1, 1, {"SS": (55.0, 30.0), "FS": (65.0, 26.0)})
        r = shell_rotation(play).iloc[0]
        assert r["shell_rotation_score"] == 1.0
        assert r["shell_rotation_label"] == "rotate_to_mof_closed"
        assert r["shown_shell"] == "2-high MOF-open"
        assert r["run_shell"] == "1-high MOF-closed"
        assert r["rotator_id"] == 11  # SS is the 11th player
        assert r["rotator_dd"] == pytest.approx(-8.0, abs=0.1)

    def test_static_shell_scores_zero(self):
        # the shell holds: both safeties just backpedal into their halves
        r = shell_rotation(_play(1, 1, HOLD)).iloc[0]
        assert r["shell_rotation_score"] == 0.0
        assert r["shell_rotation_label"] == ""

    def test_bail_is_rotation_up(self):
        # off corner bails 15 yd deep: resolves Cover 3 vs Cover 1, nobody lied about the shell
        play = _play(1, 1, {"CB2": (65.0, 46.0)})
        r = shell_rotation(play).iloc[0]
        assert r["shell_rotation_label"] == "corner_bail"
        assert r["shell_rotation_score"] == 0.25  # resolves Cover 3 vs 1, not a shell lie

    def test_slot_bailing_deep_is_rotation_up(self):
        # nickel walks out of the slot into a deep half from a two-high look
        play = _play(1, 1, {"NB": (64.0, 14.0)})
        r = shell_rotation(play).iloc[0]
        assert r["shell_rotation_label"] == "rotation_up"
        assert r["shell_rotation_score"] == 1.0

    def test_single_high_spinning_to_two_high_is_mof_open(self):
        # snap: FS alone in the deep middle, SS at 7 yd; post: FS to a half, SS bails to the other
        base = {**DEFENSE, "FS": (63.0, 26.0), "SS": (57.0, 38.0)}
        play = _play(1, 1, {"FS": (66.0, 15.0), "SS": (64.0, 38.0)}, base=base)
        r = shell_rotation(play).iloc[0]
        assert r["shown_shell"] == "1-high MOF-closed"
        assert r["shell_rotation_label"] == "rotate_to_mof_open"
        assert r["shell_rotation_score"] == 1.0

    def test_short_play_is_unreadable(self):
        r = shell_rotation(_play(1, 1, n_frames=6)).iloc[0]
        assert np.isnan(r["shell_rotation_score"])
        assert r["shell_rotation_label"] == "insufficient_frames"

    def test_many_plays_keep_their_own_run_frame(self):
        frames = pd.concat(
            [_play(1, 1, {"SS": (55.0, 30.0)}, n_frames=12), _play(1, 2, HOLD, n_frames=30)]
        )
        r = shell_rotation(frames).set_index("play_id")
        assert r.loc[1, "run_frame"] == 11
        assert r.loc[2, "run_frame"] == 20
        assert r.loc[1, "shell_rotation_score"] == 1.0
        assert r.loc[2, "shell_rotation_score"] == 0.0


class TestHarness:
    def _table(self):
        return pd.DataFrame(
            {
                "dataset": ["t"] * 4,
                "game_id": [1, 1, 1, 1],
                "play_id": [1, 2, 3, 4],
                "defense": ["KC", "KC", "CIN", "CIN"],
                "shell_rotation_score": [1.0, 0.0, 1.0, np.nan],
                "model_disagreement_score": [0.8, 0.1, 0.2, 0.9],
            }
        )

    def test_rates_and_agreement(self):
        t = self._table()
        rates = flag_rates(t, ["defense"])
        assert rates.loc["KC", "shell_rotation_mean"] == 0.5
        assert rates.loc["CIN", "shell_rotation_count"] == 1  # NaN not counted
        agree = component_agreement(t)
        assert agree.loc["shell_rotation", "model_disagreement"] == pytest.approx(1 / 3)
        assert len(audit_queue(t)) == 1  # play 3: shell flags, model does not

    def test_labels_round_trip_and_evaluate(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        record_label(1, 1, "t", True, "rotation", scores={"shell_rotation": 1.0}, path=path)
        record_label(1, 2, "t", False, "", path=path)
        record_label(1, 3, "t", False, "", note="safety just deep enough", path=path)
        record_label(1, 3, "t", True, "rotation", path=path)  # relabel wins
        labels = load_labels(path)
        assert len(labels) == 3
        assert labels.set_index("play_id").loc[3, "disguised"]
        ev = evaluate(self._table(), labels)
        assert ev.loc["shell_rotation", "precision"] == 1.0
        assert ev.loc["shell_rotation", "recall"] == 1.0
        assert ev.loc["model_disagreement", "recall"] == 0.5

    def test_bad_mechanism_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mechanism"):
            record_label(1, 1, "t", True, "vibes", path=tmp_path / "l.jsonl")


class TestContext:
    def test_bdb2021_defense_is_the_other_team(self):
        plays = pd.DataFrame(
            {
                "gameId": [1, 1],
                "playId": [10, 20],
                "possessionTeam": ["KC", "CIN"],
                "quarter": [1, 1],
                "down": [1, 3],
                "yardsToGo": [10, 8],
                "defendersInTheBox": [6, 5],
            }
        )
        games = pd.DataFrame(
            {"gameId": [1], "homeTeamAbbr": ["CIN"], "visitorTeamAbbr": ["KC"], "week": [1]}
        )
        ctx = _context_bdb2021(plays, games).set_index("play_id")
        assert ctx.loc[10, "defense"] == "CIN"
        assert ctx.loc[20, "defense"] == "KC"
        assert ctx.loc[20, "week"] == 1

    def test_bdb2026_columns(self):
        supp = pd.DataFrame(
            {
                "game_id": [1],
                "play_id": [5],
                "week": [2],
                "possession_team": ["SF"],
                "defensive_team": ["LA"],
                "quarter": [2],
                "down": [2],
                "yards_to_go": [7],
                "defenders_in_the_box": [7],
                "extra": ["x"],
            }
        )
        ctx = _context_bdb2026(supp)
        assert ctx.loc[0, "defense"] == "LA" and ctx.loc[0, "defenders_in_box"] == 7
        assert "extra" not in ctx
