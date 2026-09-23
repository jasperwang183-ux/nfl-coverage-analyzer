"""QA review flow: coverage-name resolution and the JSONL review log."""

import numpy as np
import pytest

from nfl_coverage.analysis.review import load_reviews, record_review, resolve_coverage


class TestResolveCoverage:
    def test_exact_and_case_insensitive(self):
        assert resolve_coverage("COVER_3_ZONE") == "COVER_3_ZONE"
        assert resolve_coverage("cover 2 man") == "COVER_2_MAN"
        assert resolve_coverage("cover-4-zone") == "COVER_4_ZONE"

    def test_unique_prefix(self):
        assert resolve_coverage("cover 3") == "COVER_3_ZONE"
        assert resolve_coverage("cover 0") == "COVER_0_MAN"
        assert resolve_coverage("prev") == "PREVENT"

    def test_ambiguous_prefix_rejected(self):
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_coverage("cover 2")

    def test_unknown_and_empty_rejected(self):
        with pytest.raises(ValueError, match="not a coverage"):
            resolve_coverage("tampa 2")
        with pytest.raises(ValueError, match="options"):
            resolve_coverage("  ")


class TestReviewLog:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "reviews.jsonl"
        probs = {"COVER_3_ZONE": 0.55, "COVER_1_MAN": 0.30, "COVER_2_ZONE": 0.15}
        feats = {"snap_n_defenders": 11.0, "snap_max_depth": np.nan}

        record_review("a.png", "COVER_3_ZONE", "COVER_3_ZONE", probs, feats, path=path)
        record_review(
            "b.png",
            "COVER_3_ZONE",
            "COVER_1_MAN",
            probs,
            feats,
            path=path,
            caveats=["deep zone beyond 12 yd is off-frame"],
        )

        log = load_reviews(path)
        assert len(log) == 2
        assert log["accepted"].tolist() == [True, False]
        # pipeline doubts ride along so caveated corrections can be weighted down
        assert log["caveats"].tolist() == [[], ["deep zone beyond 12 yd is off-frame"]]
        assert log["final"].tolist() == ["COVER_3_ZONE", "COVER_1_MAN"]
        assert log["confidence"].iloc[0] == pytest.approx(0.55)
        # features survive as retraining-ready values, NaN included
        assert log["features"].iloc[0]["snap_n_defenders"] == 11.0
        assert log["features"].iloc[0]["snap_max_depth"] is None

    def test_missing_log_is_empty(self, tmp_path):
        assert load_reviews(tmp_path / "nope.jsonl").empty
