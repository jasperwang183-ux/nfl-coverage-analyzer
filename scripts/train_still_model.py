"""Train and persist the still-image "shown" coverage model.

A still image yields exactly one canonical frame (the pre-snap alignment), so this
model uses only the snap_* static-alignment features — the subset a screenshot can
provide. Trained on every labeled play from both datasets (2018 + 2023).

Evaluates on the 2018 weeks 14-17 holdout first (honest estimate of still-image
ceiling), then refits on all labeled plays and saves the bundle to
models/still_shown.joblib.

Usage: python scripts/train_still_model.py
"""

from __future__ import annotations

from datetime import date

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from nfl_coverage.features.build import build_features
from nfl_coverage.models.persist import save_bundle

HOLDOUT_WEEKS = {14, 15, 16, 17}
MIN_CLASS = 100  # drop ultra-rare classes (PREVENT)
TARGET = "coverage_type"


def _make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, random_state=0)


def main() -> None:
    table = build_features()
    labeled = table[table[TARGET].notna()].copy()
    counts = labeled[TARGET].value_counts()
    labeled = labeled[labeled[TARGET].isin(counts[counts >= MIN_CLASS].index)]
    snap_cols = sorted(c for c in table.columns if c.startswith("snap_"))

    holdout = (labeled.dataset == "bdb2021") & labeled.week.isin(HOLDOUT_WEEKS)
    train, test = labeled[~holdout], labeled[holdout]
    model = _make_model().fit(train[snap_cols], train[TARGET])
    pred = model.predict(test[snap_cols])
    acc = accuracy_score(test[TARGET], pred)
    majority = test[TARGET].value_counts(normalize=True).iloc[0]
    print(
        f"still-image model (snap features only) | holdout accuracy {acc:.1%} "
        f"(n={len(test)}, majority {majority:.1%})"
    )
    print(classification_report(test[TARGET], pred, zero_division=0))

    final = _make_model().fit(labeled[snap_cols], labeled[TARGET])
    path = save_bundle(
        final,
        snap_cols,
        TARGET,
        meta={
            "trained": str(date.today()),
            "n_train": len(labeled),
            "holdout_accuracy": round(acc, 4),
            "holdout": "bdb2021 weeks 14-17",
            "features": "snap_* only (still-image compatible)",
        },
        name="still_shown",
    )
    print(f"saved {path} (refit on all {len(labeled)} labeled plays)")


if __name__ == "__main__":
    main()
