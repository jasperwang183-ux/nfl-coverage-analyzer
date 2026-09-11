"""v0.1 baseline: gradient-boosted coverage classifiers + first disguise readout.

Trains two models on BDB 2021 (2018 season):
  - "run"  : what coverage was actually played (snap + pre-snap + post-snap features)
  - "shown": what the defense showed before the ball came out (snap + pre-snap only)

Evaluates on held-out 2018 weeks and cross-season on BDB 2026 (2023) using the
features both datasets share. Disguise = shown-model confident but wrong.

Usage: python scripts/train_baseline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from nfl_coverage.features.build import build_features

HOLDOUT_WEEKS = {14, 15, 16, 17}
MIN_CLASS = 100  # drop ultra-rare classes (PREVENT) from v0.1


def _fit_eval(name, train, test, cols, target="coverage_type"):
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    model.fit(train[cols], train[target])
    pred = model.predict(test[cols])
    acc = accuracy_score(test[target], pred)
    print(f"\n=== {name} | accuracy {acc:.1%} (n={len(test)}) ===")
    print(classification_report(test[target], pred, zero_division=0))
    return model


def main() -> None:
    table = build_features()
    labeled = table[table.coverage_type.notna()].copy()
    keep = labeled.coverage_type.value_counts()
    labeled = labeled[labeled.coverage_type.isin(keep[keep >= MIN_CLASS].index)]

    d21 = labeled[labeled.dataset == "bdb2021"]
    d26 = labeled[labeled.dataset == "bdb2026"]
    train = d21[~d21.week.isin(HOLDOUT_WEEKS)]
    test = d21[d21.week.isin(HOLDOUT_WEEKS)]
    majority = test.coverage_type.value_counts(normalize=True).iloc[0]
    print(
        f"train {len(train)} plays (2018 wks 1-13) | test {len(test)} (wks 14-17) | "
        f"2023 cross-season {len(d26)} | majority-class baseline {majority:.1%}"
    )

    feat_cols = [c for c in table.columns if c.split("_")[0] in ("snap", "pre", "post")]
    shown_cols = [c for c in feat_cols if not c.startswith("post_")]
    shared_cols = [c for c in feat_cols if not c.startswith("pre_")]  # exist in bdb2026

    run_model = _fit_eval("RUN model (all features, 2018 holdout)", train, test, feat_cols)
    shown_model = _fit_eval("SHOWN model (pre-snap only, 2018 holdout)", train, test, shown_cols)

    # cross-season transfer: retrain on shared features so 2026 can be scored
    xseason = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    xseason.fit(d21[shared_cols], d21.coverage_type)
    pred26 = xseason.predict(d26[shared_cols])
    print(
        f"\n=== cross-season: trained 2018, tested 2023 | accuracy "
        f"{accuracy_score(d26.coverage_type, pred26):.1%} (n={len(d26)}) ==="
    )
    print(classification_report(d26.coverage_type, pred26, zero_division=0))

    # disguise readout on 2018 holdout: shown-model confident but wrong
    proba = shown_model.predict_proba(test[shown_cols])
    conf = proba.max(axis=1)
    shown_pred = shown_model.classes_[proba.argmax(axis=1)]
    disguised = (conf >= 0.5) & (shown_pred != test.coverage_type)
    print(f"\n=== disguise prototype (2018 holdout) ===")
    print(
        f"plays where defense showed one thing confidently but ran another: "
        f"{disguised.sum()} / {len(test)} ({disguised.mean():.1%})"
    )
    combos = (
        pd.DataFrame({"shown": shown_pred[disguised], "ran": test.coverage_type[disguised]})
        .value_counts()
        .head(8)
    )
    print("top shown->ran disguises:")
    print(combos.to_string())


if __name__ == "__main__":
    main()
