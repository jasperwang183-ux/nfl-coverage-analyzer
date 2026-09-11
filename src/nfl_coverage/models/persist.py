"""Save/load trained coverage models with the metadata needed to score new plays.

A bundle is a dict written with joblib:
    model          fitted sklearn estimator
    feature_cols   ordered list of columns the model was fit on
    target         label column name (e.g. 'coverage_type')
    meta           free-form dict (training set size, eval accuracy, date, ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from nfl_coverage.config import MODELS_DIR


def save_bundle(
    model: Any,
    feature_cols: list[str],
    target: str,
    meta: dict | None = None,
    path: Path | None = None,
    name: str = "model",
) -> Path:
    path = path or MODELS_DIR / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "target": target, "meta": meta or {}},
        path,
    )
    return path


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    missing = {"model", "feature_cols", "target"} - bundle.keys()
    if missing:
        raise ValueError(f"{path} is not a model bundle (missing {sorted(missing)})")
    return bundle


def predict_proba(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Score a per-play feature table; returns one row per input row, one column per class.

    Feature columns missing from `features` are filled with NaN (HistGradientBoosting
    handles NaN natively), so a still image can be scored even if a feature is
    uncomputable for it.
    """
    X = features.reindex(columns=bundle["feature_cols"])
    proba = bundle["model"].predict_proba(X)
    return pd.DataFrame(proba, columns=bundle["model"].classes_, index=features.index)
