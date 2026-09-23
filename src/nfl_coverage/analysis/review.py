"""Human QA for vision-pipeline coverage calls.

The still/clip pipeline ends with a model prediction that an analyst can accept or
correct. Each review is appended to a JSONL log together with the play's snap
features, so corrected calls double as labeled examples for future retraining —
the feedback loop that improves the model where the vision path actually gets
used, instead of only on Big Data Bowl plays.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl_coverage.config import COVERAGE_CLASSES, DATA_PROCESSED

REVIEWS_PATH = DATA_PROCESSED / "still_reviews.jsonl"


def resolve_coverage(text: str, classes: list[str] = COVERAGE_CLASSES) -> str:
    """Map free-form analyst input to a canonical coverage class.

    Accepts the exact class, case/separator-insensitive forms ("cover 3 zone"),
    and unique prefixes ("cover 3" -> COVER_3_ZONE). Ambiguous or unknown input
    raises ValueError naming the candidates.
    """

    def norm(s: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "_", s.upper()).strip("_")

    wanted = norm(text)
    if not wanted:
        raise ValueError(f"empty coverage; options: {' '.join(classes)}")
    by_norm = {norm(c): c for c in classes}
    if wanted in by_norm:
        return by_norm[wanted]
    prefix = [c for n, c in by_norm.items() if n.startswith(wanted)]
    if len(prefix) == 1:
        return prefix[0]
    if prefix:
        raise ValueError(f"{text!r} is ambiguous: {' '.join(sorted(prefix))}")
    raise ValueError(f"{text!r} is not a coverage; options: {' '.join(classes)}")


def record_review(
    image: str,
    predicted: str,
    final: str,
    probabilities: dict[str, float],
    features: dict[str, float],
    path: Path = REVIEWS_PATH,
    caveats: list[str] | None = None,
) -> dict:
    """Append one review to the JSONL log; returns the record written.

    `caveats` are the vision pipeline's own doubts about the input (partial field
    view, implausible defender count); a corrected call on a caveated play is
    weaker retraining evidence than one on a clean view.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image": str(image),
        "predicted": predicted,
        "confidence": round(float(probabilities[predicted]), 4),
        "final": final,
        "accepted": final == predicted,
        "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
        "features": {k: None if pd.isna(v) else float(v) for k, v in features.items()},
        "caveats": list(caveats or []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_reviews(path: Path = REVIEWS_PATH) -> pd.DataFrame:
    """Load the review log; empty DataFrame if none exist yet."""
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)
