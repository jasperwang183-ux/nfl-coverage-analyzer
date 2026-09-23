"""Evaluation harness for disguise components.

There are no charted disguise labels, so the harness fixes the success criteria
up front (docs/DISGUISE_PLAN.md):

  - hand labels (`data/processed/disguise_labels.jsonl`) give precision/recall
    per component — the only real metric;
  - rates by defense / down / distance are face-validity checks;
  - cross-component agreement is reported, and the disagreement set is the
    audit queue for the next labeling round.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_coverage.config import DATA_PROCESSED
from nfl_coverage.data.context import distance_bucket
from nfl_coverage.features.alignment import KEYS

LABELS_PATH = DATA_PROCESSED / "disguise_labels.jsonl"
SCORES_PATH = DATA_PROCESSED / "disguise_scores.parquet"
MECHANISMS = ["rotation", "corner", "bluff", "manzone", "other"]


def score_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.endswith("_score")]


def flag_rates(table: pd.DataFrame, by: list[str], threshold: float = 0.5) -> pd.DataFrame:
    """Fraction of scorable plays flagged per component, grouped by `by` columns."""
    cols = score_columns(table)
    flagged = table[by].copy()
    for c in cols:
        flagged[c.replace("_score", "")] = np.where(table[c].isna(), np.nan, table[c] >= threshold)
    out = flagged.groupby(by, observed=True).agg(["mean", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.sort_index()


def face_validity(table: pd.DataFrame, threshold: float = 0.5) -> dict[str, pd.DataFrame]:
    """The sanity views a football reader expects: by defense, down, distance, red zone."""
    t = table.copy()
    views = {}
    if "defense" in t:
        views["by_defense"] = flag_rates(t, ["defense"], threshold).sort_values(
            flag_rates(t, ["defense"], threshold).columns[0], ascending=False
        )
    if "down" in t and "yards_to_go" in t:
        t["distance"] = distance_bucket(t["yards_to_go"])
        views["by_down_distance"] = flag_rates(
            t.dropna(subset=["down"]), ["down", "distance"], threshold
        )
    return views


def component_agreement(table: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Pairwise: of plays either component flags, how often both do (Jaccard)."""
    cols = score_columns(table)
    flags = {c.replace("_score", ""): table[c] >= threshold for c in cols}
    names = list(flags)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            both = (flags[a] & flags[b]).sum()
            either = (flags[a] | flags[b]).sum()
            out.loc[a, b] = both / either if either else np.nan
    return out


def audit_queue(table: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Plays where components disagree — the most informative ones to label next."""
    cols = score_columns(table)
    flags = np.column_stack([(table[c] >= threshold).to_numpy() for c in cols])
    scorable = np.column_stack([table[c].notna().to_numpy() for c in cols]).all(axis=1)
    disagree = scorable & (flags.any(axis=1) & ~flags.all(axis=1))
    return table[disagree]


# ------------------------------------------------------ label-implied shell check

MOF_CLOSED_BY_COVERAGE = {
    "COVER_0_MAN": None,  # no deep player at all: MOF state undefined
    "COVER_1_MAN": True,
    "COVER_3_ZONE": True,
    "COVER_2_MAN": False,
    "COVER_2_ZONE": False,
    "COVER_4_ZONE": False,
    "COVER_6_ZONE": False,
    "PREVENT": None,
}


def run_shell_accuracy(table: pd.DataFrame, label_col: str = "coverage_type") -> pd.DataFrame:
    """How often the post-snap shell read agrees with the coverage label's MOF state.

    The label says what was played: Cover 1/3 close the middle, Cover 2/4/6 keep
    it open. That makes the *run* half of the shell reader measurable on every
    labeled play (the shown half is not: nobody charted the pre-snap look). This
    is the number to watch when touching shell thresholds; the calibration
    sweep that set them is in docs/DISGUISE_PLAN.md.
    """
    t = table.dropna(subset=["run_mof_closed"]).copy()
    t["mof_true"] = t[label_col].map(MOF_CLOSED_BY_COVERAGE)
    t = t.dropna(subset=["mof_true"])
    t["hit"] = t["run_mof_closed"].astype(bool) == t["mof_true"].astype(bool)
    by = t.groupby("dataset")["hit"].agg(["mean", "count"])
    by.loc["all"] = [t["hit"].mean(), len(t)]
    return by.rename(columns={"mean": "run_mof_accuracy", "count": "n"})


# ------------------------------------------------------------------ hand labels


def record_label(
    game_id: int,
    play_id: int,
    dataset: str,
    disguised: bool,
    mechanism: str,
    note: str = "",
    scores: dict[str, float] | None = None,
    path: Path = LABELS_PATH,
) -> dict:
    """Append one analyst verdict to the label log; returns the record."""
    if mechanism and mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS} (or empty), got {mechanism!r}")
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset,
        "game_id": int(game_id),
        "play_id": int(play_id),
        "disguised": bool(disguised),
        "mechanism": mechanism if disguised else "",
        "note": note,
        "scores_at_label": {
            k: (None if pd.isna(v) else float(v)) for k, v in (scores or {}).items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_labels(path: Path = LABELS_PATH) -> pd.DataFrame:
    """Label log as a DataFrame (latest verdict per play wins); empty if none yet."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["dataset", "game_id", "play_id", "disguised", "mechanism"])
    df = pd.read_json(path, lines=True)
    return df.drop_duplicates(["dataset", "game_id", "play_id"], keep="last").reset_index(drop=True)


def evaluate(table: pd.DataFrame, labels: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Precision / recall / n per component against hand labels, at `threshold`."""
    joined = labels.merge(table, on=KEYS, how="inner")
    rows = []
    truth = joined["disguised"].astype(bool)
    for c in score_columns(table):
        s = joined[c]
        ok = s.notna()
        pred = (s >= threshold) & ok
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth & ok).sum())
        rows.append(
            {
                "component": c.replace("_score", ""),
                "n_labeled": int(ok.sum()),
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "flagged": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("component")


def sample_for_labeling(
    table: pd.DataFrame,
    n: int,
    labels: pd.DataFrame | None = None,
    component: str = "shell_rotation",
    threshold: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Half flagged, half not, unlabeled plays only — so precision *and* recall are estimable."""
    t = table[table[f"{component}_score"].notna()]
    if labels is not None and len(labels):
        done = labels[KEYS].drop_duplicates()
        t = t.merge(done.assign(_done=True), on=KEYS, how="left")
        t = t[t["_done"].isna()].drop(columns="_done")
    pos = t[t[f"{component}_score"] >= threshold]
    neg = t[t[f"{component}_score"] < threshold]
    k = n // 2
    picked = pd.concat(
        [
            pos.sample(min(k, len(pos)), random_state=seed),
            neg.sample(min(n - k, len(neg)), random_state=seed),
        ]
    )
    return picked.sample(frac=1, random_state=seed).reset_index(drop=True)
