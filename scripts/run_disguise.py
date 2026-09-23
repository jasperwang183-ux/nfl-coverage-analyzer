"""Score every cached week with the disguise components and report the harness views.

Writes data/processed/disguise_scores.parquet (one row per play: component scores,
labels, evidence, play context) and prints: flag rates, face-validity views by
defense / down / distance, component agreement, the audit queue size, and
precision / recall against hand labels if any exist.

Usage:
  python scripts/run_disguise.py                 # all cached weeks, both datasets
  python scripts/run_disguise.py --weeks 1 2 --dataset bdb2026
"""

from __future__ import annotations

import argparse

import pandas as pd

from nfl_coverage.analysis.disguise import model_disagreement, shell_rotation
from nfl_coverage.analysis.harness import (
    SCORES_PATH,
    audit_queue,
    component_agreement,
    evaluate,
    face_validity,
    flag_rates,
    load_labels,
    run_shell_accuracy,
)
from nfl_coverage.data.context import load_play_context
from nfl_coverage.features.alignment import KEYS
from nfl_coverage.models.persist import load_bundle

FRAMES_DIR = SCORES_PATH.parent / "frames"
FEATURES_PATH = SCORES_PATH.parent / "features.parquet"
MODEL_PATH = SCORES_PATH.parents[2] / "models" / "still_shown.joblib"


def score_dataset(dataset: str, weeks: list[int] | None) -> pd.DataFrame:
    files = sorted(FRAMES_DIR.glob(f"{dataset}_w*.parquet"))
    if weeks:
        files = [f for f in files if int(f.stem.split("_w")[1]) in weeks]
    parts = []
    for f in files:
        frames = pd.read_parquet(f)
        part = shell_rotation(frames)
        part.insert(1, "week", int(f.stem.split("_w")[1]))
        parts.append(part)
        print(f"  {f.name}: {len(part)} plays")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bdb2021", "bdb2026"], default=None)
    ap.add_argument("--weeks", type=int, nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else ["bdb2021", "bdb2026"]

    tables = []
    for ds in datasets:
        print(f"scoring {ds}")
        t = score_dataset(ds, args.weeks)
        ctx = load_play_context(ds).drop(columns=["week"])
        t = t.merge(ctx, on=["game_id", "play_id"], how="left")
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)

    # statistical component: shown model on the cached snap features
    if FEATURES_PATH.exists() and MODEL_PATH.exists():
        feats = pd.read_parquet(FEATURES_PATH)
        md = model_disagreement(feats, load_bundle(MODEL_PATH))
        table = table.merge(md, on=KEYS, how="left")
    else:
        print("(no features.parquet / still_shown.joblib: skipping model_disagreement)")

    if "coverage_type" not in table:
        labels_by_play = feats[KEYS + ["coverage_type"]] if FEATURES_PATH.exists() else None
        if labels_by_play is not None:
            table = table.merge(labels_by_play, on=KEYS, how="left")
    table.to_parquet(SCORES_PATH, index=False)
    print(f"\nwrote {SCORES_PATH} ({len(table)} plays)\n")

    pd.set_option("display.width", 160)
    if "coverage_type" in table:
        print("=== run-shell MOF accuracy vs label (Cover 1/3 closed, 2/4/6 open) ===")
        print(run_shell_accuracy(table).round(3).to_string())
        print()
    print("=== flag rate by dataset ===")
    print(flag_rates(table, ["dataset"], args.threshold).round(3).to_string())
    for name, view in face_validity(table, args.threshold).items():
        print(f"\n=== {name} ===")
        print(view.round(3).head(40).to_string())
    print("\n=== shell_rotation labels ===")
    print(table["shell_rotation_label"].value_counts().to_string())
    print("\n=== shown -> run shell (flagged) ===")
    flagged = table[table["shell_rotation_score"] >= args.threshold]
    print(
        flagged.groupby(["shown_shell", "run_shell"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
    print("\n=== component agreement (Jaccard) ===")
    print(component_agreement(table, args.threshold).round(3).to_string())
    print(f"\naudit queue (components disagree): {len(audit_queue(table, args.threshold))} plays")

    labels = load_labels()
    if len(labels):
        print(f"\n=== vs {len(labels)} hand labels ===")
        print(evaluate(table, labels, args.threshold).round(3).to_string())
    else:
        print("\nno hand labels yet: run scripts/label_disguise.py")


if __name__ == "__main__":
    main()
