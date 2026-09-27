"""Accuracy on Big Data Bowl plays no threshold or model was tuned on.

Every post-tracking stage of the film pipeline (formation, shell reads, the
variant model, disguise) consumes the canonical frame schema, so tracking data
with real coverage labels can score them at scale. The thresholds were set on
week 1 of both seasons and the variant model is evaluated on 2018 weeks 14-17;
HOLDOUT is the frozen set nobody tunes on. `scripts/verify.py bdb` is the CLI.

Checks (each returns a flat dict of metrics, so two runs diff cleanly):

  formation  `infer_formation` offense side at the snap, every other play mirrored,
             on clean positions and with film-like noise (jitter, missed bodies, a
             mislabeled jersey, an official the striped-shirt check missed).
  shell      the tracking-data shell reader's played MOF state vs the label.
  film_read  the film path's played read (`declared_frame` + `structural_read` at
             the film deep threshold) vs the label's MOF state, on a play sample.
  variant    the snap-only variant model retrained without the holdout weeks.
  disguise   precision / recall against the hand labels, when there are any.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_coverage.analysis.disguise import RUN_FRAME, declared_frame, shell_rotation
from nfl_coverage.analysis.harness import (
    DEEP_COUNT_BY_COVERAGE,
    MOF_CLOSED_BY_COVERAGE,
    evaluate,
    load_labels,
    run_shell_accuracy,
)
from nfl_coverage.analysis.report import structural_read
from nfl_coverage.analysis.shell import _formation_center
from nfl_coverage.config import DATA_PROCESSED, FIELD_LENGTH
from nfl_coverage.features.alignment import KEYS, _los_estimate
from nfl_coverage.vision.formation import infer_formation

FRAMES_DIR = DATA_PROCESSED / "frames"
FEATURES_PATH = DATA_PROCESSED / "features.parquet"

# Frozen. Tune on anything else; these weeks only ever score a change.
HOLDOUT = {"bdb2021": [14, 15, 16, 17], "bdb2026": [14, 15, 16, 17, 18]}

MIN_CLASS = 100  # the variant model drops rarer classes (PREVENT), as train_still_model does
FILM_RUN_DEEP = 12.0  # pipeline.RUN_DEEP_FILM (not imported: keeps cv2 out of this module)
# The page's play window ends ~4 s after the snap (suggested frame -1 s / +4.5 s); tracking
# plays run to the whistle, and `declared_frame` would pick a late frame film never sees.
FILM_WINDOW_FRAMES = 40


def load_holdout_frames(dataset: str, weeks: list[int] | None = None) -> pd.DataFrame:
    weeks = weeks or HOLDOUT[dataset]
    parts = [pd.read_parquet(FRAMES_DIR / f"{dataset}_w{w:02d}.parquet") for w in weeks]
    return pd.concat(parts, ignore_index=True)


def _rate(hits: pd.Series) -> float:
    return round(float(hits.mean()), 4) if len(hits) else float("nan")


# ------------------------------------------------------------------ formation


def perturb_snap(
    xy: np.ndarray, is_off: np.ndarray, offense_right: bool, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What film does to a clean snap frame: returns (xy, color labels, is_player).

    Positions jitter 0.4 yd (feet from boxes through a homography), two bodies go
    undetected, one jersey lands in the wrong color cluster, and half the time the
    referee stands 12 yd behind the offense and is taken for a player.
    """
    n = len(xy)
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(n, size=min(2, n - 6), replace=False)] = False
    xy = xy[keep] + rng.normal(0.0, 0.4, size=(keep.sum(), 2))
    labels = is_off[keep].astype(int)
    labels[rng.integers(len(labels))] ^= 1
    is_player = np.ones(len(xy), dtype=bool)
    if rng.random() < 0.5:
        off_x = xy[labels == 1, 0] if (labels == 1).any() else xy[:, 0]
        back = off_x.max() + 12.0 if offense_right else off_x.min() - 12.0
        xy = np.vstack([xy, [back, float(np.median(xy[:, 1]))]])
        labels = np.append(labels, 1)
        is_player = np.append(is_player, True)
    return xy, labels, is_player


def formation_side(frames: pd.DataFrame, noisy: bool, seed: int = 0) -> dict:
    """Offense side at the snap: right / wrong / abstained, every other play mirrored.

    Tracking data carries few or no offensive linemen, so the LOS from the two
    fronts (what film gives `infer_formation`) cannot be tested here: the LOS is
    passed as a hint from the offense's alignment (`_los_estimate`), and what is
    scored is the side rule - the part film's officials and missed bodies break.
    """
    rng = np.random.default_rng(seed)
    snap = frames[frames["rel_frame"] == 0]
    right = wrong = abstain = errors = 0
    for k, (_, play) in enumerate(snap.groupby(KEYS, sort=True)):
        xy = play[["x", "y"]].to_numpy(float, copy=True)
        is_off = (play["side"] == "Offense").to_numpy()
        # canonical frames put the offense on the low-x side, moving right
        mirrored = k % 2 == 1
        if mirrored:
            xy[:, 0] = FIELD_LENGTH - xy[:, 0]
        truth = "right" if mirrored else "left"
        if not is_off.any() or is_off.all():
            errors += 1  # a snap frame missing one team: nothing to call
            continue
        los = float(np.quantile(xy[is_off, 0], 0.25 if mirrored else 0.75))  # _los_estimate
        if noisy:
            xy, labels, is_player = perturb_snap(xy, is_off, mirrored, rng)
        else:
            labels, is_player = is_off.astype(int), np.ones(len(xy), dtype=bool)
        try:
            got = infer_formation(xy, labels, is_player, los_hint=los).offense
        except ValueError:
            errors += 1
            continue
        if got is None:
            abstain += 1
        elif got == truth:
            right += 1
        else:
            wrong += 1
    n = right + wrong + abstain + errors
    return {
        "n": n,
        "right": round(right / n, 4) if n else float("nan"),
        "wrong": round(wrong / n, 4) if n else float("nan"),
        "abstain": round(abstain / n, 4) if n else float("nan"),
        "error": round(errors / n, 4) if n else float("nan"),
    }


# ------------------------------------------------------------------ shells


def shell_mof(frames: pd.DataFrame) -> dict:
    """The tracking-data shell reader's played MOF state vs the coverage label."""
    table = shell_rotation(frames)
    labels = frames.groupby(KEYS)["coverage_type"].first().reset_index()
    table = table.merge(labels, on=KEYS, how="left")
    acc = run_shell_accuracy(table)
    return {
        "run_mof_accuracy": round(float(acc.loc["all", "run_mof_accuracy"]), 4),
        "n": int(acc.loc["all", "n"]),
    }


@dataclass(frozen=True)
class FilmReadRule:
    """How the film path picks and reads the played frame."""

    run_deep: float = FILM_RUN_DEEP
    frame_rule: str = "declared"  # 'declared': most deep defenders; 'fixed': RUN_FRAME
    declare_max: int = FILM_WINDOW_FRAMES  # last rel_frame 'declared' may pick

    @property
    def name(self) -> str:
        if self.frame_rule == "fixed":
            return f"fixed@{RUN_FRAME / 10:.1f}s/{self.run_deep:g}yd"
        return f"declared<={self.declare_max / 10:.1f}s/{self.run_deep:g}yd"

    def played_frame(self, play: pd.DataFrame, los: float, center: float) -> int:
        if self.frame_rule == "fixed":
            return min(RUN_FRAME, int(play["rel_frame"].max()))
        capped = play[play["rel_frame"] <= self.declare_max]
        return declared_frame(capped, RUN_FRAME, self.run_deep, center=center, los=los)


def film_read_eval(
    frames: pd.DataFrame,
    rules: list[FilmReadRule],
    sample: int = 400,
    seed: int = 0,
    window: int = FILM_WINDOW_FRAMES,
) -> dict[str, dict]:
    """The film path's played read vs the labels, per rule, on one seeded play sample.

    Same calls `pipeline.read_tracked` makes after tracking: a played frame, then
    `structural_read` there, on frames up to `window` after the snap (what a page
    window shows). Scored two ways: the MOF state (Cover 1/3 closed, 2/4/6 open)
    and the deep count (Cover 1 = 1 ... Cover 4 = 4). Per play (slow), so a
    seeded sample; the n is reported so a small change is not over-read.
    """
    plays = frames.groupby(KEYS)["coverage_type"].first()
    plays = plays[plays.map(MOF_CLOSED_BY_COVERAGE).notna()]
    if sample and len(plays) > sample:
        plays = plays.sample(sample, random_state=seed)
    rows: dict[str, list[tuple[bool, bool | None, str]]] = {r.name: [] for r in rules}
    errors = dict.fromkeys(rows, 0)
    wanted = frames.set_index(KEYS).index.isin(plays.index) & (frames["rel_frame"] <= window)
    for key, play in frames[wanted].groupby(KEYS, sort=False):
        snap = play[play["rel_frame"] == 0]
        off = snap[snap["side"] == "Offense"]
        label = plays.loc[key]
        truth_mof = bool(MOF_CLOSED_BY_COVERAGE[label])
        truth_deep = DEEP_COUNT_BY_COVERAGE.get(label)
        try:
            los = float(_los_estimate(off).iloc[0])
            center = float(_formation_center(off).iloc[0])
        except (ValueError, IndexError, KeyError):
            for r in rules:
                errors[r.name] += 1
            continue
        for r in rules:
            try:
                rf = r.played_frame(play, los, center)
                read = structural_read(
                    play[play["rel_frame"] == rf], los, center, r.run_deep, "played", [], rf
                )
            except (ValueError, IndexError, KeyError):
                errors[r.name] += 1
                continue
            deep_hit = None if truth_deep is None else read.n_deep == truth_deep
            rows[r.name].append((read.mof_closed == truth_mof, deep_hit, read.confidence))
    out = {}
    for name, rs in rows.items():
        mof = pd.Series([h for h, _, _ in rs], dtype=float)
        deep = pd.Series([d for _, d, _ in rs if d is not None], dtype=float)
        m = {
            "n": len(rs),
            "mof_accuracy": _rate(mof),
            "n_deep_count": len(deep),
            "deep_count_accuracy": _rate(deep),
            "error": errors[name],
        }
        for conf in ("high", "medium", "low"):
            h = pd.Series([hit for hit, _, c in rs if c == conf], dtype=float)
            m[f"acc_{conf}"] = _rate(h)
            m[f"n_{conf}"] = len(h)
        out[name] = m
    return out


def film_read_mof(
    frames: pd.DataFrame,
    sample: int = 400,
    seed: int = 0,
    window: int = FILM_WINDOW_FRAMES,
    rule: FilmReadRule | None = None,
) -> dict:
    """The film path's current played read (or `rule`) scored on a play sample."""
    rule = rule or FilmReadRule()
    return film_read_eval(frames, [rule], sample, seed, window)[rule.name]


def tuning_weeks(dataset: str) -> list[int]:
    """Every cached week outside HOLDOUT: the only pool a threshold sweep may look at."""
    weeks = sorted(int(f.stem.split("_w")[1]) for f in FRAMES_DIR.glob(f"{dataset}_w*.parquet"))
    return [w for w in weeks if w not in HOLDOUT[dataset]]


# ------------------------------------------------------------------ variant model


def _variant_holdout_proba(
    features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit the snap-only recipe off the holdout: (train, test, classes, test proba)."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # deferred: slow import

    table = features if features is not None else pd.read_parquet(FEATURES_PATH)
    labeled = table[table["coverage_type"].notna()]
    counts = labeled["coverage_type"].value_counts()
    labeled = labeled[labeled["coverage_type"].isin(counts[counts >= MIN_CLASS].index)]
    held = pd.Series(False, index=labeled.index)
    for ds, weeks in HOLDOUT.items():
        held |= (labeled["dataset"] == ds) & labeled["week"].isin(weeks)
    train, test = labeled[~held], labeled[held]
    cols = sorted(c for c in table.columns if c.startswith("snap_"))
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    model.fit(train[cols], train["coverage_type"])
    return train, test, model.classes_, model.predict_proba(test[cols])


def variant_model(features: pd.DataFrame | None = None) -> dict:
    """The snap-only variant model retrained without the holdout weeks, scored on them.

    The saved `still_shown.joblib` is refit on every labeled play (holdout included),
    so it cannot be scored honestly; this retrains the same recipe on the rest.
    """
    train, test, classes, proba = _variant_holdout_proba(features)
    order = np.argsort(-proba, axis=1)
    truth = test["coverage_type"].to_numpy()
    top1 = classes[order[:, 0]] == truth
    top2 = top1 | (classes[order[:, 1]] == truth)
    out = {
        "n_train": len(train),
        "n": len(test),
        "top1": _rate(pd.Series(top1, dtype=float)),
        "top2": _rate(pd.Series(top2, dtype=float)),
        "majority": round(float(test["coverage_type"].value_counts(normalize=True).iloc[0]), 4),
    }
    for ds in HOLDOUT:
        m = (test["dataset"] == ds).to_numpy()
        out[f"top1_{ds}"] = _rate(pd.Series(top1[m], dtype=float))
    return out


SEQUENCE_MODEL = "sequence_v1.pt"


def _topk(proba: np.ndarray, classes: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    top = np.argsort(-proba, axis=1)[:, :k]
    return (classes[top] == truth[:, None]).any(axis=1)


def sequence_model(frames_by_ds: dict[str, pd.DataFrame], name: str = SEQUENCE_MODEL) -> dict:
    """The saved two-head sequence model (trained off the holdout) on the holdout plays.

    Parts: `all` (pre-snap head, full-play head, and the snap-only variant model
    refit off the holdout, all on the same plays, with the majority class),
    `per_class` (n and top-1 per true class), `heads` (how often the two heads
    disagree and who is right then). Needs `models/<name>` from train_sequence.py.
    """
    import torch  # deferred: heavy import, only this check needs it

    from nfl_coverage.config import MODELS_DIR
    from nfl_coverage.models.sequence import featurize, load, play_tensors, read_indices

    path = MODELS_DIR / name
    if not path.exists():
        return {"all": {"n": 0, "note": f"no {path.name}; run scripts/train_sequence.py"}}
    bundle = load(path, "cuda" if torch.cuda.is_available() else "cpu")
    trained_off = bundle.meta.get("holdout_weeks")
    if trained_off != HOLDOUT:
        raise ValueError(f"{path.name} was not trained off this holdout: {trained_off}")
    classes = np.array(bundle.classes)
    dev = next(bundle.model.parameters()).device
    keys, pv, pp = [], [], []
    for frames in frames_by_ds.values():
        t = play_tensors(frames)
        ok = ~np.isnan(t["ref"]).any(axis=1) & t["keys"]["coverage_type"].isin(classes).to_numpy()
        keys.append(t["keys"][ok])
        idx = np.flatnonzero(ok)
        for s in range(0, len(idx), 512):
            b = idx[s : s + 512]
            raw, mask, ref = (torch.from_numpy(t[k][b]).to(dev) for k in ("raw", "mask", "ref"))
            with torch.no_grad():
                lv, _ = bundle.model(featurize(raw, mask, ref), mask)
            last, pre = read_indices(mask)
            r = torch.arange(len(b), device=dev)
            pv.append(torch.softmax(lv[r, last].float(), -1).cpu().numpy())
            pp.append(torch.softmax(lv[r, pre].float(), -1).cpu().numpy())
    plays = pd.concat(keys, ignore_index=True)
    full, pre = np.concatenate(pv), np.concatenate(pp)
    truth = plays["coverage_type"].to_numpy()

    _, test, v_classes, v_proba = _variant_holdout_proba()
    v = pd.DataFrame(v_proba, columns=v_classes).assign(**{k: test[k].to_numpy() for k in KEYS})
    v = plays[KEYS].merge(v, on=KEYS, how="left")
    has_v = v[list(v_classes)].notna().all(axis=1).to_numpy()
    v_proba = v[list(v_classes)].to_numpy()

    res: dict[str, dict] = {"all": {}, "per_class": {}, "heads": {}}
    a = res["all"]
    a["n"] = len(plays)
    a["majority"] = round(float(pd.Series(truth).value_counts(normalize=True).iloc[0]), 4)
    hits = {}
    for tag, proba, cls, sel in (
        ("pre", pre, classes, slice(None)),
        ("full", full, classes, slice(None)),
        ("variant", v_proba[has_v], v_classes, has_v),
    ):
        for k in (1, 2, 3):
            h = _topk(proba, cls, truth[sel], k)
            a[f"{tag}_top{k}"] = _rate(pd.Series(h, dtype=float))
            if k == 1:
                hits[tag] = h
        for ds in HOLDOUT:
            m = plays["dataset"].to_numpy()[sel] == ds
            a[f"{tag}_top1_{ds}"] = _rate(pd.Series(hits[tag][m], dtype=float))
    a["n_variant"] = int(has_v.sum())
    for c in classes:
        m = truth == c
        res["per_class"][c] = {
            "n": int(m.sum()),
            "pre_top1": _rate(pd.Series(hits["pre"][m], dtype=float)),
            "full_top1": _rate(pd.Series(hits["full"][m], dtype=float)),
            "variant_top1": _rate(pd.Series(hits["variant"][m[has_v]], dtype=float)),
        }
    # disguise-shaped: the pre-snap head's call differs from the full-play head's
    pre_c, full_c = classes[pre.argmax(1)], classes[full.argmax(1)]
    differ = pre_c != full_c
    conf = pre.max(1) >= 0.6
    h = res["heads"]
    h["n"] = len(plays)
    h["differ_rate"] = _rate(pd.Series(differ, dtype=float))
    h["differ_full_right"] = _rate(pd.Series(hits["full"][differ], dtype=float))
    h["differ_pre_right"] = _rate(pd.Series(hits["pre"][differ], dtype=float))
    h["agree_right"] = _rate(pd.Series(hits["full"][~differ], dtype=float))
    h["n_confident_pre"] = int(conf.sum())
    h["confident_pre_differ_rate"] = _rate(pd.Series(differ[conf], dtype=float))
    h["confident_pre_differ_full_right"] = _rate(
        pd.Series(hits["full"][differ & conf], dtype=float)
    )
    h["n_confident_pre_differ"] = int((differ & conf).sum())
    return res


# ------------------------------------------------------------------ disguise


def disguise_hand_labels(frames_by_ds: dict[str, pd.DataFrame]) -> dict:
    """Precision / recall of the rotation component on hand-labeled holdout plays."""
    labels = load_labels()
    if not len(labels):
        return {"n": 0, "note": "no hand labels"}
    table = pd.concat([shell_rotation(f) for f in frames_by_ds.values()], ignore_index=True)
    ev = evaluate(table, labels)
    if not len(ev):
        return {"n": 0, "note": "no hand-labeled plays in the holdout weeks"}
    return {
        k: (round(float(v), 4) if isinstance(v, float) else v)
        for k, v in ev.iloc[0].to_dict().items()
    }


# ------------------------------------------------------------------ run


CHECKS = ("formation", "shell", "film_read", "variant", "disguise", "sequence")


def run_holdout(
    checks: tuple[str, ...] = CHECKS,
    datasets: tuple[str, ...] = ("bdb2021", "bdb2026"),
    sample: int = 400,
    log: Callable[[str], None] = print,
) -> dict:
    """Every requested check on the frozen holdout; {check: {dataset|all: metrics}}."""
    results: dict[str, dict] = {c: {} for c in checks}
    frames_by_ds: dict[str, pd.DataFrame] = {}
    needs_frames = {"formation", "shell", "film_read", "disguise", "sequence"} & set(checks)
    for ds in datasets if needs_frames else ():
        log(f"loading {ds} holdout weeks {HOLDOUT[ds]}")
        frames_by_ds[ds] = load_holdout_frames(ds)
    for ds, frames in frames_by_ds.items():
        if "formation" in checks:
            log(f"  formation {ds}")
            results["formation"][f"{ds}_clean"] = formation_side(frames, noisy=False)
            results["formation"][f"{ds}_filmlike"] = formation_side(frames, noisy=True)
        if "shell" in checks:
            log(f"  shell {ds}")
            results["shell"][ds] = shell_mof(frames)
        if "film_read" in checks:
            log(f"  film read {ds} ({sample} plays)")
            results["film_read"][ds] = film_read_mof(frames, sample)
    if "variant" in checks:
        log("  variant model")
        results["variant"]["all"] = variant_model()
    if "sequence" in checks:
        log("  sequence model")
        results["sequence"] = sequence_model(frames_by_ds)
    if "disguise" in checks:
        log("  disguise hand labels")
        results["disguise"]["all"] = disguise_hand_labels(frames_by_ds)
    return results
