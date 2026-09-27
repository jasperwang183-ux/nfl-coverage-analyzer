"""Train the two-head spatio-temporal sequence model on the canonical frames.

Trains on every week outside analysis.holdout.HOLDOUT, with a validation split of
whole games carved from those weeks; the checkpoint with the best validation
variant top-1 is kept. The holdout is scored only with --score-holdout (once, on
the chosen configuration). The variant classes and their filter match
`holdout.variant_model()` (classes with >= MIN_CLASS labeled plays).

    python scripts/train_sequence.py                       # tune: val metrics only
    python scripts/train_sequence.py --score-holdout       # final: also the frozen holdout

Tensors are cached once in data/processed/sequence_v1_tensors.pt (+ _plays.parquet).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from nfl_coverage.analysis.holdout import FEATURES_PATH, FRAMES_DIR, HOLDOUT, MIN_CLASS
from nfl_coverage.config import DATA_PROCESSED, MODELS_DIR
from nfl_coverage.models.sequence import (
    MAN_ZONE,
    Augment,
    Bundle,
    Config,
    SequenceModel,
    cosine_lr,
    encode_labels,
    featurize,
    losses,
    play_tensors,
    read_indices,
    save,
)

CACHE = DATA_PROCESSED / "sequence_v1_tensors.pt"
CACHE_PLAYS = DATA_PROCESSED / "sequence_v1_plays.parquet"


def build_cache() -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    if CACHE.exists() and CACHE_PLAYS.exists():
        return torch.load(CACHE, weights_only=True), pd.read_parquet(CACHE_PLAYS)
    raws, masks, refs, plays = [], [], [], []
    for path in sorted(FRAMES_DIR.glob("*.parquet")):
        t = play_tensors(pd.read_parquet(path))
        keys = t["keys"].assign(week=int(path.stem.split("_w")[1]))
        raws.append(t["raw"])
        masks.append(t["mask"])
        refs.append(t["ref"])
        plays.append(keys)
        print(f"{path.name}: {len(keys)} plays", flush=True)
    mask = np.concatenate(masks)
    p = int(np.flatnonzero(mask.any(axis=(0, 1))).max()) + 1
    tensors = {
        "raw": torch.from_numpy(np.concatenate(raws)[:, :, :p]),
        "mask": torch.from_numpy(mask[:, :, :p]),
        "ref": torch.from_numpy(np.concatenate(refs)),
    }
    table = pd.concat(plays, ignore_index=True)
    torch.save(tensors, CACHE)
    table.to_parquet(CACHE_PLAYS, index=False)
    return tensors, table


def variant_classes() -> list[str]:
    """Same filter as holdout.variant_model(): classes with >= MIN_CLASS labeled plays."""
    counts = pd.read_parquet(FEATURES_PATH, columns=["coverage_type"])["coverage_type"]
    counts = counts.value_counts()
    return sorted(counts[counts >= MIN_CLASS].index)


def splits(plays: pd.DataFrame, val_frac: float, seed: int) -> dict[str, np.ndarray]:
    held = np.zeros(len(plays), dtype=bool)
    for ds, weeks in HOLDOUT.items():
        held |= ((plays["dataset"] == ds) & plays["week"].isin(weeks)).to_numpy()
    games = np.sort(plays.loc[~held, "game_id"].unique())
    rng = np.random.default_rng(seed)
    val_games = rng.choice(games, size=int(len(games) * val_frac), replace=False)
    val = ~held & plays["game_id"].isin(val_games).to_numpy()
    ok = plays["ref_ok"].to_numpy()
    return {
        "train": np.flatnonzero(~held & ~val & ok),
        "val": np.flatnonzero(val & ok),
        "holdout": np.flatnonzero(held & ok),
    }


@torch.no_grad()
def evaluate(model, data, idx, y_var, y_mz, plays, batch=1024) -> dict:
    model.eval()
    out = {"var": [], "mz": [], "pvar": [], "pmz": []}
    for s in range(0, len(idx), batch):
        b = torch.as_tensor(idx[s : s + batch], device=data["raw"].device)
        raw, mask, ref = data["raw"][b], data["mask"][b], data["ref"][b]
        lv, lm = model(featurize(raw, mask, ref), mask)
        last, pre = read_indices(mask)
        r = torch.arange(len(b), device=raw.device)
        out["var"].append(torch.softmax(lv[r, last].float(), -1).cpu())
        out["mz"].append(torch.softmax(lm[r, last].float(), -1).cpu())
        out["pvar"].append(torch.softmax(lv[r, pre].float(), -1).cpu())
        out["pmz"].append(torch.softmax(lm[r, pre].float(), -1).cpu())
    probs = {k: torch.cat(v).numpy() for k, v in out.items()}
    yv, ym = y_var[idx], y_mz[idx]
    ds = plays["dataset"].to_numpy()[idx]
    has_v, has_m = yv >= 0, ym >= 0
    m: dict = {"n": int(has_v.sum()), "n_man_zone": int(has_m.sum())}
    for tag, pv, pm in (("", probs["var"], probs["mz"]), ("pre_", probs["pvar"], probs["pmz"])):
        order = np.argsort(-pv, axis=1)
        top1 = order[:, 0] == yv
        top2 = top1 | (order[:, 1] == yv)
        mz = pm.argmax(1) == ym
        m[f"{tag}top1"] = round(float(top1[has_v].mean()), 4)
        m[f"{tag}top2"] = round(float(top2[has_v].mean()), 4)
        m[f"{tag}man_zone"] = round(float(mz[has_m].mean()), 4)
        for d in ("bdb2021", "bdb2026"):
            sel = has_v & (ds == d)
            m[f"{tag}top1_{d}"] = round(float(top1[sel].mean()), 4) if sel.any() else None
    m["majority"] = round(float(np.bincount(yv[has_v]).max() / has_v.sum()), 4)
    model.train()
    return m


def hgb_reference(plays: pd.DataFrame, sp: dict, classes: list[str]) -> dict:
    """The variant model's recipe (snap features, HGB) on the same train/val split."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    feats = pd.read_parquet(FEATURES_PATH)
    cols = sorted(c for c in feats.columns if c.startswith("snap_"))
    tab = plays.merge(feats[["dataset", "game_id", "play_id", *cols]], how="left")
    tab = tab.assign(row=np.arange(len(tab)))
    lab = tab["coverage_type"].isin(classes).to_numpy()
    tr, va = tab.iloc[sp["train"]][lab[sp["train"]]], tab.iloc[sp["val"]][lab[sp["val"]]]
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    model.fit(tr[cols], tr["coverage_type"])
    proba = model.predict_proba(va[cols])
    order = np.argsort(-proba, axis=1)
    truth = va["coverage_type"].to_numpy()
    top1 = model.classes_[order[:, 0]] == truth
    top2 = top1 | (model.classes_[order[:, 1]] == truth)
    return {
        "n": len(va),
        "top1": round(float(top1.mean()), 4),
        "top2": round(float(top2.mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--spatial-layers", type=int, default=2)
    ap.add_argument("--temporal-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--step-weight", type=float, default=0.5)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train plays (smoke)")
    ap.add_argument("--name", default="sequence_v1")
    ap.add_argument("--score-holdout", action="store_true")
    ap.add_argument("--hgb-reference", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tensors, plays = build_cache()
    classes = variant_classes()
    plays["ref_ok"] = ~torch.isnan(tensors["ref"]).any(1).numpy()
    y_var = encode_labels(plays["coverage_type"], classes)
    y_mz = encode_labels(plays["man_zone"], MAN_ZONE)
    sp = splits(plays, args.val_frac, args.seed)
    if args.max_train:
        sp["train"] = np.random.default_rng(0).permutation(sp["train"])[: args.max_train]
    print(
        f"classes {classes}\nplays train {len(sp['train'])} val {len(sp['val'])} "
        f"holdout {len(sp['holdout'])} (no LOS: {(~plays['ref_ok']).sum()}), "
        f"cache {time.time() - t0:.0f}s, device {dev}",
        flush=True,
    )
    ref_metrics = hgb_reference(plays, sp, classes) if args.hgb_reference else None
    if ref_metrics:
        print(f"HGB snap-feature reference on val: {ref_metrics}", flush=True)

    data = {k: v.to(dev) for k, v in tensors.items()}
    data["ref"] = torch.nan_to_num(data["ref"])
    yv_t, ym_t = torch.as_tensor(y_var, device=dev), torch.as_tensor(y_mz, device=dev)
    cfg = Config(
        n_variant=len(classes),
        d=args.d,
        heads=4,
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        ff=2 * args.d,
        dropout=args.dropout,
    )
    model = SequenceModel(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(sp["train"]) // args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_lr(s, total, warmup=min(500, total // 10))
    )
    aug = None if args.no_aug else Augment()
    use_amp = dev == "cuda"
    best, best_state, history = -1.0, None, []
    t_train = time.time()
    for epoch in range(args.epochs):
        perm = np.random.permutation(sp["train"])
        run = []
        for s in range(steps_per_epoch):
            b = torch.as_tensor(perm[s * args.batch : (s + 1) * args.batch], device=dev)
            raw, mask, ref = data["raw"][b], data["mask"][b], data["ref"][b]
            if aug:
                raw, mask, ref = aug(raw, mask, ref)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = losses(
                    model, featurize(raw, mask, ref), mask, yv_t[b], ym_t[b], args.step_weight
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(loss.item())
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            val = evaluate(model, data, sp["val"], y_var, y_mz, plays)
        history.append({"epoch": epoch + 1, "loss": round(float(np.mean(run)), 4), **val})
        print(
            f"ep {epoch + 1:3d} loss {np.mean(run):.4f} | val top1 {val['top1']:.4f} "
            f"top2 {val['top2']:.4f} mz {val['man_zone']:.4f} | pre top1 {val['pre_top1']:.4f} "
            f"mz {val['pre_man_zone']:.4f} | {time.time() - t_train:.0f}s",
            flush=True,
        )
        if val["top1"] > best:
            best = val["top1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_s = time.time() - t_train
    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda h: h["top1"])
    meta = {
        "args": vars(args),
        "n_train": len(sp["train"]),
        "n_val": len(sp["val"]),
        "train_seconds": round(train_s),
        "best_epoch": best_epoch["epoch"],
        "val": best_epoch,
        "val_hgb_reference": ref_metrics,
        "history": history,
        "holdout_weeks": HOLDOUT,
    }
    if args.score_holdout:
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            meta["holdout"] = evaluate(model, data, sp["holdout"], y_var, y_mz, plays)
        print(f"HOLDOUT: {meta['holdout']}", flush=True)
    path = MODELS_DIR / f"{args.name}.pt"
    save(Bundle(model.cpu(), classes, meta), path)
    print(f"best val epoch {best_epoch['epoch']}: {best_epoch}\nsaved {path} ({train_s:.0f}s)")
    print(f"config {asdict(cfg)}")


if __name__ == "__main__":
    main()
