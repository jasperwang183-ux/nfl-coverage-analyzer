"""Verify the pipeline beyond the unit tests: does it run on unseen film, and is it right?

  film     every clip in samples/verify/clips.json through every stage (label-free
           health checks; `expect` facts compared when an entry has them)
  bdb      accuracy on the frozen Big Data Bowl holdout weeks (real coverage labels)
  compare  two saved runs side by side: what a refactor or algorithm swap changed
  add      a new clip: suggest the calibration frame and window, draw the numbered
           yard lines, print the manifest entry to fill in

Runs are saved under data/verify/runs/<tag>/ (gitignored; the tag defaults to the
commit, with -dirty when the tree has changes). Tracking is cached per clip,
window and calibration naming; pass --retrack after changing detection/tracking.

Usage:
  python scripts/verify.py film [--only ID ...] [--tag NAME] [--retrack]
  python scripts/verify.py bdb [--checks formation shell film_read variant disguise]
  python scripts/verify.py compare BEFORE_TAG AFTER_TAG
  python scripts/verify.py add PATH_TO_CLIP.mp4 --id NAME
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from nfl_coverage.config import MODELS_DIR, ROOT, SEQUENCE_MODEL

MANIFEST = ROOT / "samples" / "verify" / "clips.json"
RUNS = ROOT / "data" / "verify" / "runs"
CACHE = ROOT / "data" / "verify" / "cache"


def default_tag() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout

    sha = git("rev-parse", "--short", "HEAD").strip() or "nogit"
    dirty = "-dirty" if git("status", "--porcelain", "--", "src", "scripts").strip() else ""
    return sha + dirty


def save(tag: str, name: str, payload: dict) -> Path:
    d = RUNS / tag
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(payload, indent=1, default=str))
    return p


# ------------------------------------------------------------------ film


def cmd_film(args) -> None:
    from nfl_coverage.vision.verify import load_manifest, verify_clip

    bundle = None
    model = MODELS_DIR / "still_shown.joblib"
    if model.exists():
        from nfl_coverage.models.persist import load_bundle

        bundle = load_bundle(model)
    from nfl_coverage.models.sequence import load_if_present

    sequence_bundle = load_if_present(SEQUENCE_MODEL)
    clips = load_manifest(args.manifest)
    if args.only:
        clips = [c for c in clips if c["id"] in args.only]
    results = []
    for c in clips:
        print(f"{c['id']}")
        r = verify_clip(
            c,
            CACHE,
            bundle,
            retrack=args.retrack,
            imgsz=args.imgsz,
            auto_naming=args.auto_naming,
            sequence_bundle=sequence_bundle,
        )
        results.append(r)
        for s in r["stages"]:
            extra = f"  {s['error']}" if s["error"] else ""
            extra += "".join(f"\n        - {n}" for n in s["notes"][:4])
            print(f"    {s['status']:<7} {s['name']:<12} {s['seconds']:6.1f}s{extra}")
        for k, v in r.get("expect", {}).items():
            print(
                f"    {'ok' if v['ok'] else 'WRONG':<7} expect {k}: {v['expected']} got {v['got']}"
            )
    tag = args.tag or default_tag()
    p = save(tag, "film", {"tag": tag, "manifest": str(args.manifest), "clips": results})
    print_film_table(results)
    print(f"\nsaved {p}")


def print_film_table(results: list[dict]) -> None:
    from nfl_coverage.vision.verify import STAGE_ORDER

    seen = {s["name"] for r in results for s in r["stages"]}
    names = [n for n in STAGE_ORDER if n in seen]
    mark = {"ok": "ok", "warn": "warn", "fail": "FAIL", "skipped": "-"}
    print("\n" + f"{'clip':<22}" + "".join(f"{n:<12}" for n in names) + "verdict")
    for r in results:
        by = {s["name"]: s["status"] for s in r["stages"]}
        cells = "".join(f"{mark.get(by.get(n, ''), '-'):<12}" for n in names)
        print(f"{r['id']:<22}{cells}{r['status']}")
    n_ok = sum(not r["status"].startswith("fail") for r in results)
    print(f"\n{n_ok}/{len(results)} clips produce a read")


# ------------------------------------------------------------------ bdb


def cmd_bdb(args) -> None:
    from nfl_coverage.analysis.holdout import HOLDOUT, run_holdout

    res = run_holdout(tuple(args.checks), sample=args.sample)
    tag = args.tag or default_tag()
    p = save(tag, "bdb", {"tag": tag, "holdout": HOLDOUT, "sample": args.sample, "results": res})
    for check, parts in res.items():
        print(f"\n=== {check} ===")
        for part, metrics in parts.items():
            print(f"  {part:<20} " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    print(f"\nsaved {p}")


# ------------------------------------------------------------------ compare


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-9) / n) if n else float("inf")


def cmd_compare(args) -> None:
    a_dir, b_dir = RUNS / args.before, RUNS / args.after
    for d in (a_dir, b_dir):
        if not d.exists():
            sys.exit(f"no run {d.name}; runs: {', '.join(sorted(p.name for p in RUNS.iterdir()))}")
    if (a_dir / "film.json").exists() and (b_dir / "film.json").exists():
        compare_film(
            json.loads((a_dir / "film.json").read_text()),
            json.loads((b_dir / "film.json").read_text()),
        )
    if (a_dir / "bdb.json").exists() and (b_dir / "bdb.json").exists():
        compare_bdb(
            json.loads((a_dir / "bdb.json").read_text()),
            json.loads((b_dir / "bdb.json").read_text()),
        )


def compare_film(a: dict, b: dict) -> None:
    print(f"=== film: {a['tag']} -> {b['tag']} ===")
    before = {r["id"]: r for r in a["clips"]}
    keys = (
        "mean_error",
        "n_landmarks",
        "worst_residual",
        "n_offense",
        "n_defense",
        "snap_t",
        "offense",
        "los",
        "shown",
        "played",
        "shown_confidence",
        "played_confidence",
    )
    for r in b["clips"]:
        old = before.get(r["id"])
        if old is None:
            print(f"  {r['id']:<22} new: {r['status']}")
            continue
        flag = "" if old["status"] == r["status"] else "   <-- verdict changed"
        print(f"  {r['id']:<22} {old['status']} -> {r['status']}{flag}")
        om = {k: v for s in old["stages"] for k, v in s["metrics"].items()}
        nm = {k: v for s in r["stages"] for k, v in s["metrics"].items()}
        for k in keys:
            if k in om or k in nm:
                if om.get(k) != nm.get(k):
                    print(f"      {k}: {om.get(k)} -> {nm.get(k)}")
        oe, ne = old.get("expect", {}), r.get("expect", {})
        for k in ne:
            if oe.get(k, {}).get("ok") != ne[k]["ok"]:
                print(
                    f"      expect {k}: {'ok' if oe.get(k, {}).get('ok') else 'wrong'} -> "
                    f"{'ok' if ne[k]['ok'] else 'wrong'}"
                )
    for rid in set(before) - {r["id"] for r in b["clips"]}:
        print(f"  {rid:<22} (not in the new run)")


RATE_KEYS = (
    "right",
    "wrong",
    "abstain",
    "run_mof_accuracy",
    "mof_accuracy",
    "top1",
    "top2",
    "acc_high",
    "acc_medium",
    "precision",
    "recall",
)


def compare_bdb(a: dict, b: dict) -> None:
    print(f"\n=== bdb holdout: {a['tag']} -> {b['tag']} ===")
    print("  (moves under 2 standard errors are marked ~ : likely noise)")
    for check, parts in b["results"].items():
        for part, m in parts.items():
            old = a["results"].get(check, {}).get(part, {})
            for k, v in m.items():
                ov = old.get(k)
                if k not in RATE_KEYS or not isinstance(v, (int, float)) or ov is None:
                    continue
                if isinstance(ov, float) and math.isnan(ov):
                    continue
                n = m.get("n") or 0
                d = v - ov
                se = math.hypot(_se(ov, old.get("n") or 0), _se(v, n))
                sig = "~" if abs(d) < 2 * se else ("+" if d > 0 else "-")
                print(f"  {sig} {check}/{part}/{k}: {ov:.3f} -> {v:.3f} ({d:+.3f}, n={n})")


# ------------------------------------------------------------------ add


def cmd_add(args) -> None:
    import cv2

    from nfl_coverage.vision.fieldlines import find_marks
    from nfl_coverage.vision.pipeline import _frame_at
    from nfl_coverage.web.app import draw_numbered_lines, sample_video, suggest_calibration_frame

    video = Path(args.video).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        samples = sample_video(video, Path(tmp))
    i = suggest_calibration_frame(samples)
    t = args.t if args.t is not None else next(s["t"] for s in samples if s["i"] == i)
    img = _frame_at(video, t)
    marks = find_marks(img)
    out = ROOT / "data" / "verify" / "add"
    out.mkdir(parents=True, exist_ok=True)
    lines_img = out / f"{args.id}_lines_{t:.1f}s.jpg"
    cv2.imwrite(str(lines_img), draw_numbered_lines(img, marks.lines, marks.offsets))
    duration = samples[-1]["t"] if samples else 0.0
    x_mid = [round(ln.x_at(marks.ref_y)) for ln in marks.lines]
    entry = {
        "id": args.id,
        "video": str(video).replace("\\", "/"),
        "calib": {"t": round(t, 2), "named_x": x_mid[0], "painted": None, "going_up": True},
        "window": [round(max(0.0, t - 1.0), 2), round(min(duration, t + 4.5), 2)],
        "source": "",
        "expect": {},
    }
    print(
        f"{len(marks.lines)} lines found at {t:.2f} s "
        f"({sum(o is not None for o in marks.offsets)} numbered as yard lines)"
    )
    print(f"numbered lines drawn on {lines_img}")
    print("line x at mid-frame: " + ", ".join(f"{k + 1}={x}" for k, x in enumerate(x_mid)))
    print("fill in `painted` (the painted yard of the line at named_x - line 1 - 0-50) and")
    print("`going_up` (whether the painted numbers rise to its right); name another line by")
    print("putting its x in named_x. Then add this to the manifest:\n")
    print(json.dumps(entry, indent=2))


def main() -> None:
    # empty-slice means on frames with no bodies in a band are expected; they drown the report
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("film")
    f.add_argument("--manifest", type=Path, default=MANIFEST)
    f.add_argument("--only", nargs="*")
    f.add_argument("--tag")
    f.add_argument("--retrack", action="store_true")
    f.add_argument("--imgsz", type=int, default=1920)
    f.add_argument(
        "--auto-naming", action="store_true", help="name the yard lines as the page does, unaided"
    )
    b = sub.add_parser("bdb")
    b.add_argument(
        "--checks", nargs="*", default=["formation", "shell", "film_read", "variant", "disguise"]
    )
    b.add_argument("--sample", type=int, default=400, help="plays per dataset for film_read")
    b.add_argument("--tag")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    a = sub.add_parser("add")
    a.add_argument("video")
    a.add_argument("--id", required=True)
    a.add_argument("--t", type=float, help="calibration frame time (default: the page's pick)")
    args = ap.parse_args()
    {"film": cmd_film, "bdb": cmd_bdb, "compare": cmd_compare, "add": cmd_add}[args.cmd](args)


if __name__ == "__main__":
    main()
