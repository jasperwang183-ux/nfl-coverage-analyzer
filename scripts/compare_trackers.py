"""Which multi-object tracker keeps defenders' identities on film? Scored against truth.

Re-tracks one page clip of a play that is also in the Big Data Bowl tracking data with
each tracker in TRACKERS, reusing the clip's calibration chain (it depends on the frames,
not on the tracker), reads the play, and scores every defender's identity from the snap
to +2.5 s with `vision.identity`. Prints one row per tracker and the named player's row.

Usage:
  python scripts/compare_trackers.py --clip 187dcd12b5d2 --dataset bdb2026 --week 2 \
      --game 2023091709 --play 3503 --player "Fred Warner"
"""

from __future__ import annotations

import argparse
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nfl_coverage.analysis.mechanisms import mechanism_hits
from nfl_coverage.config import BDB2026, DATA_PROCESSED, ROOT
from nfl_coverage.features.phases import defender_phases
from nfl_coverage.vision.identity import identity_report
from nfl_coverage.vision.pipeline import TrackedClip, _canonical, read_tracked
from nfl_coverage.vision.track import track_players

# name -> (ultralytics base config, overrides)
TRACKERS: dict[str, tuple[str, dict]] = {
    "bytetrack": ("bytetrack", {}),
    "botsort": ("botsort", {}),
    "botsort+reid": ("botsort", {"with_reid": True}),
    "deepocsort+reid": ("deepocsort", {"with_reid": True, "gmc_method": "sparseOptFlow"}),
    "tracktrack+reid": (
        "tracktrack",
        {"with_reid": True, "track_high_thresh": 0.3, "new_track_thresh": 0.35},
    ),
}
HORIZON = 25  # rel_frames scored: the snap to +2.5 s, where the disguise phases live


def tracker_yaml(base: str, overrides: dict, out_dir: Path) -> str:
    import ultralytics

    cfg = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{base}.yaml"
    conf = yaml.safe_load(cfg.read_text()) | overrides
    path = out_dir / f"{base}-{abs(hash(frozenset(overrides.items())))}.yaml"
    path.write_text(yaml.safe_dump(conf))
    return str(path)


def truth_play(dataset: str, week: int, game: int, play: int) -> pd.DataFrame:
    f = pd.read_parquet(DATA_PROCESSED / "frames" / f"{dataset}_w{week:02d}.parquet")
    return f[(f["game_id"] == game) & (f["play_id"] == play)]


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="page clip id under data/web/")
    ap.add_argument("--dataset", default="bdb2026")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--play", type=int, required=True)
    ap.add_argument("--player", default=None, help="a player to report on by name")
    ap.add_argument("--only", nargs="*", default=None, help="tracker names to run")
    args = ap.parse_args()

    d = ROOT / "data" / "web" / args.clip
    base = TrackedClip.load(d / "tracked")
    snap_t = read_tracked(base).snap_t
    start, end = base.frames[0].t, base.frames[-1].t
    truth = truth_play(args.dataset, args.week, args.game, args.play)
    player_id = None
    if args.player:
        names = pd.read_csv(
            BDB2026 / "train" / f"input_2023_w{args.week:02d}.csv",
            usecols=["game_id", "play_id", "nfl_id", "player_name"],
        )
        hit = names[
            (names["game_id"] == args.game)
            & (names["play_id"] == args.play)
            & (names["player_name"] == args.player)
        ]
        player_id = int(hit["nfl_id"].iloc[0]) if len(hit) else None

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (cfg, overrides) in TRACKERS.items():
            if args.only and name not in args.only:
                continue
            t0 = time.time()
            frames = track_players(
                d / "clip.mp4",
                start,
                end + 1e-3,
                base.step,
                imgsz=1920,
                tracker=tracker_yaml(cfg, overrides, Path(tmp)),
            )
            if len(frames) != len(base.frames):
                print(f"{name}: {len(frames)} frames vs {len(base.frames)}: skipped")
                continue
            clip = TrackedClip(frames, base.calibrations, base.step, list(base.notes))
            read = read_tracked(clip, snap_t=snap_t)
            rep = identity_report(read.tracks, truth, HORIZON)
            scored = [p for p in rep if p.frames]
            los_c = _canonical(read.offense, read.los, read.ball_y)[0]
            phases = defender_phases(
                read.tracks, los=los_c, center=read.centers.mechanisms, step=read.step
            )
            me = next((p for p in rep if p.truth_id == player_id), None)
            rows.append(
                {
                    "tracker": name,
                    "s": round(time.time() - t0),
                    "defenders": len(scored),
                    "matched_at_snap": sum(p.snap_film_id is not None for p in scored),
                    "kept_share": round(
                        sum(p.kept for p in scored) / sum(p.frames for p in scored), 3
                    ),
                    "switches": sum(p.switches for p in scored),
                    "error_yd": round(float(np.nanmean([p.mean_error_yd for p in scored])), 2),
                    "player_kept": f"{me.kept}/{me.frames}" if me else "-",
                    "player_switches": me.switches if me else "-",
                    "mechanisms": ",".join(mechanism_hits(phases)["mechanism"]) or "-",
                }
            )
            print(rows[-1])
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
