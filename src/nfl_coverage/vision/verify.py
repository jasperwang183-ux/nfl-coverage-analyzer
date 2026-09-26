"""Does the film pipeline run, stage by stage, on clips it was not built on?

Each clip in the manifest (`samples/verify/clips.json`) goes through what the
page does - lines, calibration from a stored line naming, carry into the play
window, tracking, the read - and every stage gets a record: ok / warn / fail /
skipped, seconds, metrics, and the error when it fails. None of it needs a
coverage label. The checks are self-consistency:

  calibration  the fit reproduces its landmarks; the painted numbers do not prefer
               the naming shifted 5 yd; not anchored on the hashes alone.
  grid         along the tracked window, yard lines found independently on the
               frame project to whole 5-yd lines through the chained calibration
               (drift shows as a residual; a whole 5-yd slip does not - the
               numbers check covers the anchor only).
  read         a snap was found, the sides count ~11 each, the offense side is
               called with some confidence, shown and played reads exist.

A manifest entry may also carry `expect` (what a person knows about the play);
those are compared and reported, but a clip without them is fully checked.
`scripts/verify.py film` is the CLI.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GRID_SAMPLE_EVERY = 1.0  # s between frames the grid-drift check looks at
GRID_WARN = 0.75  # yd median residual of detected yard lines off the 5-yd grid
GRID_FAIL = 1.5
# Defenders at the snap: the coverage read is built on them, so their count decides.
DEFENSE_OK = (10, 11)  # one may be hidden behind another
DEFENSE_WARN = (8, 12)
OFFENSE_WARN = 9  # engaged linemen merge into a few boxes: a short offense is common, warn only
OFFENSE_FAIL = 6  # below this the formation (LOS, offense side, ball) is a guess
STAGE_ORDER = ["probe", "lines", "calibration", "carry", "tracking", "grid", "read"]
# Source files whose change invalidates cached tracking (detection, tracking, calibration chain)
TRACKING_SOURCES = [
    "vision/detect.py",
    "vision/track.py",
    "vision/teams.py",
    "vision/homography.py",
    "vision/clip.py",
    "vision/fieldlines.py",
    "vision/pipeline.py",
]


@dataclass
class Stage:
    name: str
    status: str = "ok"  # ok / warn / fail / skipped
    seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def warn(self, note: str) -> None:
        self.notes.append(note)
        if self.status == "ok":
            self.status = "warn"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 1),
            "metrics": self.metrics,
            "notes": self.notes,
            "error": self.error,
        }


class StageFailed(Exception):
    """A stage cannot produce what the next one needs; the rest are skipped."""


@contextmanager
def _stage(stages: list[Stage], name: str):
    st = Stage(name)
    stages.append(st)
    t0 = time.perf_counter()
    try:
        yield st
    except StageFailed as e:
        st.status, st.error = "fail", str(e)
        raise
    except Exception as e:
        st.status = "fail"
        tb = traceback.format_exc().strip().splitlines()
        st.error = f"{type(e).__name__}: {e} | " + " / ".join(tb[-3:-1])
        raise StageFailed(st.error) from e
    finally:
        st.seconds = time.perf_counter() - t0


# ------------------------------------------------------------------ pure checks


def grid_residual(xs: np.ndarray) -> float:
    """Median distance (yd) of projected yard-line x's from the nearest whole 5 yd."""
    xs = np.asarray(xs, float)
    if not len(xs):
        return float("nan")
    return float(np.median(np.abs(xs - 5.0 * np.round(xs / 5.0))))


def side_status(n_offense: int, n_defense: int) -> str:
    """ok / warn / fail for the snap's side counts; the defense decides a fail."""
    if not DEFENSE_WARN[0] <= n_defense <= DEFENSE_WARN[1] or n_offense < OFFENSE_FAIL:
        return "fail"
    if not DEFENSE_OK[0] <= n_defense <= DEFENSE_OK[1] or not OFFENSE_WARN <= n_offense <= 11:
        return "warn"
    return "ok"


def compare_expect(expect: dict, result: dict) -> dict[str, dict]:
    """Each known fact vs what the read said: {key: {expected, got, ok}}."""
    rep = result.get("report", {})
    shown, played = rep.get("shown") or {}, rep.get("played") or {}
    got = {
        "offense": result.get("formation", {}).get("used", {}).get("offense"),
        "snap_t": result.get("snap", {}).get("t"),
        "los": result.get("formation", {}).get("used", {}).get("los"),
        "shown_mof_closed": shown.get("mof_closed"),
        "played_mof_closed": played.get("mof_closed"),
        "shown_n_deep": shown.get("n_deep"),
        "played_n_deep": played.get("n_deep"),
    }
    tol = {"snap_t": 0.25, "los": 1.0}
    out = {}
    for key, want in expect.items():
        if key not in got:
            continue
        g = got[key]
        if g is None:
            ok = False
        elif key in tol:
            ok = abs(float(g) - float(want)) <= tol[key]
        else:
            ok = g == want
        out[key] = {"expected": want, "got": g, "ok": bool(ok)}
    return out


def naming_yards(
    offsets: list[int | None], calib: dict, x_mid: list[float] | None = None
) -> list[float | None]:
    """Field x per detected line from the stored naming (as the page's calibrate does).

    The named line is found by `named_x` (its pixel x at the marks' reference row)
    when given - robust to detection gaining or losing a line elsewhere - else by
    its 1-based `named_line` index, as the page asks for it.
    """
    from nfl_coverage.vision.fieldlines import field_x_for_lines

    if calib.get("painted") is None:
        raise ValueError("the manifest entry has no painted yard for its named line")
    if calib.get("named_x") is not None and x_mid:
        k = int(np.argmin(np.abs(np.asarray(x_mid) - calib["named_x"])))
        if abs(x_mid[k] - calib["named_x"]) > 60:
            raise ValueError(
                f"no detected line near x={calib['named_x']} (nearest at {x_mid[k]:.0f})"
            )
    else:
        k = calib.get("named_line", 1) - 1
    return field_x_for_lines(offsets, k, calib["painted"], calib.get("going_up", True))


def tracking_code_hash() -> str:
    """Content hash of the code tracking depends on: a change there re-tracks by itself."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha1()
    for rel in TRACKING_SOURCES:
        h.update((root / rel).read_bytes())
    return h.hexdigest()[:8]


def cache_key(video: Path, entry: dict, imgsz: int) -> str:
    st = video.stat()
    # tracking depends on the anchor calibration and the window, not on read overrides
    blob = json.dumps(
        [video.name, st.st_size, entry["calib"], entry["window"], imgsz, tracking_code_hash()],
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ------------------------------------------------------------------ one clip


def verify_clip(
    entry: dict,
    cache_root: Path,
    bundle: dict | None = None,
    retrack: bool = False,
    imgsz: int = 1920,
    log: Callable[[str], None] = print,
) -> dict:
    """Run one manifest entry through every stage; never raises."""
    import cv2

    from nfl_coverage.vision.fieldlines import (
        find_marks,
        five_yard_shift_warning,
        n_number_edges,
        search_calibration,
    )
    from nfl_coverage.vision.pipeline import (
        ReadError,
        TrackedClip,
        _frame_at,
        carry_calibration,
        read_tracked,
        track_frames,
    )

    stages: list[Stage] = []
    out: dict = {"id": entry["id"], "video": entry["video"], "stages": stages}
    video = Path(entry["video"])
    if not video.is_absolute():
        from nfl_coverage.config import ROOT

        video = ROOT / video
    cal_t = float(entry["calib"]["t"])
    start, end = (float(v) for v in entry["window"])
    result: dict | None = None
    try:
        with _stage(stages, "probe") as st:
            if not video.exists():
                raise StageFailed(f"video not found: {video}")
            cap = cv2.VideoCapture(str(video))
            fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if not fps or not n:
                raise StageFailed("OpenCV cannot read the video")
            st.metrics = {"fps": round(fps, 2), "duration": round(n / fps, 2)}
            if not 0 <= start < end <= n / fps + 0.05:
                raise StageFailed(f"window {start}-{end} s is outside the clip")

        with _stage(stages, "lines") as st:
            img = _frame_at(video, cal_t)
            marks = find_marks(img)
            n_yard = sum(o is not None for o in marks.offsets)
            st.metrics = {"n_lines": len(marks.lines), "n_yard_lines": n_yard}
            if n_yard < 2:
                raise StageFailed(f"only {n_yard} yard lines at {cal_t} s")
            if n_yard < 3:
                st.warn("fewer than 3 yard lines: a thin base for the fit")

        with _stage(stages, "calibration") as st:
            x_mid = [ln.x_at(marks.ref_y) for ln in marks.lines]
            yards = naming_yards(marks.offsets, entry["calib"], x_mid)
            search = search_calibration(img, yards, marks=marks)
            best = search.best
            st.metrics = {
                "n_landmarks": len(best.pixel),
                "n_rows": search.n_rows,
                "n_number_edges": n_number_edges(best),
                "mean_error": round(float(best.errors.mean()), 3),
                "max_error": round(float(best.errors.max()), 3),
                "n_ok": search.n_ok,
                "n_tried": search.n_tried,
            }
            if search.warning:
                st.warn("hash rows only: depth toward the sidelines is extrapolated")
            shift = five_yard_shift_warning(img, yards, search, marks=marks)
            if shift:
                st.warn("numbers prefer a 5-yd shift: " + shift)
            anchor, anchor_t = best.calibration, cal_t

        if not start <= cal_t <= end:
            with _stage(stages, "carry") as st:
                anchor, notes = carry_calibration(video, anchor, cal_t, start)
                anchor_t = start
                st.notes = notes
                if any("kept the carried" in n for n in notes):
                    st.warn("refit in the window failed; the carried calibration was kept")
                if any(n.startswith("calibrated on the hash") for n in notes):
                    st.warn("refit on the hash rows alone: depth toward the sidelines extrapolated")

        with _stage(stages, "tracking") as st:
            key = cache_key(video, entry, imgsz)
            cdir = Path(cache_root) / f"{entry['id']}-{key}"
            if retrack or not (cdir / "tracked.json").exists():
                log(f"    tracking {start}-{end} s (minutes on a CPU)")
                clip = track_frames(video, start, end, anchor, anchor_t, imgsz=imgsz)
                clip.caveats = [search.warning] if search.warning else []
                clip.save(cdir)
                st.metrics["cached"] = False
            else:
                st.metrics["cached"] = True
            clip = TrackedClip.load(cdir)
            per_frame = [int((f.ids >= 0).sum()) for f in clip.frames]
            st.metrics.update(
                {
                    "frames": len(clip.frames),
                    "bodies_median": float(np.median(per_frame)),
                    "bodies_min": int(min(per_frame)),
                    "cache": str(cdir),
                }
            )
            st.notes = list(clip.notes)
            if np.median(per_frame) < 18:
                st.warn(f"median {np.median(per_frame):.0f} tracked bodies per frame")

        with _stage(stages, "grid") as st:
            residuals, found = [], []
            every = max(1, round(GRID_SAMPLE_EVERY / clip.step))
            for i in range(0, len(clip.frames), every):
                fr, cal = clip.frames[i], clip.calibrations[i]
                m = find_marks(fr.image)
                ys = [m.ref_y - 0.1 * fr.image.shape[0], m.ref_y, m.ref_y + 0.1 * fr.image.shape[0]]
                pts = np.float32([[ln.x_at(y), y] for ln in m.lines for y in ys])
                if len(pts) == 0:
                    continue
                xs = cal.to_field(pts)[:, 0]
                residuals.append(grid_residual(xs))
                found.append(len(m.lines))
            if not residuals:
                raise StageFailed("no yard lines found on any tracked frame")
            worst = float(np.nanmax(residuals))
            st.metrics = {
                "frames_checked": len(residuals),
                "median_residual": round(float(np.nanmedian(residuals)), 3),
                "worst_residual": round(worst, 3),
                "worst_at_s": round(float(clip.frames[int(np.nanargmax(residuals)) * every].t), 2),
            }
            if worst >= GRID_FAIL:
                st.status = "fail"
                st.error = f"grid drifts {worst:.2f} yd off the painted lines"
            elif worst >= GRID_WARN:
                st.warn(f"grid drifts {worst:.2f} yd off the painted lines")

        with _stage(stages, "read") as st:
            try:
                given = entry.get("given", {})
                read = read_tracked(
                    clip,
                    snap_t=given.get("snap_t"),
                    los=given.get("los"),
                    offense=given.get("offense"),
                    ball_y=given.get("ball_y"),
                    bundle=bundle,
                )
            except ReadError as e:
                raise StageFailed(f"needs input: {e}") from e
            result = read.to_dict()
            rep = result["report"]
            sides = result["sides"]
            st.metrics = {
                "snap_t": round(result["snap"]["t"], 2),
                "snap_confidence": result["snap"]["confidence"],
                "offense": result["formation"]["used"]["offense"],
                "offense_confidence": result["formation"]["inferred"]["offense_confidence"],
                "los": round(result["formation"]["used"]["los"], 1),
                "los_confidence": result["formation"]["inferred"]["los_confidence"],
                "n_offense": sides["offense"],
                "n_defense": sides["defense"],
                "shown": (rep.get("shown") or {}).get("shell"),
                "shown_confidence": (rep.get("shown") or {}).get("confidence"),
                "played": (rep.get("played") or {}).get("shell"),
                "played_confidence": (rep.get("played") or {}).get("confidence"),
                "disguise": rep.get("disguise_label"),
                "variant_top": (rep.get("variant") or [[None]])[0][0],
            }
            s = side_status(sides["offense"], sides["defense"])
            if s != "ok":
                st.warn(f"{sides['offense']} offense / {sides['defense']} defense at the snap")
            if s == "fail":
                st.status = "fail"
                st.error = (
                    f"{sides['offense']} offense / {sides['defense']} defense at the snap: "
                    "too few bodies for the read to mean much"
                )
            if result["formation"]["inferred"]["offense_confidence"] == "low":
                st.warn("offense side called at low confidence")
            if not rep.get("played"):
                st.warn("no played read (the window ends too early?)")
            for w in result.get("warnings", []):
                st.warn(w)
    except StageFailed:
        pass

    if entry.get("expect") and result is not None:
        out["expect"] = compare_expect(entry["expect"], result)
    out["stages"] = [s.to_dict() for s in stages]
    out["status"] = _rollup(out["stages"])
    return out


def _rollup(stages: list[dict]) -> str:
    """The clip's verdict: its first failing stage, else warn / ok."""
    for s in stages:
        if s["status"] == "fail":
            return f"fail@{s['name']}"
    return "warn" if any(s["status"] == "warn" for s in stages) else "ok"


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["clips"] if isinstance(data, dict) else data
