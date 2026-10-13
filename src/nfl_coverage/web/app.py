"""The page: upload an mp4, get its pre-snap and post-snap reads, correct them.

One FastAPI app serving a single static page and a small JSON API. Each uploaded
clip lives in its own folder under the data directory (gitignored `data/web/`):

  clip.mp4, meta.json            the upload and its state (stage, progress, choices)
  samples/sample_XXX.jpg         one thumbnail per SAMPLE_EVERY s, for the frame picker
  lines.jpg                      detected yard lines, numbered, on the chosen frame
  calibration.json, grid.jpg     the fitted calibration and the field drawn back on it
  tracked/                       the cached tracked frames (`pipeline.TrackedClip`)
  result.json, snap.jpg, topdown.png   the read, the annotated snap frame, the render

Flow: POST /api/clips (upload) starts the whole job in a background thread: the
yard lines are named on a still pre-snap frame with no person, then the play is
tracked and read; poll GET /api/clips/{id} for its `stage` (see STAGES). POST
.../correction records a person's verdict on either read. The manual steps are
the fallback and the adjust path: POST .../lines, .../calibrate (a person names
one line), .../analyze (another window) and .../reread (overrides, from the
cached frames).

Run: `python scripts/serve.py` and open http://127.0.0.1:8000.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nfl_coverage.config import DATA_PROCESSED, MODELS_DIR, ROOT, SEQUENCE_MODEL

STATIC = Path(__file__).parent / "static"
DEFAULT_DATA = ROOT / "data" / "web"
SAMPLE_EVERY = 0.5  # s between picker thumbnails
THUMB_WIDTH = 640
STILL_SCORE = 1.0  # motion score of a paused / pre-snap still stretch
MOVING_SCORE = 3.0  # ... and of a play in motion
PLAY_PEAK = 10.0  # a run peaking below this is a camera move, not the play
WINDOW_BEFORE = 1.0  # s tracked before the sample where the play starts moving
# ...and after it: the snap comes within ~0.5 s of the last still frame, the played read
# is taken 1.5 s after the snap and the sequence model reads to +3 s
WINDOW_AFTER = 6.0
MAX_AUTO_FRAMES = 3  # ~5 s a try; neighbouring still frames fail together, so ask sooner
SNAP_LABELS = DATA_PROCESSED / "snap_labels.jsonl"
CORRECTIONS = DATA_PROCESSED / "coverage_corrections.jsonl"

# the job's state machine (meta.json `stage`): each stage and the stages it may move to
STAGES: dict[str, tuple[str, ...]] = {
    "uploaded": ("calibrating", "calibrated"),
    "calibrating": ("calibrating", "tracking", "needs_input", "error"),
    "calibrated": ("calibrated", "queued"),
    "queued": ("calibrating", "tracking", "error"),
    "tracking": ("reading", "error"),
    "reading": ("done", "needs_input", "error"),
    "done": ("calibrated", "queued", "done"),
    "needs_input": ("calibrated", "queued", "done"),
    "error": ("calibrated", "queued", "done"),
}
ACTIVE = ("calibrating", "queued", "tracking", "reading")
NEEDS = ("calibration", "snap", "offense")  # what a needs_input stage asks a person for


# ---------------------------------------------------------------- video helpers


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("not a readable video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0 or w <= 0:
        raise ValueError("the video reports no frames")
    return {"fps": fps, "n_frames": n, "duration": n / fps, "width": w, "height": h}


def sample_video(path: Path, out_dir: Path, every: float = SAMPLE_EVERY) -> list[dict]:
    """One thumbnail per `every` seconds (sequential read) with a motion score each.

    The score is the mean absolute gray change against the previous sample on a
    160x90 thumbnail: ~0 on a paused or set pre-snap look, 6-18 once a play runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    samples, prev, fi, next_t = [], None, 0, 0.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = fi / fps
        fi += 1
        if t + 1e-6 < next_t:
            continue
        next_t += every
        ok, img = cap.retrieve()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        score = 0.0 if prev is None else float(np.abs(g - prev).mean())
        prev = g
        i = len(samples)
        th = cv2.resize(img, (THUMB_WIDTH, int(THUMB_WIDTH * img.shape[0] / img.shape[1])))
        name = f"sample_{i:03d}.jpg"
        cv2.imwrite(str(out_dir / name), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
        samples.append({"i": i, "t": round(t, 3), "score": round(score, 2), "image": name})
    cap.release()
    return samples


def first_motion(samples: list[dict]) -> int | None:
    """Where the play starts: the first sample of the first run of two or more moving
    samples that peaks like a play (None: nothing moves).

    A camera pan or zoom before the snap moves the frame too (peak 8 on the Dolphins
    wide shot) and a replay after the play can move it more than the play (1532), so
    the pick is the first run peaking at PLAY_PEAK, else the first run at all.
    """
    runs, start = [], None
    for k in range(1, len(samples) + 1):
        moving = k < len(samples) and samples[k]["score"] > MOVING_SCORE
        if moving and start is None:
            start = k
        elif not moving and start is not None:
            if k - start >= 2:
                runs.append((start, max(s["score"] for s in samples[start:k])))
            start = None
    played = [s for s, peak in runs if peak >= PLAY_PEAK]
    return played[0] if played else (runs[0][0] if runs else None)


def play_window(samples: list[dict], t_cal: float, duration: float) -> dict:
    """Seconds to track, around the sample just before the play starts (the calibration
    frame when no play is found). The calibration is carried to the window when it is
    outside it."""
    k = first_motion(samples)
    anchor = samples[k - 1]["t"] if k is not None else t_cal
    return {
        "start": round(max(0.0, anchor - WINDOW_BEFORE), 2),
        "end": round(min(duration, anchor + WINDOW_AFTER), 2),
    }


def suggest_calibration_frame(samples: list[dict]) -> int:
    """The last still sample before the first sustained motion (the pre-snap look)."""
    k = first_motion(samples)
    if k is not None:
        for j in range(k - 1, -1, -1):
            if samples[j]["score"] < STILL_SCORE or j == 0:
                return samples[j]["i"]
    return samples[0]["i"] if samples else 0


def calibration_candidates(samples: list[dict], suggested: int) -> list[int]:
    """Samples to name the yard lines on, in order: the suggested one, then the other
    still samples before the play moves, nearest to it first."""
    k = first_motion(samples)
    before = samples if k is None else samples[:k]
    still = [
        s["i"] for s in before if s["i"] != suggested and (s["score"] < STILL_SCORE or s["i"] == 0)
    ]
    return [suggested] + sorted(still, key=lambda i: (abs(i - suggested), i))


def frame_at(path: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"no frame at {t:.2f} s")
    return img


def draw_numbered_lines(image: np.ndarray, lines, offsets=None) -> np.ndarray:
    """Every detected line, numbered from the left; lines that are not yard lines in gray."""
    out = image.copy()
    h, w = image.shape[:2]
    offsets = offsets if offsets is not None else [0] * len(lines)
    for k, (ln, off) in enumerate(zip(lines, offsets), start=1):
        color = (0, 230, 255) if off is not None else (150, 150, 150)
        cv2.line(out, (int(ln.x_at(0)), 0), (int(ln.x_at(h - 1)), h - 1), color, 4, cv2.LINE_AA)
        # put the number where this line is in frame (slanted lines leave it at the edges)
        for fy in (0.5, 0.38, 0.62, 0.25, 0.75, 0.12, 0.88):
            y = int(h * fy)
            x = int(ln.x_at(y))
            if 50 <= x < w - 50:
                break
        else:
            continue
        cv2.circle(out, (x, y), 34, (0, 0, 0), -1)
        cv2.circle(out, (x, y), 34, color, 3)
        label = str(k) if off is not None else "x"
        cv2.putText(
            out,
            label,
            (x - 14 * len(label), y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            color,
            3,
            cv2.LINE_AA,
        )
    return out


def lines_info(d: Path, image: np.ndarray, t: float, marks) -> dict:
    """Draw the detected lines numbered on the frame (lines.jpg) and describe them."""
    cv2.imwrite(str(d / "lines.jpg"), draw_numbered_lines(image, marks.lines, marks.offsets))
    return {
        "t": t,
        "n": len(marks.lines),
        "offsets": marks.offsets,
        "n_yard_lines": sum(o is not None for o in marks.offsets),
        "x_mid": [round(ln.x_at(marks.ref_y)) for ln in marks.lines],
    }


def save_fit(d: Path, image: np.ndarray, t: float, yards: list, search) -> dict:
    """Save a fitted calibration (calibration.json, grid.jpg) and describe it for meta.json."""
    from nfl_coverage.vision.fieldlines import (
        draw_field_grid,
        n_number_edges,
        painted_from_field_x,
    )

    best = search.best
    best.calibration.save(d / "calibration.json")
    grid = draw_field_grid(image, best.calibration)
    for x, y in best.pixel:  # the landmarks the fit used
        cv2.circle(grid, (int(x), int(y)), 8, (255, 0, 255), 2)
    cv2.imwrite(str(d / "grid.jpg"), grid)
    return {
        "t": t,
        "yards": yards,
        "painted": [None if y is None else painted_from_field_x(y) for y in yards],
        "n_landmarks": len(best.pixel),
        "n_rows": search.n_rows,
        "n_number_edges": n_number_edges(best),
        "mean_error": round(float(best.errors.mean()), 3),
        "max_error": round(float(best.errors.max()), 3),
        "n_ok": search.n_ok,
        "n_tried": search.n_tried,
        "notes": best.notes,
        "depth_warning": search.warning,
    }


def auto_calibrate(d: Path, meta: dict) -> tuple[dict | None, list[str]]:
    """Name the yard lines with no person on the first still pre-snap frame that allows it.

    Returns the calibration's description (None if no frame could be named) and why
    each frame tried before it failed.
    """
    from nfl_coverage.vision.fieldlines import auto_name, find_marks

    failed = []
    for i in calibration_candidates(meta["samples"], meta["suggested_sample"])[:MAX_AUTO_FRAMES]:
        t = meta["samples"][i]["t"]
        img = frame_at(d / "clip.mp4", t)
        marks = find_marks(img)
        try:
            naming = auto_name(img, marks)
        except ValueError as e:
            failed.append(f"{t:.1f} s: {e}")
            continue
        cal = save_fit(d, img, t, naming.yards, naming.search)
        return cal | {"naming": "auto", "edge_margin": naming.edge_margin}, failed
    return None, failed


# ---------------------------------------------------------------- clip store


class ClipStore:
    """One folder per clip; meta.json is the durable state, a lock guards writes."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def dir(self, clip_id: str) -> Path:
        if not clip_id.isalnum():
            raise HTTPException(404, "no such clip")
        d = self.root / clip_id
        if not (d / "meta.json").exists():
            raise HTTPException(404, "no such clip")
        return d

    def meta(self, clip_id: str) -> dict:
        return json.loads((self.dir(clip_id) / "meta.json").read_text())

    def advance(self, clip_id: str, stage: str, **fields) -> dict:
        """Move the job to `stage` if STAGES allows it from the current one. `needs` is
        set only on needs_input and the message is cleared unless one is given."""
        if stage == "needs_input" and fields.get("needs") not in NEEDS:
            raise ValueError(f"needs_input must name what it needs, one of {NEEDS}")
        return self._change(clip_id, {"message": "", "needs": None, **fields, "stage": stage})

    def update(self, clip_id: str, **fields) -> dict:
        return self._change(clip_id, fields)

    def _change(self, clip_id: str, fields: dict) -> dict:
        with self._lock:
            d = self.dir(clip_id)
            meta = json.loads((d / "meta.json").read_text())
            if "stage" in fields and fields["stage"] not in STAGES[meta["stage"]]:
                raise HTTPException(409, f"cannot go from {meta['stage']} to {fields['stage']}")
            meta.update(fields)
            tmp = d / "meta.json.tmp"
            tmp.write_text(json.dumps(meta, allow_nan=False))
            tmp.replace(d / "meta.json")
            return meta

    def create(self, name: str) -> tuple[str, Path]:
        clip_id = uuid.uuid4().hex[:12]
        d = self.root / clip_id
        d.mkdir(parents=True)
        return clip_id, d


# ---------------------------------------------------------------- API models


class LinesRequest(BaseModel):
    t: float
    min_length: int = 250
    votes: int = 120


class CalibrateRequest(BaseModel):
    t: float
    named_line: int = 1  # 1-based: the detected line the person read
    painted: int | None = None  # its painted yard (0 = goal line, 50 = midfield)
    going_up: bool = True  # painted numbers increase to its right
    first_painted: int | None = None  # older clients: the painted yard of line 1
    yards: list[float | None] | None = None  # explicit field x per line (None = not a yard line)
    min_length: int = 250
    votes: int = 120


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    phase: Literal["pre", "post"]
    target: Literal["shell", "exact"] = "shell"  # the structural call, or the model's coverage
    correct: bool
    call: str | None = Field(None, min_length=1, max_length=60)
    note: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def _a_wrong_read_names_the_call(self):
        if not self.correct and self.call is None:
            raise ValueError("a wrong read needs the call it should have been")
        return self


class AnalyzeRequest(BaseModel):
    start: float
    end: float
    imgsz: int = 1920


class RereadRequest(BaseModel):
    snap_t: float | None = None
    los: float | None = None
    offense: str | None = None
    ball_y: float | None = None


# ---------------------------------------------------------------- the app


def create_app(
    data_dir: Path | None = None,
    model_path: Path | None = None,
    auto: bool = True,
    sequence_path: Path | None = None,
) -> FastAPI:
    """`auto`: an upload runs the whole job (naming, tracking, reading) with no person."""
    store = ClipStore(data_dir or DEFAULT_DATA)
    model_path = model_path if model_path is not None else MODELS_DIR / "still_shown.joblib"
    sequence_path = sequence_path if sequence_path is not None else SEQUENCE_MODEL
    app = FastAPI(title="NFL Coverage Analyzer")
    track_lock = threading.Lock()  # one YOLO job at a time
    cache: dict = {}

    def bundle():
        if "bundle" not in cache:
            cache["bundle"] = None
            if model_path.exists():
                from nfl_coverage.models.persist import load_bundle

                cache["bundle"] = load_bundle(model_path)
        return cache["bundle"]

    def sequence_bundle():
        if "sequence" not in cache:
            from nfl_coverage.models.sequence import load_if_present

            cache["sequence"] = load_if_present(sequence_path)
        return cache["sequence"]

    @app.exception_handler(Exception)
    async def unexpected(request, exc: Exception):
        traceback.print_exception(exc)
        return JSONResponse(
            status_code=500,
            content={"detail": f"unexpected error ({type(exc).__name__}: {exc})"},
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/clips")
    def list_clips():
        out = []
        for d in sorted(store.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (d / "meta.json").exists():
                m = json.loads((d / "meta.json").read_text())
                out.append({k: m.get(k) for k in ("id", "name", "created", "stage")})
        return out[:30]

    @app.post("/api/clips")
    async def upload(file: Annotated[UploadFile, File()]):
        name = Path(file.filename or "clip.mp4").name
        clip_id, d = store.create(name)
        video = d / "clip.mp4"
        with video.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
        try:
            info = probe_video(video)
        except ValueError as e:
            raise HTTPException(400, f"{name}: {e}") from e
        samples = sample_video(video, d / "samples")
        suggested = suggest_calibration_frame(samples)
        t_cal = samples[suggested]["t"] if samples else 0.0
        meta = {
            "id": clip_id,
            "name": name,
            "created": time.time(),
            **info,
            "samples": samples,
            "suggested_sample": suggested,
            "window": play_window(samples, t_cal, info["duration"]),
            "stage": "uploaded",
            "needs": None,
            "progress": 0.0,
            "message": "",
            "corrections": {},
        }
        (d / "meta.json").write_text(json.dumps(meta, allow_nan=False))
        if not auto:
            return meta
        meta = store.advance(clip_id, "calibrating", progress=0.0)
        threading.Thread(target=run_auto, args=(clip_id,), daemon=True).start()
        return meta

    @app.get("/api/calls")
    def calls():
        """What a person may say a read should have been: shells, the model's classes, Other."""
        from nfl_coverage.analysis.shell import shell_label

        b, seq = bundle(), sequence_bundle()
        shells = dict.fromkeys(shell_label(n, c) for n in range(5) for c in (False, True))
        if seq is not None:
            variants = list(seq.classes)
        else:
            variants = [str(c) for c in b["model"].classes_] if b is not None else []
        return {"shells": list(shells), "variants": variants, "other": "Other"}

    @app.get("/api/clips/{clip_id}")
    def get_clip(clip_id: str):
        meta = store.meta(clip_id)
        result = store.dir(clip_id) / "result.json"
        if result.exists():
            meta["result"] = json.loads(result.read_text())
        return meta

    @app.get("/api/clips/{clip_id}/video")
    def video(clip_id: str):
        return FileResponse(store.dir(clip_id) / "clip.mp4", media_type="video/mp4")

    @app.get("/api/clips/{clip_id}/files/{name:path}")
    def files(clip_id: str, name: str):
        d = store.dir(clip_id).resolve()
        p = (d / name).resolve()
        if d not in p.parents or not p.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(p, headers={"Cache-Control": "no-store"})

    @app.post("/api/clips/{clip_id}/lines")
    def lines(clip_id: str, req: LinesRequest):
        from nfl_coverage.vision.fieldlines import find_marks

        d = store.dir(clip_id)
        img = frame_at(d / "clip.mp4", req.t)
        info = lines_info(d, img, req.t, find_marks(img, None, req.min_length, req.votes))
        store.update(clip_id, lines=info)
        return info

    @app.post("/api/clips/{clip_id}/calibrate")
    def calibrate(clip_id: str, req: CalibrateRequest):
        from nfl_coverage.vision.fieldlines import (
            field_x_for_lines,
            find_marks,
            five_yard_shift_warning,
            search_calibration,
        )

        d = store.dir(clip_id)
        if store.meta(clip_id)["stage"] in ACTIVE:
            raise HTTPException(409, "wait for the analysis running now")
        img = frame_at(d / "clip.mp4", req.t)
        marks = find_marks(img, None, req.min_length, req.votes)
        n_yard = sum(o is not None for o in marks.offsets)
        if n_yard < 2:
            raise HTTPException(
                422,
                f"only {n_yard} yard line{'s' if n_yard != 1 else ''} found on this frame - "
                "pick a frame where more of the field is in view, or search harder",
            )
        try:
            if req.yards:
                if len(req.yards) != len(marks.lines):
                    raise ValueError(
                        f"{len(req.yards)} values given for {len(marks.lines)} detected lines"
                    )
                yards = [None if y is None else float(y) for y in req.yards]
            else:
                painted = req.painted if req.painted is not None else req.first_painted
                if painted is None:
                    raise ValueError("give the painted yard of one line")
                yards = field_x_for_lines(marks.offsets, req.named_line - 1, painted, req.going_up)
            search = search_calibration(img, yards, None, req.min_length, req.votes, marks)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        cal = save_fit(d, img, req.t, yards, search) | {
            "naming": "person",
            "warning": five_yard_shift_warning(
                img, yards, search, req.min_length, req.votes, marks
            ),
        }
        store.update(clip_id, calibration=cal)
        store.advance(clip_id, "calibrated")
        return cal

    def write_outputs(clip_id: str, clip, read) -> dict:
        from nfl_coverage.analysis.viz import render_play
        from nfl_coverage.vision.overlay import overlay_frames
        from nfl_coverage.vision.pipeline import annotate_snap

        d = store.dir(clip_id)
        result = read.to_dict()
        cv2.imwrite(str(d / "snap.jpg"), annotate_snap(clip, read))
        (d / "overlay.json").write_text(json.dumps(overlay_frames(clip, read)))
        render_play(
            read.tracks.assign(dataset="clip", coverage_type="?", man_zone="?"),
            str(d / "topdown.png"),
            note="tracked",
            mark_frame=read.played_frame,
        )
        read.tracks.to_parquet(d / "tracks.parquet", index=False)
        (d / "result.json").write_text(json.dumps(result, allow_nan=False))
        store.update(clip_id, corrections={})  # they were verdicts on the read just replaced
        return result

    def run_analysis(clip_id: str, req: AnalyzeRequest):
        from nfl_coverage.vision.homography import FieldCalibration
        from nfl_coverage.vision.pipeline import (
            ReadError,
            TrackedClip,
            carry_calibration,
            read_tracked,
            track_frames,
        )

        d = store.dir(clip_id)
        meta = store.meta(clip_id)
        cal_meta = meta["calibration"]
        last = [0.0]

        def progress(frac: float):
            if frac - last[0] >= 0.02 or frac >= 1.0:
                last[0] = frac
                store.update(clip_id, progress=round(frac, 3))

        try:
            with track_lock:
                anchor = FieldCalibration.load(d / "calibration.json")
                anchor_t = cal_meta["t"]
                caveats = [cal_meta["depth_warning"]] if cal_meta.get("depth_warning") else []
                carry_notes: list[str] = []
                if not req.start <= anchor_t <= req.end:
                    # the calibrated frame is outside the play window: follow the camera
                    # from it to the window's first frame and refit there
                    store.advance(clip_id, "calibrating", progress=0.0)
                    try:
                        anchor, carry_notes = carry_calibration(
                            d / "clip.mp4", anchor, anchor_t, req.start
                        )
                    except ValueError as e:
                        store.advance(clip_id, "needs_input", message=str(e), needs="calibration")
                        return
                    anchor_t = req.start
                    caveats = [n for n in carry_notes if n.startswith("calibrated on the hash")]
                store.advance(clip_id, "tracking", progress=0.0)
                clip = track_frames(
                    d / "clip.mp4",
                    req.start,
                    req.end,
                    anchor,
                    anchor_t,
                    imgsz=req.imgsz,
                    progress=progress,
                )
                clip.notes = carry_notes + clip.notes
                clip.caveats = caveats
                clip.save(d / "tracked")
            store.advance(clip_id, "reading", progress=1.0)
            clip = TrackedClip.load(d / "tracked")
            try:
                read = read_tracked(clip, bundle=bundle(), sequence_bundle=sequence_bundle())
            except ReadError as e:
                store.advance(clip_id, "needs_input", message=str(e), needs=e.needs)
                return
            write_outputs(clip_id, clip, read)
            store.advance(clip_id, "done")
        except Exception as e:  # noqa: BLE001 - a background thread: surface any failure to the page
            traceback.print_exc()
            store.advance(clip_id, "error", message=f"{type(e).__name__}: {e}")

    def run_auto(clip_id: str):
        """The main path: name the yard lines with no person, then track and read."""
        try:
            cal, failed = auto_calibrate(store.dir(clip_id), store.meta(clip_id))
        except Exception as e:  # noqa: BLE001 - a background thread: surface any failure to the page
            traceback.print_exc()
            store.advance(clip_id, "error", message=f"{type(e).__name__}: {e}")
            return
        if cal is None:
            store.advance(
                clip_id,
                "needs_input",
                needs="calibration",
                message="the yard lines could not be named on any still pre-snap frame ("
                + "; ".join(failed)
                + "): pick a frame below and name one line",
            )
            return
        window = store.update(clip_id, calibration=cal)["window"]
        run_analysis(clip_id, AnalyzeRequest(start=window["start"], end=window["end"]))

    @app.post("/api/clips/{clip_id}/analyze")
    def analyze(clip_id: str, req: AnalyzeRequest):
        meta = store.meta(clip_id)
        if "calibration" not in meta:
            raise HTTPException(409, "calibrate a frame first")
        if not 0 <= req.start < req.end <= meta["duration"] + 0.05:
            raise HTTPException(
                422, f"the window must lie inside the clip (0 - {meta['duration']:.1f} s)"
            )
        if req.end - req.start < 1.5:
            raise HTTPException(422, "the window must be at least 1.5 s long")
        if meta["stage"] in ACTIVE:
            raise HTTPException(409, "already analyzing")
        (store.dir(clip_id) / "result.json").unlink(missing_ok=True)
        store.advance(clip_id, "queued", window={"start": req.start, "end": req.end}, progress=0.0)
        threading.Thread(target=run_analysis, args=(clip_id, req), daemon=True).start()
        return store.meta(clip_id)

    @app.post("/api/clips/{clip_id}/reread")
    def reread(clip_id: str, req: RereadRequest):
        from nfl_coverage.vision.pipeline import (
            ReadError,
            TrackedClip,
            read_tracked,
        )

        d = store.dir(clip_id)
        if not (d / "tracked" / "tracked.json").exists():
            raise HTTPException(409, "analyze the clip first")
        if store.meta(clip_id)["stage"] in ACTIVE:
            raise HTTPException(409, "wait for the analysis running now")
        if req.offense not in (None, "left", "right"):
            raise HTTPException(422, "offense must be left or right")
        clip = TrackedClip.load(d / "tracked")
        try:
            read = read_tracked(
                clip,
                req.snap_t,
                req.los,
                req.offense,
                req.ball_y,
                bundle(),
                sequence_bundle(),
            )
        except ReadError as e:
            raise HTTPException(422, str(e)) from e
        result = write_outputs(clip_id, clip, read)
        store.advance(clip_id, "done", overrides=req.model_dump())
        if req.snap_t is not None:
            log_snap_label(store.meta(clip_id), read)
        return result

    @app.post("/api/clips/{clip_id}/correction")
    def correction(clip_id: str, req: CorrectionRequest):
        d = store.dir(clip_id)
        if not (d / "result.json").exists():
            raise HTTPException(409, "there is no read to correct yet")
        meta = store.meta(clip_id)
        res = json.loads((d / "result.json").read_text())
        read = res["report"]["shown" if req.phase == "pre" else "played"]
        if read is None:
            raise HTTPException(409, "there is no post-snap read to correct")
        model = res["report"].get("pre_model" if req.phase == "pre" else "post_model")
        if req.target == "exact" and not model:
            raise HTTPException(409, "there is no model call to confirm on this card")
        judged = model["variant"][0][0] if req.target == "exact" else read["shell"]
        rec = {
            "logged": time.time(),
            "clip": meta["name"],
            "clip_id": clip_id,
            "phase": req.phase,
            "target": req.target,
            "correct": req.correct,
            "call": req.call or judged,
            "note": req.note,
            "model_call": read["shell"],
            "model_confidence": read["confidence"],
            "model_variant": model["variant"][0][0] if model else None,
            "model_variant_confidence": model["confidence"] if model else None,
            "snap_t": res["snap"]["t"],
            "read_t": res["snap"]["t"] if req.phase == "pre" else res["played_frame"]["t"],
        }
        log_correction(rec)
        key = req.phase if req.target == "shell" else f"{req.phase}_exact"
        store.update(clip_id, corrections={**meta.get("corrections", {}), key: rec})
        return rec

    return app


def log_correction(rec: dict) -> None:
    """A person's verdict on a read, next to the model's call: labels for retraining."""
    CORRECTIONS.parent.mkdir(parents=True, exist_ok=True)
    with CORRECTIONS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def log_snap_label(meta: dict, read) -> None:
    """A snap time a person chose, next to the detector's call: labels for the learned detector."""
    SNAP_LABELS.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "clip": meta.get("name"),
        "clip_id": meta.get("id"),
        "chosen_t": read.snap_t,
        "detected_t": read.snap.t,
        "detected_confidence": read.snap.confidence,
        "logged": time.time(),
    }
    with SNAP_LABELS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


app = create_app()
