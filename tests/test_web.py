"""The page's API and the pieces under it, on synthetic video (no YOLO, no real film)."""

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nfl_coverage.vision.fieldlines import field_x_from_painted, painted_from_field_x
from nfl_coverage.vision.homography import FieldCalibration
from nfl_coverage.vision.pipeline import TrackedClip
from nfl_coverage.vision.track import TrackedFrame
from nfl_coverage.web.app import (
    calibration_candidates,
    create_app,
    suggest_calibration_frame,
)


def _video(path, seconds=4.0, fps=10, still_until=2.0, size=(320, 180), spacing=50, width=2):
    """Still for `still_until` s (a paused pre-snap look), then a block sweeps across."""
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        img = np.full((h, w, 3), (40, 120, 40), np.uint8)
        for x in range(20, w - 20, spacing):
            cv2.line(img, (x, 0), (x, h - 1), (240, 240, 240), width)
        t = i / fps
        if t >= still_until:
            x = int(20 + (t - still_until) * 120)
            cv2.rectangle(img, (x, 60), (x + 60, 140), (200, 60, 60), -1)
        out.write(img)
    out.release()
    return path


# ------------------------------------------------------------- painted yards


def test_field_x_from_painted_crosses_midfield():
    xs = field_x_from_painted(40, True, 5)
    assert xs == [50.0, 55.0, 60.0, 65.0, 70.0]
    assert [painted_from_field_x(x) for x in xs] == [40, 45, 50, 45, 40]


def test_field_x_from_painted_on_the_high_half():
    xs = field_x_from_painted(30, False, 4)
    assert xs == [80.0, 85.0, 90.0, 95.0]
    assert [painted_from_field_x(x) for x in xs] == [30, 25, 20, 15]


@pytest.mark.parametrize("bad", [-5, 55, 32])
def test_field_x_from_painted_rejects_impossible_yards(bad):
    with pytest.raises(ValueError):
        field_x_from_painted(bad, True, 3)


def test_field_x_from_painted_rejects_running_past_the_goal_line():
    with pytest.raises(ValueError):
        field_x_from_painted(5, False, 4)  # 5, 0, then past the goal line


# ------------------------------------------------------------- tracked cache


def test_tracked_clip_round_trips(tmp_path):
    px = np.float32([[10, 10], [300, 10], [300, 170], [10, 170]])
    fp = np.float32([[40, 50], [60, 50], [60, 5], [40, 5]])
    cal = FieldCalibration.from_points(px, fp)
    frames = []
    for i in range(3):
        img = np.full((180, 320, 3), 60 + 20 * i, np.uint8)
        boxes = np.array([[100.0, 50.0, 120.0, 100.0], [200.0, 60.0, 222.0, 110.0]])
        frames.append(
            TrackedFrame(i, 1.0 + 0.1 * i, img, boxes, np.array([3, 7]), np.array([0.9, 0.8]))
        )
    TrackedClip(frames, [cal] * 3, 0.1, ["a note"]).save(tmp_path / "tracked")
    back = TrackedClip.load(tmp_path / "tracked")
    assert len(back.frames) == 3 and back.step == 0.1 and back.notes == ["a note"]
    assert back.frames[2].t == pytest.approx(1.2)
    assert back.frames[1].ids.tolist() == [3, 7]
    assert np.allclose(back.frames[0].boxes, frames[0].boxes)
    assert np.allclose(back.calibrations[1].matrix, cal.matrix)
    assert back.calibrations[0].field_points.shape == (4, 2)  # the visibility check needs them
    assert back.index_at(1.14) == 1


# ------------------------------------------------------------- the API


def test_suggested_frame_is_the_last_still_before_motion():
    scores = [0, 0.1, 0.2, 0.1, 7.0, 9.0, 12.0]
    samples = [{"i": i, "t": i * 0.5, "score": s} for i, s in enumerate(scores)]
    assert suggest_calibration_frame(samples) == 3


def test_naming_tries_the_suggested_frame_then_the_nearest_still_ones():
    scores = [0, 0.1, 5.0, 0.2, 0.1, 0.3, 7.0, 9.0, 0.1]
    samples = [{"i": i, "t": i * 0.5, "score": s} for i, s in enumerate(scores)]
    assert suggest_calibration_frame(samples) == 5
    # 2 moved (a camera jolt), 6 on is the play: neither is tried
    assert calibration_candidates(samples, 5) == [5, 4, 3, 1, 0]


def test_naming_without_a_play_in_motion_tries_every_still_frame():
    samples = [{"i": i, "t": i * 0.5, "score": 0.1 * i} for i in range(4)]
    assert calibration_candidates(samples, 2) == [2, 1, 3, 0]


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        data_dir=tmp_path / "web",
        model_path=tmp_path / "none.joblib",
        auto=False,
        sequence_path=tmp_path / "none.pt",
    )
    return TestClient(app)


def _upload(client, tmp_path) -> str:
    video = _video(tmp_path / "play.mp4")
    with video.open("rb") as fh:
        r = client.post("/api/clips", files={"file": ("play.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _wait(client, cid) -> dict:
    for _ in range(400):
        meta = client.get(f"/api/clips/{cid}").json()
        if meta["stage"] not in ("queued", "calibrating", "tracking", "reading"):
            return meta
        time.sleep(0.05)
    raise TimeoutError(meta["stage"])


def test_upload_samples_frames_and_suggests_the_pre_snap_look(client, tmp_path):
    video = _video(tmp_path / "play.mp4")
    with video.open("rb") as fh:
        r = client.post("/api/clips", files={"file": ("play.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["name"] == "play.mp4" and meta["stage"] == "uploaded"
    assert meta["duration"] == pytest.approx(4.0, abs=0.2)
    assert len(meta["samples"]) == 8  # one per 0.5 s
    # still until 2.0 s: the suggestion is the last still sample, 1.5 s or 2.0 s
    assert 1.5 <= meta["samples"][meta["suggested_sample"]]["t"] <= 2.0
    assert meta["window"]["start"] <= meta["samples"][meta["suggested_sample"]]["t"]

    cid = meta["id"]
    assert client.get(f"/api/clips/{cid}").json()["id"] == cid
    img = client.get(f"/api/clips/{cid}/files/samples/sample_000.jpg")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    assert client.get(f"/api/clips/{cid}/video").status_code == 200
    assert any(c["id"] == cid for c in client.get("/api/clips").json())


def test_files_outside_the_clip_folder_are_refused(client, tmp_path):
    video = _video(tmp_path / "play.mp4")
    with video.open("rb") as fh:
        cid = client.post("/api/clips", files={"file": ("play.mp4", fh, "video/mp4")}).json()["id"]
    assert client.get(f"/api/clips/{cid}/files/../../secret.txt").status_code == 404
    assert client.get(f"/api/clips/{cid}/files/meta.json.nope").status_code == 404
    assert client.get("/api/clips/not-a-clip").status_code == 404


def test_steps_out_of_order_are_refused(client, tmp_path):
    video = _video(tmp_path / "play.mp4")
    with video.open("rb") as fh:
        cid = client.post("/api/clips", files={"file": ("play.mp4", fh, "video/mp4")}).json()["id"]
    r = client.post(f"/api/clips/{cid}/analyze", json={"start": 0.5, "end": 3.5})
    assert r.status_code == 409 and "calibrate" in r.json()["detail"]
    r = client.post(f"/api/clips/{cid}/reread", json={})
    assert r.status_code == 409 and "analyze" in r.json()["detail"]
    r = client.post(f"/api/clips/{cid}/calibrate", json={"t": 1.0})
    assert r.status_code == 422  # no yards given
    # a window outside the clip is refused before any tracking starts
    meta_path = tmp_path / "web" / cid / "meta.json"
    import json

    meta = json.loads(meta_path.read_text())
    meta["calibration"] = {"t": 1.0}
    meta_path.write_text(json.dumps(meta))
    r = client.post(f"/api/clips/{cid}/analyze", json={"start": 3.0, "end": 9.0})
    assert r.status_code == 422 and "inside the clip" in r.json()["detail"]


def test_lines_endpoint_numbers_the_detected_lines(client, tmp_path):
    # yard lines as film shows them: far apart and a few pixels wide
    video = _video(tmp_path / "play.mp4", size=(1280, 720), spacing=160, width=4)
    with video.open("rb") as fh:
        cid = client.post("/api/clips", files={"file": ("play.mp4", fh, "video/mp4")}).json()["id"]
    r = client.post(f"/api/clips/{cid}/lines", json={"t": 1.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] >= 6  # 8 painted lines; the edge ones may be cut
    assert body["x_mid"] == sorted(body["x_mid"])
    assert client.get(f"/api/clips/{cid}/files/lines.jpg").status_code == 200


def test_a_non_video_upload_is_rejected(client, tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_bytes(b"not a video at all")
    with bad.open("rb") as fh:
        r = client.post("/api/clips", files={"file": ("notes.mp4", fh, "video/mp4")})
    assert r.status_code == 400


def _still_clip(n=30):
    """A tracked window where nobody moves: there is no snap to find."""
    px = np.float32([[10, 10], [300, 10], [300, 170], [10, 170]])
    fp = np.float32([[40, 50], [60, 50], [60, 5], [40, 5]])
    cal = FieldCalibration.from_points(px, fp)
    boxes = np.array([[40.0 + 20 * k, 60.0, 52.0 + 20 * k, 100.0] for k in range(12)])
    frames = [
        TrackedFrame(
            i, 0.1 * i, np.full((180, 320, 3), 90, np.uint8), boxes, np.arange(12), np.ones(12)
        )
        for i in range(n)
    ]
    return TrackedClip(frames, [cal] * n, 0.1)


def test_a_clip_without_a_snap_asks_for_one(client, tmp_path, monkeypatch):
    video = _video(tmp_path / "play.mp4")
    with video.open("rb") as fh:
        cid = client.post("/api/clips", files={"file": ("play.mp4", fh, "video/mp4")}).json()["id"]
    d = tmp_path / "web" / cid
    clip = _still_clip()
    clip.calibrations[0].save(d / "calibration.json")
    meta = json.loads((d / "meta.json").read_text())
    meta["calibration"] = {"t": 0.5}
    meta["stage"] = "calibrated"
    (d / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr("nfl_coverage.vision.pipeline.track_frames", lambda *a, **k: clip)
    r = client.post(f"/api/clips/{cid}/analyze", json={"start": 0.0, "end": 2.9})
    assert r.status_code == 200, r.text
    for _ in range(100):
        meta = client.get(f"/api/clips/{cid}").json()
        if meta["stage"] not in ("queued", "calibrating", "tracking", "reading"):
            break
        time.sleep(0.05)
    assert meta["stage"] == "needs_input"
    assert meta["needs"] == "snap"
    assert "no snap found" in meta["message"]


# ------------------------------------------------------------- the automatic path


def test_an_upload_that_cannot_be_named_asks_a_person(tmp_path, monkeypatch):
    def no_fit(image, marks):
        raise ValueError("only 1 yard lines found")

    monkeypatch.setattr("nfl_coverage.vision.fieldlines.auto_name", no_fit)
    client = TestClient(
        create_app(
            data_dir=tmp_path / "web",
            model_path=tmp_path / "x.joblib",
            sequence_path=tmp_path / "x.pt",
        )
    )
    cid = _upload(client, tmp_path)
    meta = _wait(client, cid)
    assert meta["stage"] == "needs_input" and meta["needs"] == "calibration"
    assert "only 1 yard lines found" in meta["message"]
    assert "calibration" not in meta
    # the manual naming is the way on from here
    r = client.post(f"/api/clips/{cid}/analyze", json={"start": 0.5, "end": 3.5})
    assert r.status_code == 409 and "calibrate" in r.json()["detail"]


def test_an_upload_runs_on_to_tracking_and_reading_by_itself(tmp_path, monkeypatch):
    clip = _still_clip()

    def named(d, meta):
        clip.calibrations[0].save(d / "calibration.json")
        return {"t": 0.5, "naming": "auto", "edge_margin": 3}, ["0.0 s: no fit"]

    monkeypatch.setattr("nfl_coverage.web.app.auto_calibrate", named)
    monkeypatch.setattr("nfl_coverage.vision.pipeline.track_frames", lambda *a, **k: clip)
    client = TestClient(
        create_app(
            data_dir=tmp_path / "web",
            model_path=tmp_path / "x.joblib",
            sequence_path=tmp_path / "x.pt",
        )
    )
    cid = _upload(client, tmp_path)
    meta = _wait(client, cid)
    assert meta["calibration"]["naming"] == "auto"
    assert meta["stage"] == "needs_input" and meta["needs"] == "snap"  # the still clip has none


def test_a_stage_the_machine_does_not_allow_is_refused(client, tmp_path):
    from fastapi import HTTPException

    from nfl_coverage.web.app import ClipStore

    cid = _upload(client, tmp_path)
    store = ClipStore(tmp_path / "web")
    with pytest.raises(HTTPException):
        store.advance(cid, "done")  # uploaded -> done skips the whole job
    store.advance(cid, "calibrating")
    with pytest.raises(ValueError):
        store.advance(cid, "needs_input")  # without saying what it needs
    meta = store.advance(cid, "needs_input", needs="calibration", message="name a line")
    assert meta["stage"] == "needs_input" and meta["needs"] == "calibration"
    meta = store.advance(cid, "calibrated")
    assert meta["needs"] is None and meta["message"] == ""


# ------------------------------------------------------------- corrections


def _with_result(client, tmp_path) -> str:
    cid = _upload(client, tmp_path)
    shown = {"shell": "2-high MOF-open", "confidence": "high"}
    played = {"shell": "4-deep", "confidence": "medium"}
    result = {
        "snap": {"t": 5.5},
        "played_frame": {"t": 7.7},
        "report": {"shown": shown, "played": played},
    }
    (tmp_path / "web" / cid / "result.json").write_text(json.dumps(result))
    return cid


@pytest.mark.parametrize(
    "body",
    [
        {"phase": "during", "correct": True},
        {"phase": "pre"},
        {"phase": "pre", "correct": False},  # wrong, but not what it should have been
        {"phase": "pre", "correct": False, "call": "   "},
        {"phase": "pre", "correct": False, "call": "x" * 61},
        {"phase": "pre", "correct": True, "extra": 1},
    ],
)
def test_a_malformed_correction_is_refused(client, tmp_path, monkeypatch, body):
    monkeypatch.setattr("nfl_coverage.web.app.CORRECTIONS", tmp_path / "c.jsonl")
    cid = _with_result(client, tmp_path)
    assert client.post(f"/api/clips/{cid}/correction", json=body).status_code == 422
    assert not (tmp_path / "c.jsonl").exists()


def test_a_correction_before_any_read_is_refused(client, tmp_path):
    cid = _upload(client, tmp_path)
    r = client.post(f"/api/clips/{cid}/correction", json={"phase": "pre", "correct": True})
    assert r.status_code == 409


def test_corrections_are_logged_against_the_model_call(client, tmp_path, monkeypatch):
    log = tmp_path / "c.jsonl"
    monkeypatch.setattr("nfl_coverage.web.app.CORRECTIONS", log)
    cid = _with_result(client, tmp_path)
    body = {"phase": "post", "correct": False, "call": "COVER_6_ZONE", "note": " boundary "}
    r = client.post(f"/api/clips/{cid}/correction", json=body)
    assert r.status_code == 200, r.text
    r = client.post(f"/api/clips/{cid}/correction", json={"phase": "pre", "correct": True})
    assert r.status_code == 200, r.text
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    keys = {"logged", "clip", "clip_id", "phase", "correct", "call", "note"}
    keys |= {"model_call", "model_confidence", "snap_t", "read_t"}
    keys |= {"model_variant", "model_variant_confidence"}
    assert [set(rec) for rec in recs] == [keys, keys]
    post, pre = recs
    assert post["model_variant"] is None  # no sequence model in this app
    assert post["clip"] == "play.mp4" and post["clip_id"] == cid
    assert (post["phase"], post["correct"], post["call"], post["note"]) == (
        "post",
        False,
        "COVER_6_ZONE",
        "boundary",
    )
    assert (post["model_call"], post["model_confidence"], post["read_t"]) == (
        "4-deep",
        "medium",
        7.7,
    )
    assert (pre["call"], pre["model_call"], pre["read_t"], pre["snap_t"]) == (
        "2-high MOF-open",
        "2-high MOF-open",
        5.5,
        5.5,
    )
    kept = client.get(f"/api/clips/{cid}").json()["corrections"]
    assert kept["post"]["call"] == "COVER_6_ZONE" and kept["pre"]["correct"] is True


def test_the_calls_a_person_may_pick(client):
    calls = client.get("/api/calls").json()
    assert "2-high MOF-open" in calls["shells"] and "4-deep" in calls["shells"]
    assert calls["variants"] == []  # no model bundle in this app
    assert calls["other"] == "Other"
