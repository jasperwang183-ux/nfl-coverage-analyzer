"""The page end to end on real film: upload and nothing else, then correct a read.

Runs YOLO on the committed sample clip (samples/sf-lar/clip1.mp4), so it takes
tens of seconds on a GPU and minutes on a CPU; `pytest -m "not slow"` skips it.
The expected values are the facts recorded for this play in
samples/verify/clips.json (the analyst's Cover 4 call), less the absolute LOS:
the yard lines are named with no person, so field x is arbitrary along the field.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from nfl_coverage.config import ROOT
from nfl_coverage.web.app import create_app

CLIP = ROOT / "samples" / "sf-lar" / "clip1.mp4"
MANIFEST = ROOT / "samples" / "verify" / "clips.json"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CLIP.exists(), reason="sample clip not present"),
    pytest.mark.skipif(not (ROOT / "yolo11s.pt").exists(), reason="YOLO weights not present"),
]


def _entry(clip_id: str) -> dict:
    return next(c for c in json.loads(MANIFEST.read_text())["clips"] if c["id"] == clip_id)


def _wait(client: TestClient, cid: str, timeout: float = 600.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        meta = client.get(f"/api/clips/{cid}").json()
        if meta["stage"] not in ("queued", "calibrating", "tracking", "reading"):
            return meta
        time.sleep(1.0)
    raise TimeoutError(f"analysis still at {meta['stage']} after {timeout} s")


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    client = TestClient(create_app(data_dir=tmp_path_factory.mktemp("web")))
    with CLIP.open("rb") as fh:
        meta = client.post("/api/clips", files={"file": ("clip1.mp4", fh, "video/mp4")}).json()
    assert meta["stage"] == "calibrating"
    cid = meta["id"]
    return client, cid, _wait(client, cid), _entry("sf-lar-clip1")["expect"]


def test_the_yard_lines_are_named_with_no_person(analyzed):
    _, _, meta, _ = analyzed
    cal = meta["calibration"]
    assert cal["naming"] == "auto" and cal["edge_margin"] > 0
    assert cal["mean_error"] < 0.3


def test_the_read_matches_what_is_known_about_the_play(analyzed):
    _, _, meta, expect = analyzed
    assert meta["stage"] == "done", meta.get("message")
    res = meta["result"]
    assert res["formation"]["used"]["offense"] == expect["offense"]
    assert abs(res["snap"]["t"] - expect["snap_t"]) <= 0.25
    shown, played = res["report"]["shown"], res["report"]["played"]
    assert shown["n_deep"] == expect["shown_n_deep"]
    assert shown["mof_closed"] == expect["shown_mof_closed"]
    assert played["n_deep"] == expect["played_n_deep"]
    assert played["mof_closed"] == expect["played_mof_closed"]
    assert 2.0 - 1e-6 <= res["played_frame"]["t"] - res["snap"]["t"] <= 5.0 + 1e-6


def test_the_overlay_covers_the_window_and_marks_the_deep_defenders(analyzed):
    client, cid, meta, expect = analyzed
    ov = client.get(f"/api/clips/{cid}/files/overlay.json").json()
    ts = [f["t"] for f in ov["frames"]]
    assert ts[0] <= meta["result"]["snap"]["t"] <= ts[-1]
    snap = min(ov["frames"], key=lambda f: abs(f["t"] - ov["snap_t"]))
    played = min(ov["frames"], key=lambda f: abs(f["t"] - ov["played_t"]))
    assert sum(b[4] == "S" for b in snap["boxes"]) == expect["shown_n_deep"]
    assert sum(b[4] == "S" for b in played["boxes"]) == expect["played_n_deep"]
    assert len(snap["los"]) >= 2
    assert ov["shown"] == meta["result"]["report"]["shown"]["shell"]


def test_a_correction_lands_in_the_log(analyzed, tmp_path, monkeypatch):
    log = tmp_path / "coverage_corrections.jsonl"
    monkeypatch.setattr("nfl_coverage.web.app.CORRECTIONS", log)
    client, cid, meta, _ = analyzed
    body = {"phase": "post", "correct": False, "call": "COVER_6_ZONE", "note": "e2e"}
    r = client.post(f"/api/clips/{cid}/correction", json=body)
    assert r.status_code == 200, r.text
    (rec,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert rec["clip"] == "clip1.mp4" and rec["call"] == "COVER_6_ZONE"
    assert rec["model_call"] == meta["result"]["report"]["played"]["shell"]
    assert rec["read_t"] == meta["result"]["played_frame"]["t"]
    assert client.get(f"/api/clips/{cid}").json()["corrections"]["post"] == rec


def test_a_snap_override_rereads_from_the_tracking_cache(analyzed, tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_coverage.web.app.SNAP_LABELS", tmp_path / "snap_labels.jsonl")
    client, cid, meta, _ = analyzed
    t0 = time.time()
    r = client.post(f"/api/clips/{cid}/reread", json={"snap_t": meta["result"]["snap"]["t"] + 0.1})
    assert r.status_code == 200, r.text
    assert r.json()["snap"]["source"] == "given"
    assert time.time() - t0 < 30  # no re-tracking
    assert (tmp_path / "snap_labels.jsonl").exists()  # the pick is logged against the detector
    assert client.get(f"/api/clips/{cid}").json()["corrections"] == {}  # they judged the old read
