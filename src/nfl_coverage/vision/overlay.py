"""The read drawn back on the playing clip: per-frame boxes and the LOS, in video pixels.

The page draws this on a canvas over the video, so the overlay stays in sync
while the user scrubs, and the clip itself is never re-encoded. Each box has a
one-letter kind:

  O offense, D defense, S a deep defender (the ones the shell read counts),
  L a line player (set aside from the coverage read), X an official or anyone
  the read did not side.

"Deep" uses the read's own tests: SHOWN_DEEP up to the snap, then where each defender
is headed (`report.COUNT_TEST`), the same test the played card counts with.
"""

from __future__ import annotations

import numpy as np

from nfl_coverage.analysis.report import COUNT_TEST, ProjectedDepth, drop_rates
from nfl_coverage.config import FIELD_LENGTH, FIELD_WIDTH
from nfl_coverage.vision.pipeline import SHOWN_DEEP, ClipRead, TrackedClip

LOS_SAMPLES = (
    12  # points along the projected LOS (a straight field line bends under zoom distortion)
)


def deep_ids(read: ClipRead, rel_frame: int, test: ProjectedDepth = COUNT_TEST) -> set[int]:
    """Defenders the shell read counts as deep on a frame, by the tests it uses."""
    los_c = FIELD_LENGTH - read.los if read.offense == "right" else read.los
    t = read.tracks
    rows = t[(t["rel_frame"] == rel_frame) & (t["side"] == "Defense")]
    depth = rows["x"] - los_c
    if rel_frame <= 0:
        deep = depth >= SHOWN_DEEP
    else:
        rate = rows["nfl_id"].map(drop_rates(t, rel_frame, test.window, read.step)).fillna(0.0)
        deep = depth + rate * test.lookahead >= test.line
    return {int(i) for i in rows.loc[deep, "nfl_id"]}


def los_pixels(calib, los: float, width: int, height: int) -> list[list[int]]:
    """The LOS as a pixel polyline, dropping points that fall far outside the frame."""
    hinv = np.linalg.inv(calib.matrix)
    ys = np.linspace(0.0, FIELD_WIDTH, LOS_SAMPLES)
    p = hinv @ np.vstack([np.full_like(ys, los), ys, np.ones_like(ys)])
    ok = p[2] > 0 if np.mean(p[2]) > 0 else p[2] < 0  # in front of the camera
    px = (p[:2] / p[2]).T
    inside = (
        ok
        & (px[:, 0] > -width)
        & (px[:, 0] < 2 * width)
        & (px[:, 1] > -height)
        & (px[:, 1] < 2 * height)
    )
    return [[round(float(x)), round(float(y))] for x, y in px[inside]]


def overlay_frames(clip: TrackedClip, read: ClipRead) -> dict:
    """JSON-able per-frame overlay for the page (`overlay.json`)."""
    h, w = clip.frames[0].image.shape[:2]
    side_by_id, line_ids = read.sides.side_by_id, read.sides.line_ids
    frames = []
    for fr, calib in zip(clip.frames, clip.calibrations):
        rel = fr.index - read.snap_index
        deep = deep_ids(read, rel)
        boxes = []
        for box, tid in zip(fr.boxes, fr.ids):
            tid = int(tid)
            side = side_by_id.get(tid)
            if tid in line_ids:
                kind = "L"
            elif tid in deep:
                kind = "S"
            elif side == "Offense":
                kind = "O"
            elif side == "Defense":
                kind = "D"
            else:
                kind = "X"
            boxes.append([*(round(float(v)) for v in box), kind])
        frames.append({"t": fr.t, "boxes": boxes, "los": los_pixels(calib, read.los, w, h)})
    rep = read.report
    return {
        "width": w,
        "height": h,
        "snap_t": read.snap_t,
        "played_t": read.snap_t + read.played_frame * read.step,
        "shown": rep.shown.shell,
        "played": rep.played.shell if rep.played else None,
        "frames": frames,
        "field": field_view(read),
    }


def field_view(read: ClipRead) -> dict:
    """Every tracked player in field yards per rel_frame, for the page's coverage diagram.

    Canonical coordinates (offense moving right), the same frame as the diagram's zones.
    """
    los_c = FIELD_LENGTH - read.los if read.offense == "right" else read.los
    t = read.tracks.drop_duplicates(["nfl_id", "rel_frame"])
    frames = {
        int(rel): [[int(r.nfl_id), r.side[0], round(float(r.x), 2), round(float(r.y), 2)]
                   for r in g.itertuples()]
        for rel, g in t.groupby("rel_frame")
    }  # fmt: skip
    return {
        "step": read.step,
        "los": los_c,
        "ball": read.centers.mechanisms,
        "frames": frames,
    }
