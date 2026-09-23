"""Field-visibility guard: is the region the coverage read depends on actually in frame?

The coverage model cannot tell a *cropped* player from an *absent* one. A camera
angle that cuts off a deep safety makes a two-high shell read as single-high or
Cover 0, silently — and the defender count cannot catch it, because training data
carries coverage players only (mean ~7 defenders), so 5-9 bodies is in-distribution.

The guard runs the calibration homography the other way round: take the
analysis-critical field region (full width, a few yards behind the LOS to safety
depth beyond it), project it into pixel space, and require it to lie inside the
image. A rectangle on the ground plane projects to a convex quadrilateral, so the
four corners decide containment; a sampled grid gives the visible fraction and
how deep the full-width view reaches, for the caveat text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nfl_coverage.config import FIELD_LENGTH, FIELD_WIDTH
from nfl_coverage.vision.homography import FieldCalibration

# depth window (yards from the LOS) the shown-coverage read depends on: shallow
# enough to keep press corners and the box edge, deep enough for two-high safeties
BEHIND_LOS = 3.0
BEYOND_LOS = 18.0

# plausibility band on coverage defenders (after line filtering): training mean is
# ~7.2; fewer than 5 means bodies were missed or cropped, more than 11 is impossible
MIN_DEFENDERS = 5
MAX_DEFENDERS = 11


def required_region(
    los: float, offense: str, behind: float = BEHIND_LOS, beyond: float = BEYOND_LOS
) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) in field yards that a coverage read needs to see.

    `offense` is the side of the LOS the offense occupies ('left' = low x), so the
    defensive depth extends the other way. Clamped to the field.
    """
    if offense not in ("left", "right"):
        raise ValueError(f"offense must be 'left' or 'right', got {offense!r}")
    if offense == "left":
        x_min, x_max = los - behind, los + beyond
    else:
        x_min, x_max = los - beyond, los + behind
    return max(0.0, x_min), min(FIELD_LENGTH, x_max), 0.0, FIELD_WIDTH


def field_to_pixel(calib: FieldCalibration, field_xy) -> tuple[np.ndarray, np.ndarray]:
    """Project field points into the image. Returns (pixels (N, 2), in_front (N,)).

    A homography is only defined up to scale, so the sign of the homogeneous w
    that separates "in front of the camera" from "behind it" is read off the
    calibration points, which are visible by construction. Points behind the
    camera get finite but meaningless pixels; callers must mask them out.
    """
    inv = np.linalg.inv(calib.matrix)
    pts = np.asarray(field_xy, dtype=float).reshape(-1, 2)

    def homog(p):
        h = np.hstack([p, np.ones((len(p), 1))]) @ inv.T
        return h[:, :2] / h[:, 2:], h[:, 2]

    _, w_calib = homog(calib.field_points.astype(float))
    front_sign = np.sign(np.median(w_calib)) or 1.0
    px, w = homog(pts)
    return px, np.sign(w) == front_sign


def in_image(pixels: np.ndarray, image_shape: tuple[int, int], margin: int = 0) -> np.ndarray:
    """Boolean mask of pixels inside an (H, W) image, shrunk by `margin` on each side."""
    h, w = image_shape[:2]
    x, y = pixels[:, 0], pixels[:, 1]
    eps = 1e-3  # float32 slop from the homography solve; far below a pixel
    return (
        (x >= margin - eps)
        & (x <= w - 1 - margin + eps)
        & (y >= margin - eps)
        & (y <= h - 1 - margin + eps)
    )


@dataclass
class VisibilityReport:
    """Outcome of the field-visibility check for one calibrated image.

    The visible part of the required region is described as a band: the y-range
    that is in frame at every depth (`band`), and how deep that band stays in
    frame (`depth_visible`). A cropped near sideline therefore reads as a narrow
    band, not as "nothing visible" — the two failure modes need different caveats.
    """

    region: tuple[float, float, float, float]  # field yards: x_min, x_max, y_min, y_max
    region_pixels: np.ndarray  # (4, 2) projected corners, for drawing
    corner_visible: dict[str, bool]
    fraction_visible: float  # of a sampled grid over the region
    band: tuple[float, float]  # y-range in frame at every sampled depth
    depth_visible: float  # yards beyond the LOS the band stays in frame
    depth_required: float
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(self.corner_visible.values())

    def summary(self) -> str:
        if self.ok:
            return (
                f"field visibility OK: full width in frame to {self.depth_required:.0f} yd "
                "beyond the LOS"
            )
        return "PARTIAL VIEW: " + "; ".join(self.warnings)


def check_visibility(
    calib: FieldCalibration,
    image_shape: tuple[int, int],
    los: float,
    offense: str,
    behind: float = BEHIND_LOS,
    beyond: float = BEYOND_LOS,
    depth_slack: float = 1.0,
    sideline_slack: float = 2.0,
    margin: int = 0,
    step: float = 0.5,
) -> VisibilityReport:
    """Is the analysis-critical region inside the image?

    image_shape : (H, W[, C]) as from `img.shape`.
    los/offense : as given to the still pipeline (raw calibration field frame).
    *_slack     : yards the region is shrunk by before the pass/fail test —
                  calibration error is ~0.5 yd, and nobody aligns on the paint
                  (the widest realistic split is ~2 yd off the sideline).
    margin      : pixels to shrink the image by (a foot point on the very edge
                  means the body is already leaving the frame).
    """
    x_min, x_max, y_min, y_max = region = required_region(los, offense, behind, beyond)
    sign = 1.0 if offense == "left" else -1.0  # +1: defense at higher x

    # pass/fail on the corners of the slack-shrunk region: a ground-plane rectangle
    # projects to a convex quad, so corners inside a rectangular image <=> all inside
    sx0, sx1 = x_min + depth_slack, x_max - depth_slack
    sy0, sy1 = y_min + sideline_slack, y_max - sideline_slack
    deep_x, shallow_x = (sx1, sx0) if offense == "left" else (sx0, sx1)
    # named by football meaning, not camera side: which sideline is "near" depends
    # on where the camera stands, and the caveat must read the same either way
    corners = {
        "behind LOS, y=0 sideline": (shallow_x, sy0),
        "behind LOS, far sideline": (shallow_x, sy1),
        "deep, y=0 sideline": (deep_x, sy0),
        "deep, far sideline": (deep_x, sy1),
    }
    corner_px, front = field_to_pixel(calib, list(corners.values()))
    corner_visible = {
        name: bool(f and v)
        for name, f, v in zip(corners, front, in_image(corner_px, image_shape, margin))
    }
    region_px, _ = field_to_pixel(
        calib, [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
    )

    # sampled grid over the full region for the descriptive numbers
    depths = np.arange(-behind, beyond + 1e-9, step)
    ys = np.arange(y_min, y_max + 1e-9, step)
    if ys[-1] < y_max - 1e-6:
        ys = np.append(ys, y_max)
    grid = np.array([(los + sign * d, y) for d in depths for y in ys])
    px, fr = field_to_pixel(calib, grid)
    vis = (fr & in_image(px, image_shape, margin)).reshape(len(depths), len(ys))
    fraction = float(vis.mean())

    # describe what *is* usable as the largest fully visible rectangle that
    # contains the LOS row. Image edges are slanted in field space, so a per-column
    # or per-row walk would be pessimistic on one axis and optimistic on the other.
    los_row = int(np.argmin(np.abs(depths)))
    best = None  # (area, shallow_row, deep_row, y_lo, y_hi)
    for r0 in range(los_row, -1, -1):
        if not vis[r0].any():
            break
        for r1 in range(los_row, len(depths)):
            col_ok = vis[r0 : r1 + 1].all(axis=0)
            if not col_ok.any():
                break
            # longest contiguous run of visible columns
            padded = np.concatenate([[0], col_ok.astype(int), [0]])
            edges = np.flatnonzero(np.diff(padded))
            starts, ends = edges[::2], edges[1::2]
            k = int(np.argmax(ends - starts))
            y_lo, y_hi = float(ys[starts[k]]), float(ys[ends[k] - 1])
            area = (y_hi - y_lo) * float(depths[r1] - depths[r0])
            if best is None or area > best[0]:
                best = (area, r0, r1, y_lo, y_hi)
    if best is not None:
        _, r0, r1, y_lo, y_hi = best
        shallow_visible, depth_visible = float(depths[r0]), float(depths[r1])
        band = (y_lo, y_hi)
    else:
        depth_visible = shallow_visible = float("nan")
        band = (float("nan"), float("nan"))

    warnings: list[str] = []
    if not all(corner_visible.values()):
        warnings.append(f"{1 - fraction:.0%} of the analysis region is off-frame")
        if np.isnan(depth_visible):
            warnings.append("the line of scrimmage itself is not in frame")
        else:
            if band[0] > y_min + sideline_slack:
                warnings.append(
                    f"sideline strip y < {band[0]:.0f} yd is off-frame — "
                    "wide defenders on that side may be cut off"
                )
            if band[1] < y_max - sideline_slack:
                warnings.append(
                    f"sideline strip y > {band[1]:.0f} yd is off-frame — "
                    "wide defenders on that side may be cut off"
                )
            if depth_visible < beyond - depth_slack:
                warnings.append(
                    f"deep zone beyond {depth_visible:.0f} yd is off-frame "
                    f"(need {beyond:.0f}) — deep safeties may be cropped, not absent"
                )
            if shallow_visible > -behind + depth_slack:
                warnings.append(
                    f"backfield beyond {-shallow_visible:.0f} yd behind the LOS is off-frame"
                )
    return VisibilityReport(
        region=region,
        region_pixels=region_px,
        corner_visible=corner_visible,
        fraction_visible=fraction,
        band=band,
        depth_visible=depth_visible,
        depth_required=beyond,
        warnings=warnings,
    )


def defender_count_warnings(
    n_defenders: int, low: int = MIN_DEFENDERS, high: int = MAX_DEFENDERS
) -> list[str]:
    """Plausibility band on the number of coverage defenders fed to the model."""
    if n_defenders < low:
        msg = (
            f"only {n_defenders} coverage defenders detected (expect {low}-{high}) — "
            "missed or cropped bodies will bias the read toward thin shells"
        )
        return [msg]
    if n_defenders > high:
        msg = (
            f"{n_defenders} coverage defenders detected (max {high}) — "
            "side assignment or line filtering is off"
        )
        return [msg]
    return []
