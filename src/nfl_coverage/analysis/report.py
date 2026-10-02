"""Coverage report: structural reads, their confidence, and the model's variant call, kept apart.

The 8-class model tops out near 50% because the snap frame is ambiguous by
design, while the structural facts underneath it (how many deep, middle open or
closed, corners pressed or off) are read directly from positions with far
higher reliability. A report therefore says three separate things:

  1. structure  — deep count, middle of field, corner technique, with a
                  confidence that depends on how far each fact sits from its
                  threshold and on the pipeline's caveats (partial view, counts);
  2. family     — the coverage family that structure implies (Cover 2/4/6 vs
                  Cover 1/3 vs Cover 0), which is what a quarterback reads;
  3. variant    — the model's probabilities over the exact coverage, with their
                  honest, lower confidence.

Plus the disguise verdict (shown vs played) and the reasons in words. This is
the object a UI shows under the video.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from nfl_coverage.analysis.shell import (
    MOF_HALFWIDTH,
    MOF_PARTNER_MIN,
    MOF_SINGLE_HALFWIDTH,
    deep_defenders,
    shell_label,
)

CORNER_DEEP_DEPTH = 12.0  # the widest defender per side must be this deep to count as a deep third

# The played structure is read at two times (s after the snap, deep line yd). The middle is
# read once rotations are done but before routes pull man defenders deep (after ~2 s a
# corner carrying a go route reads as a second safety); the deep count at the landmark
# (features.phases), once quarters and Cover 3 corners have mostly bailed. Tuning weeks,
# 2,400 plays: middle 0.768, count 0.392 at 2.0 s (0.406 at 2.2 s, inside noise, but film
# has no throw frame and defenders break on a quick throw by 2.2 s), vs 0.559 / 0.290 for
# the most-deep frame in 2-4 s it replaced.
MOF_READ = (1.5, 11.0)
COUNT_READ = (2.0, 12.0)
LEVELS = ["low", "medium", "high"]

FAMILY = {
    (0, False): "Cover 0 (no deep help)",
    (1, True): "Cover 1 or Cover 3 (single-high, middle closed)",
    (1, False): "single-high shaded to a side",
    (2, False): "Cover 2 / Cover 4 / Cover 6 (two-high, middle open)",
    (2, True): "post safety plus a deep half (rotated two-high)",
    (3, True): "Cover 3 (three deep, middle closed)",
    (3, False): "three deep, middle open (quarters/halves rotation)",
    (4, True): "Cover 4 (four deep, quarters)",
    (4, False): "Cover 4 (four deep, quarters)",
}


@dataclass
class StructuralRead:
    frame: str  # 'snap' or e.g. '+2.0 s'
    n_deep: int
    mof_closed: bool
    shell: str
    family: str
    n_defenders: int
    deep_depths: list[float]
    deep_offsets: list[float]  # |y - ball| of the deep players
    corner_depths: tuple[float, float]  # low-y side, high-y side
    corners_pressed: tuple[bool, bool]
    depth_margin: float  # yards the deep count is from flipping
    mof_margin: float  # yards the MOF verdict is from flipping
    confidence: str  # high / medium / low
    reasons: list[str] = field(default_factory=list)
    confidence_why: list[str] = field(default_factory=list)  # what kept it from high


def _confidence(
    depth_margin: float, mof_margin: float, caveats: list[str]
) -> tuple[str, list[str]]:
    serious = [c for c in caveats if "off-frame" in c or "defenders detected" in c]
    why = [c.split(" — ")[0] for c in serious]
    if depth_margin < 2.0:
        why.append(
            f"a defender is {depth_margin:.1f} yd from the deep line, so the count could flip"
        )
    if mof_margin < 2.0:
        why.append(f"the middle open/closed call is {mof_margin:.1f} yd from flipping")
    if any("defenders detected" in c for c in caveats):
        return "low", why  # missing or surplus bodies: the structure itself is in doubt
    if depth_margin >= 2.0 and mof_margin >= 2.0 and not serious:
        return "high", []
    if depth_margin >= 1.0 and mof_margin >= 1.0 and len(serious) <= 1:
        return "medium", why
    return "low", why


def structural_read(
    analysis: pd.DataFrame,
    los: float,
    center: float,
    deep_depth: float,
    frame: str,
    caveats: list[str],
    rel_frame: int = 0,
) -> StructuralRead:
    """Read one canonical frame's defense into structure + confidence + reasons.

    `analysis` is a canonical rel_frame-0 frame (coverage players only),
    `los`/`center` in canonical coords, `deep_depth` the deep threshold to use
    (DEEP_DEPTH at the snap, deeper after it).
    """
    d = deep_defenders(analysis, rel_frame, deep_depth=deep_depth, los=los, center=center)
    # an off corner at 9-11 yd is technique, not a deep safety: the widest defender
    # on each side only counts as deep past CORNER_DEEP_DEPTH
    corner_deep = max(deep_depth, CORNER_DEEP_DEPTH)
    d["deep"] = np.where(d["widest"], d["depth"] >= corner_deep, d["deep"])
    deep = d[d["deep"]].sort_values("depth", ascending=False)
    n_deep = len(deep)
    offsets = np.sort(deep["offset"].to_numpy())
    if n_deep == 0:
        mof_closed, mof_margin = False, float("inf")
    elif n_deep == 1:
        mof_closed = offsets[0] <= MOF_SINGLE_HALFWIDTH + 0.05
        mof_margin = abs(MOF_SINGLE_HALFWIDTH - offsets[0])
    else:
        mof_closed = offsets[0] <= MOF_HALFWIDTH + 0.05 and offsets[1] >= MOF_PARTNER_MIN - 0.05
        near_slack = MOF_HALFWIDTH - offsets[0]  # >0 while the nearest is inside the band
        partner_slack = offsets[1] - MOF_PARTNER_MIN  # >0 while the partner is split away
        if mof_closed:  # either condition failing flips it
            mof_margin = min(near_slack, partner_slack)
        else:  # both must be met to flip it: the larger deficit is the distance
            mof_margin = max(-near_slack, -partner_slack)
    # how far the *counted* deep players sit above their threshold, and how close
    # the deepest uncounted one is to joining them
    thr = np.where(d["widest"], corner_deep, deep_depth)
    gaps = np.abs(d["depth"].to_numpy() - thr)
    depth_margin = float(gaps[d["deep"].to_numpy()].min()) if n_deep else float("inf")
    if (~d["deep"]).any():
        depth_margin = min(depth_margin, float(gaps[~d["deep"].to_numpy()].min()))

    low = d.loc[d["y"].idxmin()] if len(d) else None
    high = d.loc[d["y"].idxmax()] if len(d) else None
    at = analysis[analysis["rel_frame"] == rel_frame]
    off = at[(at["side"] == "Offense") & (at["position"] != "QB")]

    def pressed(row) -> bool:
        if row is None or len(off) == 0:
            return False
        return bool(np.hypot(off["x"] - row["x"], off["y"] - row["y"]).min() < 1.5)

    reasons = []
    if n_deep:
        reasons.append(
            f"{n_deep} defender{'s' if n_deep != 1 else ''} at least {deep_depth:.0f} yd deep "
            f"({', '.join(f'{v:.1f}' for v in deep['depth'])} yd)"
        )
        near = f"nearest deep player {offsets[0]:.1f} yd off the ball laterally"
        if n_deep >= 2:
            near += f", next {offsets[1]:.1f} yd"
        reasons.append(near + (" -> middle closed" if mof_closed else " -> middle open"))
    else:
        reasons.append(f"nobody {deep_depth:.0f} yd deep -> no deep help")
    if low is not None:
        reasons.append(
            f"widest defenders at {low['depth']:.1f} and {high['depth']:.1f} yd "
            f"({'press' if pressed(low) else 'off'} / {'press' if pressed(high) else 'off'})"
        )
    fam = FAMILY.get((min(n_deep, 4), bool(mof_closed)), shell_label(n_deep, bool(mof_closed)))
    confidence, confidence_why = _confidence(depth_margin, mof_margin, caveats)
    return StructuralRead(
        frame=frame,
        n_deep=n_deep,
        mof_closed=bool(mof_closed),
        shell=shell_label(n_deep, bool(mof_closed)),
        family=fam,
        n_defenders=len(d),
        deep_depths=[round(float(v), 1) for v in deep["depth"]],
        deep_offsets=[round(float(v), 1) for v in offsets],
        corner_depths=(
            round(float(low["depth"]), 1) if low is not None else float("nan"),
            round(float(high["depth"]), 1) if high is not None else float("nan"),
        ),
        corners_pressed=(pressed(low), pressed(high)),
        depth_margin=round(depth_margin, 2),
        mof_margin=round(mof_margin, 2) if np.isfinite(mof_margin) else 99.0,
        confidence=confidence,
        reasons=reasons,
        confidence_why=confidence_why,
    )


def played_frames(
    last_rel_frame: int,
    step: float,
    mof_read: tuple[float, float] = MOF_READ,
    count_read: tuple[float, float] = COUNT_READ,
) -> tuple[int, int]:
    """The rel_frames of the middle read and the count read, capped at the last tracked one."""
    return tuple(min(round(t / step), last_rel_frame) for t, _ in (mof_read, count_read))


def played_read(
    tracks: pd.DataFrame,
    los: float,
    center: float,
    step: float,
    caveats: list[str],
    mof_read: tuple[float, float] = MOF_READ,
    count_read: tuple[float, float] = COUNT_READ,
    hit: dict | None = None,
) -> StructuralRead:
    """The played structure: the middle from the early frame, the deep count from the later one.

    When nobody is deep yet at the middle read (a post safety dropping late from a
    0-high look), the later frame decides the middle too. `hit` is a mechanism hit
    (`analysis.mechanisms`) that says what a runner in the deep middle is: a pole runner
    covers the hole, not a half, so the structure is read without him (Tampa 2 when two
    halves remain); a late post is the single-high safety even short of the count line.
    A corner roll-up never reaches the count line, so it names the family, not the count:
    two halves behind him are Cover 2, the trap rather than quarters.
    """
    kind = hit["mechanism"] if hit else None
    if kind == "pole_run":
        tracks = tracks[tracks["nfl_id"] != int(hit["nfl_id"])]
    mf, cf = played_frames(int(tracks["rel_frame"].max()), step, mof_read, count_read)

    def at(rf: int, line: float) -> StructuralRead:
        frame = tracks[tracks["rel_frame"] == rf]
        return structural_read(frame, los, center, line, f"+{rf * step:.1f} s", caveats, rf)

    middle, count = at(mf, mof_read[1]), at(cf, count_read[1])
    src = middle if middle.n_deep else count
    closed = src.mof_closed
    n = count.n_deep
    worse = min(src, count, key=lambda r: LEVELS.index(r.confidence))
    why = list(dict.fromkeys(src.confidence_why + count.confidence_why))
    mid_reason = src.reasons[1] if src.n_deep else f"nobody {mof_read[1]:.0f} yd deep"
    reasons = [
        f"at {count.frame}: {count.reasons[0]}",
        f"middle at {src.frame}: {mid_reason}",
        *count.reasons[2 if count.n_deep else 1 :],
    ]
    if kind == "late_post" and n == 0:
        n, closed = 1, True
        reasons.append("the box defender who dropped to the deep middle is the post safety")
    family = FAMILY.get((min(n, 4), closed), shell_label(n, closed))
    if kind == "pole_run":
        reasons.append("a box defender ran the pole; he is not counted as a deep safety")
        if n == 2 and not closed:
            family = "Tampa 2 (Cover 2 zone, a runner down the middle)"
    if kind == "corner_roll_up":
        reasons.append("a soft corner rolled up to squat the flat; he is not a deep third")
        if n == 2 and not closed:
            family = "Cover 2 (a corner squats the flat: the trap, not quarters)"
    return replace(
        count,
        frame=f"{middle.frame} (middle) / {count.frame} (count)",
        n_deep=n,
        mof_closed=closed,
        shell=shell_label(n, closed),
        family=family,
        mof_margin=src.mof_margin,
        confidence=worse.confidence,
        confidence_why=why,
        reasons=reasons,
    )


@dataclass
class ModelRead:
    """One sequence-model head's call on a clip: what it read up to, ranked, with a level."""

    source: str
    t: float  # seconds after the snap of the last frame the head read
    variant: list[tuple[str, float]]  # descending
    confidence: str
    man_zone: list[tuple[str, float]]  # descending


def model_read(source: str, t: float, variant: dict, man_zone: dict) -> ModelRead:
    ranked = sorted(variant.items(), key=lambda kv: -kv[1])
    return ModelRead(
        source=source,
        t=round(t, 1),
        variant=ranked,
        confidence=variant_confidence(ranked),
        man_zone=sorted(man_zone.items(), key=lambda kv: -kv[1]),
    )


@dataclass
class CoverageReport:
    shown: StructuralRead
    played: StructuralRead | None
    variant: list[tuple[str, float]]  # snap-frame model probabilities, descending
    variant_confidence: str
    disguise_label: str
    disguise_score: float
    disguise_text: str
    caveats: list[str]
    pre_model: ModelRead | None = None  # sequence model, frames up to the snap
    post_model: ModelRead | None = None  # sequence model, the whole play
    mechanisms: list[dict] = field(default_factory=list)  # analysis.mechanisms hits

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"SHOWN  [{self.shown.confidence} confidence]: {self.shown.shell} -> {self.shown.family}"
        ]
        lines += [f"    - {r}" for r in self.shown.reasons]
        top = self.variant[0]
        lines.append(
            f"VARIANT [{self.variant_confidence} confidence]: "
            + ", ".join(f"{c} {p:.0%}" for c, p in self.variant[:3])
            + f"  (model's best guess {top[0]})"
        )
        if self.played is not None:
            lines.append(
                f"PLAYED {self.played.frame} [{self.played.confidence} confidence]: "
                f"{self.played.shell} -> {self.played.family}"
            )
            lines += [f"    - {r}" for r in self.played.reasons]
        lines.append(f"DISGUISE: {self.disguise_text} (score {self.disguise_score:.2f})")
        for c in self.caveats:
            lines.append(f"    ! {c}")
        return "\n".join(lines)


def disguise_verdict(
    rotation_label: str,
    shown: StructuralRead,
    played: StructuralRead | None,
    hits: list[dict],
) -> tuple[str, str]:
    """One disguise call for the card: (label, text), from the strongest evidence.

    A coverage mechanism seen in the first second leads (it is the disguise itself);
    otherwise a shell rotation between the snap and the middle read; otherwise the
    look held. The rotation labels come from `disguise.shell_rotation`.
    """
    coverage = [h for h in hits if h.get("kind") == "coverage"]
    if coverage and played is not None:
        h = coverage[0]
        return h["mechanism"], (f"{h['text']} (showed {shown.family}, played {played.family})")
    return rotation_label, disguise_text(rotation_label, shown, played)


def disguise_text(label: str, shown: StructuralRead, played: StructuralRead | None) -> str:
    if played is None:
        return "no post-snap frame read"
    if not label:
        return f"none: the {shown.shell} look held ({played.family})"
    texts = {
        "rotate_to_mof_closed": "two-high look rotated to a post safety: showed "
        f"{shown.family}, played {played.family}",
        "rotate_to_mof_open": "single-high look split into two deep: showed "
        f"{shown.family}, played {played.family}",
        "rotation_down": f"a deep defender came down: showed {shown.family}, played {played.family}",
        "rotation_up": f"more deep than shown, corners or a slot bailed: showed {shown.family}, "
        f"played {played.family} (a man look that plays Cover 3 reads this way)",
        "corner_bail": f"no disguise: corners bailed deep after the snap, the technique of "
        f"the coverage shown, not a change of it (showed {shown.family}, played "
        f"{played.family})",
        "threshold_flicker": f"no disguise: the shell label changed only because a defender "
        f"sits on the deep threshold; nobody moved 3 yd (showed {shown.family}, played "
        f"{played.family})",
    }
    return texts.get(label, label)


def variant_confidence(probs: list[tuple[str, float]]) -> str:
    if len(probs) < 2:
        return "low"
    top, second = probs[0][1], probs[1][1]
    if top >= 0.6 and top - second >= 0.3:
        return "high"
    if top >= 0.4 and top - second >= 0.15:
        return "medium"
    return "low"
