"""Two-head spatio-temporal transformer over canonical frames.

Each frame is a set of players (attention across players, no player order); the
frame embeddings then go through a causal temporal transformer, so the output at
frame t is a read from frames up to t only. Two heads sit on every output:
coverage variant and man/zone. The full-play read is the output at the last
frame, the pre-snap read the output at the snap (rel_frame 0) - the same model,
one pass, and training on every prefix doubles as random truncation. Causal also
gives a per-frame read, which the live overlay wants.

Inputs are positions only (x, y, side, which frames a player is seen in).
Velocities come from finite differences of the positions, as on film; tracking's
`s`/`a`/`dir`/`o` are ignored so a film play looks like a training play. x is
measured from the line of scrimmage and y from the formation center at the snap,
both estimated from the offense like the snap features do.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from nfl_coverage.config import FIELD_WIDTH, FRAMES_PER_SECOND

KEYS = ["dataset", "game_id", "play_id"]
PRE_FRAMES = 10  # bdb2021 carries 1 s before the snap; bdb2026 starts at the snap
POST_FRAMES = 30  # 3 s: past most throws (bdb2026 ends at the throw, median 2.6 s)
T = PRE_FRAMES + 1 + POST_FRAMES
SNAP_T = PRE_FRAMES  # grid index of rel_frame 0
P_MAX = 24
RAW = 3  # x, y, is_offense
N_FEAT = 10
MAN_ZONE = ["MAN_COVERAGE", "ZONE_COVERAGE"]


# ------------------------------------------------------------------ tensors


def play_tensors(frames: pd.DataFrame, p_max: int = P_MAX) -> dict[str, np.ndarray]:
    """Canonical frames (any number of plays) -> fixed grids, one row per play.

    raw   [N, T, P, 3] x, y, is_offense on the rel_frame grid -PRE..POST
    mask  [N, T, P]    player seen in that frame
    ref   [N, 2]       LOS and formation center y at the snap (NaN if no offense)
    keys  DataFrame of the play keys, row-aligned (plus the labels when present)

    Player slots take defenders first, so a crowded film frame drops offense.
    Plays with no row at rel_frame 0 keep ref from their earliest post-snap frame.
    """
    df = frames[(frames["rel_frame"] >= -PRE_FRAMES) & (frames["rel_frame"] <= POST_FRAMES)]
    df = df[df["side"].isin(["Offense", "Defense"])].copy()
    for k in KEYS:
        if k not in df:
            df[k] = 0 if k != "dataset" else "clip"
    labels = [c for c in ("coverage_type", "man_zone") if c in df]
    play_keys = (
        df.groupby(KEYS, sort=True)[labels].first().reset_index()
        if labels
        else (df[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True))
    )
    n = len(play_keys)
    play_idx = pd.Series(np.arange(n), index=pd.MultiIndex.from_frame(play_keys[KEYS]))
    df["pi"] = play_idx.reindex(pd.MultiIndex.from_frame(df[KEYS])).to_numpy()

    players = df[["pi", "nfl_id", "side"]].drop_duplicates(["pi", "nfl_id"])
    players = players.assign(off=players["side"].eq("Offense")).sort_values(["pi", "off", "nfl_id"])
    players["slot"] = players.groupby("pi").cumcount()
    ref = np.full((n, 2), np.nan, dtype=np.float32)
    post = df[(df["rel_frame"] >= 0) & (df["side"] == "Offense")]
    at_ref = post[post["rel_frame"].eq(post.groupby("pi")["rel_frame"].transform("min"))]
    off = at_ref.groupby("pi")
    los = off["x"].quantile(0.75)  # features.alignment._los_estimate
    ref[los.index.to_numpy(), 0] = los.to_numpy()
    cy = off["y"].mean()
    ref[cy.index.to_numpy(), 1] = cy.to_numpy()

    df = df.merge(players[["pi", "nfl_id", "slot"]], on=["pi", "nfl_id"], how="left")
    df = df[df["slot"] < p_max]

    raw = np.zeros((n, T, p_max, RAW), dtype=np.float32)
    mask = np.zeros((n, T, p_max), dtype=bool)
    pi = df["pi"].to_numpy()
    ti = df["rel_frame"].to_numpy() + PRE_FRAMES
    si = df["slot"].to_numpy()
    raw[pi, ti, si, 0] = df["x"].to_numpy(np.float32)
    raw[pi, ti, si, 1] = df["y"].to_numpy(np.float32)
    raw[pi, ti, si, 2] = df["side"].eq("Offense").to_numpy(np.float32)
    mask[pi, ti, si] = True

    return {"raw": raw, "mask": mask, "ref": ref, "keys": play_keys}


def featurize(raw: torch.Tensor, mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """[B, T, P, 3] positions -> [B, T, P, N_FEAT] token features (zero where masked)."""
    x, y, off = raw[..., 0], raw[..., 1], raw[..., 2]
    m = mask.float()
    los = ref[:, 0].view(-1, 1, 1)
    cy = ref[:, 1].view(-1, 1, 1)
    dt = 1.0 / FRAMES_PER_SECOND

    # backward differences only: a centered one would let the snap read see frame +1
    def vel(c: torch.Tensor) -> torch.Tensor:
        prev_c, prev_m = torch.roll(c, 1, dims=1), torch.roll(m, 1, dims=1)
        prev_m[:, 0] = 0
        return prev_m * m * (c - prev_c) / dt

    vx, vy = vel(x), vel(y)
    speed = torch.sqrt(vx**2 + vy**2 + 1e-6)
    feats = torch.stack(
        [
            (x - los) / 10.0,
            (y - cy) / 10.0,
            (y - FIELD_WIDTH / 2) / (FIELD_WIDTH / 2),
            vx.clamp(-12, 12) / 5.0,
            vy.clamp(-12, 12) / 5.0,
            speed.clamp(max=12) / 5.0,
            off,
            1.0 - off,
            ((x - los) > 8).float(),
            torch.ones_like(x),
        ],
        dim=-1,
    )
    return torch.nan_to_num(feats) * m.unsqueeze(-1)


@dataclass
class Augment:
    """Film-like damage for training: missing bodies, position noise, mirrored plays."""

    drop_player: float = 0.12  # whole track missing
    drop_frame: float = 0.05  # one body missed in one frame
    jitter_play: float = 0.3  # yd, per player per play (homography bias)
    jitter_frame: float = 0.08  # yd, per frame (box noise)
    ref_jitter: float = 0.4  # yd, LOS / center estimate error
    mirror: float = 0.5  # flip y about the formation center: labels are symmetric
    min_pre: int = 0  # pre-snap frames kept are drawn from [min_pre, PRE_FRAMES]

    def __call__(
        self, raw: torch.Tensor, mask: torch.Tensor, ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, p, _ = raw.shape
        dev = raw.device
        raw, ref = raw.clone(), ref.clone()
        keep = torch.rand(b, 1, p, device=dev) >= self.drop_player
        keep = keep & (torch.rand(b, t, p, device=dev) >= self.drop_frame)
        mask = mask & keep
        noise = torch.randn(b, 1, p, 2, device=dev) * self.jitter_play
        noise = noise + torch.randn(b, t, p, 2, device=dev) * self.jitter_frame
        raw[..., :2] = raw[..., :2] + noise
        ref = ref + torch.randn(b, 2, device=dev) * self.ref_jitter
        flip = torch.rand(b, device=dev) < self.mirror
        raw[flip, ..., 1] = FIELD_WIDTH - raw[flip, ..., 1]
        ref[flip, 1] = FIELD_WIDTH - ref[flip, 1]
        n_pre = torch.randint(self.min_pre, PRE_FRAMES + 1, (b, 1), device=dev)
        grid = torch.arange(t, device=dev).view(1, t)
        mask = mask & (grid >= SNAP_T - n_pre).unsqueeze(-1)
        return raw, mask, ref


# ------------------------------------------------------------------ model


@dataclass
class Config:
    n_variant: int
    d: int = 128
    heads: int = 4
    spatial_layers: int = 2
    temporal_layers: int = 3
    ff: int = 256
    dropout: float = 0.1


class SequenceModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.embed = nn.Sequential(nn.Linear(N_FEAT, d), nn.GELU(), nn.Linear(d, d))

        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d, cfg.heads, cfg.ff, cfg.dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )  # fmt: skip

        self.spatial = nn.TransformerEncoder(
            layer(), cfg.spatial_layers, enable_nested_tensor=False
        )
        self.pool = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d))
        self.time_pos = nn.Embedding(T, d)
        self.temporal = nn.TransformerEncoder(
            layer(), cfg.temporal_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)
        self.variant_head = nn.Linear(d, cfg.n_variant)
        self.mz_head = nn.Linear(d, 2)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, T, P, F], [B, T, P] -> variant [B, T, C] and man/zone [B, T, 2] logits.

        Output t reads frames <= t. Outputs at frames with nobody seen are garbage.
        """
        b, t, p, _ = feats.shape
        h = self.embed(feats).view(b * t, p, -1)
        m = mask.reshape(b * t, p)
        pad = ~m
        pad[:, 0] &= m.any(dim=1)  # an empty frame attends to one dummy slot, not nothing
        h = self.spatial(h, src_key_padding_mask=pad)
        mf = m.unsqueeze(-1).float()
        cnt = mf.sum(1).clamp(min=1)
        off = feats.reshape(b * t, p, -1)[..., 6:7] * mf
        mean = (h * mf).sum(1) / cnt
        mean_def = (h * (mf - off)).sum(1) / (mf - off).sum(1).clamp(min=1)
        mx = h.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values * m.any(1, keepdim=True)
        z = self.pool(torch.cat([mean, mean_def, mx], dim=-1)).view(b, t, -1)
        z = z + self.time_pos.weight[:t]

        frame_ok = mask.any(-1)
        idx = torch.arange(t, device=feats.device)
        causal = idx.view(1, t, 1) >= idx.view(1, 1, t)
        allowed = causal & (
            frame_ok.unsqueeze(1) | torch.eye(t, dtype=torch.bool, device=idx.device)
        )
        blocked = (~allowed).repeat_interleave(self.cfg.heads, dim=0)
        z = self.norm(self.temporal(z, mask=blocked))
        return self.variant_head(z), self.mz_head(z)


def read_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid index of the full-play read (last seen frame) and the pre-snap read.

    The pre-snap read is the last seen frame at or before the snap; a play with
    nothing seen by the snap (film that starts late) falls back to its first frame.
    """
    frame_ok = mask.any(-1)
    t = frame_ok.shape[1]
    idx = torch.arange(t, device=mask.device).expand_as(frame_ok)
    last = torch.where(frame_ok, idx, -1).max(1).values.clamp(min=0)
    pre_ok = frame_ok & (idx <= SNAP_T)
    first = torch.where(frame_ok, idx, t).min(1).values.clamp(max=t - 1)
    pre = torch.where(pre_ok.any(1), torch.where(pre_ok, idx, -1).max(1).values, first)
    return last, pre


def losses(
    model: SequenceModel,
    feats: torch.Tensor,
    mask: torch.Tensor,
    y_var: torch.Tensor,
    y_mz: torch.Tensor,
    step_weight: float = 0.5,
    smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """CE at the full-play read and the pre-snap read, plus a mean over every seen frame.

    Labels of -1 are missing (the head is skipped for that play).
    """
    lv, lm = model(feats, mask)
    last, pre = read_indices(mask)
    b = torch.arange(len(last), device=feats.device)
    frame_ok = mask.any(-1).float()

    def ce(logits_bt: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ok = y >= 0
        if not ok.any():
            return logits_bt.sum() * 0.0
        yy = y.clamp(min=0)

        def at(i: torch.Tensor) -> torch.Tensor:
            return nn.functional.cross_entropy(
                logits_bt[b, i][ok], yy[ok], label_smoothing=smoothing
            )

        tb, tt, c = logits_bt.shape
        every = nn.functional.cross_entropy(
            logits_bt.reshape(-1, c),
            yy.view(-1, 1).expand(tb, tt).reshape(-1),
            label_smoothing=smoothing,
            reduction="none",
        ).view(tb, tt)
        w = frame_ok * ok.float().view(-1, 1)
        return at(last) + 0.5 * at(pre) + step_weight * (every * w).sum() / w.sum().clamp(min=1)

    lvar, lmz = ce(lv, y_var), ce(lm, y_mz)
    return lvar + 0.5 * lmz, {"variant": float(lvar.detach()), "man_zone": float(lmz.detach())}


# ------------------------------------------------------------------ persistence


@dataclass
class Bundle:
    model: SequenceModel
    classes: list[str]
    meta: dict = field(default_factory=dict)


def save(bundle: Bundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": bundle.model.state_dict(),
            "config": asdict(bundle.model.cfg),
            "classes": bundle.classes,
            "meta": bundle.meta,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {"config": asdict(bundle.model.cfg), "classes": bundle.classes, **bundle.meta},
            indent=2,
        )
    )


def load(path: str | Path, device: str = "cpu") -> Bundle:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SequenceModel(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return Bundle(model, list(ckpt["classes"]), ckpt.get("meta", {}))


def load_if_present(path: str | Path) -> Bundle | None:
    """The bundle on the GPU when there is one; None when the checkpoint is missing."""
    if not Path(path).exists():
        return None
    return load(path, "cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ inference


def _softmax_dict(logits: torch.Tensor, names: list[str]) -> dict[str, float]:
    p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return {n: float(v) for n, v in zip(names, p)}


@torch.no_grad()
def predict(bundle: Bundle, frames: pd.DataFrame) -> dict:
    """Read ONE play of canonical frames (tracking or film; `o`, `s`, `a` unused).

    Returns {class: prob} dicts: `variant` and `man_zone` from the whole play,
    `pre_variant` and `pre_man_zone` from frames up to and including the snap,
    plus `last_rel_frame` / `pre_rel_frame`, the frames each read was taken at.
    """
    model = bundle.model
    if frames.empty:
        raise ValueError("no frames")
    one = frames.copy()
    for k in KEYS:
        one[k] = 0 if k != "dataset" else "clip"
    t = play_tensors(one)
    if not t["mask"].any():
        raise ValueError(f"no player seen between rel_frame -{PRE_FRAMES} and {POST_FRAMES}")
    if np.isnan(t["ref"]).any():
        raise ValueError("no offense after the snap: LOS and formation center are unknown")
    dev = next(model.parameters()).device
    raw = torch.from_numpy(t["raw"]).to(dev)
    mask = torch.from_numpy(t["mask"]).to(dev)
    ref = torch.from_numpy(t["ref"]).to(dev)
    model.eval()
    lv, lm = model(featurize(raw, mask, ref), mask)
    last, pre = read_indices(mask)
    li, pi = int(last[0]), int(pre[0])
    return {
        "variant": _softmax_dict(lv[0, li], bundle.classes),
        "man_zone": _softmax_dict(lm[0, li], MAN_ZONE),
        "pre_variant": _softmax_dict(lv[0, pi], bundle.classes),
        "pre_man_zone": _softmax_dict(lm[0, pi], MAN_ZONE),
        "last_rel_frame": li - PRE_FRAMES,
        "pre_rel_frame": pi - PRE_FRAMES,
    }


def encode_labels(values: pd.Series, classes: list[str]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes)}
    return values.map(lookup).fillna(-1).astype(np.int64).to_numpy().copy()


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
