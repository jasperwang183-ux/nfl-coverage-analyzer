import numpy as np
import pandas as pd
import pytest
import torch

from nfl_coverage.models.sequence import (
    MAN_ZONE,
    N_FEAT,
    PRE_FRAMES,
    SNAP_T,
    Augment,
    Bundle,
    Config,
    SequenceModel,
    T,
    encode_labels,
    featurize,
    load,
    losses,
    play_tensors,
    predict,
    read_indices,
    save,
)

CLASSES = ["COVER_1_MAN", "COVER_3_ZONE"]


def synthetic_play(play_id: int = 1, pre: int = 5, post: int = 20, deep_drop: bool = True):
    """3 offense + 4 defense; the safeties either bail deep or stay (a toy Cover 3/1 split)."""
    rows = []
    start = {
        1: ("Offense", 38.0, 10.0),
        2: ("Offense", 38.0, 43.0),
        3: ("Offense", 34.0, 26.0),
        11: ("Defense", 45.0, 10.0),
        12: ("Defense", 45.0, 43.0),
        13: ("Defense", 50.0, 20.0),
        14: ("Defense", 50.0, 33.0),
    }
    for rf in range(-pre, post + 1):
        for pid, (side, x, y) in start.items():
            t = max(rf, 0) / 10.0
            if side == "Offense":
                x += 5.0 * t
            elif pid in (13, 14) and deep_drop:
                x += 3.0 * t
            rows.append(
                {
                    "dataset": "bdb2021",
                    "game_id": 1,
                    "play_id": play_id,
                    "nfl_id": pid,
                    "frame_id": rf + pre + 1,
                    "rel_frame": rf,
                    "x": x,
                    "y": y,
                    "s": np.nan,
                    "a": np.nan,
                    "o": np.nan,
                    "dir": np.nan,
                    "side": side,
                    "coverage_type": "COVER_3_ZONE" if deep_drop else "COVER_1_MAN",
                    "man_zone": "ZONE_COVERAGE" if deep_drop else "MAN_COVERAGE",
                }
            )
    return pd.DataFrame(rows)


def tiny_model(seed: int = 0) -> SequenceModel:
    torch.manual_seed(seed)
    return SequenceModel(
        Config(n_variant=len(CLASSES), d=32, heads=2, spatial_layers=1, temporal_layers=1, ff=64)
    )


def test_play_tensors_shapes_masks_and_reference():
    frames = pd.concat([synthetic_play(1, pre=5), synthetic_play(2, pre=0, post=8)])
    t = play_tensors(frames, p_max=6)
    assert t["raw"].shape == (2, T, 6, 3)
    assert t["mask"].shape == (2, T, 6)
    # play 1: frames -5..20 seen, earlier pre-snap and later post-snap slots empty
    seen = t["mask"][0].any(-1)
    assert seen[SNAP_T - 5 : SNAP_T + 21].all() and not seen[: SNAP_T - 5].any()
    assert not seen[SNAP_T + 21 :].any()
    # p_max 6 of 7 players: defenders fill slots first, so one offensive player is dropped
    assert t["mask"][0, SNAP_T].sum() == 6
    assert t["raw"][0, SNAP_T, :4, 2].sum() == 0  # slots 0-3 are the defense
    # LOS = 75th pct of offense x at the snap, center = offense mean y
    assert t["ref"][0, 0] == pytest.approx(np.quantile([38.0, 38.0, 34.0], 0.75))
    assert t["ref"][0, 1] == pytest.approx(np.mean([10.0, 43.0, 26.0]))
    assert list(t["keys"]["coverage_type"]) == ["COVER_3_ZONE", "COVER_3_ZONE"]


def test_featurize_velocity_from_positions_and_zero_when_masked():
    t = play_tensors(synthetic_play())
    raw, mask, ref = (torch.from_numpy(t[k]) for k in ("raw", "mask", "ref"))
    f = featurize(raw, mask, ref)
    assert f.shape == (1, T, raw.shape[2], N_FEAT)
    # an offensive receiver runs +5 yd/s after the snap: vx feature = 5 / 5
    off_slot = int(np.flatnonzero(t["raw"][0, SNAP_T, :, 2] == 1)[0])
    assert f[0, SNAP_T + 5, off_slot, 3].item() == pytest.approx(1.0, abs=1e-4)
    assert (f[~mask] == 0).all()


def test_augment_drops_and_truncates_only():
    t = play_tensors(synthetic_play(pre=10))
    raw, mask, ref = (torch.from_numpy(t[k]) for k in ("raw", "mask", "ref"))
    torch.manual_seed(0)
    _, m2, _ = Augment(drop_player=0.5)(
        raw.repeat(8, 1, 1, 1), mask.repeat(8, 1, 1), ref.repeat(8, 1)
    )
    assert (m2 <= mask).all()
    assert m2.sum() < mask.sum() * 8


def test_model_outputs_and_causality():
    t = play_tensors(synthetic_play())
    raw, mask, ref = (torch.from_numpy(t[k]) for k in ("raw", "mask", "ref"))
    model = tiny_model().eval()
    f = featurize(raw, mask, ref)
    lv, lm = model(f, mask)
    assert lv.shape == (1, T, len(CLASSES)) and lm.shape == (1, T, 2)
    assert torch.isfinite(lv).all() and torch.isfinite(lm).all()
    # changing a post-snap frame must not change the read at the snap
    f2 = f.clone()
    f2[:, SNAP_T + 3 :] += 1.0
    lv2, _ = model(f2, mask)
    assert torch.allclose(lv[0, : SNAP_T + 1], lv2[0, : SNAP_T + 1], atol=1e-5)
    assert not torch.allclose(lv[0, -1], lv2[0, -1])


def test_read_indices_last_and_pre_snap():
    mask = torch.zeros(3, T, 4, dtype=torch.bool)
    mask[0, SNAP_T - 3 : SNAP_T + 8, 0] = True
    mask[1, SNAP_T : SNAP_T + 2, 0] = True
    mask[2, SNAP_T + 4 : SNAP_T + 6, 0] = True  # film that starts after the snap
    last, pre = read_indices(mask)
    assert last.tolist() == [SNAP_T + 7, SNAP_T + 1, SNAP_T + 5]
    assert pre.tolist() == [SNAP_T, SNAP_T, SNAP_T + 4]


def test_predict_on_a_film_like_play_and_pre_snap_only(tmp_path):
    bundle = Bundle(tiny_model(), CLASSES, {"note": "test"})
    path = tmp_path / "seq.pt"
    save(bundle, path)
    assert path.with_suffix(".json").exists()
    loaded = load(path)
    film = synthetic_play(pre=12, post=35).drop(
        columns=["dataset", "game_id", "play_id", "coverage_type", "man_zone", "frame_id"]
    )
    film = film.drop(index=film.sample(frac=0.1, random_state=0).index)  # missed bodies
    out = predict(loaded, film)
    for k in ("variant", "pre_variant"):
        assert set(out[k]) == set(CLASSES) and sum(out[k].values()) == pytest.approx(1.0)
    assert set(out["man_zone"]) == set(MAN_ZONE)
    assert out["pre_rel_frame"] == 0 and out["last_rel_frame"] == 30
    # pre-snap only: the full read equals the pre-snap read, both at the snap
    pre_only = predict(loaded, film[film["rel_frame"] <= 0])
    assert pre_only["last_rel_frame"] == 0
    assert pre_only["variant"] == pytest.approx(pre_only["pre_variant"])
    assert pre_only["pre_variant"] == pytest.approx(out["pre_variant"], abs=1e-5)


def test_predict_rejects_a_play_without_offense():
    frames = synthetic_play()
    with pytest.raises(ValueError):
        predict(Bundle(tiny_model(), CLASSES), frames[frames["side"] == "Defense"])


def test_one_training_step_reduces_loss_on_two_plays():
    frames = pd.concat([synthetic_play(1, deep_drop=True), synthetic_play(2, deep_drop=False)])
    t = play_tensors(frames)
    raw, mask, ref = (torch.from_numpy(t[k]) for k in ("raw", "mask", "ref"))
    y_var = torch.from_numpy(encode_labels(t["keys"]["coverage_type"], CLASSES))
    y_mz = torch.from_numpy(encode_labels(t["keys"]["man_zone"], MAN_ZONE))
    assert y_var.tolist() == [1, 0] and y_mz.tolist() == [1, 0]
    model = tiny_model().eval()  # no dropout: one step must be comparable
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    f = featurize(raw, mask, ref)
    before, _ = losses(model, f, mask, y_var, y_mz)
    opt.zero_grad()
    before.backward()
    opt.step()
    after, parts = losses(model, f, mask, y_var, y_mz)
    assert after.item() < before.item()
    assert set(parts) == {"variant", "man_zone"}


def test_missing_labels_are_skipped():
    y = encode_labels(pd.Series(["COVER_3_ZONE", None, "PREVENT"]), CLASSES)
    assert y.tolist() == [1, -1, -1]
    assert PRE_FRAMES == SNAP_T


def test_film_reads_skip_the_model_off_the_10_hz_grid():
    from nfl_coverage.vision.pipeline import sequence_reads

    assert sequence_reads(None, None, None, 0, "left", 0.1) == (None, None)
    assert sequence_reads(object(), None, None, 0, "left", 0.2) == (None, None)
