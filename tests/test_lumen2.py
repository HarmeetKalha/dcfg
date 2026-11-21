"""Tests for dcfg.lumen2 (model parts, label stream, trainer hooks, Monday trigger). Tiny CLIP-like base, fake frame source."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from dcfg.data.degradations import TrainDegradation  # noqa: E402
from dcfg.data.ffpp import FrameRecord  # noqa: E402
from dcfg.degrade.labels import ops_to_labels  # noqa: E402
from dcfg.degrade.sensor import DamageSensor  # noqa: E402
from dcfg.gend.crop import gend_resize  # noqa: E402
from dcfg.gend.train_gend_aug import GendTrainSet, load_cfg  # noqa: E402
from dcfg.lumen2 import model as M  # noqa: E402
from dcfg.lumen2 import train as T  # noqa: E402
from dcfg.models.gate import FrequencyGate  # noqa: E402
from dcfg.train.train_frontend import TrainDataset, epoch_order  # noqa: E402

D = 8


class TinyBase(nn.Module):
    """Mimics GendCLIP's interface: ``vision`` with CLIP-named LayerNorms (all width D), ``linear`` head, ``features(x01 224)``."""

    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Module()
        self.vision.proj = nn.Linear(3, D)
        self.vision.pre_layrnorm = nn.LayerNorm(D)
        self.vision.layer_norm1 = nn.LayerNorm(D)
        self.vision.layer_norm2 = nn.LayerNorm(D)
        self.vision.post_layernorm = nn.LayerNorm(D)
        self.linear = nn.Linear(D, 2)

    def features(self, x01):
        t = self.vision.proj(x01.mean((2, 3))).unsqueeze(1).repeat(1, 3, 1) * torch.tensor([1.0, 2.0, 3.0], device=x01.device).view(1, 3, 1)
        t = self.vision.layer_norm2(self.vision.layer_norm1(self.vision.pre_layrnorm(t)))
        z = torch.nn.functional.normalize(self.vision.post_layernorm(t[:, 0]), dim=1)
        return self.linear(z), z

    def forward(self, x01):
        l, _ = self.features(x01)
        return l[:, 1] - l[:, 0]


def crops(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [np.clip(rng.normal(120, 40, (256, 256, 3)), 0, 255).astype(np.uint8) for _ in range(n)]


def packed_batch(imgs):
    return torch.stack([torch.from_numpy(M.pack_views(i)).permute(2, 0, 1) for i in imgs]).float() / 255.0


# ------------------------------------------------------------------------------------------------ model parts
def test_pack_views_layout():
    img = crops(1)[0]
    p = M.pack_views(img)
    assert p.shape == (480, 256, 3) and p.dtype == np.uint8
    assert np.array_equal(p[:256], img) and np.array_equal(p[256:, :224], gend_resize(img)) and not p[256:, 224:].any()
    c, r = M.unpack_views(torch.from_numpy(p).permute(2, 0, 1)[None])
    assert c.shape == (1, 3, 256, 256) and r.shape == (1, 3, 224, 224)


def test_cond_layernorm_equals_plain_at_init_and_with_zero_delta():
    torch.manual_seed(0)
    ln = nn.LayerNorm(D)
    nn.init.normal_(ln.weight), nn.init.normal_(ln.bias)
    c = M.CondLayerNorm.from_layernorm(ln)
    assert set(c.state_dict()) == set(ln.state_dict())  # same key names: GenD weights load unchanged
    for x in (torch.randn(3, 5, D), torch.randn(3, D)):
        z = torch.zeros(3, D)
        assert torch.equal(c(x), ln(x))  # no condition: IS the plain LN
        c.set_cond((z, z))
        assert torch.allclose(c(x), ln(x), atol=1e-5)  # zero delta: equal up to fp rounding
        c.set_cond(None)


def test_cond_layernorm_formula_and_token_sharing():
    torch.manual_seed(1)
    ln = nn.LayerNorm(D)
    c = M.CondLayerNorm.from_layernorm(ln)
    x, dg, db = torch.randn(2, 4, D), torch.randn(2, D), torch.randn(2, D)
    c.set_cond((dg, db))
    xh = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + ln.eps)
    want = xh * (ln.weight + dg)[:, None] + (ln.bias + db)[:, None]
    assert torch.allclose(c(x), want, atol=1e-5)


def test_cond_layernorm_checkpointed_grads_equal_plain():
    torch.manual_seed(2)
    c = M.CondLayerNorm(D)
    x = torch.randn(2, 3, D, requires_grad=True)
    dg, db = torch.randn(2, D, requires_grad=True), torch.randn(2, D, requires_grad=True)
    c.set_cond((dg, db))
    g1 = torch.autograd.grad(c(x).pow(2).sum(), [x, dg, db, c.weight, c.bias])
    g2 = torch.autograd.grad(c._modulate(x, dg, db).pow(2).sum(), [x, dg, db, c.weight, c.bias])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g1, g2))


def test_generator_zero_init_and_rank():
    g = M.LnGenerator(5, D, embed_dim=128)
    dg, db = g(torch.randn(3, 128))
    assert dg.shape == db.shape == (3, 5, D) and not dg.any() and not db.any()
    assert g.U.shape == (5, D, M.RANK) and M.RANK == 8
    with torch.no_grad():
        g.U.add_(1.0)
    dg2, _ = g(torch.randn(3, 128))
    assert dg2.abs().sum() > 0


def test_param_budget_real_sizes():
    """Real CLIP ViT-L/14 sizes without transformers: 50 LayerNorms of width 1024."""
    sensor, gate, gen = DamageSensor(), FrequencyGate(use_sensor=True), M.LnGenerator(50, 1024)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert (n(sensor), n(gate), n(gen)) == (772_264, 132_160, 837_000)
    assert n(sensor) + n(gate) + n(gen) < 3_000_000
    assert M.GENDTRAIN_PARAMS == 104_450 == 50 * 2 * 1024 + 1024 * 2 + 2


def build_tiny():
    torch.manual_seed(0)
    return M.Lumen2Model(TinyBase())


def test_lumen2_is_identity_at_init():
    m = build_tiny().eval()
    imgs = crops(4)
    x = packed_batch(imgs)
    x224 = torch.stack([torch.from_numpy(gend_resize(i)).permute(2, 0, 1) for i in imgs]).float() / 255.0
    with torch.no_grad():
        logit, sens, w = m.predict_with_sensor(x)
        assert torch.equal(w, torch.ones_like(w))  # gate weights exactly 1
        assert torch.allclose(logit, m.base(x224), atol=1e-4)  # same output as the base GenD model
    assert set(sens) == {"embedding", "presence_logit", "strength"} and sens["embedding"].shape == (4, 128)
    assert all(mod._cond is None for mod in m.cond_lns)  # condition cleared after forward


def test_lumen2_trains_all_parts_and_condition_matters(cpu_only):
    m = build_tiny().train()
    for n, p in m.named_parameters():
        p.requires_grad = M.Lumen2Model.is_trainable_name(n)
    x = packed_batch(crops(4))
    logits, z, sens, w = m.forward_train(x)
    (logits.sum() + sens["presence_logit"].sum()).backward()
    assert m.gen.U.grad.abs().sum() > 0 and m.gen.V.grad.abs().sum() > 0  # U,V get gradient although zero-init
    assert m.gate.fc2.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.sensor.parameters())
    assert not m.base.vision.proj.weight.requires_grad  # frozen
    with torch.no_grad():  # a non-zero U changes the output
        m.eval()
        y0 = m(x)
        m.gen.U.copy_(torch.randn_like(m.gen.U)); m.gen.V.copy_(torch.randn_like(m.gen.V))
        assert not torch.allclose(m(x), y0)


def test_trainable_names_and_state_roundtrip():
    m = build_tiny()
    names = {k for k in m.state_dict() if M.Lumen2Model.is_trainable_name(k)}
    assert "base.vision.layer_norm1.weight" in names and "base.linear.bias" in names and "base.vision.proj.weight" not in names
    assert any(k.startswith("sensor.") and "running_mean" in k for k in names)  # BatchNorm statistics are part of the state
    assert all(k.startswith(("base.", "sensor.", "gate.", "gen.")) for k in names)
    sd = M.trainable_state(m)
    m2 = build_tiny()
    with torch.no_grad():
        m2.gen.U.add_(1.0)
    M.load_trainable(m2, sd)
    assert torch.equal(m2.gen.U, m.gen.U)
    with pytest.raises(KeyError):
        M.load_trainable(m2, {k: v for k, v in sd.items() if k != "gen.U"})


# ------------------------------------------------------------------------------------------------ data stream
class FakeSource:
    def __init__(self, nvid: int = 6, nfr: int = 8) -> None:
        self.recs = {}
        for split in ("train", "val", "test"):
            r = []
            for v in range(nvid):
                lab = v % 2
                name = ("FF-real" if lab == 0 else "FF-DF") + f"/{v:03d}"
                r += [FrameRecord(path="", key=f"{split}/{name}/{f:03d}.png", video=name, label=lab) for f in range(nfr)]
            self.recs[split] = r

    def records(self, split, comp):
        return self.recs[split]

    def load(self, rec):
        rng = np.random.default_rng(int(hashlib.sha1(rec.key.encode()).hexdigest()[:8], 16))
        return np.clip(rng.normal(60 + 120 * rec.label, 20, (256, 256, 3)), 0, 255).astype(np.uint8)


def test_label_stream_bit_identical_to_gend_aug_stream():
    """The damaged images (256 crop and 224 resize) hashed over whole batches equal GenD-aug's stream; labels = ops_to_labels."""
    src = FakeSource()
    recs = src.records("train", "c23")
    labels = np.array([r.label for r in recs])
    order = epoch_order(labels, 42, 3, "full")
    deg = TrainDegradation()
    ref = GendTrainSet(TrainDataset(src, recs, order, 5, 42, 3, deg))
    new = T.LabelledTrainDataset(src, recs, order, 5, 42, 3, deg)
    plain = TrainDataset(src, recs, order, 5, 42, 3, deg)
    h_ref, h_new, h_256 = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    n_deg = 0
    for j in range(min(len(ref), 40)):
        a224, _, ridx_a = ref[j]
        packed, y, ridx_b, applied, strength = new[j]
        p256 = plain[j][0]
        assert ridx_a == ridx_b and packed.shape == (3, 480, 256)
        h_ref.update(a224.numpy().tobytes())
        h_new.update(packed[:, 256:, :224].contiguous().numpy().tobytes())
        h_256.update(p256.numpy().tobytes())
        assert torch.equal(packed[:, :256], p256)
        _, info = deg(src.load(recs[ridx_b]), np.random.default_rng(0))  # labels only checked structurally below
        assert applied.shape == (4,) and strength.shape == (4,) and ((strength > 0) <= (applied > 0)).all()
        n_deg += int(applied.sum() > 0)
    assert h_ref.hexdigest() == h_new.hexdigest()
    assert n_deg > 0


def test_labels_match_ops():
    from dcfg.data.degradations import make_rng
    from dcfg.degrade.labels import apply_with_labels

    img = crops(1)[0]
    deg = TrainDegradation()
    for pos in range(12):
        out, info, a, s = apply_with_labels(deg, img, make_rng(42, 0, pos))
        ref, ref_info = deg(img, make_rng(42, 0, pos))
        assert np.array_equal(out, ref)
        ea, es = ops_to_labels([(o["op"], o["param"]) for o in ref_info["ops"]])
        assert np.array_equal(a, ea) and np.array_equal(s, es)


# ------------------------------------------------------------------------------------------------ Monday trigger
def test_should_run_partial_truth_table():
    d = datetime(2026, 11, 23, 12, 30, tzinfo=timezone.utc)
    before, after = d - timedelta(seconds=1), d
    assert not T.should_run_partial(before, d, False, False)  # too early
    assert T.should_run_partial(after, d, False, False)  # at/after 18:00 IST (12:30 UTC)
    assert T.should_run_partial(after + timedelta(days=1), d, False, False)
    assert not T.should_run_partial(after, d, True, False)  # already done (idempotent across resumes)
    assert not T.should_run_partial(after, d, False, True)  # training finished
    assert T.MONDAY_DEADLINE_UTC == datetime(2026, 11, 23, 18, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def cfg(**kw):
    c = load_cfg("configs/train/lumen2.yaml")
    c.update(batch_size=4, accum=2, num_workers=0, val_workers=0, max_epochs=3, val_frames_per_video=4, ckpt_every_steps=3,
             log_every=2, lr=1e-2, min_lr=1e-3, patience=6, require_pass_marker=None)
    c.update(kw)
    return c


def test_config_identical_to_gend_aug_except_additions():
    a, b = load_cfg("configs/train/gend_aug.yaml"), load_cfg("configs/train/lumen2.yaml")
    assert {k: v for k, v in b.items() if k not in ("run_id", "w_damage", "w_gate_penalty", "require_pass_marker")} == \
           {k: v for k, v in a.items() if k != "run_id"}
    assert (b["w_damage"], b["w_gate_penalty"]) == (0.1, 1e-4)


@pytest.fixture
def cpu_only(monkeypatch):
    """CPU is deterministic (GPU conv nondeterminism breaks exact resume comparisons at the test's large lr)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_trainer_runs_resumes_and_monday_backup_once(tmp_path, monkeypatch, cpu_only):
    monkeypatch.setattr(T, "RESULTS_ROOT", tmp_path / "results")
    calls = []
    monkeypatch.setattr(T.Lumen2Trainer, "_publish", staticmethod(lambda job, summary: calls.append(job)))
    now = datetime(2026, 11, 23, 12, 31, tzinfo=timezone.utc)  # after the deadline from the start
    tr = T.Lumen2Trainer(cfg(), tmp_path / "run", model_fn=build_tiny, source=FakeSource(), expected_gend_trainable=None,
                         now_fn=lambda: now)
    res = tr.train()
    assert res["stop_reason"] == "epoch_cap" and res["epochs_run"] == 3 and "part_params" in res
    # the backup ran after epoch 1 only (once), not after the final epoch, and is marked done
    parts = sorted(p.name for p in (tmp_path / "results" / "lumen2").glob("partial_epoch*.json"))
    assert parts == ["partial_epoch1.json"] and calls == ["lumen2_partial_epoch1"]
    pj = json.loads((tmp_path / "results/lumen2/partial_epoch1.json").read_text())
    assert pj["label"] == "partial, epoch 1, 8 frames/video" and pj["frames_per_video"] == 8 and set(pj["sensor"]) == {"per_op", "n_labelled_frames"}
    assert (tmp_path / "run/monday_done.json").exists()
    ck = torch.load(tmp_path / "run/best_trainable.pt", weights_only=False)
    assert set(ck["trainable"]) == set(M.trainable_state(build_tiny()))
    # weights restored after the partial evaluation: epoch-1 best did not overwrite the live model (3 epochs trained)
    assert tr.st["step"] == 3 * tr.spe
    # a resumed run of a finished job returns the same result and does not run the backup again
    tr2 = T.Lumen2Trainer(cfg(), tmp_path / "run", model_fn=build_tiny, source=FakeSource(), expected_gend_trainable=None, now_fn=lambda: now)
    assert tr2.train()["best_epoch"] == res["best_epoch"] and len(calls) == 1


def test_resume_gives_same_batches_and_params(tmp_path, cpu_only):
    def run(d, steps, extra=None):
        t = T.Lumen2Trainer(cfg(max_steps=steps, **(extra or {})), d, model_fn=build_tiny, source=FakeSource(),
                            expected_gend_trainable=None, now_fn=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
        t.train()
        return t

    a = run(tmp_path / "a", 4)
    run(tmp_path / "b", 2)
    (tmp_path / "b" / "done.json").unlink()
    b = run(tmp_path / "b", 4)
    ha = [ln.split()[-1] for ln in (tmp_path / "a/batches.log").read_text().splitlines() if not ln.startswith("#")]
    hb = [ln.split()[-1] for ln in (tmp_path / "b/batches.log").read_text().splitlines() if not ln.startswith("#")]
    assert ha == hb and len(ha) == 4
    sa, sb = M.trainable_state(a.model), M.trainable_state(b.model)
    assert max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa) < 1e-5
