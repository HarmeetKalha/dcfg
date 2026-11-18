"""Design-selection variants: consistency loss, FiLM (identity at init, hooks, teacher isolation), engine, eval."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from dcfg.detectors.adapter import Detector, FilmHooks
from dcfg.eval import evaluate as ev
from dcfg.models.film import FILM_KINDS, FiLMBank
from dcfg.models.frontend import FrontEnd
from dcfg.paths import REPO_ROOT
from dcfg.train import consistency as cons
from dcfg.train import train_frontend as tf
from test_train_frontend import DummyDetector, cfg_for, data  # noqa: F401

# ------------------------------------------------------------------ consistency loss
def test_bernoulli_kl_equals_two_class_softmax_kl():
    torch.manual_seed(0)
    t, s = torch.randn(16) * 3, torch.randn(16) * 3
    two = lambda l: torch.stack([torch.zeros_like(l), l], 1)  # noqa: E731  cls0=0, cls1=l -> logit = cls1 - cls0
    pt, ps = F.softmax(two(t), 1), F.softmax(two(s), 1)
    ref = (pt * (pt.log() - ps.log())).sum(1)
    assert torch.allclose(cons.bernoulli_kl(t, s), ref, atol=1e-5)
    assert torch.allclose(cons.bernoulli_kl(t, t), torch.zeros(16), atol=1e-6)
    assert (cons.bernoulli_kl(t, s) >= -1e-6).all()


def test_consistency_loss_terms_and_gradients():
    s_l = torch.randn(8, requires_grad=True)
    s_f = torch.randn(8, 5, requires_grad=True)
    t_l, t_f, y = torch.randn(8), torch.randn(8, 5), torch.randint(0, 2, (8,)).float()
    total, terms = cons.consistency_loss(s_l, s_f, t_l, t_f, y, {"kl": 2.0, "feat": 0.5})
    exp = terms["bce"] + 2.0 * terms["kl"] + 0.5 * terms["feat"]
    assert torch.allclose(total.detach(), exp, atol=1e-6)
    assert set(terms) == {"bce", "kl", "feat"} and all(not v.requires_grad for v in terms.values())
    total.backward()
    assert s_l.grad is not None and s_f.grad is not None
    with pytest.raises(AssertionError):  # teacher must be no-grad
        cons.consistency_loss(s_l, s_f, t_l.clone().requires_grad_(), t_f, y, {"kl": 1, "feat": 1})
    assert cons.parse_config(None) is None and cons.parse_config({"kl": 1, "feat": 1}) == {"kl": 1.0, "feat": 1.0}
    with pytest.raises(KeyError):
        cons.parse_config({"kl": 1, "bogus": 1})


# ------------------------------------------------------------------ real Xception (CPU, random-init head not needed)
@pytest.fixture(scope="module")
def xcep():
    return Detector("xception", device="cpu").freeze()


def _film_front(det, kind="film"):
    ch = {k: c for k, (_, c) in det.film_sites().items()}
    return FrontEnd("dcfg", film=FiLMBank(kind, ch)).eval()


def test_film_site_channels_from_actual_module(xcep):
    sites = xcep.film_sites()
    assert {k: c for k, (_, c) in sites.items()} == {"entry": 728, "mid4": 728, "mid8": 728, "exit": 2048}
    bb = xcep.model.backbone
    assert sites["entry"][0] is bb.block3 and sites["mid4"][0] is bb.block7
    assert sites["mid8"][0] is bb.block11 and sites["exit"][0] is bb.bn4
    # measured output channels of the real hooked modules
    got = {}
    hs = [m.register_forward_hook(lambda _m, _i, o, k=k: got.__setitem__(k, o.shape[1])) for k, (m, _) in sites.items()]
    with torch.no_grad():
        xcep(xcep.normalize(torch.rand(1, 3, 256, 256)))
    for h in hs:
        h.remove()
    assert got == {"entry": 728, "mid4": 728, "mid8": 728, "exit": 2048}


def test_film_param_counts():
    ch = {"entry": 728, "mid4": 728, "mid8": 728, "exit": 2048}
    full, lite = FiLMBank("film", ch), FiLMBank("filmlite", ch)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(full) == 128 * 256 + 256 + 256 * 2 * 4232 + 2 * 4232 == 2_208_272
    assert n(lite) == 128 * 64 + 64 + 64 * 2 * 2776 + 2 * 2776 == 369_136
    assert lite.sites == ("entry", "exit") and full.sites == ("entry", "mid4", "mid8", "exit")


def test_film_exact_identity_at_init_then_differs(xcep):
    torch.manual_seed(0)
    x = torch.rand(2, 3, 256, 256)
    fe = _film_front(xcep)
    assert all(float(p.detach().abs().sum()) == 0 for p in fe.film.fc2.parameters())  # zero-init weight + bias
    hooks = FilmHooks(xcep, fe.film.sites)
    try:
        with torch.no_grad():
            y, _, fp = fe.forward_full(x)
            plain = xcep(xcep.normalize(y))
            with hooks.conditioned(fp):
                with_film = xcep(xcep.normalize(y))
            assert (plain - with_film).abs().max() <= 1e-5 and torch.equal(plain, with_film)
            for dg, b in fp.values():
                assert float(dg.abs().max()) == 0 and float(b.abs().max()) == 0
            # perturbation -> outputs differ
            with torch.no_grad():
                fe.film.fc2.weight.normal_(0, 0.05)
                fe.film.fc2.bias.normal_(0, 0.05)
            _, _, fp2 = fe.forward_full(x)
            with hooks.conditioned(fp2):
                pert = xcep(xcep.normalize(y))
            assert (pert - plain).abs().max() > 1e-3
            # hooks off: pass-through even with params set and conditioned
            with hooks.conditioned(fp2), hooks.disabled():
                off = xcep(xcep.normalize(y))
            assert torch.equal(off, plain)
            assert torch.equal(xcep(xcep.normalize(y)), plain)  # outside the context: plain again
    finally:
        hooks.remove()


def test_film_lite_only_entry_exit_hooks(xcep):
    fe = _film_front(xcep, "filmlite")
    hooks = FilmHooks(xcep, fe.film.sites)
    try:
        assert len(hooks.handles) == 2 and set(hooks.channels) == {"entry", "exit"}
    finally:
        hooks.remove()
    assert not xcep.model.backbone.block7._forward_hooks


def test_teacher_has_hooks_off_and_detector_unchanged(xcep):
    """consistency_forward: teacher == plain detector(clean) even with perturbed FiLM; student differs;
    after optimizer steps the detector params/buffers are bit-identical and frozen."""
    torch.manual_seed(1)
    fe = _film_front(xcep)
    with torch.no_grad():
        fe.film.fc2.weight.normal_(0, 0.05)
    fe.train()
    hooks = FilmHooks(xcep, fe.film.sites)
    pipe = tf.Pipeline(fe, xcep, amp=False, train_detector=False, film_hooks=hooks)
    before = {k: v.clone() for k, v in xcep.state_dict().items()}
    opt = torch.optim.AdamW([p for p in fe.parameters() if p.requires_grad], lr=1e-3)
    x_deg, x_clean = torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256)
    y = torch.tensor([0.0, 1.0])
    try:
        for _ in range(2):
            o = pipe.consistency_forward(x_deg, x_clean)
            with torch.no_grad():
                plain_t = xcep(xcep.normalize(x_clean))
            assert torch.equal(o["t_logit"], plain_t) and not o["t_logit"].requires_grad
            loss, _ = cons.consistency_loss(o["s_logit"], o["s_feat"], o["t_logit"], o["t_feat"], y, {"kl": 1, "feat": 1})
            opt.zero_grad()
            loss.backward()
            assert fe.film.fc2.weight.grad is not None and float(fe.film.fc2.weight.grad.abs().sum()) > 0
            assert all(p.grad is None for p in xcep.parameters())  # frozen: no grads on detector
            opt.step()
        tf.assert_detector_state(xcep, False)
        after = xcep.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert o["t_feat"].shape == (2, 2048) and o["s_feat"].shape == (2, 2048)
        assert not xcep.training
    finally:
        hooks.remove()


def test_forward_with_features_matches_forward(xcep):
    z = xcep.normalize(torch.rand(2, 3, 256, 256))
    with torch.no_grad():
        l, f = xcep.forward_with_features(z)
        assert f.shape == (2, 2048) and torch.equal(l, xcep(z)) and (f >= 0).all()  # post-ReLU pooled
        assert torch.allclose(xcep.model.backbone.last_linear(f)[:, 1] - xcep.model.backbone.last_linear(f)[:, 0], l, atol=1e-5)


# ------------------------------------------------------------------ engine (dummy detector)
def test_dataset_returns_clean_and_degraded(data, tmp_path):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_cons"))
    ds = tf.TrainDataset(t.source, t.records, np.arange(len(t.records)), 0, 42, 0, t.deg, return_clean=True)
    ds0 = tf.TrainDataset(t.source, t.records, np.arange(len(t.records)), 0, 42, 0, t.deg)
    n_clean = n_deg = 0
    for j in range(len(ds)):
        deg, y, r, s, clean = ds[j]
        assert deg.shape == clean.shape and torch.equal(ds0[j][0], deg)  # same rng/degradation as before
        same = torch.equal(deg, clean)
        n_clean += same
        n_deg += not same
    assert n_clean > 0 and n_deg > 0 and len(ds0[0]) == 4


@pytest.mark.parametrize("variant", ["dcfg_cons", "dcfg_film", "dcfg_filmlite"])
def test_new_variants_train_log_terms_and_artifacts(data, tmp_path, variant):
    cfg = cfg_for(data, tmp_path, variant, max_epochs=1, log_every=2)
    cfg["consistency"] = {"kl": 1.0, "feat": 1.0}
    t = tf.Trainer(cfg)
    t.train()
    rd = tmp_path / f"run_{variant}"
    hdr = (rd / "train_log.csv").read_text().splitlines()[0].split(",")
    assert {"loss", "loss_bce", "loss_kl", "loss_feat"} <= set(hdr) and ("gnorm_film" in hdr) == (variant != "dcfg_cons")
    ck = torch.load(rd / "frontend.pt", weights_only=False)
    assert (ck["film"] is None) == (variant == "dcfg_cons") and ck["variant"] == variant
    assert not (rd / "detector.pt").exists() and t.train_det is False
    gate_group = [g for g in t.opt.param_groups if g["name"] == "gate"][0]
    n_film = sum(p.numel() for p in t.frontend.film.parameters()) if t.frontend.film is not None else 0
    assert sum(p.numel() for p in gate_group["params"]) == sum(p.numel() for p in t.frontend.gate.parameters()) + n_film
    assert gate_group["lr"] == pytest.approx(1e-3)


def test_consistency_defaults_and_rejected_for_other_variants(data, tmp_path):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_cons"))
    assert t.cons == {"kl": 1.0, "feat": 1.0} and t.cfg["consistency"] == {"kl": 1.0, "feat": 1.0}
    with pytest.raises(ValueError, match="consistency"):
        tf.Trainer(cfg_for(data, tmp_path, "fixed", consistency={"kl": 1.0}))


def test_consistency_normalization_and_autocast_same_for_teacher_and_student(data, tmp_path):
    DummyDetector.n_normalize = 0
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_cons"))
    DummyDetector.n_normalize = 0
    x = torch.rand(4, 3, 64, 64, device=t.dev)
    t.model.consistency_forward(x, x)
    assert DummyDetector.n_normalize == 2  # one normalization per pass (teacher, student), same code path


def test_frozen_detector_with_film_passes_check(data, tmp_path):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_film", max_steps=3))
    t.train()
    t.check_frozen()
    assert t.film_hooks is not None and len(t.film_hooks.handles) == 4


def test_frontend_roundtrip_with_film_and_eval_applies_film(data, tmp_path, monkeypatch):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_filmlite", max_epochs=1))
    t.train()
    rd = tmp_path / "run_dcfg_filmlite"
    fe, meta = tf.load_frontend(rd / "frontend.pt")
    assert fe.film is not None and fe.film.kind == "filmlite" and meta["film"]["channels"] == {"entry": 4, "exit": 4}
    assert all(torch.equal(a, b.cpu()) for a, b in zip(fe.state_dict().values(), t.frontend.state_dict().values()))
    w = tmp_path / "w.pt"
    torch.save({"state_dict": DummyDetector().model.state_dict()}, w)
    seen = []
    orig = FilmHooks.conditioned

    def spy(self, params):
        seen.append(params is not None and set(params) == {"entry", "exit"})
        return orig(self, params)

    monkeypatch.setattr(FilmHooks, "conditioned", spy)
    out = tmp_path / "res"
    rc = ev.main(["--run", "t_dcfg_filmlite", "--detector", "dummy", "--frontend", str(rd / "frontend.pt"),
                  "--weights", str(w), "--conditions", "clean", "--json", data["json"], "--frames-root", data["frames_root"],
                  "--out-dir", str(out), "--runs-dir", str(tmp_path / "runs"), "--workers", "0", "--n-boot", "10"])
    assert rc == 0 and seen and all(seen)
    assert json.loads((out / "t_dcfg_filmlite__clean.json").read_text())["variant"] == "dcfg_filmlite"


def test_eval_film_changes_scores_vs_identity(data, tmp_path):
    """Same frontend.pt, film weights zeroed vs trained: eval output must differ (FiLM actually applied)."""
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_film", max_epochs=1))
    t.train()
    rd = tmp_path / "run_dcfg_film"
    ck = torch.load(rd / "frontend.pt", weights_only=False)
    ck["state_dict"] = {k: (torch.randn_like(v) * 0.5 if k.startswith("film.fc2") else v) for k, v in ck["state_dict"].items()}
    torch.save(ck, rd / "frontend_perturbed.pt")
    ck0 = copy.deepcopy(ck)
    ck0["state_dict"] = {k: (torch.zeros_like(v) if k.startswith("film.fc2") else v) for k, v in ck["state_dict"].items()}
    torch.save(ck0, rd / "frontend_zero.pt")
    w = tmp_path / "w.pt"
    torch.save({"state_dict": DummyDetector().model.state_dict()}, w)
    res = {}
    for name in ("perturbed", "zero"):
        out = tmp_path / f"r_{name}"
        ev.main(["--run", "t_dcfg_film", "--detector", "dummy", "--frontend", str(rd / f"frontend_{name}.pt"),
                 "--weights", str(w), "--conditions", "clean", "--json", data["json"], "--frames-root", data["frames_root"],
                 "--out-dir", str(out), "--runs-dir", str(tmp_path / "runs"), "--workers", "0", "--n-boot", "10"])
        res[name] = np.load(out / "scores" / "t_dcfg_film__clean.npz")["frame_prob"]
    assert not np.allclose(res["perturbed"], res["zero"], atol=1e-6)


# ------------------------------------------------------------------ pilot configs / isolation (new dirs)
NEW = ["jointft", "dcfgcons", "dcfgfilm", "dcfgfilmlite"]


@pytest.mark.parametrize("x", NEW)
def test_new_pilot_configs_isolated_and_standard(x):
    c = tf.load_config(REPO_ROOT / f"configs/train/pilot/pilot_{x}_xception.yaml")
    assert c["run_id"] == f"pilot_{x}_xception" and c["pilot"] is True and c["stop_after_epochs"] == 5
    assert c["run_dir"] == f"runs/pilot/pilot_{x}_xception" and c["wandb_id"] == f"pilot_{x}_xception_r1"
    assert c["timing_sync"] is True and c["max_epochs"] == 30 and c["batch_size"] == 32 and c["seed"] == 42
    assert c["epoch_mode"] == "full" and c["detector"] == "xception"
    assert (c["variant"] in tf.CONSISTENCY_VARIANTS) == (x != "jointft")
    if x != "jointft":
        assert c["consistency"] == {"kl": 1.0, "feat": 1.0}


def test_jointft_pilot_equals_real_jointft_except_pilot_keys():
    real = tf.load_config(REPO_ROOT / "configs/train/jointft_xception.yaml")
    pil = tf.load_config(REPO_ROOT / "configs/train/pilot/pilot_jointft_xception.yaml")
    diff = {k for k in set(real) | set(pil) if real.get(k) != pil.get(k)}
    assert diff == {"run_id", "pilot", "stop_after_epochs", "timing_sync", "run_dir", "wandb_id"}
    assert pil["lr_gate"] == 1e-3 and pil["lr_sensor"] == 1e-4 and pil["lr_detector"] == 2e-5


def test_real_configs_untouched_by_new_variants():
    for p in (REPO_ROOT / "configs/train").glob("*.yaml"):
        if p.name == "base.yaml":
            continue
        c = tf.load_config(p)
        assert not c.get("consistency")  # lumen variants get the default 1/1 from the engine, not from the config
        assert "consistency" not in yaml.safe_load(p.read_text())


def test_pilot_check_out_dir_new_ids():
    for x in NEW:
        with pytest.raises(ValueError, match="pilot"):
            ev.check_out_dir("test", ev.RAW_DIR, f"pilot_{x}_xception")


def test_pilot_eval_script_handles_new_run_ids():
    import subprocess
    sh = (REPO_ROOT / "ops/jobs/pilot_eval.sh").read_text()
    assert 'det="${run#pilot_*_}"' in sh and "frontend.pt" in sh and "detector.pt" in sh
    assert subprocess.run(["bash", "-n", str(REPO_ROOT / "ops/jobs/pilot_eval.sh")]).returncode == 0
    for x in NEW:  # detector name parsed from the run id
        rid = f"pilot_{x}_xception"
        assert subprocess.check_output(["bash", "-c", f'run={rid}; echo "${{run#pilot_*_}}"'], text=True).strip() == "xception"
