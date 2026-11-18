"""LUMEN: variants / alias, per-detector FiLM sites + identity at init, hidden-width sizing, bnft, logging fields."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from dcfg.detectors.adapter import Detector, FilmHooks
from dcfg.detectors.registry import weights_available
from dcfg.eval import evaluate as ev
from dcfg.models.film import (GATE_SENSOR_PARAMS, LUMEN_HIDDEN, FiLMBank, choose_hidden, film_param_count)
from dcfg.models.frontend import FrontEnd
from dcfg.train import train_frontend as tf
from test_train_frontend import DummyDetector, cfg_for, data  # noqa: F401

JOBS = Path(__file__).resolve().parents[1] / "ops" / "jobs"
sys.path.insert(0, str(JOBS))
import gen_matrix as gm  # noqa: E402

SITES = {"xception": {"entry": ("block3", 728), "exit": ("bn4", 2048)},
         "f3net": {"entry": ("block3", 728), "exit": ("bn4", 2048)},
         "efficientnetb4": {"entry": ("_blocks.9", 56), "exit": ("_bn1", 1792)},
         "ucf": {"entry": ("block3", 728), "exit": ("bn4", 2048)}}


# ------------------------------------------------------------------ variants / alias
def test_alias_and_new_variants_registered():
    assert tf.canonical_variant("dcfg_filmlite") == "lumen" and tf.canonical_variant("lumen") == "lumen"
    for v in ("lumen", "lumen_nocons", "lumen_nolive", "lumen_frozensensor", "bnft", "dcfg_filmlite"):
        assert v in tf.VARIANTS
    with pytest.raises(ValueError):
        tf.canonical_variant("nope")
    assert "lumen_nocons" not in tf.CONSISTENCY_VARIANTS and "lumen_nolive" in tf.CONSISTENCY_VARIANTS


@pytest.mark.parametrize("variant", ["lumen", "dcfg_filmlite", "lumen_nocons", "lumen_nolive", "lumen_frozensensor"])
def test_lumen_family_structure(data, tmp_path, variant):  # noqa: F811
    t = tf.Trainer(cfg_for(data, tmp_path, variant, max_epochs=1))
    assert t.frontend.film is not None and t.frontend.film.sites == ("entry", "exit")
    assert (t.cons is None) == (variant == "lumen_nocons")
    assert t.frontend.freeze_sensor == (variant == "lumen_frozensensor")
    sens = t.frontend.sensor
    assert all(p.requires_grad for p in sens.parameters()) != (variant == "lumen_frozensensor")
    assert not t.train_det and not t.bn_only
    t.check_frozen()


def test_nolive_has_random_sensor_and_live_has_loaded(data, tmp_path):  # noqa: F811
    live = tf.Trainer(cfg_for(data, tmp_path, "lumen")).frontend.sensor
    nolive = tf.Trainer(cfg_for(data, tmp_path, "lumen_nolive", run_dir=str(tmp_path / "nl"))).frontend.sensor
    ref = torch.load(tmp_path / "sensor.pt", weights_only=True)
    assert all(torch.equal(live.state_dict()[k].cpu(), v) for k, v in ref.items())
    assert not all(torch.equal(nolive.state_dict()[k].cpu(), v) for k, v in ref.items())


def test_alias_run_matches_lumen_and_keeps_name(data, tmp_path):  # noqa: F811
    a = tf.Trainer(cfg_for(data, tmp_path, "dcfg_filmlite", max_epochs=1))
    a.train()
    ck = torch.load(tmp_path / "run_dcfg_filmlite" / "frontend.pt", weights_only=False)
    assert ck["variant"] == "dcfg_filmlite" and ck["film"]["kind"] == "filmlite"
    assert a.cons == {"kl": 1.0, "feat": 1.0}


# ------------------------------------------------------------------ hidden width sizing
def test_choose_hidden_matches_table_and_is_3_to_4_percent():
    n = {"xception": 21_861_162, "efficientnetb4": 17_552_202, "f3net": 22_519_114}
    ch = {"xception": [728, 2048], "f3net": [728, 2048], "efficientnetb4": [56, 1792]}
    for d in n:
        h = choose_hidden(n[d], ch[d])
        assert h == LUMEN_HIDDEN[d]
        pct = 100 * (GATE_SENSOR_PARAMS + film_param_count(ch[d], h)) / n[d]
        assert 3.0 <= pct <= 4.0
    assert GATE_SENSOR_PARAMS + film_param_count([728, 2048], 64) == 763_505  # the pilot's dcfg_filmlite


def test_film_hidden_override_and_meta_roundtrip():
    b = FiLMBank("filmlite", {"entry": 5, "exit": 7}, hidden=16)
    assert b.fc1.out_features == 16 and b.meta()["hidden"] == 16
    assert sum(p.numel() for p in b.parameters()) == film_param_count([5, 7], 16)
    assert FiLMBank("filmlite", {"entry": 5, "exit": 7}).hidden == 64  # old default (pilot checkpoints)


# ------------------------------------------------------------------ per-detector sites + identity at init
@pytest.fixture(scope="module")
def dets():
    cache: dict[str, Detector] = {}

    def get(name: str) -> Detector:
        if not weights_available(name):
            pytest.skip(f"weights for {name} absent")
        if name not in cache:
            cache[name] = Detector(name, device="cpu").freeze()
        return cache[name]
    return get


@pytest.mark.parametrize("name", list(SITES))
def test_film_sites_modules_channels(dets, name):
    d = dets(name)
    sites = d.film_sites()
    named = dict(d.model.named_modules())
    for site, (modname, ch) in SITES[name].items():
        mod, c = sites[site]
        assert c == ch
        bb = {"efficientnetb4": "backbone.efficientnet.", "ucf": "encoder_f."}.get(name, "backbone.")
        assert named[bb + modname] is mod


@pytest.mark.parametrize("name", list(SITES))
def test_lumen_identity_at_init(dets, name):
    d = dets(name)
    ch = {k: c for k, (_, c) in d.film_sites().items() if k in ("entry", "exit")}
    fe = FrontEnd("dcfg", film=FiLMBank("filmlite", ch, hidden=LUMEN_HIDDEN[name])).eval()
    hooks = FilmHooks(d, fe.film.sites)
    try:
        torch.manual_seed(0)
        x = torch.rand(2, 3, 256, 256)
        with torch.no_grad():
            y, _, fp = fe.forward_full(x)
            plain = d(d.normalize(y))
            with hooks.conditioned(fp):
                with_film = d(d.normalize(y))
            assert (plain - with_film).abs().max() <= 1e-5
            with torch.no_grad():  # and it is not a no-op once the last layer is nonzero
                fe.film.fc2.weight.normal_(0, 0.05)
            y, _, fp = fe.forward_full(x)
            with hooks.conditioned(fp):
                moved = d(d.normalize(y))
            assert (plain - moved).abs().max() > 1e-4
    finally:
        hooks.remove()


def test_ucf_feature_is_what_head_sha_classifies_on(dets):
    d = dets("ucf")
    z = d.normalize(torch.rand(2, 3, 256, 256))
    with torch.no_grad():
        logit, feat = d.forward_with_features(z)
        assert feat.shape == (2, 256) and torch.equal(logit, d(z))
        cls = d.model.head_sha.mlp(feat)  # pooled block_sha map -> head MLP == the logit source
        assert torch.allclose(cls[:, 1] - cls[:, 0], logit, atol=1e-5)


def test_ucf_teacher_pass_ignores_film_hooks(dets):
    d = dets("ucf")
    ch = {k: c for k, (_, c) in d.film_sites().items() if k in ("entry", "exit")}
    fe = FrontEnd("dcfg", film=FiLMBank("filmlite", ch, hidden=LUMEN_HIDDEN["ucf"])).eval()
    with torch.no_grad():
        fe.film.fc2.weight.normal_(0, 0.05)
    hooks = FilmHooks(d, fe.film.sites)
    try:
        pipe = tf.Pipeline(fe, d, amp=False, train_detector=False, film_hooks=hooks)
        x = torch.rand(2, 3, 256, 256)
        o = pipe.consistency_forward(x, x)
        with torch.no_grad():
            plain = d(d.normalize(x))
        assert torch.equal(o["t_logit"], plain)  # teacher = plain detector despite non-zero FiLM params
        assert (o["s_logit"] - plain).abs().max() > 1e-4  # student is modulated
    finally:
        hooks.remove()


def test_ucf_hidden_from_choose_hidden_on_inference_path():
    n_inf = 21_861_162 + 262_912 + 132_610  # encoder_f + block_sha + head_sha
    h = choose_hidden(n_inf, [728, 2048])
    assert h == LUMEN_HIDDEN["ucf"]
    assert 3.0 <= 100 * (GATE_SENSOR_PARAMS + film_param_count([728, 2048], h)) / n_inf <= 4.0


def test_train_s_banked_at_mid_epoch_checkpoint(data, tmp_path):  # noqa: F811
    class Crash(Exception):
        pass

    class Crashing(tf.Trainer):
        def after_step(self, step):
            if step == 3:
                raise Crash

    cfg = cfg_for(data, tmp_path, "fixed", ckpt_every_steps=2)
    with pytest.raises(Crash):
        Crashing(cfg).train()
    ck = torch.load(tmp_path / "run_fixed" / "last.pt", weights_only=False)
    assert ck["state"]["step"] == 2 and ck["state"]["train_s"] > 0


def test_frozen_sensor_excluded_from_trainable_mb(data, tmp_path):  # noqa: F811
    live = tf.Trainer(cfg_for(data, tmp_path, "lumen")).trainable_state_dict()
    fz = tf.Trainer(cfg_for(data, tmp_path, "lumen_frozensensor", run_dir=str(tmp_path / "fz"))).trainable_state_dict()
    assert any(k.startswith("frontend.sensor.") for k in live)
    assert not any(k.startswith("frontend.sensor.") for k in fz) and any(k.startswith("frontend.gate.") for k in fz)


# ------------------------------------------------------------------ bnft
def test_bnft_trainable_set_and_modes(data, tmp_path):  # noqa: F811
    t = tf.Trainer(cfg_for(data, tmp_path, "bnft", max_epochs=1))
    bn = t.det.model[1]
    assert t.n_bn_trainable == 8 and t.frontend is None and not t.train_det and t.bn_only
    assert [n for n, p in t.det.named_parameters() if p.requires_grad] == ["model.1.weight", "model.1.bias"]
    assert [g["name"] for g in t.opt.param_groups] == ["bn"] and t.opt.param_groups[0]["initial_lr"] == pytest.approx(2e-5)
    t.model.train()
    assert bn.training and not t.det.model[0].training and not t.det.training
    t.check_frozen()  # BN in train mode allowed for bnft
    t.model.eval()
    assert not bn.training
    t.check_frozen()
    # the same BN-in-train-mode state is REJECTED for a frozen detector
    ft = tf.Trainer(cfg_for(data, tmp_path, "fixed", run_dir=str(tmp_path / "f")))
    ft.det.model[1].train()
    with pytest.raises(AssertionError):
        ft.check_frozen()


def test_bnft_trains_only_bn_and_updates_running_stats(data, tmp_path):  # noqa: F811
    t = tf.Trainer(cfg_for(data, tmp_path, "bnft", max_epochs=1))
    before = {k: v.clone() for k, v in t.det.model.state_dict().items()}
    res = t.train()
    after = t.det.model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.startswith("1.") for k in changed)  # only BN (module 1) tensors
    assert "1.running_mean" in changed and "1.num_batches_tracked" in changed
    assert "1.weight" in changed or "1.bias" in changed
    rd = tmp_path / "run_bnft"
    assert (rd / "bn.pt").exists() and not (rd / "detector.pt").exists() and not (rd / "frontend.pt").exists()
    sd = torch.load(rd / "bn.pt", weights_only=True)
    assert set(sd["state_dict"]) == {"model.1." + k for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")} \
        or set(sd["state_dict"]) == {"1." + k for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")}
    assert res["params"] == 8 and res["trainable_mb"] < res["detector_mb"]


def test_bnft_resume_restores_bn_state(data, tmp_path):  # noqa: F811
    cfg = cfg_for(data, tmp_path, "bnft", max_steps=5, ckpt_every_steps=2)
    t = tf.Trainer(cfg)
    t.train()
    ck = torch.load(tmp_path / "run_bnft" / "last.pt", weights_only=False)
    assert "bn" in ck["model"] and int(ck["model"]["bn"]["1.num_batches_tracked"]) > 0
    t2 = tf.Trainer(cfg_for(data, tmp_path, "bnft", max_steps=5, ckpt_every_steps=2, run_dir=str(tmp_path / "r2")))
    t2._load_model_state(ck["model"])
    assert torch.equal(t2.det.model[1].running_mean.cpu(), ck["model"]["bn"]["1.running_mean"])


def test_eval_bn_ckpt_applies_bn_state_and_logs_fields(data, tmp_path):  # noqa: F811
    t = tf.Trainer(cfg_for(data, tmp_path, "bnft", max_epochs=1))
    t.train()
    rd = tmp_path / "run_bnft"
    w = tmp_path / "w.pt"
    stock = DummyDetector().model.state_dict()
    torch.save({"state_dict": stock}, w)
    out = tmp_path / "res"
    args = ["--run", "t_bnft", "--detector", "dummy", "--bn-ckpt", str(rd / "bn.pt"), "--weights", str(w),
            "--conditions", "clean", "--json", data["json"], "--frames-root", data["frames_root"], "--out-dir", str(out),
            "--runs-dir", str(tmp_path / "runs"), "--workers", "0", "--n-boot", "10"]
    assert ev.main(args) == 0
    r = json.loads((out / "t_bnft__clean.json").read_text())
    assert r["bn_ckpt"] and r["checkpoints"]["bn_sha256"] and r["best_epoch"] is not None
    # BN state differs from the stock one => scores differ from the plain detector's
    assert not torch.equal(torch.load(rd / "bn.pt", weights_only=True)["state_dict"]["1.running_mean"], stock["1.running_mean"])
    with pytest.raises(SystemExit):
        ev.main(args + ["--detector-ckpt", str(w), "--force"])


# ------------------------------------------------------------------ logging fields
LOG_KEYS = ("best_epoch", "early_stop_epoch", "train_wall_time_s", "trainable_mb", "detector_mb", "params")


def test_result_json_and_done_json_have_training_fields(data, tmp_path):  # noqa: F811
    t = tf.Trainer(cfg_for(data, tmp_path, "lumen", max_epochs=2))
    res = t.train()
    rd = tmp_path / "run_lumen"
    done = json.loads((rd / "done.json").read_text())
    for k in LOG_KEYS:
        assert k in done
    assert done["early_stop_epoch"] is None and done["train_wall_time_s"] > 0
    assert 0 < done["trainable_mb"] and 0 < done["detector_mb"] and done["params"] > 0
    w = tmp_path / "w.pt"
    torch.save({"state_dict": DummyDetector().model.state_dict()}, w)
    out = tmp_path / "res"
    assert ev.main(["--run", "t_lumen", "--detector", "dummy", "--frontend", str(rd / "frontend.pt"), "--weights", str(w),
                    "--conditions", "clean", "--json", data["json"], "--frames-root", data["frames_root"],
                    "--out-dir", str(out), "--runs-dir", str(tmp_path / "runs"), "--workers", "0", "--n-boot", "10"]) == 0
    r = json.loads((out / "t_lumen__clean.json").read_text())
    for k in LOG_KEYS:
        assert r[k] == done[k] and r["train"][k] == done[k]


def test_early_stop_epoch_recorded(data, tmp_path):  # noqa: F811
    t = tf.Trainer(cfg_for(data, tmp_path, "fixed", max_epochs=5, patience=1))
    t.val_mean = iter([0.9, 0.8, 0.7, 0.6, 0.5])
    orig = t.validate

    def fake() -> dict:
        v = orig()
        v["mean_auc"] = next(t.val_mean)
        return v

    t.validate = fake  # type: ignore[method-assign]
    res = t.train()
    assert res["early_stop_epoch"] == 1 and res["best_epoch"] == 0 and res["epochs_run"] == 2


def test_eval_fields_null_without_training(data, tmp_path):  # noqa: F811
    w = tmp_path / "w.pt"
    torch.save({"state_dict": DummyDetector().model.state_dict()}, w)
    out = tmp_path / "res"
    assert ev.main(["--run", "t_base", "--detector", "dummy", "--weights", str(w), "--conditions", "clean",
                    "--json", data["json"], "--frames-root", data["frames_root"], "--out-dir", str(out),
                    "--runs-dir", str(tmp_path / "runs"), "--workers", "0", "--n-boot", "10"]) == 0
    r = json.loads((out / "t_base__clean.json").read_text())
    assert all(k in r and r[k] is None for k in LOG_KEYS)


def test_state_dict_mb_counts_fp32_bytes():
    sd = {"a": torch.zeros(1024, 256, dtype=torch.float16), "n": torch.zeros((), dtype=torch.long)}
    mb = tf.state_dict_mb(sd)
    assert 1.0 <= mb < 1.01  # 262144 fp32 values = 1 MiB (+ pickle overhead)


def test_train_log_has_scaler_columns(data, tmp_path):  # noqa: F811
    tf.Trainer(cfg_for(data, tmp_path, "lumen", max_epochs=1, log_every=2)).train()
    hdr = (tmp_path / "run_lumen" / "train_log.csv").read_text().splitlines()[0].split(",")
    assert {"scaler_scale", "skipped_steps"} <= set(hdr)


# ------------------------------------------------------------------ gen_matrix
def test_matrix_exact_runs_groups_and_order():
    runs = gm.matrix_runs()
    ids = [r["run_id"] for r in runs]
    g1 = [f"{v}_{d}" for d in ("xception", "efficientnetb4", "f3net", "ucf") for v in ("lumen", "fixed", "augft")]
    assert ids == g1 + ["jointft_xception", "bnft_xception", "dcfg_xception", "dcfgcons_xception", "lumen_nocons_xception",
                        "lumen_nolive_xception", "lumen_frozensensor_xception", "lumen_xception_s43", "lumen_xception_s44",
                        "augft_xception_s43", "augft_xception_s44"]
    assert [r["group"] for r in runs] == [1] * 12 + [2] * 2 + [3] * 5 + [4] * 4
    by = {r["run_id"]: r for r in runs}
    assert by["jointft_xception"]["variant"] == "dcfg_jointft" and by["dcfgcons_xception"]["variant"] == "dcfg_cons"
    assert by["lumen_xception_s44"]["seed"] == 44 and by["augft_xception_s43"]["seed"] == 43
    assert not any(r["run_id"].startswith("base_") for r in runs)


def test_queue_lines_adjacency_and_findings(tmp_path, capsys):
    q = tmp_path / "q.txt"
    assert gm.main(["--out-dir", str(tmp_path / "cfg"), "--batch-yaml", str(tmp_path / "none.yaml"), "--queue-out", str(q)]) == 0
    lines = q.read_text().splitlines()
    assert lines == capsys.readouterr().out.strip().splitlines()
    ids = [ln.split(" | ")[0] for ln in lines]
    assert len(ids) == len(set(ids)) == 2 * 23 + 4
    for r in gm.matrix_runs():
        i = ids.index(r["run_id"])
        assert ids[i + 1] == f"eval_{r['run_id']}"
        ev_line = lines[i + 1]
        assert "--conditions all --split test" in ev_line and f"--run {r['run_id']} " in ev_line
    last_in_group = {1: "eval_augft_ucf", 2: "eval_bnft_xception", 3: "eval_lumen_frozensensor_xception",
                     4: "eval_augft_xception_s44"}
    for g, last in last_in_group.items():
        i = ids.index(last)
        assert lines[i + 1] == f"findings_group{g} | .venv/bin/python ops/jobs/group_findings.py --group {g} #noheartbeat"
    by = {ln.split(" | ")[0]: ln for ln in lines}
    assert "--bn-ckpt runs/bnft_xception/bn.pt" in by["eval_bnft_xception"] and "--frontend" not in by["eval_bnft_xception"]
    assert "--frontend runs/lumen_xception/frontend.pt" in by["eval_lumen_xception"] and "--detector-ckpt" not in by["eval_lumen_xception"]
    j = by["eval_jointft_xception"]
    assert "--frontend runs/jointft_xception/frontend.pt" in j and "--detector-ckpt runs/jointft_xception/detector.pt" in j
    assert "--detector-ckpt runs/augft_ucf/detector.pt" in by["eval_augft_ucf"] and "--frontend" not in by["eval_augft_ucf"]
    assert all(ln.endswith("#hb=30") for ln in lines if not ln.startswith("findings_"))


def test_configs_real_settings(tmp_path):
    from dcfg.paths import REPO_ROOT
    (tmp_path / "base.yaml").write_text((REPO_ROOT / "configs/train/base.yaml").read_text())
    (tmp_path / "bf.yaml").write_text("xception: 32\nefficientnetb4: 24\nf3net: 32\nucf: 16\n")
    gm.main(["--out-dir", str(tmp_path), "--batch-yaml", str(tmp_path / "bf.yaml"), "--no-write"])
    assert not list(tmp_path.glob("lumen_*.yaml"))
    gm.main(["--out-dir", str(tmp_path), "--batch-yaml", str(tmp_path / "bf.yaml")])
    for r in gm.matrix_runs():
        c = tf.load_config(tmp_path / f"{r['run_id']}.yaml")
        assert c["pilot"] is False and c["timing_sync"] is False and c["stop_after_epochs"] is None
        assert c["max_epochs"] == 30 and c["patience"] == 6 and c["epoch_mode"] == "full"
        assert c["seed"] == r["seed"] and c["variant"] == r["variant"] and c["detector"] == r["detector"]
        assert c["batch_size"] == {"xception": 32, "efficientnetb4": 24, "f3net": 32, "ucf": 16}[r["detector"]]
    assert tf.load_config(tmp_path / "lumen_frozensensor_xception.yaml")["freeze_sensor"] is True


def test_prune_only_removes_generated_stale_configs(tmp_path):
    (tmp_path / "old_gen.yaml").write_text("# generated by ops/jobs/gen_matrix.py (PLAN 7.4); do not edit by hand\nrun_id: old_gen\n")
    (tmp_path / "hand.yaml").write_text("# my own\nrun_id: hand\n")
    (tmp_path / "base.yaml").write_text("# Base training config\n")
    gm.main(["--out-dir", str(tmp_path), "--batch-yaml", str(tmp_path / "x.yaml"), "--prune"])
    assert not (tmp_path / "old_gen.yaml").exists() and (tmp_path / "hand.yaml").exists() and (tmp_path / "base.yaml").exists()
    assert (tmp_path / "lumen_xception.yaml").exists()


# ------------------------------------------------------------------ ETA
def test_eta_scaling_and_totals():
    import lumen_eta as le
    r = {"lumen": {"xception": 100.0, "f3net": 50.0, "efficientnetb4": 80.0}, "lumen_nocons": {"xception": 125.0}, "bnft": {"xception": 200.0},
         "fixed": {"xception": 300.0, "ucf": 150.0, "efficientnetb4": 150.0, "f3net": 250.0}, "augft": {"xception": 200.0, "ucf": 100.0, "efficientnetb4": 100.0, "f3net": 200.0},
         "dcfg": {"xception": 200.0, "ucf": 100.0}}
    by = {x["run_id"]: x for x in gm.matrix_runs()}
    f = lambda rid: le.epoch_minutes(by[rid], r)  # noqa: E731
    assert f("lumen_xception")[0] == pytest.approx(22.2)
    assert f("lumen_f3net")[0] == pytest.approx(44.4)
    assert f("lumen_ucf")[0] == pytest.approx(44.4) and f("lumen_ucf")[1].startswith("ASSUMPTION")
    assert f("lumen_nocons_xception")[0] == pytest.approx(22.2 * 100 / 125)
    assert f("bnft_xception")[0] == pytest.approx(17.8 * 200 / 200)
    assert f("fixed_ucf")[0] == pytest.approx(28.0) and f("augft_efficientnetb4")[0] == pytest.approx(35.6)
    assert f("jointft_xception")[0] == 22.9 and f("dcfgcons_xception")[0] == 22.1 and f("lumen_xception_s43")[0] == 22.2
    res = le.build(r, {d: 30.0 for d in gm.DETECTORS})
    assert [g["n"] for g in res["groups"].values()] == [12, 2, 5, 4]
    row = {x["run_id"]: x for x in res["rows"]}["lumen_xception"]
    assert row["worst_min"] == pytest.approx(30 * 22.2 + 30) and row["likely_min"] == pytest.approx(15 * 22.2 + 30)
    g = res["groups"]
    assert g[4]["cum_worst"] == pytest.approx(res["worst"]) and g[2]["cum_likely"] == pytest.approx(g[1]["likely"] + g[2]["likely"])
    assert "GPU-days" in le.markdown(res, ["n"]) and "ASSUMPTION" in le.markdown(res, ["n"])
