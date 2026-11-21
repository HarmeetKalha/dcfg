"""Damage sensor model/loss/eval analysis, heatmap helpers, job summary robustness."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

from dcfg.analysis.gate_heatmaps import band_mask
from dcfg.data.degradations import EvalCondition
from dcfg.degrade.eval_damage_sensor import analyse, has_labels
from dcfg.degrade.sensor import DamageSensor, damage_loss


def test_shapes_and_param_budget():
    s = DamageSensor()
    assert s.n_params() < 1_000_000
    o = s(torch.rand(2, 3, 256, 256))
    assert o["embedding"].shape == (2, 128) and o["presence_logit"].shape == (2, 4)
    assert o["strength"].shape == (2, 4) and 0 <= float(o["strength"].min()) and float(o["strength"].max()) <= 1


def test_loss_ignores_unapplied_strength():
    out = {"presence_logit": torch.zeros(2, 4), "strength": torch.tensor([[0.2, 0.9, 0.5, 0.5]] * 2)}
    applied = torch.tensor([[1.0, 0, 0, 0]] * 2)
    l1, p1 = damage_loss(out, applied, torch.tensor([[0.2, 0, 0, 0]] * 2))
    l2, p2 = damage_loss(out, applied, torch.tensor([[0.2, 1.0, 1.0, 1.0]] * 2))  # unapplied targets differ
    assert torch.isclose(l1, l2) and float(p1["mse"]) == 0.0
    l3, p3 = damage_loss(out, torch.zeros(2, 4), torch.zeros(2, 4))  # nothing applied: finite
    assert torch.isfinite(l3)


def test_analyse_perfect_predictions():
    conds = {"clean": EvalCondition("clean", "clean"), "jpeg_q50": EvalCondition("jpeg_q50", "jpeg", (("jpeg", 50),)),
             "webp_q50": EvalCondition("webp_q50", "held_out", (("webp", 50),))}
    assert has_labels(conds["jpeg_q50"]) and not has_labels(conds["webp_q50"])
    n = 10
    preds = {"clean": (np.zeros((n, 4)), np.zeros((n, 4))),
             "jpeg_q50": (np.tile([1.0, 0, 0, 0], (n, 1)), np.tile([0.6, 0, 0, 0], (n, 1)) + np.random.default_rng(0).normal(0, 0.01, (n, 4))),
             "webp_q50": (np.full((n, 4), 0.3), np.full((n, 4), 0.2))}
    r = analyse(preds, conds)
    assert r["per_op"]["jpeg"]["presence_acc"] == 1.0 and r["per_op"]["jpeg"]["presence_f1"] == 1.0
    assert r["confusion"]["jpeg_q50"]["predicted"] == {"jpeg": 1.0} and "webp_q50" in r["unseen"]


def test_band_masks_partition_the_grid():
    m = sum(band_mask(b).astype(int) for b in ("low", "mid", "high"))
    assert (m == 1).all()


def test_job_summary_robust_for_unknown_and_missing():
    out = subprocess.run([sys.executable, "ops/jobs/job_summary.py", "nonexistent_job"], capture_output=True, text=True)
    assert out.returncode == 0 and "nonexistent_job" in out.stdout
