"""Evaluate the standalone damage sensor on the FF++ c23/c40 TEST split under all 23 conditions.

    python -m dcfg.degrade.eval_damage_sensor --ckpt runs/damage_sensor/best.pt --out results/sensor/standalone

Ground truth (``dcfg.degrade.labels``): clean -> no op; jpeg/blur/noise/resize conditions -> that op at its
parameter; mixed_* -> blur+noise+jpeg. Conditions with ops the sensor was never trained on (webp, motion) or with real
compression (c40) have no label: they are reported as mean predicted presence probability and strength ("unseen").
Metrics over the 18 labelled conditions pooled (clean + 17 synthetic): presence accuracy / F1 per op (threshold 0.5), strength
MAE / Pearson / Spearman per op over frames where the op is applied, a per-condition table, and a confusion table
(true op set -> distribution of predicted op sets) for every labelled condition. Writes ``standalone.md`` + ``standalone.json``.
Test split is used for reporting only (no selection happens here).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from dcfg.data.degradations import EvalCondition, load_conditions
from dcfg.data.ffpp import DFBJsonSource
from dcfg.degrade.labels import OPS, ops_to_labels
from dcfg.degrade.sensor import DamageSensor
from dcfg.eval.evaluate import CONDITIONS_YAML, ConditionDataset, condition_comp
from dcfg.paths import REPO_ROOT, RUNS_ROOT


def has_labels(c: EvalCondition) -> bool:
    """True for clean and for conditions made only of ops the sensor was trained on."""
    return c.synthetic and all(op in OPS for op, _ in c.ops)


@torch.no_grad()
def predict(model: DamageSensor, source: DFBJsonSource, cond: EvalCondition, dev: str, workers: int, max_frames: int = 0
            ) -> tuple[np.ndarray, np.ndarray]:
    """(presence prob (N,4), strength (N,4)) over the test frames of ``cond``."""
    recs = source.records("test", condition_comp(cond))
    if max_frames:
        recs = recs[:: max(1, len(recs) // max_frames)]
    dl = DataLoader(ConditionDataset(source, recs, cond), batch_size=128, shuffle=False, num_workers=workers)
    p, s = [], []
    for xb in dl:
        out = model(xb.to(dev).float().div_(255.0))
        p.append(torch.sigmoid(out["presence_logit"]).cpu().numpy()); s.append(out["strength"].cpu().numpy())
    return np.concatenate(p), np.concatenate(s)


def _set_name(bits: np.ndarray) -> str:
    return "+".join(o for o, b in zip(OPS, bits) if b) or "none"


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(1, 2 * tp + fp + fn)


def analyse(preds: dict[str, tuple[np.ndarray, np.ndarray]], conds: dict[str, EvalCondition]) -> dict:
    lab = [c for c in preds if has_labels(conds[c])]
    P = np.concatenate([preds[c][0] for c in lab]); S = np.concatenate([preds[c][1] for c in lab])
    A = np.concatenate([np.tile(ops_to_labels(conds[c].ops)[0], (len(preds[c][0]), 1)) for c in lab])
    T = np.concatenate([np.tile(ops_to_labels(conds[c].ops)[1], (len(preds[c][0]), 1)) for c in lab])
    per_op = {}
    for i, op in enumerate(OPS):
        pred, tru = P[:, i] > 0.5, A[:, i] > 0.5
        tp, fp, fn = int((pred & tru).sum()), int((pred & ~tru).sum()), int((~pred & tru).sum())
        m = tru
        r = {"presence_acc": float((pred == tru).mean()), "presence_f1": f1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
             "n_applied": int(m.sum()), "strength_mae": float(np.abs(S[m, i] - T[m, i]).mean())}
        r["strength_pearson"] = float(pearsonr(S[m, i], T[m, i])[0]) if T[m, i].std() > 0 else None
        r["strength_spearman"] = float(spearmanr(S[m, i], T[m, i])[0]) if T[m, i].std() > 0 else None
        per_op[op] = r
    per_cond, confusion = {}, {}
    for c in lab:
        p, s = preds[c]
        a, t = ops_to_labels(conds[c].ops)
        pb = p > 0.5
        per_cond[c] = {"true_ops": _set_name(a > 0.5), "true_strength": {o: float(t[i]) for i, o in enumerate(OPS) if a[i]},
                       "mean_presence": p.mean(0).tolist(), "mean_strength_when_present":
                       {o: (float(s[pb[:, i], i].mean()) if pb[:, i].any() else None) for i, o in enumerate(OPS)},
                       "mean_strength_all": s.mean(0).tolist(), "exact_set_acc": float((pb == (a > 0.5)).all(1).mean())}
        names = [_set_name(b) for b in pb]
        u, n = np.unique(names, return_counts=True)
        confusion[c] = {"true": _set_name(a > 0.5), "predicted": {k: float(v / len(names)) for k, v in sorted(zip(u, n), key=lambda z: -z[1])[:5]}}
    unseen = {c: {"mean_presence": dict(zip(OPS, preds[c][0].mean(0).tolist())),
                  "mean_strength_all": dict(zip(OPS, preds[c][1].mean(0).tolist())),
                  "frac_any_present": float((preds[c][0] > 0.5).any(1).mean())}
              for c in preds if not has_labels(conds[c])}
    return {"labelled_conditions": lab, "per_op": per_op, "per_condition": per_cond, "confusion": confusion, "unseen": unseen,
            "n_labelled_frames": int(len(P))}


def write_md(res: dict, path: Path, meta: dict) -> None:
    L = ["# Standalone damage sensor: test-split evaluation", "",
         f"Checkpoint `{meta['ckpt']}` (best val loss epoch {meta.get('epoch')}, {meta.get('params')} parameters; val loss "
         f"{meta.get('val', {}).get('loss', float('nan')):.4f}). 128x128 native centre crop + 192 DCT log-energy statistics. "
         "Test split, 23 conditions, same degradation code and per-frame seeds as `results/raw`. Test only: nothing was selected on it.",
         "", f"Labelled conditions pooled: {len(res['labelled_conditions'])} (clean + 17 synthetic), {res['n_labelled_frames']} frames. "
         "Presence threshold 0.5. Strength is scaled 0..1 over the training range (jpeg q95->0, q20->1; blur 0.5->4; noise 2->20; "
         "resize 0.9->0.25), clipped; strength metrics only over frames where the op is truly applied.", "",
         "## Per operation", "", "| op | presence acc | presence F1 | strength MAE | Pearson | Spearman | n applied | TP | FP | FN |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for op, r in res["per_op"].items():
        L.append(f"| {op} | {r['presence_acc']:.3f} | {r['presence_f1']:.3f} | {r['strength_mae']:.3f} | {fmt(r['strength_pearson'])} | "
                 f"{fmt(r['strength_spearman'])} | {r['n_applied']} | {r['tp']} | {r['fp']} | {r['fn']} |")
    L += ["", "## Per condition (labelled)", "",
          "| condition | true ops | true strength | mean P(jpeg/blur/noise/resize) | mean strength where present | exact op-set acc |",
          "|---|---|---|---|---|---|"]
    for c, r in res["per_condition"].items():
        ts = ", ".join(f"{o} {v:.2f}" for o, v in r["true_strength"].items()) or "-"
        ms = ", ".join(f"{o} {v:.2f}" for o, v in r["mean_strength_when_present"].items() if v is not None) or "-"
        L.append(f"| {c} | {r['true_ops']} | {ts} | {' / '.join(f'{x:.2f}' for x in r['mean_presence'])} | {ms} | {r['exact_set_acc']:.3f} |")
    L += ["", "## Confusion: true op set -> predicted op set (top 5, fraction of frames)", "", "| condition | true | predicted |", "|---|---|---|"]
    for c, r in res["confusion"].items():
        L.append(f"| {c} | {r['true']} | " + "; ".join(f"{k} {v:.2f}" for k, v in r["predicted"].items()) + " |")
    L += ["", "## Unseen conditions (no ground truth: c40 real compression, WebP, motion blur)", "",
          "| condition | frac any op present | P(jpeg) | P(blur) | P(noise) | P(resize) | str jpeg | str blur | str noise | str resize |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in res["unseen"].items():
        L.append(f"| {c} | {r['frac_any_present']:.2f} | " + " | ".join(f"{r['mean_presence'][o]:.2f}" for o in OPS) + " | "
                 + " | ".join(f"{r['mean_strength_all'][o]:.2f}" for o in OPS) + " |")
    path.write_text("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.degrade.eval_damage_sensor")
    ap.add_argument("--ckpt", default=str(RUNS_ROOT / "damage_sensor" / "best.pt"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results/sensor/standalone"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0, help="smoke only")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = DamageSensor().to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    conds = load_conditions(CONDITIONS_YAML)
    source = DFBJsonSource()
    preds = {}
    for name, c in conds.items():
        preds[name] = predict(model, source, c, dev, a.workers, a.max_frames)
        print(f"{name}: {len(preds[name][0])} frames", flush=True)
    res = analyse(preds, conds)
    meta = {**ck["meta"], "ckpt": str(a.ckpt)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps({"meta": meta, **res}, indent=2, default=float) + "\n")
    write_md(res, Path(str(out) + ".md"), meta)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
