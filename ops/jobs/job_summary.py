#!/usr/bin/env python
"""One-line result summary of a finished meeting-queue job: python ops/jobs/job_summary.py <job_id>

Robust: unknown job or missing files give a generic line, never an exception. Importable (``summarize``)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results" / "raw"


def _j(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def raw_aggregates(run: str) -> dict[str, float] | None:
    """clean / synthetic mean / held-out mean / c40 / heavy mean frame AUC of a results/raw run (None if incomplete)."""
    import yaml
    conds = yaml.safe_load((REPO / "configs/eval/conditions.yaml").read_text())["conditions"]
    auc = {}
    for c in conds:
        d = _j(RAW / f"{run}__{c['name']}.json")
        if d is None:
            return None
        auc[c["name"]] = d["frame_auc"]
    synth = [c["name"] for c in conds if c["name"] != "clean" and not c.get("held_out") and c.get("synthetic", True)]
    held = [c["name"] for c in conds if c.get("held_out")]
    heavy = ["jpeg_q10", "blur_s4", "noise_s20", "resize_0.25", "mixed_heavy"]
    return {"clean": auc["clean"], "synth": float(np.mean([auc[c] for c in synth])), "heldout": float(np.mean([auc[c] for c in held])),
            "c40": auc["c40"], "heavy": float(np.mean([auc[c] for c in heavy]))}


def _gate_heatmaps() -> str:
    npz = REPO / "results/gate/lumen1/gate_means.npz"
    with np.load(npz) as z:
        m = lambda c: float(z[f"{c}__all__sum"].sum() / z[f"{c}__all__n"] / 192)  # noqa: E731
        return (f"gate heatmaps done: mean w clean {m('clean'):.3f}, noise_s20 {m('noise_s20'):.3f}, jpeg_q10 {m('jpeg_q10'):.3f}, "
                f"blur_s4 {m('blur_s4'):.3f}, c40 {m('c40'):.3f}")


def _train(run_dir: str, what: str) -> str:
    d = _j(REPO / "runs" / run_dir / "done.json") or {}
    if "stop_reason" in d:  # sensor
        return (f"{what}: {d['epochs_run']} mini-epochs, best {d['best_epoch']}, val loss {d['best_val_loss']:.4f}, "
                f"{d['train_s'] / 3600:.2f} h train, {d['params']} params")
    return (f"{what}: {d.get('epochs_run')} epochs, best epoch {d.get('best_epoch')}, val mean AUC {d.get('val_mean_auc', float('nan')):.4f}, "
            f"early stop={d.get('stopped_early')}, {d.get('total_time_s', 0) / 3600:.1f} h")


def _sensor_eval() -> str:
    d = _j(REPO / "results/sensor/standalone.json")
    if not d:
        return "sensor eval finished (no json)"
    po = d["per_op"]
    return "sensor eval: presence F1 " + " ".join(f"{o} {r['presence_f1']:.2f}" for o, r in po.items()) + "; strength Spearman " + \
           " ".join(f"{o} {r['strength_spearman'] if r['strength_spearman'] is None else round(r['strength_spearman'], 2)}" for o, r in po.items())


def _nogate_eval() -> str:
    a = raw_aggregates("lumen_nogate_xception")
    if a is None:
        return "nogate eval finished (incomplete results/raw)"
    b = raw_aggregates("lumen_xception") or {}
    return ("LUMEN1-no-gate frame AUC: " + ", ".join(f"{k} {v:.3f} (LUMEN1 {b.get(k, float('nan')):.3f})" for k, v in a.items()))


def _ablation() -> str:
    d = _j(REPO / "results/gate/ablation_xception.json")
    if not d:
        return "ablation report written (no json)"
    g = d["diffs"].get("frame/lumen_xception-lumen_nogate_xception", {})
    s = g.get("synth_mean")
    h = g.get("heavy_mean")
    f = lambda v: "n/a" if v is None else f"{v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]"  # noqa: E731
    return f"gate ablation: LUMEN1 - no-gate frame AUC synth {f(s)}, heavy {f(h)}"


def summarize(job: str) -> str:
    """One line for ``job`` (the id without the ``after_`` prefix)."""
    try:
        if job.startswith("gate_heatmaps"):
            return _gate_heatmaps()
        if job == "sensor_train":
            return _train("damage_sensor", "damage sensor trained")
        if job == "sensor_eval":
            return _sensor_eval()
        if job == "lumen_nogate_xception":
            return _train("lumen_nogate_xception", "LUMEN1-no-gate trained")
        if job == "eval_lumen_nogate_xception":
            return _nogate_eval()
        if job == "gate_ablation":
            return _ablation()
    except Exception as e:  # noqa: BLE001
        return f"{job} done (summary failed: {type(e).__name__})"
    return f"{job} done"


if __name__ == "__main__":
    print(summarize(sys.argv[1] if len(sys.argv) > 1 else ""))
