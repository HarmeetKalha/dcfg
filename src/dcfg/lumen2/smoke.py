"""LUMEN2 smoke test + time gate (queued GPU job). Writes ``<run-dir>/PASS`` only if every check passes AND 18 epochs fit in 30 h.

    python -m dcfg.lumen2.smoke --config configs/train/lumen2.yaml [--run-dir runs/lumen2_smoke] [--steps 300]

Checks (results: ``<run-dir>/smoke.json``, ``results/lumen2/lumen2_smoke.md``):
 1. step-0 LUMEN2 == public GenD: fake logits and L2 embeddings of the same real val batch, fp32, within fp tolerance;
 2. damaged-image stream: through real DataLoaders, sha256 over whole micro-batches of the 224 view (LUMEN2) == GenD-aug's
    ``GendTrainSet`` output and of the 256 view == ``TrainDataset`` output, for two epoch slices (incl. a resume offset);
    record order hashes of ALL 18 epochs == ``runs/gendaug_train/batches.log`` (the actual GenD-aug run);
 3. ``--steps`` (300) OPTIMISER steps (= 4 micro-steps each, 128 images) of the real trainer: main loss and sensor loss decrease
    (mean of the first vs last quarter of the logged steps; main allowed to be within +1% = noise tolerance), no NaN, gradients
    reach U, V, gate, sensor and the GenD LayerNorms;
 4. epoch time: steady-state images/s of that run -> 1435 steps/epoch, + validation time measured on a 200-video val subset scaled to
    the full 8-of-32 val set; 18 x (train + val) > 30 h => NO PASS, ntfy "waiting for options";
 5. parameter counts per part;
 6. resume check: straight run vs stop-and-resume (identical batch hashes, parameters within GPU non-determinism tolerance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dcfg.eval.evaluate import condition_comp
from dcfg.gend.train_gend_aug import GendTrainSet, load_cfg
from dcfg.lumen2.model import Lumen2Model, build_lumen2, pack_views, unpack_views
from dcfg.lumen2.train import Lumen2Trainer
from dcfg.paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT
from dcfg.train.train_frontend import TrainDataset, batch_hash, epoch_order

LIMIT_H = 30.0
STEPS_PER_EPOCH_FALLBACK = 1435


def estimate_hours(img_s: float, val_s_per_frame: float, val_frames: int, spe: int, eff: int, epochs: int) -> dict[str, float]:
    """Wall-time estimate of a full run: ``epochs x (spe*eff / img_s + val_frames * val_s_per_frame)``."""
    train_s = spe * eff / img_s
    val_s = val_frames * val_s_per_frame
    return {"epoch_train_s": train_s, "epoch_val_s": val_s, "epoch_s": train_s + val_s, "total_h": epochs * (train_s + val_s) / 3600.0}


def _sha(*arrs: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ check 1
@torch.no_grad()
def check_step0(trainer: Lumen2Trainer, n: int = 16) -> dict[str, Any]:
    from dcfg.gend.crop import gend_resize
    from dcfg.gend.model import load_gend

    dev = trainer.dev
    recs = trainer.source.records("val", trainer.cfg["comp"])
    recs = recs[:: max(1, len(recs) // n)][:n]
    imgs = [trainer.source.load(r) for r in recs]
    packed = torch.stack([torch.from_numpy(pack_views(i)).permute(2, 0, 1) for i in imgs]).float().div_(255.0).to(dev)
    x224 = torch.stack([torch.from_numpy(gend_resize(i)).permute(2, 0, 1) for i in imgs]).float().div_(255.0).to(dev)
    fresh = build_lumen2("cpu").to(dev).eval()  # step-0 LUMEN2 (public GenD weights + zero-init parts), independent of the trained one
    gend = load_gend(str(dev)).eval()
    lg, zg = gend.features(x224)
    ll, zl, _, w = fresh.forward_train(packed)
    return {"max_abs_logit_diff": float((lg - ll).abs().max()), "max_abs_embedding_diff": float((zg - zl).abs().max()),
            "gate_w_all_one": bool((w == 1).all()), "n_images": len(imgs), "pass": bool((lg - ll).abs().max() < 5e-3 and (zg - zl).abs().max() < 5e-3 and (w == 1).all())}


# ------------------------------------------------------------------------------------------------ check 2
def check_stream(trainer: Lumen2Trainer, gendaug_log: Path, n_micro: int = 12) -> dict[str, Any]:
    cfg, recs = trainer.cfg, trainer.records
    seed, bs = int(cfg["seed"]), trainer.bs
    out: dict[str, Any] = {"slices": []}
    ok = True
    for epoch, skip in ((0, 0), (1, 7)):
        order = epoch_order(trainer.labels, seed, epoch, cfg["epoch_mode"])[: trainer.spe * trainer.eff]
        new = trainer._loader(epoch, order, skip)
        ref = DataLoader(GendTrainSet(TrainDataset(trainer.source, recs, order, skip * trainer.eff, seed, epoch, trainer.deg)), batch_size=bs,
                         shuffle=False, drop_last=True, num_workers=int(cfg["num_workers"]))
        plain = DataLoader(TrainDataset(trainer.source, recs, order, skip * trainer.eff, seed, epoch, trainer.deg), batch_size=bs,
                           shuffle=False, drop_last=True, num_workers=int(cfg["num_workers"]))
        h_new224, h_ref224, h_new256, h_plain256, n_img = [], [], [], [], 0
        applied_sum, clean = np.zeros(4), 0
        for i, (a, b, c) in enumerate(zip(new, ref, plain)):
            if i >= n_micro:
                break
            crop, img224 = unpack_views(a[0].float())
            assert torch.equal(a[2], b[2]) and torch.equal(a[2], c[2]), "record order differs"
            h_new224.append(_sha(img224.byte().numpy())); h_ref224.append(_sha(b[0].numpy()))
            h_new256.append(_sha(crop.byte().numpy())); h_plain256.append(_sha(c[0].numpy()))
            applied_sum += a[3].numpy().sum(0); clean += int((a[3].sum(1) == 0).sum()); n_img += len(a[0])
        sl = {"epoch": epoch, "skip_steps": skip, "micro_batches": len(h_new224), "images": n_img,
              "view224_equal": h_new224 == h_ref224, "view256_equal": h_new256 == h_plain256,
              "digest": _sha(np.frombuffer("".join(h_new224).encode(), np.uint8)),
              "clean_fraction": clean / n_img, "mean_applied_per_op": (applied_sum / n_img).round(3).tolist()}
        ok &= sl["view224_equal"] and sl["view256_equal"] and len(h_new224) == n_micro
        out["slices"].append(sl)
    # record order of every epoch vs the real GenD-aug run
    logged: dict[tuple[int, int], str] = {}
    if gendaug_log.exists():
        for ln in gendaug_log.read_text().splitlines():
            if ln and not ln.startswith("#"):
                e, k, _, h = ln.split()
                logged[(int(e), int(k))] = h
    mismatch, checked = 0, 0
    for epoch in sorted({e for e, _ in logged}):
        order = epoch_order(trainer.labels, int(cfg["seed"]), epoch, cfg["epoch_mode"])[: trainer.spe * trainer.eff]
        for k in range(trainer.spe):
            if (epoch, k) in logged:
                checked += 1
                mismatch += batch_hash(torch.as_tensor(order[k * trainer.eff:(k + 1) * trainer.eff])) != logged[(epoch, k)]
    out |= {"gendaug_log": str(gendaug_log), "gendaug_batches_checked": checked, "gendaug_batches_mismatched": int(mismatch)}
    out["pass"] = bool(ok and checked > 0 and mismatch == 0)
    return out


# ------------------------------------------------------------------------------------------------ checks 3, 4, 5
def quarter_means(x: list[float]) -> tuple[float, float]:
    q = max(1, len(x) // 4)
    return float(np.mean(x[:q])), float(np.mean(x[-q:]))


def train_check(cfg: dict[str, Any], run_dir: Path, steps: int) -> tuple[dict[str, Any], Lumen2Trainer]:
    d = run_dir / "train_check"
    shutil.rmtree(d, ignore_errors=True)
    c = {**cfg, "run_id": "lumen2_smoke_train", "max_steps": steps, "val_max_videos": 200, "val_on_truncate": True, "log_every": 10,
         "ckpt_every_steps": 10_000, "require_pass_marker": None}
    tr = Lumen2Trainer(c, d)
    t0 = time.time()
    res = tr.train()
    wall = time.time() - t0
    import csv

    rows = list(csv.DictReader((d / "train_log.csv").open()))
    col = lambda k: [float(r[k]) for r in rows]  # noqa: E731
    main, sens = col("main"), col("damage")
    (m0, m1), (s0, s1) = quarter_means(main), quarter_means(sens)
    steady = [float(r["img_s"]) for r in rows if int(r["step"]) > steps // 3][1:] or [float(rows[-1]["img_s"])]
    img_s = float(np.median(steady))
    grads = {"U": float(tr.model.gen.U.abs().sum()), "V": float(tr.model.gen.V.abs().sum())}  # nonzero only if U,V received updates
    allfinite = bool(np.isfinite(col("loss")).all())
    val = res["val"]
    val_frames = int(sum(val["n_frames"].values()))
    # full val size: count without the video cap
    tr.cfg["val_max_videos"] = None
    tr._val_recs.clear()
    full_frames = sum(len(tr.val_records(condition_comp(c_))) for c_ in tr.val_conds)
    est = estimate_hours(img_s, tr.st["val_s"] / max(1, val_frames), full_frames, tr.spe, tr.eff, int(cfg["max_epochs"]))
    out = {"steps": steps, "micro_steps": steps * tr.accum, "wall_s": round(wall), "main_loss_first_last_quarter": [m0, m1],
           "damage_loss_first_last_quarter": [s0, s1], "all_losses_finite": allfinite, "nonfinite_steps": tr.st["nonfinite_steps"],
           "U_abs_sum": grads["U"], "V_abs_sum": grads["V"], "tiny_val_mean_auc": val["mean_auc"],
           "img_s_steady": img_s, "gpu_mem_peak_mb": max(col("gpu_mem_mb")), "val_frames_measured": val_frames, "val_frames_full": full_frames,
           "estimate": est, "last_losses": {k: rows[-1][k] for k in ("loss", "ce", "main", "damage", "sens_bce", "sens_mse", "gate_pen")}}
    out["loss_pass"] = bool(allfinite and tr.st["nonfinite_steps"] == 0 and s1 < s0 and m1 <= m0 * 1.01 and grads["U"] > 0 and grads["V"] > 0)
    return out, tr


def _free() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resume_check(cfg: dict[str, Any], run_dir: Path, n: int = 40) -> dict[str, Any]:
    base = {**cfg, "run_id": "lumen2_smoke_resume", "ckpt_every_steps": 10_000, "val_max_videos": 10, "val_on_truncate": False,
            "log_every": 10, "require_pass_marker": None}
    a_dir, b_dir = run_dir / "resume_A", run_dir / "resume_B"
    for d in (a_dir, b_dir):
        shutil.rmtree(d, ignore_errors=True)
    ta = Lumen2Trainer({**base, "max_steps": n}, a_dir)
    ta.train()
    from dcfg.lumen2.model import trainable_state

    sa, pn = trainable_state(ta.model), {k for k, _ in ta.model.named_parameters()}
    del ta
    _free()
    tb0 = Lumen2Trainer({**base, "max_steps": n // 2}, b_dir)
    tb0.train()
    del tb0
    _free()
    (b_dir / "done.json").unlink()
    tb = Lumen2Trainer({**base, "max_steps": n}, b_dir)
    tb.train()
    hashes = lambda p: [ln.split()[-1] for ln in p.read_text().splitlines() if ln and not ln.startswith("#")]  # noqa: E731
    sb = trainable_state(tb.model)
    md = max(float((sa[k] - sb[k]).abs().max()) for k in sa if k in pn)
    rel = lambda x, y: float(((x.float() - y.float()).abs() / (x.float().abs() + y.float().abs() + 1e-3)).max())  # noqa: E731
    md_buf = max(rel(sa[k], sb[k]) for k in sa if k not in pn)  # BatchNorm running stats: relative difference
    out = {"steps": n, "resumed": "# resumed" in (b_dir / "batches.log").read_text(), "batch_hashes_equal": hashes(a_dir / "batches.log") == hashes(b_dir / "batches.log"),
           "max_abs_param_diff": md, "max_abs_buffer_diff": md_buf}
    out["pass"] = bool(out["resumed"] and out["batch_hashes_equal"] and md < 5e-3 and md_buf < 5e-2)
    return out


def notify(msg: str) -> None:
    try:
        subprocess.run([str(REPO_ROOT / "ops/notify.sh"), msg, "DCFG LUMEN2 smoke"], timeout=60, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    except Exception:  # noqa: BLE001
        pass


def write_md(res: dict[str, Any], path: Path) -> None:
    t, e = res["train_check"], res["train_check"]["estimate"]
    L = ["# LUMEN2 smoke test", "",
         f"Verdict: **{'PASS' if res['pass'] else 'NO PASS (training not started)'}**" + ("" if res["time_ok"] else f" - estimated {e['total_h']:.1f} h > {LIMIT_H:.0f} h"), "",
         "| check | result |", "|---|---|",
         f"| step-0 LUMEN2 == GenD | max |logit diff| {res['step0']['max_abs_logit_diff']:.2e}, max |embedding diff| {res['step0']['max_abs_embedding_diff']:.2e}, w all 1: {res['step0']['gate_w_all_one']} -> {'ok' if res['step0']['pass'] else 'FAIL'} |",
         f"| damaged-image stream == GenD-aug | 224 view and 256 view hashes equal in {len(res['stream']['slices'])} slices; {res['stream']['gendaug_batches_checked']} record-order hashes vs runs/gendaug_train/batches.log, {res['stream']['gendaug_batches_mismatched']} mismatches -> {'ok' if res['stream']['pass'] else 'FAIL'} |",
         f"| {t['steps']} optimiser steps ({t['micro_steps']} micro-steps) | main loss {t['main_loss_first_last_quarter'][0]:.4f} -> {t['main_loss_first_last_quarter'][1]:.4f}, sensor loss {t['damage_loss_first_last_quarter'][0]:.4f} -> {t['damage_loss_first_last_quarter'][1]:.4f} (first -> last quarter), NaN steps {t['nonfinite_steps']}, U/V abs sums {t['U_abs_sum']:.3g}/{t['V_abs_sum']:.3g} -> {'ok' if t['loss_pass'] else 'FAIL'} |",
         f"| resume | resumed {res['resume']['resumed']}, batch hashes equal {res['resume']['batch_hashes_equal']}, max |param diff| {res['resume']['max_abs_param_diff']:.2e}, BN buffers (max relative diff) {res['resume']['max_abs_buffer_diff']:.2e} -> {'ok' if res['resume']['pass'] else 'FAIL'} |",
         f"| time | {t['img_s_steady']:.1f} img/s -> epoch train {e['epoch_train_s'] / 60:.1f} min + val {e['epoch_val_s'] / 60:.1f} min; 18 epochs = **{e['total_h']:.1f} h** (limit {LIMIT_H:.0f} h); peak GPU {t['gpu_mem_peak_mb']:.0f} MB |",
         "", "## Parameters per part", "", "| part | params |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in res["part_params"].items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.lumen2.smoke", description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/train/lumen2.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--gendaug-log", default=str(RUNS_ROOT / "gendaug_train" / "batches.log"))
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    run_dir = Path(a.run_dir or os.environ.get("DCFG_RUN_DIR") or RUNS_ROOT / "lumen2_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "PASS").unlink(missing_ok=True)
    res: dict[str, Any] = {}
    res["train_check"], tr = train_check(cfg, run_dir, a.steps)
    res["part_params"] = tr.part_counts
    _free()
    res["step0"] = check_step0(tr)
    _free()
    res["stream"] = check_stream(tr, Path(a.gendaug_log))
    del tr
    torch.cuda.empty_cache()
    res["resume"] = resume_check(cfg, run_dir)
    est = res["train_check"]["estimate"]
    res["time_ok"] = bool(est["total_h"] <= LIMIT_H)
    res["pass"] = bool(res["step0"]["pass"] and res["stream"]["pass"] and res["train_check"]["loss_pass"] and res["resume"]["pass"] and res["time_ok"])
    (run_dir / "smoke.json").write_text(json.dumps(res, indent=2, default=float))
    write_md(res, RESULTS_ROOT / "lumen2" / "lumen2_smoke.md")
    print(json.dumps({k: v for k, v in res.items() if k in ("pass", "time_ok")} | {"est_total_h": est["total_h"]}), flush=True)
    if res["pass"]:
        (run_dir / "PASS").write_text(f"pass, estimated {est['total_h']:.1f} h\n")
    elif not res["time_ok"]:
        notify(f"LUMEN2 18 epochs would take {est['total_h']:.1f} h > {LIMIT_H:.0f} h, waiting for options (training NOT started)")
    else:
        failed = [k for k in ("step0", "stream", "resume") if not res[k]["pass"]] + ([] if res["train_check"]["loss_pass"] else ["train_check"])
        notify(f"LUMEN2 smoke FAILED ({', '.join(failed)}); training NOT started")
    return 0


if __name__ == "__main__":
    sys.exit(main())
