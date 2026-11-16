"""LIVE pretraining of the degradation sensor (PLAN 6.1-6.2, 7.5 style optimisation).

Usage:
    python -m dcfg.train.train_sensor --config configs/sensor_pretrain.yaml [--protocol10]
        [--run-dir DIR] [--overwrite] [--set key=value ...]

Main run: MSE on train-normalised MOS, AdamW, 1-epoch warmup + cosine, early stopping on
val SRCC (center crop); final test SRCC/PLCC (center + 5-crop) from the best checkpoint
-> results/raw/sensor_live__test.json and ~/data/weights/sensor_live.pt (backbone only).
``--protocol10``: 10 random 80/20 splits, fixed epoch count = main run's best epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader

from dcfg.data.live import (LiveDataset, LiveRecords, get_split, load_records, mos_stats,
                            random_80_20_splits, subset_records)
from dcfg.models.sensor import DegradationSensor
from dcfg.paths import REPO_ROOT, RUNS_ROOT
from dcfg.utils.heartbeat import Heartbeat
from dcfg.utils.seed import seed_everything


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _path(p: str | os.PathLike[str]) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _save_atomic(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _loader(rec: LiveRecords, idx: list[int], mode: str, mean: float, std: float, cfg: dict,
            shuffle: bool, bs: int | None = None) -> DataLoader:
    nw = int(cfg["num_workers"])
    return DataLoader(LiveDataset(rec, idx, mode, mean, std), batch_size=bs or int(cfg["batch_size"]),
                      shuffle=shuffle, drop_last=shuffle, num_workers=nw, pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Predictions and targets (normalised MOS). Handles 5-crop batches by averaging the crops."""
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if x.dim() == 5:
            b, c = x.shape[:2]
            p = model(x.flatten(0, 1))[1].view(b, c).float().mean(1)
        else:
            p = model(x)[1].float()
        ps.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """SRCC and PLCC (invariant to the affine MOS normalisation)."""
    return {"srcc": float(spearmanr(p, y)[0]), "plcc": float(pearsonr(p, y)[0])}


def _lr_lambda(steps_per_epoch: int, warmup_epochs: int, max_epochs: int):
    warm, total = max(1, steps_per_epoch * warmup_epochs), steps_per_epoch * max_epochs

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
    return f


def train_one(cfg: dict, rec: LiveRecords, train_idx: list[int], val_idx: list[int] | None,
              run_dir: Path, mean: float, std: float, max_epochs: int, early_stop: bool,
              job_id: str, tag: str = "") -> dict[str, Any]:
    """Train (resumably) in ``run_dir``. With ``early_stop`` select best on val SRCC, else
    run exactly ``max_epochs`` and keep the last model. Returns state incl. best/last paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg["seed"]))
    model = DegradationSensor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    tr_loader = _loader(rec, train_idx, "train", mean, std, cfg, True)
    va_loader = _loader(rec, val_idx, "center", mean, std, cfg, False) if val_idx else None
    spe = len(tr_loader)
    # cosine horizon is cfg max_epochs (not the fixed-epoch run length) so protocol10 matches the main recipe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(spe, int(cfg["warmup_epochs"]), int(cfg["max_epochs"])))
    use_amp = bool(cfg["amp"]) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    st = {"epoch": 0, "best_srcc": -float("inf"), "best_epoch": 0, "bad": 0, "stopped": False, "step": 0}
    last, best = run_dir / "last.pt", run_dir / "best.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        st = ck["state"]; _set_rng_state(ck["rng"])
        print(f"[{tag}] resumed from epoch {st['epoch']}", flush=True)
    hb = Heartbeat(run_dir, float(cfg["heartbeat_s"]))
    wb = None
    if not tag:
        try:
            import wandb
            wb = wandb.init(project=cfg["wandb_project"], job_type="sensor", name=job_id, id=job_id,
                            resume="allow", config=cfg, dir=str(run_dir))
        except Exception as e:  # noqa: BLE001
            print(f"wandb disabled: {e}", flush=True)
    csv_path = run_dir / "metrics.csv"
    new_csv = not csv_path.exists()
    ep_times: list[float] = []
    loss_fn = nn.MSELoss()
    while st["epoch"] < max_epochs and not st["stopped"]:
        ep = st["epoch"] + 1
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                p = model(x)[1]
            loss = loss_fn(p.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            st["step"] += 1
            run_loss += float(loss.detach()); nb += 1
            hb.beat(st["step"], epoch=ep, val_srcc=st["best_srcc"], eta="?")
        train_loss, dt = run_loss / max(1, nb), time.time() - t0
        ep_times.append(dt)
        val_srcc = float("nan")
        if va_loader is not None:
            val_srcc = metrics(*predict(model, va_loader, dev))["srcc"]
            if val_srcc > st["best_srcc"]:
                st.update(best_srcc=val_srcc, best_epoch=ep, bad=0)
                _save_atomic({"model": model.state_dict(), "epoch": ep}, best)
            else:
                st["bad"] += 1
        st["epoch"] = ep
        if va_loader is not None and st["bad"] >= int(cfg["patience"]) and early_stop:
            st["stopped"] = True
        _save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                      "scaler": scaler.state_dict(), "state": st, "rng": _rng_state(), "epoch": ep}, last)
        eta = int(np.mean(ep_times) * (max_epochs - ep))
        hb.beat(st["step"], force=True, epoch=ep, val_srcc=val_srcc, best_srcc=st["best_srcc"], eta=f"{eta}s")
        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new_csv:
                w.writerow(["epoch", "step", "train_loss", "val_srcc", "lr", "epoch_s"]); new_csv = False
            w.writerow([ep, st["step"], f"{train_loss:.5f}", f"{val_srcc:.5f}", f"{sched.get_last_lr()[0]:.3e}", f"{dt:.1f}"])
        if wb is not None:
            wb.log({"epoch": ep, "train_loss": train_loss, "val_srcc": val_srcc, "lr": sched.get_last_lr()[0]}, step=ep)
        print(f"[{tag}] ep {ep} loss {train_loss:.4f} val_srcc {val_srcc:.4f} {dt:.1f}s", flush=True)
    if wb is not None:
        wb.finish()
    # final model for fixed-epoch mode is last.pt; otherwise best.pt
    final = best if early_stop and best.exists() else last
    ck = torch.load(final, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return {"model": model, "dev": dev, "state": st, "epoch_s": float(np.mean(ep_times)) if ep_times else None}


def _evaluate(model: nn.Module, dev: torch.device, rec: LiveRecords, idx: list[int],
              mean: float, std: float, cfg: dict) -> dict[str, dict[str, float]]:
    out = {}
    for mode, name in (("center", "center"), ("five", "five_crop")):
        out[name] = metrics(*predict(model, _loader(rec, idx, mode, mean, std, cfg, False, bs=16), dev))
    return out


def _records(cfg: dict) -> LiveRecords:
    rec = load_records(_path(cfg["live_root"]))
    return subset_records(rec, int(cfg["subset"]), 0) if cfg.get("subset") else rec


def run_main(cfg: dict, run_dir: Path, overwrite: bool) -> dict[str, Any]:
    job_id = os.environ.get("DCFG_JOB_ID", run_dir.name)
    rec = _records(cfg)
    sp = get_split(rec, _path(cfg["split_path"]), 42)
    mean, std = mos_stats(rec, sp["train"])
    r = train_one(cfg, rec, sp["train"], sp["val"], run_dir, mean, std, int(cfg["max_epochs"]), True, job_id)
    model, dev, st = r["model"], r["dev"], r["state"]
    ev = _evaluate(model, dev, rec, sp["test"], mean, std, cfg)
    res = {"job_id": job_id, "test": ev, "n_train": len(sp["train"]), "n_val": len(sp["val"]),
           "n_test": len(sp["test"]), "best_epoch": st["best_epoch"], "best_val_srcc": st["best_srcc"],
           "epochs_run": st["epoch"], "mos_mean": mean, "mos_std": std, "git_commit": _git_commit(),
           "config": cfg}
    _write_json_atomic(_path(cfg["results_dir"]) / "sensor_live__test.json", res)
    wpath = _path(cfg["weights_path"])
    meta = wpath.with_suffix(".job")
    prev = meta.read_text().strip() if meta.exists() else None
    if not wpath.exists() or overwrite or prev != job_id:
        wpath.parent.mkdir(parents=True, exist_ok=True)
        _save_atomic(model.backbone_state_dict(), wpath)
        meta.write_text(job_id)
    else:
        print(f"{wpath} exists from the same job_id; not overwriting (use --overwrite)", flush=True)
    print(json.dumps({k: res[k] for k in ("test", "best_epoch")}), flush=True)
    return res


def run_protocol10(cfg: dict, run_dir: Path) -> dict[str, Any]:
    rdir = _path(cfg["results_dir"])
    main = json.loads((rdir / "sensor_live__test.json").read_text())
    n_ep = int(main["best_epoch"])
    job_id = os.environ.get("DCFG_JOB_ID", run_dir.name)
    rec = _records(cfg)
    per = []
    for sp in random_80_20_splits(len(rec.names)):
        d = run_dir / f"split{sp['seed']}"
        done = d / "result.json"
        if done.exists():
            per.append(json.loads(done.read_text())); continue
        mean, std = mos_stats(rec, sp["train"])
        c = dict(cfg, seed=int(cfg["seed"]) + sp["seed"])
        r = train_one(c, rec, sp["train"], None, d, mean, std, n_ep, False, job_id, tag=f"p10_s{sp['seed']}")
        ev = _evaluate(r["model"], r["dev"], rec, sp["test"], mean, std, cfg)
        item = {"split_seed": sp["seed"], "test": ev, "n_train": len(sp["train"]), "n_test": len(sp["test"]),
                "mos_mean": mean, "mos_std": std}
        _write_json_atomic(done, item); per.append(item)
        for f in ("last.pt", "best.pt"):  # per-split checkpoints are no longer needed once the result is saved
            (d / f).unlink(missing_ok=True)
    med = {m: {k: float(np.median([p["test"][m][k] for p in per])) for k in ("srcc", "plcc")}
           for m in ("center", "five_crop")}
    res = {"job_id": job_id, "epochs": n_ep, "n_splits": len(per), "median": med, "per_split": per,
           "git_commit": _git_commit(), "config": cfg}
    _write_json_atomic(rdir / "sensor_live__protocol10.json", res)
    print(json.dumps(med), flush=True)
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sensor_pretrain.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--protocol10", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="YAML-parsed config overrides")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(_path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_dir = Path(a.run_dir or os.environ.get("DCFG_RUN_DIR") or RUNS_ROOT / ("sensor_live_p10" if a.protocol10 else "sensor_live"))
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.protocol10:
        run_protocol10(cfg, run_dir)
    else:
        run_main(cfg, run_dir, a.overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])
