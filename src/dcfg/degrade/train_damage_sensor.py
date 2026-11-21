"""Train the standalone damage sensor (test run, capped wall time).

    python -m dcfg.degrade.train_damage_sensor --run-dir runs/damage_sensor [--cap-hours 1.5]

FF++ c23 train frames (real + fake), our PLAN 7.2 degradation sampler with labels, seed 42, AdamW. A "mini-epoch" is
``samples_per_epoch`` (40,000) frames of a fresh class-balanced permutation (``epoch_order``, mode full); after each one the
model is validated on a fixed degraded val subset (8 frames/video, sampler draws fixed by seed) and the best val loss kept.
LR: 100-step linear warm-up, then cosine over the wall-time cap (so the schedule adapts to throughput). Resume-safe:
``last.pt`` is written after every mini-epoch; the train-seconds counter accumulates over resumes.
Outputs in run-dir: ``best.pt`` ({state_dict, meta}), ``last.pt``, ``metrics.csv``, ``done.json``, ``heartbeat``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dcfg.data.degradations import TrainDegradation, make_rng
from dcfg.data.ffpp import DFBJsonSource, FrameRecord
from dcfg.degrade.labels import apply_with_labels
from dcfg.degrade.sensor import DamageSensor, damage_loss
from dcfg.paths import RUNS_ROOT
from dcfg.train.train_frontend import epoch_order
from dcfg.utils.heartbeat import Heartbeat
from dcfg.utils.seed import seed_everything

VAL_SEED = 4242  # fixed degradation draws for validation (separate from the training stream)


class LabelledDegradedSet(Dataset):
    """uint8 CHW degraded crop, applied (4), strength (4), binary label; rng = make_rng(seed, epoch, position)."""

    def __init__(self, source: DFBJsonSource, records: list[FrameRecord], order: np.ndarray, seed: int, epoch: int) -> None:
        self.source, self.records, self.order, self.seed, self.epoch = source, records, order, seed, epoch
        self.deg = TrainDegradation()

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, pos: int) -> tuple[torch.Tensor, ...]:
        rec = self.records[int(self.order[pos])]
        img, _, a, s = apply_with_labels(self.deg, self.source.load(rec), make_rng(self.seed, self.epoch, pos))
        return (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), torch.from_numpy(a), torch.from_numpy(s),
                torch.tensor(float(rec.label)))


def val_records(source: DFBJsonSource, per_video: int = 8) -> list[FrameRecord]:
    """Val frames subsampled to ``per_video`` per video with linspace over the key-sorted frames."""
    by: dict[str, list[FrameRecord]] = {}
    for r in source.records("val", "c23"):
        by.setdefault(r.video, []).append(r)
    out: list[FrameRecord] = []
    for v in sorted(by):
        fr = sorted(by[v], key=lambda r: r.key)
        idx = np.unique(np.linspace(0, len(fr) - 1, min(per_video, len(fr))).round().astype(int))
        out += [fr[i] for i in idx]
    return out


def _save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model: DamageSensor, loader: DataLoader, dev: str, w_strength: float) -> dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "bce": 0.0, "mse": 0.0}
    n = 0
    acc = 0.0
    for xb, a, s, _ in loader:
        x = xb.to(dev).float().div_(255.0)
        a, s = a.to(dev), s.to(dev)
        out = model(x)
        loss, parts = damage_loss(out, a, s, w_strength)
        k = len(x)
        tot["loss"] += float(loss) * k; tot["bce"] += float(parts["bce"]) * k; tot["mse"] += float(parts["mse"]) * k
        acc += float(((out["presence_logit"] > 0).float() == a).float().mean()) * k
        n += k
    return {**{k: v / n for k, v in tot.items()}, "presence_acc": acc / n, "n": n}


def train(a: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(a.seed)
    source = DFBJsonSource()
    recs = source.records("train", "c23")
    labels = np.array([r.label for r in recs])
    vrecs = val_records(source, a.val_per_video)
    if a.val_max:
        vrecs = vrecs[:: max(1, len(vrecs) // a.val_max)]
    vloader = DataLoader(LabelledDegradedSet(source, vrecs, np.arange(len(vrecs)), VAL_SEED, 0), batch_size=128,
                         shuffle=False, num_workers=a.val_workers)
    model = DamageSensor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    st = {"epoch": 0, "step": 0, "train_s": 0.0, "best": float("inf"), "best_epoch": -1}
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); st = ck["state"]
        print(f"resumed at epoch {st['epoch']} step {st['step']} train_s {st['train_s']:.0f}", flush=True)
    hb = Heartbeat(Path(os.environ.get("DCFG_RUN_DIR") or run_dir), 60.0)
    cap_s = a.cap_hours * 3600.0
    n_params = model.n_params()
    print(f"DamageSensor params={n_params} train frames={len(recs)} val frames={len(vrecs)}", flush=True)

    def lr_at(step: int, t: float) -> float:
        warm = min(1.0, (step + 1) / 100)
        return a.lr * warm * 0.5 * (1 + math.cos(math.pi * min(1.0, t / cap_s)))

    stop_reason = None
    while True:
        if st["train_s"] >= cap_s:
            stop_reason = "time_cap"; break
        if a.max_epochs and st["epoch"] >= a.max_epochs:
            stop_reason = "max_epochs"; break
        e = st["epoch"]
        order = epoch_order(labels, a.seed, e, "full")[: a.samples_per_epoch]
        loader = DataLoader(LabelledDegradedSet(source, recs, order, a.seed, e), batch_size=a.batch_size, shuffle=False,
                            num_workers=a.workers, drop_last=True, persistent_workers=False, prefetch_factor=4)
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for xb, ap, sg, _ in loader:
            if a.max_steps and st["step"] >= a.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(st["step"], st["train_s"] + (time.time() - t0))
            x = xb.to(dev, non_blocking=True).float().div_(255.0)
            loss, parts = damage_loss(model(x), ap.to(dev), sg.to(dev), a.w_strength)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            st["step"] += 1
            nb += 1
            run_loss += float(loss.detach()) if nb % 20 == 0 else 0.0
            if nb % 20 == 0:
                hb.beat(st["step"], epoch=e, loss=float(loss.detach()), train_s=round(st["train_s"] + time.time() - t0))
        st["train_s"] += time.time() - t0
        v = validate(model, vloader, dev, a.w_strength)
        better = v["loss"] < st["best"]
        if better:
            st["best"], st["best_epoch"] = v["loss"], e
            _save({"state_dict": model.state_dict(), "meta": {"epoch": e, "val": v, "params": n_params, "seed": a.seed,
                                                              "crop": 128}}, run_dir / "best.pt")
        st["epoch"] = e + 1
        _save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st}, last)
        row = {"epoch": e, "step": st["step"], "train_s": f"{st['train_s']:.0f}", "val_loss": f"{v['loss']:.5f}",
               "val_bce": f"{v['bce']:.5f}", "val_mse": f"{v['mse']:.5f}", "val_presence_acc": f"{v['presence_acc']:.4f}",
               "lr": f"{opt.param_groups[0]['lr']:.2e}", "img_s": f"{nb * a.batch_size / max(1e-9, time.time() - t0):.0f}"}
        new = not (run_dir / "metrics.csv").exists()
        with open(run_dir / "metrics.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(json.dumps(row), flush=True)
        if a.max_steps and st["step"] >= a.max_steps:
            stop_reason = "max_steps"; break
    res = {"stop_reason": stop_reason, "epochs_run": st["epoch"], "best_epoch": st["best_epoch"], "best_val_loss": st["best"],
           "train_s": st["train_s"], "params": n_params, "cap_hours": a.cap_hours, "seed": a.seed}
    (run_dir / "done.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.degrade.train_damage_sensor")
    ap.add_argument("--run-dir", default=str(RUNS_ROOT / "damage_sensor"))
    ap.add_argument("--cap-hours", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--samples-per-epoch", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--w-strength", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--val-workers", type=int, default=3)
    ap.add_argument("--val-per-video", type=int, default=8)
    ap.add_argument("--val-max", type=int, default=0, help="smoke only: cap val frames")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only")
    ap.add_argument("--max-epochs", type=int, default=0, help="smoke only")
    a = ap.parse_args(argv)
    print(json.dumps(train(a)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
