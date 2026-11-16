"""Dummy GPU job (~3 min): fp16 matmul loop + heartbeat + dummy W&B run.

Usage: python ops/smoke_gpu.py [--seconds 180]. wandb mode comes from WANDB_MODE.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import wandb

from dcfg.utils.heartbeat import Heartbeat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    args = ap.parse_args()

    job = os.environ.get("DCFG_JOB_ID", "smoke")
    run_dir = os.environ.get("DCFG_RUN_DIR", f"runs/{job}")
    hb = Heartbeat(run_dir, interval_s=15)
    run = wandb.init(project="dcfg", job_type="smoke", name=job)

    dev = "cuda"
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    t0 = time.time()
    step = 0
    while time.time() - t0 < args.seconds:
        for _ in range(10):
            c = a @ b
            a = (c / c.abs().max()).half()
        torch.cuda.synchronize()
        step += 10
        el = time.time() - t0
        tput = step / el
        run.log({"matmuls_per_s": tput}, step=step)
        hb.beat(step, throughput=round(tput, 2), eta=f"{max(args.seconds - el, 0):.0f}s")
    hb.beat(step, force=True, throughput=round(step / (time.time() - t0), 2), eta="0s")
    run.finish()
    print(f"final throughput: {step / (time.time() - t0):.1f} matmuls/s over {step} steps")


if __name__ == "__main__":
    main()
