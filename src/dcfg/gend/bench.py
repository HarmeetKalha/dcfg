"""GPU cost of public GenD: fp16 inference memory + images/s, and a short LayerNorm-only fwd+bwd timing.

    python -m dcfg.gend.bench [--out results/lumen2/gend_bench.json]

Random input, no data, no real training, no weights saved (<~2 min). LayerNorm-only training = GenD's recipe:
only ``pre_layrnorm``/``layer_norm1``/``layer_norm2``/``post_layernorm`` (+ the linear head) require grad, bf16 autocast.
The epoch estimate multiplies the measured seconds/image by ``--epoch-frames`` (full-epoch mode, FF++ c23 train);
it EXCLUDES data loading/degradation (CPU side), val passes and the optimizer's negligible cost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from dcfg.gend.model import load_gend
from dcfg.paths import RESULTS_ROOT


def _sync() -> None:
    torch.cuda.synchronize()


def bench_inference(model: torch.nn.Module, bs: int, iters: int = 20) -> dict:
    x = torch.rand(bs, 3, 224, 224, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            model(x)
        _sync()
        t = time.time()
        for _ in range(iters):
            model(x)
        _sync()
        dt = time.time() - t
    return {"batch": bs, "img_per_s": round(bs * iters / dt, 1), "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def bench_ln_train(model: torch.nn.Module, bs: int, iters: int = 8) -> dict:
    for n, p in model.named_parameters():
        p.requires_grad = ("layer_norm" in n) or ("layrnorm" in n) or ("post_layernorm" in n) or n.startswith("linear.")
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=1e-4)
    x = torch.rand(bs, 3, 224, 224, device="cuda")
    y = torch.randint(0, 2, (bs,), device="cuda").float()
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def step() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(2):
        step()
    _sync()
    t = time.time()
    for _ in range(iters):
        step()
    _sync()
    dt = time.time() - t
    return {"batch": bs, "trainable_params": n_tr, "img_per_s": round(bs * iters / dt, 1),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.gend.bench")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "lumen2" / "gend_bench.json"))
    ap.add_argument("--epoch-frames", type=int, default=184_000)
    a = ap.parse_args(argv)
    model = load_gend("cuda").eval()
    res: dict = {"gpu": torch.cuda.get_device_name(0), "weights_fp32_gb": round(sum(p.numel() for p in model.parameters()) * 4 / 2**30, 2)}
    res["inference_fp16"] = [bench_inference(model, bs) for bs in (32, 64)]
    train = []
    for bs in (32, 16):
        try:
            train.append(bench_ln_train(model, bs))
        except torch.cuda.OutOfMemoryError:
            train.append({"batch": bs, "oom": True})
            torch.cuda.empty_cache()
    res["ln_only_train_bf16"] = train
    ok = [t for t in train if "img_per_s" in t]
    if ok:
        best = max(ok, key=lambda t: t["img_per_s"])
        res["epoch_estimate_h"] = {"frames": a.epoch_frames, "batch": best["batch"],
                                   "hours": round(a.epoch_frames / best["img_per_s"] / 3600, 2),
                                   "note": "GPU fwd+bwd only; excludes data loading/degradation and validation"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
