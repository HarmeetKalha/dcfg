"""T4: front-end overhead (params, FLOPs, added ms/image) per variant, plus each detector's own cost.

    python -m dcfg.analysis.overhead [--out results/raw/overhead.json]

Timing: CUDA events around the module forward only (eval mode, no_grad, fp32 input 256x256; the sensor CNN
uses its default fp16 autocast on CUDA as in training/eval), warmup 20, 100 iterations, median; reported
per batch and per image, at batch 1 and 32. "Added ms/image" is the front-end alone. FLOPs (forward, per
image) come from fvcore or thop if importable, else torch.utils.flop_counter (built in, counts matmul/conv
only, FLOPs = 2*MACs); the method used is recorded. Backward cost is not measured.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from dcfg.models.frontend import FrontEnd
from dcfg.paths import RESULTS_ROOT

VARIANTS = ("dcfg", "fixed", "nosensor")
DETECTORS = ("xception", "efficientnetb4", "f3net", "ucf")
RES = 256


def count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def count_flops(m: nn.Module, x: torch.Tensor) -> tuple[float | None, str]:
    """Forward FLOPs for input ``x`` and the method name."""
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
        return float(FlopCountAnalysis(m, x).total()) * 2, "fvcore(2*MACs)"
    except ImportError:
        pass
    try:
        from thop import profile  # type: ignore
        macs, _ = profile(m, inputs=(x,), verbose=False)
        return float(macs) * 2, "thop(2*MACs)"
    except ImportError:
        pass
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x)
        return float(fc.get_total_flops()), "torch.flop_counter"
    except Exception:  # noqa: BLE001
        return None, "unavailable"


@torch.no_grad()
def time_module(fn: Callable[[torch.Tensor], object], batch: int, device: str, warmup: int = 20,
                iters: int = 100, res: int = RES) -> dict[str, float]:
    """Median forward time (ms) per batch and per image (CUDA events on GPU, perf_counter on CPU)."""
    import time
    x = torch.rand(batch, 3, res, res, device=device)
    for _ in range(warmup):
        fn(x)
    ts: list[float] = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(x)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    else:
        for _ in range(iters):
            t = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - t) * 1e3)
    med = statistics.median(ts)
    return {"ms_per_batch": med, "ms_per_image": med / batch}


def measure(module: nn.Module, fn: Callable[[torch.Tensor], object], device: str, batches=(1, 32),
            warmup: int = 20, iters: int = 100, res: int = RES) -> dict:
    flops, method = count_flops(module, torch.rand(1, 3, res, res, device=device)) \
        if isinstance(module, nn.Module) else (None, "n/a")
    out: dict = {"params": count_params(module), "flops_per_image": flops, "flops_method": method}
    for b in batches:
        out[f"batch{b}"] = time_module(fn, b, device, warmup, iters, res)
    return out


def run(device: str, detectors=DETECTORS, variants=VARIANTS, batches=(1, 32), warmup: int = 20,
        iters: int = 100, res: int = RES, with_detectors: bool = True) -> dict:
    res_d: dict = {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                   "resolution": res, "warmup": warmup, "iters": iters, "statistic": "median",
                   "torch": torch.__version__, "timestamp": datetime.now(timezone.utc).isoformat(),
                   "frontends": {}, "detectors": {}}
    for v in variants:
        fe = FrontEnd(v).to(device).eval()
        r = measure(fe, lambda x, fe=fe: fe(x), device, batches, warmup, iters, res)
        r["params_gate"] = count_params(fe.gate)
        r["params_sensor"] = count_params(fe.sensor) if fe.sensor is not None else 0
        res_d["frontends"][v] = r
        del fe
    if with_detectors:
        from dcfg.detectors.adapter import Detector
        for name in detectors:
            det = Detector(name, device=device).freeze().eval()
            r = measure(det, lambda x, d=det: d(d.normalize(x)), device, batches, warmup, iters, res)
            res_d["detectors"][name] = r
            del det
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return res_d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(RESULTS_ROOT / "raw" / "overhead.json"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run(device, warmup=a.warmup, iters=a.iters)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    for kind in ("frontends", "detectors"):
        for k, r in res[kind].items():
            print(f"{kind[:3]} {k:<15} params={r['params']:>10,} flops={r['flops_per_image']} "
                  f"b1={r['batch1']['ms_per_image']:.3f}ms b32={r['batch32']['ms_per_image']:.3f}ms/img")
    return 0


if __name__ == "__main__":
    sys.exit(main())
