"""T4: front-end overhead (params, FLOPs, added ms/image) per variant, plus each detector's own cost.

    python -m dcfg.analysis.overhead [--out results/raw/overhead.json]
    python -m dcfg.analysis.overhead --design-selection --out <path/outside/results/raw>/overhead.json
    python -m dcfg.analysis.overhead --lumen --out results/qa/lumen_overhead.json

Timing: CUDA events around the module forward only (eval mode, no_grad, fp32 input 256x256; the sensor CNN
uses its default fp16 autocast on CUDA as in training/eval), warmup 20, 100 iterations, median; reported
per batch and per image, at batch 1 and 32. "Added ms/image" is the front-end alone. FLOPs (forward, per
image) come from fvcore or thop if importable, else torch.utils.flop_counter (built in, counts matmul/conv
only, FLOPs = 2*MACs); the method used is recorded. Backward cost is not measured.

``--design-selection`` (Xception only, pilot study, requires an explicit ``--out`` that is not under
results/raw|tables|figures): end-to-end ms/image of detector, fixed+det, dcfg+det, film+det, filmlite+det
(front-end + FiLM hooks + detector, fp32 detector, sensor fp16 as in eval) at batch 1 and 32, the added
ms/image over the plain detector, and trainable-parameter counts of every design-selection row.

``--lumen`` (requires an explicit ``--out`` outside results/raw|tables|figures, e.g. results/qa/lumen_overhead.json):
for every detector with FiLM sites (UCF is skipped while its sites are undecided) the LUMEN pipeline
(sensor + DCFG gate + 2-site FiLM + detector) vs the plain detector: added trainable params (+ % of the detector's
params), ms/image at batch 1 and 32 (same protocol as ``--design-selection``).
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


def trainable_params_by_model(det_params: int | None = None) -> dict[str, dict]:
    """Trainable params ADDED by each design-selection row (Xception). ``det_params`` = Xception total
    (all DFB submodules, incl. the unused adjust_channel head); built from a weight-less detector if None.
    augft / jointft train the whole detector, so its params count as trainable (flag ``detector_trained``)."""
    from dcfg.detectors.adapter import Detector
    from dcfg.models.film import FiLMBank

    det = Detector("xception", device="cpu", load_weights=False)
    n_det = count_params(det.model) if det_params is None else det_params
    ch = {k: c for k, (_, c) in det.film_sites().items()}

    def tr(m: nn.Module) -> int:
        return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

    fe = {"fixed": FrontEnd("fixed"), "dcfg": FrontEnd("dcfg"),
          "film": FrontEnd("dcfg", film=FiLMBank("film", ch)), "filmlite": FrontEnd("dcfg", film=FiLMBank("filmlite", ch))}
    rows: dict[str, dict] = {"base": {"trainable": 0, "detector_trained": False}}
    rows["fixed"] = {"trainable": tr(fe["fixed"]), "detector_trained": False}
    rows["augft"] = {"trainable": n_det, "detector_trained": True}
    rows["dcfg"] = {"trainable": tr(fe["dcfg"]), "detector_trained": False}
    rows["jointft"] = {"trainable": tr(fe["dcfg"]) + n_det, "detector_trained": True}
    rows["dcfgcons"] = dict(rows["dcfg"])  # same module as dcfg; only the loss differs
    rows["dcfgfilm"] = {"trainable": tr(fe["film"]), "detector_trained": False}
    rows["dcfgfilmlite"] = {"trainable": tr(fe["filmlite"]), "detector_trained": False}
    for v in rows.values():
        v["pct_of_xception"] = 100.0 * v["trainable"] / n_det
    rows["_xception_params"] = {"trainable": n_det}  # type: ignore[assignment]
    return rows


def run_design_selection(device: str, batches=(1, 32), warmup: int = 20, iters: int = 100) -> dict:
    """Inference overhead of the design-selection pipelines (see module docstring)."""
    from dcfg.detectors.adapter import Detector, FilmHooks
    from dcfg.models.film import FiLMBank

    det = Detector("xception", device=device).freeze().eval()
    ch = {k: c for k, (_, c) in det.film_sites().items()}
    out: dict = {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                 "warmup": warmup, "iters": iters, "statistic": "median", "torch": torch.__version__,
                 "timestamp": datetime.now(timezone.utc).isoformat(), "pipelines": {},
                 "note": "inference, fp32 detector, no_grad, eval mode; added = pipeline minus plain detector. "
                         "dcfgcons/jointft inference == dcfg (+ detector fine-tune for jointft: same cost), augft == base."}

    def plain(x: torch.Tensor) -> torch.Tensor:
        return det(det.normalize(x))

    def make(kind: str):
        film = FiLMBank(kind, ch) if kind in ("film", "filmlite") else None
        fe = FrontEnd("fixed" if kind == "fixed" else "dcfg", film=film).to(device).eval()
        hooks = FilmHooks(det, fe.film.sites) if film is not None else None

        def fn(x: torch.Tensor) -> torch.Tensor:
            y, _, fp = fe.forward_full(x)
            if hooks is None:
                return plain(y)
            with hooks.conditioned(fp):
                return plain(y)
        return fn, hooks

    out["pipelines"]["base"] = {f"batch{b}": time_module(plain, b, device, warmup, iters) for b in batches}
    for kind in ("fixed", "dcfg", "film", "filmlite"):
        fn, hooks = make(kind)
        out["pipelines"][kind] = {f"batch{b}": time_module(fn, b, device, warmup, iters) for b in batches}
        if hooks is not None:
            hooks.remove()
    for kind, r in out["pipelines"].items():
        for b in batches:
            r[f"batch{b}"]["added_ms_per_image"] = (r[f"batch{b}"]["ms_per_image"]
                                                    - out["pipelines"]["base"][f"batch{b}"]["ms_per_image"])
    out["trainable_params"] = trainable_params_by_model()
    return out


def run_lumen_overhead(device: str, detectors=DETECTORS, batches=(1, 32), warmup: int = 20,
                       iters: int = 100) -> dict:
    """Params and inference overhead of the LUMEN pipeline per detector (see module docstring)."""
    from dcfg.detectors.adapter import Detector, FilmHooks
    from dcfg.models.film import LUMEN_HIDDEN, FiLMBank

    out: dict = {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                 "warmup": warmup, "iters": iters, "statistic": "median", "torch": torch.__version__,
                 "timestamp": datetime.now(timezone.utc).isoformat(), "detectors": {},
                 "note": "inference, fp32 detector, no_grad, eval mode; sensor CNN fp16 autocast as in eval; "
                         "added = LUMEN pipeline minus the plain detector (front-end + FiLM MLP + hooks)."}
    for name in detectors:
        det = Detector(name, device=device).freeze().eval()
        n_det = count_params(det.model)
        try:
            sites = det.film_sites()
        except NotImplementedError as e:
            out["detectors"][name] = {"skipped": str(e), "detector_params": n_det}
            del det
            continue
        ch = {k: c for k, (_, c) in sites.items() if k in ("entry", "exit")}
        fe = FrontEnd("dcfg", film=FiLMBank("filmlite", ch, hidden=LUMEN_HIDDEN[name])).to(device).eval()
        hooks = FilmHooks(det, fe.film.sites)

        def plain(x: torch.Tensor, d=det) -> torch.Tensor:
            return d(d.normalize(x))

        def lumen(x: torch.Tensor, d=det, f=fe, h=hooks) -> torch.Tensor:
            y, _, fp = f.forward_full(x)
            with h.conditioned(fp):
                return d(d.normalize(y))

        n_gate, n_sensor, n_film = count_params(fe.gate), count_params(fe.sensor), count_params(fe.film)
        r: dict = {"detector_params": n_det, "sites": {k: {"channels": c} for k, c in ch.items()},
                   "film_hidden": LUMEN_HIDDEN[name], "params_gate": n_gate, "params_sensor": n_sensor,
                   "params_film": n_film, "added_trainable_params": count_params(fe),
                   "added_pct_of_detector": 100.0 * count_params(fe) / n_det}
        for b in batches:
            base_t, lum_t = time_module(plain, b, device, warmup, iters), time_module(lumen, b, device, warmup, iters)
            r[f"batch{b}"] = {"plain_ms_per_image": base_t["ms_per_image"], "lumen_ms_per_image": lum_t["ms_per_image"],
                              "added_ms_per_image": lum_t["ms_per_image"] - base_t["ms_per_image"]}
        out["detectors"][name] = r
        hooks.remove()
        del det, fe
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(RESULTS_ROOT / "raw" / "overhead.json"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--design-selection", action="store_true", help="design-selection pilot overhead (needs --out)")
    ap.add_argument("--lumen", action="store_true", help="LUMEN overhead per detector (needs --out)")
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.lumen:
        if "--out" not in (argv if argv is not None else sys.argv):
            raise SystemExit("--lumen requires an explicit --out")
        out = Path(a.out).resolve()
        for forbidden in ("raw", "tables", "figures"):
            if out.is_relative_to((RESULTS_ROOT / forbidden).resolve()):
                raise SystemExit(f"refusing to write LUMEN overhead numbers into results/{forbidden}")
        res = run_lumen_overhead(device, warmup=a.warmup, iters=a.iters)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2) + "\n")
        for k, r in res["detectors"].items():
            if "skipped" in r:
                print(f"{k:<15} skipped: {r['skipped'][:80]}")
            else:
                print(f"{k:<15} +{r['added_trainable_params']:,} params ({r['added_pct_of_detector']:.2f}%) "
                      f"b1 +{r['batch1']['added_ms_per_image']:.3f} b32 +{r['batch32']['added_ms_per_image']:.3f} ms/img")
        return 0
    if a.design_selection:
        if "--out" not in (argv if argv is not None else sys.argv):
            raise SystemExit("--design-selection requires an explicit --out")
        out = Path(a.out).resolve()
        for forbidden in ("raw", "tables", "figures"):
            if out.is_relative_to((RESULTS_ROOT / forbidden).resolve()):
                raise SystemExit(f"refusing to write design-selection numbers into results/{forbidden}")
        res = run_design_selection(device, warmup=a.warmup, iters=a.iters)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2) + "\n")
        for k, r in res["pipelines"].items():
            print(f"{k:<9} b1={r['batch1']['ms_per_image']:.3f} ms/img (+{r['batch1']['added_ms_per_image']:.3f})  "
                  f"b32={r['batch32']['ms_per_image']:.3f} ms/img (+{r['batch32']['added_ms_per_image']:.3f})")
        return 0
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
