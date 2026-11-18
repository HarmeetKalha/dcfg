"""LUMEN smoke tests (300 steps lumen x {xception, efficientnetb4, f3net}; 100 steps bnft / lumen_nocons on Xception;
kill/resume check). UCF is skipped: its FiLM sites are undecided (docs/DECISIONS.md).

    .venv/bin/python ops/jobs/lumen_smoke.py [--root /tmp/claude-1000/lumen_smoke] [--qa-dir results/qa]

Reuses the trainer-subprocess helpers of phase8_smoke.py. Runs live under --root (W&B disabled). Per smoke:
loss first/last window, checkpoint written, val on a 64-video stratified subset, fp16 health (min GradScaler scale,
skipped steps, non-finite grad-norm counts, abort), steady img/s, peak VRAM. Resume: lumen_xception, SIGTERM at
~150 steps (ckpt_every_steps=100), restart to 300, batches.log verified against the epoch permutation.
Writes results/qa/lumen_smoke.{json,md}. Each sub-step is resumable through marker files.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))
import phase8_smoke as ps  # noqa: E402

REPO = ps.REPO
BASE_CONFIG = "configs/train/lumen_xception.yaml"
ps.BASE_CONFIG = BASE_CONFIG  # run_trainer / trainer_cmd read the module global


class _Ctx:
    def beat(self, **kw: Any) -> None:  # heartbeat not needed for a manual run
        pass


def smoke(root: Path, name: str, det: str, variant: str, steps: int, min_step: int, ckpt_every: int = 100) -> dict[str, Any]:
    d = root / name
    shutil.rmtree(d, ignore_errors=True)
    sets = {"variant": variant, "run_id": f"smoke_{name}", "detector": det, "max_steps": steps, "log_every": 25 if steps <= 100 else 50,
            "ckpt_every_steps": ckpt_every, "val_on_truncate": True, "val_max_videos": 64, "warmup_epochs": 0,
            "heartbeat_s": 30}
    r = ps.run_trainer(_Ctx(), d, sets)
    done = json.loads((d / "done.json").read_text()) if (d / "done.json").exists() else {}
    rows = ps.read_csv(d / "train_log.csv")
    ll = ps.loss_first_last(rows)
    val = done.get("val", {})
    scales = [float(x["scaler_scale"]) for x in rows if "scaler_scale" in x]
    out = {"name": name, "detector": det, "variant": variant, "rc": r["rc"], "steps": done.get("step"), "loss": ll,
           "ckpt_written": (d / "last.pt").exists(), "val_mean_auc": val.get("mean_auc"),
           "val_frames": val.get("n_frames"), "img_s_steady": ps.steady_img_s(rows, min_step),
           "peak_vram_mb_allocated": ps.peak_mb(rows), "scaler_scale_min": min(scales) if scales else None,
           "skipped_steps": int(rows[-1]["skipped_steps"]) if rows and "skipped_steps" in rows[-1] else None,
           "nonfinite_grad_windows": sum(int(x["nonfinite_sensor"]) + int(x["nonfinite_gate"]) for x in rows),
           "aborted": "NonFiniteAbort" in r["tail"] or "ABORT" in r["tail"], "wall_s": r["wall_s"], "tail": r["tail"][-300:] if r["rc"] else ""}
    out["pass"] = bool(r["rc"] == 0 and ll["decreased"] and out["ckpt_written"] and not out["aborted"]
                       and (out["scaler_scale_min"] or 0) >= 1.0 and math.isfinite(val.get("mean_auc") or float("nan")))
    return out


def resume_check(root: Path, det: str = "xception") -> dict[str, Any]:
    import numpy as np
    from dcfg.data.ffpp import DFBJsonSource
    from dcfg.train.train_frontend import load_config
    d = root / ("resume_lumen" if det == "xception" else f"resume_lumen_{det}")
    shutil.rmtree(d, ignore_errors=True)
    base = {"variant": "lumen", "run_id": f"smoke_resume_lumen_{det}", "detector": det, "max_steps": 300, "log_every": 50,
            "ckpt_every_steps": 100, "warmup_epochs": 0, "heartbeat_s": 30}
    r1 = ps.run_trainer(_Ctx(), d, base, kill_at_lines=150)
    if not r1["killed"] or (d / "done.json").exists():
        return {"pass": False, "error": f"not killed mid-way: {r1['tail'][-200:]}"}
    r2 = ps.run_trainer(_Ctx(), d, base)
    if r2["rc"] != 0 or not (d / "done.json").exists():
        return {"pass": False, "error": f"resume failed rc={r2['rc']}: {r2['tail'][-200:]}"}
    cfg = load_config(BASE_CONFIG, [f"{k}={ps.fmt(v)}" for k, v in base.items()])
    labels = np.array([x.label for x in DFBJsonSource(cfg["json"], cfg["frames_root"]).records("train", cfg["comp"])])
    ref = ps.reference_hasher(labels, int(cfg["seed"]), int(cfg["batch_size"]), cfg["epoch_mode"], None)
    v = ps.verify_resume((d / "batches.log").read_text().splitlines(), ref, 300)
    return {"killed_at_lines": r1["kill_lines"], **v, "pass": bool(v["ok"])}


def markdown(res: dict[str, Any]) -> str:
    L = ["# LUMEN smoke tests", "",
         "300-step smokes (lumen) and 100-step smokes (bnft, lumen_nocons), batch 32, real config except warmup_epochs=0, val on a "
         "64-video stratified subset, W&B disabled, runs in " + res["root"] + ". Loss = first vs last log window (lumen: BCE+KL+feat-MSE). "
         "img/s = mean of log windows after the warm-up (step >= 100 for 300-step runs, >= 50 for 100-step runs).", "",
         "| run | det | variant | loss first -> last | ckpt | val AUC (64 vid) | img/s | peak VRAM MB | min scaler scale | skipped steps | pass |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["smokes"]:
        ll = s["loss"]
        f = lambda v, n=4: "n/a" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))  # noqa: E731
        L.append(f"| {s['name']} | {s['detector']} | {s['variant']} | {f(ll['first'])} -> {f(ll['last'])} | {s['ckpt_written']} | "
                 f"{f(s['val_mean_auc'])} | {f(s['img_s_steady'], 1)} | {f(s['peak_vram_mb_allocated'], 0)} | "
                 f"{f(s['scaler_scale_min'], 0)} | {s['skipped_steps']} | {s['pass']} |")
    for det, r in res.get("resumes", {"xception": res.get("resume", {})}).items():
        L += ["", f"## Resume (lumen, {det})", "",
              f"- pass: {r.get('pass')}; killed at ~{r.get('killed_at_lines')} steps (ckpt every 100), resumed to 300; "
              f"repeated hashes {r.get('repeated_batch_hashes', 'n/a')}; "
              f"hash conflicts {r.get('hash_conflicts', 'n/a')}; reference mismatches {r.get('ref_mismatches', 'n/a')}"]
    L += ["", "## Notes", "", f"- UCF: {res['ucf_note']}", f"- bnft trainable params (Xception): {res['bnft_params']:,}",
          f"- total GPU wall time of this job: {res['total_wall_min']:.1f} min", f"- overall pass: {res['pass']}"]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/tmp/claude-1000/lumen_smoke")
    ap.add_argument("--qa-dir", default=str(REPO / "results/qa"))
    ap.add_argument("--ucf", action="store_true", help="only lumen_ucf (300 steps + resume); merged into the existing lumen_smoke.json")
    a = ap.parse_args(argv)
    root, qa = Path(a.root), Path(a.qa_dir)
    root.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.ucf:
        res = json.loads((qa / "lumen_smoke.json").read_text())
        res.setdefault("resumes", {"xception": res.pop("resume", {})})
        res["smokes"] = [s for s in res["smokes"] if s["name"] != "lumen_ucf"]
        res["smokes"].append(smoke(root, "lumen_ucf", "ucf", "lumen", 300, 100))
        res["resumes"]["ucf"] = resume_check(root, "ucf")
        res["ucf_note"] = "lumen_ucf smoked separately (sites encoder_f.block3 + encoder_f.bn4, user decision 2026-11-19)."
        res["total_wall_min"] += (time.time() - t0) / 60
        res["pass"] = all(s["pass"] for s in res["smokes"]) and all(r.get("pass") for r in res["resumes"].values())
        (qa / "lumen_smoke.json").write_text(json.dumps(res, indent=1, default=str))
        (qa / "lumen_smoke.md").write_text(markdown(res))
        print(f"ucf smoke: pass={res['smokes'][-1]['pass']} img/s={res['smokes'][-1]['img_s_steady']} resume={res['resumes']['ucf'].get('pass')}")
        return 0
    plan = [("lumen_xception", "xception", "lumen", 300, 100), ("lumen_efficientnetb4", "efficientnetb4", "lumen", 300, 100),
            ("lumen_f3net", "f3net", "lumen", 300, 100), ("bnft_xception", "xception", "bnft", 100, 50),
            ("lumen_nocons_xception", "xception", "lumen_nocons", 100, 50)]
    smokes = []
    for name, det, var, steps, ms in plan:
        print(f"[run] {name}", flush=True)
        smokes.append(smoke(root, name, det, var, steps, ms))
        print(f"[done] {name} pass={smokes[-1]['pass']} img/s={smokes[-1]['img_s_steady']}", flush=True)
    print("[run] resume", flush=True)
    resume = resume_check(root)
    from dcfg.detectors.adapter import Detector
    bn_params = Detector("xception", device="cpu", load_weights=False).freeze_except_bn()
    res = {"root": str(root), "smokes": smokes, "resumes": {"xception": resume}, "bnft_params": bn_params,
           "ucf_note": "lumen_ucf smoke added by --ucf (sites decided 2026-11-19: encoder_f.block3 + encoder_f.bn4).",
           "total_wall_min": (time.time() - t0) / 60}
    res["pass"] = all(s["pass"] for s in smokes) and bool(resume.get("pass"))
    (qa / "lumen_smoke.json").write_text(json.dumps(res, indent=1, default=str))
    (qa / "lumen_smoke.md").write_text(markdown(res))
    print(f"overall pass: {res['pass']}  ({res['total_wall_min']:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
