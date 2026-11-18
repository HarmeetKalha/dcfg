"""Phase 8 smoke / resume / timing job (PLAN 8.2, 8.3). One resumable job for the runner.

    queue line:  phase8_smoke | .venv/bin/python ops/jobs/phase8_smoke.py #hb=30
    dry run:     .venv/bin/python ops/jobs/phase8_smoke.py --quick --run-dir /tmp/x/run --qa-dir /tmp/x/qa

Sub-steps (each writes ``<run-dir>/markers/<name>.json`` when finished; a rerun skips finished ones):
  a_<variant>  300 training steps on real c23 train data (Xception) + one capped val pass (64 videos, stratified)
  b_<det>      3 steps for dcfg/fixed/augft on each detector, largest batch <= 32 for dcfg and augft,
               UCF augft (native loss) with amp true and false
  c_resume     kill an Xception dcfg run after ~250 steps, restart to 400, verify batches.log
  d_<det>_<v>  steady-state img/s (~200 steps after warmup) and val time on a ~10% sample, extrapolated
Final outputs (always regenerated): results/qa/phase8_smoke.{json,md}, phase8_batch.yaml, phase8_eta.md.
Training runs live in runs/smoke/<name>/ (W&B disabled). Smoke runs use warmup_epochs=0 so that the
lr is non-negligible within 300 steps (a 1-epoch warmup is ~6k steps); everything else is the real config.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_matrix import DETECTORS, matrix_runs  # noqa: E402

BASE_CONFIG = "configs/train/dcfg_xception.yaml"
SMOKE_VARIANTS = ["dcfg", "fixed", "nosensor", "dcfg_nolive", "dcfg_frozensensor", "augft", "dcfg_jointft"]
BATCH_TRIES = (32, 24, 16)
EPOCHS_WORST, EPOCHS_LIKELY = 30, 15
#: trainer variant -> timing class measured in step d
TIMING_CLASS = {"dcfg": "dcfg", "dcfg_nolive": "dcfg", "dcfg_jointft": "jointft", "fixed": "fixed",
                "nosensor": "fixed", "augft": "augft",
                # LUMEN matrix variants (legacy phase-8 approximation; the LUMEN ETA is ops/jobs/lumen_eta.py)
                "lumen": "dcfg", "lumen_nocons": "dcfg", "lumen_nolive": "dcfg", "lumen_frozensensor": "dcfg",
                "dcfg_cons": "dcfg", "bnft": "augft"}


# ============================================================================ pure logic
def variant_sets(name: str) -> dict[str, Any]:
    """Config overrides for a smoke variant name (``dcfg_frozensensor`` = dcfg + freeze_sensor)."""
    if name == "dcfg_frozensensor":
        return {"variant": "dcfg", "freeze_sensor": True}
    return {"variant": name}


def loss_first_last(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Mean loss of the first and the last logging window of train_log.csv (log_every steps each)."""
    if len(rows) < 2:
        return {"first": None, "last": None, "decreased": False, "windows": len(rows)}
    first, last = float(rows[0]["loss"]), float(rows[-1]["loss"])
    return {"first": first, "last": last, "decreased": bool(last < first), "windows": len(rows)}


def steady_img_s(rows: list[dict[str, str]], min_step: int) -> float | None:
    """Mean img/s over the log windows ending at step >= min_step (after warmup)."""
    v = [float(r["img_s"]) for r in rows if int(r["step"]) >= min_step]
    return sum(v) / len(v) if v else None


def is_oom(text: str) -> bool:
    return "out of memory" in text.lower()


def largest_fitting(try_fn: Callable[[int], dict[str, Any]], sizes: tuple[int, ...] = BATCH_TRIES
                    ) -> tuple[int | None, list[dict[str, Any]]]:
    """First size (descending) for which ``try_fn(bs)['ok']``; stops (no smaller tries) on a non-OOM failure."""
    log: list[dict[str, Any]] = []
    for bs in sizes:
        r = try_fn(bs)
        log.append({"batch_size": bs, **r})
        if r["ok"]:
            return bs, log
        if not r.get("oom"):
            break
    return None, log


def pick_batch(per_variant: dict[str, int | None]) -> int | None:
    """Detector batch = the smallest of the per-variant largest-fitting sizes (None if any failed)."""
    v = list(per_variant.values())
    return None if not v or any(x is None for x in v) else min(x for x in v if x is not None)


def parse_batches_log(lines: list[str]) -> tuple[list[tuple[int, int, int, str]], list[dict[str, int]]]:
    """-> ([(epoch, batch_in_epoch, step, hash)], [resume markers {epoch, step_in_epoch, step}])."""
    recs, resumes = [], []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("# resumed"):
            resumes.append({k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", ln)})
        elif ln and not ln.startswith("#"):
            e, b, s, h = ln.split()
            recs.append((int(e), int(b), int(s), h))
    return recs, resumes


def verify_resume(lines: list[str], reference: Callable[[int, int], str], total_steps: int) -> dict[str, Any]:
    """Check a killed+resumed run's batches.log against the deterministic reference permutation.

    Steps re-executed after a resume (between the last checkpoint and the kill) must reproduce the same
    hash; every step 0..total_steps-1 must appear; hashes must match ``reference(epoch, batch)``."""
    recs, resumes = parse_batches_log(lines)
    by_step: dict[int, set[str]] = {}
    for e, b, s, h in recs:
        by_step.setdefault(s, set()).add(h)
    steps = sorted(by_step)
    conflicts = [s for s in steps if len(by_step[s]) > 1]
    mism = [s for (e, b, s, h) in recs if reference(e, b) != h]
    first = {}
    for e, b, s, h in recs:  # canonical hash per step (first occurrence)
        first.setdefault(s, h)
    n_dup_hash = len(first) - len(set(first.values()))
    # the step counter continues: the first record after each resume marker is the marker's step
    cont = True
    for i, ln in enumerate(lines):
        if ln.strip().startswith("# resumed"):
            m = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", ln)}
            nxt = next((x.split() for x in lines[i + 1:] if x.strip() and not x.startswith("#")), None)
            cont &= nxt is not None and int(nxt[2]) == m["step"]
    missing = sorted(set(range(total_steps)) - set(steps))
    reexec = sorted({s for s in steps if sum(1 for r in recs if r[2] == s) > 1})
    ok = (bool(resumes) and not conflicts and not mism and not missing and n_dup_hash == 0 and cont
          and steps[-1] == total_steps - 1)
    return {"ok": ok, "n_resumes": len(resumes), "resume_markers": resumes, "steps_logged": len(steps),
            "last_step": steps[-1] if steps else None, "missing_steps": missing[:20],
            "hash_conflicts": conflicts[:20], "ref_mismatches": mism[:20], "repeated_batch_hashes": n_dup_hash,
            "counter_continues": cont,
            "reexecuted_steps": [reexec[0], reexec[-1]] if reexec else None}


def parse_eval_log(text: str) -> dict[str, float]:
    """Per-condition wall seconds from a runs/base_<det>/log.txt (last occurrence per condition)."""
    out: dict[str, float] = {}
    for m in re.finditer(r"^\s{2}(\S+): \S+\.json \((\d+)s\)\s*$", text, re.M):
        out[m.group(1)] = float(m.group(2))
    return out


def steps_per_epoch(n_fake: int, bs: int, native_ucf: bool) -> int:
    """Full epoch mode: every fake once + reals oversampled 1:1 => 2*n_fake frames (UCF native: pairs of bs/2)."""
    return n_fake // (bs // 2) if native_ucf else (2 * n_fake) // bs


def run_eta(spe: int, bs: int, img_s: float, val_epoch_s: float, eval_s: float, epochs: int) -> dict[str, float]:
    """Seconds: train per epoch, per epoch incl. val, total for ``epochs`` epochs + the test evaluation."""
    train_e = spe * bs / img_s
    epoch_s = train_e + val_epoch_s
    return {"train_epoch_s": train_e, "val_epoch_s": val_epoch_s, "epoch_s": epoch_s,
            "eval_s": eval_s, "total_s": epochs * epoch_s + eval_s}


def eval_seconds(per_cond_log: dict[str, float], n_conds: int, est_per_cond: float | None
                 ) -> tuple[float | None, str]:
    """Test-eval time of one run: from the base_<det> log (mean per condition x n_conds) else the estimate."""
    if len(per_cond_log) >= 3:
        mean = sum(per_cond_log.values()) / len(per_cond_log)
        return mean * n_conds, f"log ({len(per_cond_log)}/{n_conds} conds)"
    if est_per_cond is not None:
        return est_per_cond * n_conds, "estimate"
    return None, "missing"


def build_eta(runs: list[dict[str, Any]], bs_det: dict[str, int], rates: dict[str, dict[str, float]],
              val_full_s: dict[str, dict[str, float]], eval_det: dict[str, tuple[float | None, str]],
              n_fake: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """ETA rows for the matrix runs + totals (worst = 30 epochs, likely = 15 epochs, in GPU-days)."""
    rows, miss = [], []
    for r in runs:
        det, v = r["detector"], r["variant"]
        cls = TIMING_CLASS[v]
        img_s = rates.get(det, {}).get(cls)
        vs = val_full_s.get(det, {}).get("augft" if cls == "augft" else "dcfg")
        ev, ev_src = eval_det.get(det, (None, "missing"))
        bs = bs_det.get(det, 32)
        if not img_s or vs is None or ev is None:
            miss.append(r["run_id"])
            continue
        spe = steps_per_epoch(n_fake, bs, v == "augft" and det == "ucf")
        w, l = run_eta(spe, bs, img_s, vs, ev, EPOCHS_WORST), run_eta(spe, bs, img_s, vs, ev, EPOCHS_LIKELY)
        rows.append({"run_id": r["run_id"], "group": r.get("group", r["tier"]), "tier": r["tier"], "variant": v, "detector": det, "bs": bs,
                     "timing_class": cls, "img_s": img_s, "steps_per_epoch": spe, "eval_src": ev_src,
                     **{k: w[k] for k in ("train_epoch_s", "val_epoch_s", "epoch_s", "eval_s")},
                     "total_worst_s": w["total_s"], "total_likely_s": l["total_s"]})
    tot = {"worst_gpu_days": sum(x["total_worst_s"] for x in rows) / 86400,
           "likely_gpu_days": sum(x["total_likely_s"] for x in rows) / 86400, "n_runs": len(rows),
           "missing": miss}
    groups: dict[int, dict[str, float]] = {}
    for x in rows:
        g = groups.setdefault(x["group"], {"n_runs": 0, "worst_gpu_days": 0.0, "likely_gpu_days": 0.0})
        g["n_runs"] += 1
        g["worst_gpu_days"] += x["total_worst_s"] / 86400
        g["likely_gpu_days"] += x["total_likely_s"] / 86400
    tot["groups"] = groups
    return rows, tot


def eta_markdown(rows: list[dict[str, Any]], tot: dict[str, Any], notes: list[str]) -> str:
    h = lambda s: f"{s / 3600:.1f}"  # noqa: E731
    out = ["# Phase 8 ETA (Tiers 1-2)", "", "Worst case = 30 full epochs (no early stop). Likely case = early stop at "
           f"epoch {EPOCHS_LIKELY} (ASSUMPTION, not measured). Each run = epochs x (train + val over the 6 val "
           "conditions) + test evaluation over all conditions.", "",
           "| run | variant | det | bs | img/s | steps/ep | train/ep (min) | val/ep (min) | epoch (min) | test eval (min) "
           "| 30 ep total (h) | 15 ep total (h) |", "|" + "---|" * 12]
    for x in rows:
        out.append(f"| {x['run_id']} | {x['variant']} | {x['detector']} | {x['bs']} | {x['img_s']:.0f} | "
                   f"{x['steps_per_epoch']} | {x['train_epoch_s'] / 60:.1f} | {x['val_epoch_s'] / 60:.1f} | "
                   f"{x['epoch_s'] / 60:.1f} | {x['eval_s'] / 60:.1f} ({x['eval_src']}) | "
                   f"{h(x['total_worst_s'])} | {h(x['total_likely_s'])} |")
    names = {1: "dcfg/fixed/augft x 4 detectors", 2: "Xception ablations (nosensor, frozensensor, nolive, jointft)",
             3: "dcfg_xception extra seeds (43, 44)", 4: "fixed_xception extra seeds (43, 44)"}
    out += ["", "## Per priority group (you can stop after any group)", "",
            "| group | contents | runs | worst (GPU-days) | likely (GPU-days) | cumulative worst | cumulative likely |",
            "|" + "---|" * 7]
    cw = cl = 0.0
    for g in sorted(tot.get("groups", {})):
        d = tot["groups"][g]
        cw += d["worst_gpu_days"]; cl += d["likely_gpu_days"]
        out.append(f"| {g} | {names.get(g, '')} | {d['n_runs']} | {d['worst_gpu_days']:.2f} | "
                   f"{d['likely_gpu_days']:.2f} | {cw:.2f} | {cl:.2f} |")
    out += ["", f"**Total ({tot['n_runs']} runs): worst case {tot['worst_gpu_days']:.2f} GPU-days; likely case "
            f"(15 epochs, assumption) {tot['likely_gpu_days']:.2f} GPU-days.**"]
    if tot["missing"]:
        out += ["", f"Runs without data (not in totals): {', '.join(tot['missing'])}"]
    out += [""] + [f"- {n}" for n in notes]
    return "\n".join(out) + "\n"


def reference_hasher(labels: Any, seed: int, bs: int, mode: str, paired_half: int | None
                     ) -> Callable[[int, int], str]:
    """Deterministic expected batch hash (epoch, batch index) from the engine's own ``epoch_order``."""
    from dcfg.train.train_frontend import epoch_order
    cache: dict[int, Any] = {}

    def f(epoch: int, b: int) -> str:
        if epoch not in cache:
            cache[epoch] = epoch_order(labels, seed, epoch, mode, paired_half)
        idx = cache[epoch][b * bs:(b + 1) * bs]
        import numpy as np
        return hashlib.sha1(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]
    return f


# ============================================================================ job plumbing
class Ctx:
    """Paths, flags and the heartbeat of one job invocation."""

    def __init__(self, a: argparse.Namespace) -> None:
        from dcfg.utils.heartbeat import Heartbeat
        self.quick = a.quick
        self.run_dir = Path(a.run_dir or os.environ.get("DCFG_RUN_DIR") or REPO / "runs/phase8_smoke").expanduser()
        self.qa_dir = Path(a.qa_dir) if a.qa_dir else REPO / "results/qa"
        self.smoke_root = Path(a.smoke_root) if a.smoke_root else (
            self.run_dir / "smoke" if self.quick else REPO / "runs/smoke")
        self.markers = self.run_dir / "markers"
        for d in (self.run_dir, self.markers, self.qa_dir, self.smoke_root):
            d.mkdir(parents=True, exist_ok=True)
        self.hb = Heartbeat(self.run_dir, 60.0)
        self.stage = "init"
        # scales (quick = dry run)
        q = self.quick
        self.smoke_steps, self.log_every, self.ckpt_every = (20, 10, 10) if q else (300, 50, 100)
        self.val_videos = 8 if q else 64
        self.fit_steps = 3
        self.time_steps, self.time_log, self.time_min_step = (20, 5, 10) if q else (250, 50, 100)
        self.res_kill, self.res_end, self.res_ckpt = (12, 24, 5) if q else (250, 400, 100)

    def beat(self, **kw: Any) -> None:
        self.hb.beat(0, stage=self.stage, **kw)

    def marker(self, name: str) -> Path:
        return self.markers / f"{name}.json"

    def step(self, name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
        """Run a sub-step unless its done marker exists; write the marker on success."""
        m = self.marker(name)
        if m.exists():
            print(f"[skip] {name} (done)", flush=True)
            return json.loads(m.read_text())
        self.stage = name
        self.hb.beat(0, force=True, stage=name)
        t0 = time.time()
        print(f"[run ] {name}", flush=True)
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", flush=True)
            return None
        res["wall_s"] = round(time.time() - t0, 1)
        tmp = m.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(m)
        print(f"[done] {name} ({res['wall_s']}s)", flush=True)
        return res


def fmt(v: Any) -> str:
    return "true" if v is True else "false" if v is False else "null" if v is None else str(v)


def trainer_cmd(run_dir: Path, sets: dict[str, Any]) -> list[str]:
    return [sys.executable, "-m", "dcfg.train.train_frontend", "--config", BASE_CONFIG, "--run-dir", str(run_dir),
            "--set", *[f"{k}={fmt(v)}" for k, v in sets.items()]]


def run_trainer(ctx: Ctx, d: Path, sets: dict[str, Any], kill_at_lines: int | None = None,
                timeout_s: float = 4 * 3600) -> dict[str, Any]:
    """Run train_frontend as a subprocess (own process group) in run dir ``d``; optionally SIGTERM it once
    ``batches.log`` holds ``kill_at_lines`` step records. Keeps the job heartbeat alive while waiting."""
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "WANDB_MODE": "disabled"}
    env.pop("DCFG_RUN_DIR", None)
    out = d / "stdout.log"
    t0, killed, kill_step = time.time(), False, None
    with out.open("ab") as fo:
        p = subprocess.Popen(trainer_cmd(d, sets), cwd=REPO, env=env, stdout=fo, stderr=subprocess.STDOUT,
                             start_new_session=True)
        while p.poll() is None:
            time.sleep(0.1 if kill_at_lines and not killed else 2.0)
            bl = d / "batches.log"
            n = sum(1 for ln in bl.read_text().splitlines() if ln and ln[0] != "#") if bl.exists() else 0
            ctx.beat(run=d.name, lines=n)
            if kill_at_lines and not killed and n >= kill_at_lines and (d / "last.pt").exists():
                os.killpg(p.pid, signal.SIGTERM)
                killed, kill_step = True, n
            if time.time() - t0 > timeout_s:
                os.killpg(p.pid, signal.SIGKILL)
                break
        if killed:
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        try:  # reap workers
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    tail = "\n".join(out.read_text(errors="replace").splitlines()[-25:])
    return {"rc": p.returncode, "killed": killed, "kill_lines": kill_step, "oom": is_oom(tail), "tail": tail[-1500:],
            "wall_s": round(time.time() - t0, 1)}


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def peak_mb(rows: list[dict[str, str]]) -> float | None:
    v = [float(r["gpu_mem_mb"]) for r in rows]
    return max(v) if v else None


def read_batch_yaml(ctx: Ctx) -> dict[str, int]:
    p = ctx.qa_dir / "phase8_batch.yaml"
    return {str(k): int(v) for k, v in (yaml.safe_load(p.read_text()) or {}).items()} if p.exists() else {}


# ============================================================================ sub-steps
def step_a(ctx: Ctx, variant: str) -> dict[str, Any]:
    d = ctx.smoke_root / variant
    sets = {**variant_sets(variant), "run_id": f"smoke_{variant}", "detector": "xception", "max_steps": ctx.smoke_steps,
            "log_every": ctx.log_every, "ckpt_every_steps": ctx.ckpt_every, "val_on_truncate": True,
            "val_max_videos": ctx.val_videos, "warmup_epochs": 0, "heartbeat_s": 30}
    r = run_trainer(ctx, d, sets)
    done = json.loads((d / "done.json").read_text()) if (d / "done.json").exists() else {}
    if r["rc"] != 0 or not done:
        raise RuntimeError(f"training failed rc={r['rc']}: {r['tail'][-400:]}")
    rows = read_csv(d / "train_log.csv")
    ll = loss_first_last(rows)
    warm = [r for r in rows[1:]] or rows
    val = done.get("val", {})
    return {"variant": variant, "steps": done["step"], "loss": ll, "window_steps": ctx.log_every,
            "val_mean_auc": val.get("mean_auc"), "val_aucs": val.get("aucs"), "val_frames": val.get("n_frames"),
            "val_videos_cap": ctx.val_videos, "ckpt_written": (d / "last.pt").exists(),
            "img_s": (sum(float(x["img_s"]) for x in warm) / len(warm)) if warm else None,
            "peak_vram_mb_allocated": peak_mb(rows),
            "nonfinite_sensor": sum(int(x["nonfinite_sensor"]) for x in rows),
            "nonfinite_gate": sum(int(x["nonfinite_gate"]) for x in rows),
            "pass": bool(ll["decreased"] and (d / "last.pt").exists() and val.get("mean_auc") is not None
                         and math.isfinite(val.get("mean_auc") or float("nan")))}


def fit_try(ctx: Ctx, det: str, variant: str, bs: int, amp: bool = True) -> dict[str, Any]:
    """3 steps at batch ``bs`` (log_every=1): ok / OOM / losses finite / peak VRAM."""
    d = ctx.smoke_root / "b" / f"{det}_{variant}_bs{bs}{'' if amp else '_fp32'}"
    shutil.rmtree(d, ignore_errors=True)
    sets = {**variant_sets(variant), "run_id": f"fit_{det}_{variant}", "detector": det, "batch_size": bs,
            "max_steps": ctx.fit_steps, "log_every": 1, "amp": amp, "warmup_epochs": 0, "heartbeat_s": 30}
    r = run_trainer(ctx, d, sets, timeout_s=1800)
    rows = read_csv(d / "train_log.csv")
    losses = [float(x["loss"]) for x in rows]
    ok = r["rc"] == 0 and len(rows) >= ctx.fit_steps
    return {"ok": ok, "oom": r["oom"], "rc": r["rc"], "amp": amp, "losses": losses,
            "losses_finite": bool(losses) and all(math.isfinite(x) for x in losses),
            "peak_vram_mb_allocated": peak_mb(rows), "err": None if ok else r["tail"][-300:]}


def step_b(ctx: Ctx, det: str) -> dict[str, Any]:
    res: dict[str, Any] = {"detector": det, "search": {}, "largest": {}}
    for v in ("dcfg", "augft"):
        bs, log = largest_fitting(lambda b, v=v: fit_try(ctx, det, v, b))
        res["search"][v], res["largest"][v] = log, bs
    res["batch_size"] = pick_batch(res["largest"])
    bs = res["batch_size"] or BATCH_TRIES[-1]
    if det != "xception":  # xception variants are covered by step a
        res["fixed_one_step"] = fit_try(ctx, det, "fixed", bs)
    if det == "ucf":  # augft uses the native DFB loss: fp16 and fp32
        res["ucf_augft_native"] = {"amp_true": fit_try(ctx, det, "augft", bs, True),
                                   "amp_false": fit_try(ctx, det, "augft", bs, False)}
        res["ucf_fp16_losses_finite"] = res["ucf_augft_native"]["amp_true"]["losses_finite"]
    return res


def step_c(ctx: Ctx) -> dict[str, Any]:
    """Kill an xception dcfg run mid-epoch, resume it, verify batches.log against the epoch permutation."""
    from dcfg.data.ffpp import DFBJsonSource
    d = ctx.smoke_root / "resume_dcfg"
    p1 = ctx.marker("c_resume_phase1")
    base = {"variant": "dcfg", "run_id": "smoke_resume_dcfg", "detector": "xception", "max_steps": ctx.res_end,
            "log_every": ctx.log_every, "ckpt_every_steps": ctx.res_ckpt, "warmup_epochs": 0, "heartbeat_s": 30}
    if not p1.exists():
        shutil.rmtree(d, ignore_errors=True)
        r1 = run_trainer(ctx, d, base, kill_at_lines=ctx.res_kill)
        if not r1["killed"] or (d / "done.json").exists():
            raise RuntimeError(f"run was not killed mid-way (killed={r1['killed']}): {r1['tail'][-300:]}")
        p1.write_text(json.dumps({k: r1[k] for k in ("killed", "kill_lines", "rc")}))
    kill = json.loads(p1.read_text())
    r2 = run_trainer(ctx, d, base)
    if r2["rc"] != 0 or not (d / "done.json").exists():
        raise RuntimeError(f"resumed run failed rc={r2['rc']}: {r2['tail'][-400:]}")
    from dcfg.train.train_frontend import load_config
    cfg = load_config(BASE_CONFIG, [f"{k}={fmt(v)}" for k, v in base.items()])
    src = DFBJsonSource(cfg["json"], cfg["frames_root"])
    import numpy as np
    labels = np.array([x.label for x in src.records("train", cfg["comp"])])
    ref = reference_hasher(labels, int(cfg["seed"]), int(cfg["batch_size"]), cfg["epoch_mode"], None)
    v = verify_resume((d / "batches.log").read_text().splitlines(), ref, ctx.res_end)
    return {"kill": kill, "ckpt_every_steps": ctx.res_ckpt, "target_steps": ctx.res_end,
            "reference": "recomputed deterministic permutation (train_frontend.epoch_order), no second run", **v,
            "pass": v["ok"]}


def count_frames() -> dict[str, Any]:
    """Frame counts (no GPU): train fakes, val per condition, test per condition, #val videos."""
    from dcfg.data.degradations import load_conditions, load_val_conditions
    from dcfg.data.ffpp import DFBJsonSource
    from dcfg.eval.evaluate import condition_comp
    src = DFBJsonSource()
    tr = src.records("train", "c23")
    allc = load_conditions(REPO / "configs/eval/conditions.yaml")
    val_names = load_val_conditions(REPO / "configs/eval/conditions.yaml")
    n = {c: len(src.records("val", c)) for c in ("c23", "c40")}
    t = {c: len(src.records("test", c)) for c in ("c23", "c40")}
    return {"n_fake_train": sum(r.label for r in tr), "n_real_train": sum(1 - r.label for r in tr),
            "val_frames": {c: n[condition_comp(allc[c])] for c in val_names},
            "test_frames": {c: t[condition_comp(x)] for c, x in allc.items()},
            "n_val_videos": len({r.video for r in src.records("val", "c23")})}


def step_d(ctx: Ctx, det: str, variant: str, counts: dict[str, Any], with_val: bool) -> dict[str, Any]:
    bs = read_batch_yaml(ctx).get(det, 32)
    d = ctx.smoke_root / "d" / f"{det}_{variant}"
    cap = max(5, round(0.1 * counts["n_val_videos"])) if not ctx.quick else ctx.val_videos
    sets = {**variant_sets(variant), "run_id": f"time_{det}_{variant}", "detector": det, "batch_size": bs,
            "max_steps": ctx.time_steps, "log_every": ctx.time_log, "ckpt_every_steps": 100000, "warmup_epochs": 0,
            "heartbeat_s": 30, "val_on_truncate": with_val, "val_max_videos": cap}
    r = run_trainer(ctx, d, sets)
    done = json.loads((d / "done.json").read_text()) if (d / "done.json").exists() else {}
    if r["rc"] != 0 or not done:
        raise RuntimeError(f"timing run failed rc={r['rc']}: {r['tail'][-400:]}")
    rows = read_csv(d / "train_log.csv")
    out: dict[str, Any] = {"detector": det, "variant": variant, "batch_size": bs,
                           "img_s": steady_img_s(rows, ctx.time_min_step),
                           "steps_measured": ctx.time_steps - ctx.time_min_step + ctx.time_log,
                           "peak_vram_mb_allocated": peak_mb(rows)}
    if with_val:
        v = done["val"]
        full = counts["val_frames"]
        per = {c: v["times_s"][c] * full[c] / v["n_frames"][c] for c in v["times_s"]}
        out["val"] = {"sample_videos": cap, "sample_frames": v["n_frames"], "sample_s": v["times_s"],
                      "full_frames": full, "full_s_extrapolated": per, "full_total_s": sum(per.values()),
                      "frames_per_s": sum(v["n_frames"].values()) / max(1e-9, sum(v["times_s"].values()))}
    return out


# ============================================================================ outputs
def assemble(ctx: Ctx, markers: dict[str, dict[str, Any]], counts: dict[str, Any] | None) -> dict[str, Any]:
    """Write phase8_batch.yaml, phase8_eta.md, phase8_smoke.json/.md from whatever markers exist."""
    notes = ["Per-epoch val time = sum over the 6 val conditions at full val size, extrapolated linearly from a "
             "~10% video sample (includes DataLoader start-up per condition: slightly conservative).",
             "Timing classes: dcfg (also nolive, frozensensor: frozen sensor is slightly cheaper), fixed (also "
             "nosensor), augft, jointft (Xception only). Val time of module variants = measured dcfg val time.",
             "Test eval time is measured without the front-end; the module adds a small cost per image."]
    batch = {}
    for det in DETECTORS:
        b = markers.get(f"b_{det}")
        if b and b.get("batch_size"):
            batch[det] = b["batch_size"]
    if batch:
        (ctx.qa_dir / "phase8_batch.yaml").write_text(
            "# largest batch <= 32 that fits for BOTH dcfg and augft (3-step test, amp); written by phase8_smoke.py\n"
            + yaml.safe_dump(batch, sort_keys=False))
    eta_md = None
    if counts:
        rates: dict[str, dict[str, float]] = {}
        val_full: dict[str, dict[str, float]] = {}
        val_rate: dict[str, float] = {}
        bs_det = dict(batch)
        for k, m in markers.items():
            if k.startswith("d_"):
                rates.setdefault(m["detector"], {})[TIMING_CLASS[m["variant"]]] = m["img_s"]
                bs_det.setdefault(m["detector"], m["batch_size"])
                if "val" in m:
                    cls = "augft" if m["variant"] == "augft" else "dcfg"
                    val_full.setdefault(m["detector"], {})[cls] = m["val"]["full_total_s"]
                    if cls == "dcfg":
                        val_rate[m["detector"]] = m["val"]["frames_per_s"]
        n_conds = len(counts["test_frames"])
        tf = sum(counts["test_frames"].values()) / n_conds
        logs = {}
        for det in DETECTORS:
            p = REPO / "runs" / f"base_{det}" / "log.txt"
            logs[det] = parse_eval_log(p.read_text(errors="replace")) if p.exists() else {}
        calib = [sum(lg.values()) / len(lg) / (tf / val_rate[d]) for d, lg in logs.items()
                 if len(lg) >= 3 and d in val_rate]
        cal = sum(calib) / len(calib) if calib else 1.0
        eval_det = {det: eval_seconds(logs[det], n_conds, tf / val_rate[det] * cal if det in val_rate else None)
                    for det in DETECTORS}
        if not calib:
            notes.append("No finished base_<det> eval log yet: test eval estimated from the val-pass frame rate "
                         "(fp32 eval and the 4 data workers may make it slower).")
        rows, tot = build_eta(matrix_runs(), bs_det, rates, val_full, eval_det, counts["n_fake_train"])
        eta_md = eta_markdown(rows, tot, notes)
        (ctx.qa_dir / "phase8_eta.md").write_text(eta_md)
    # smoke report
    a = {k[2:]: m for k, m in markers.items() if k.startswith("a_")}
    b = {k[2:]: m for k, m in markers.items() if k.startswith("b_")}
    c = markers.get("c_resume")
    d = {k[2:]: m for k, m in markers.items() if k.startswith("d_")}
    report = {"quick": ctx.quick, "a_smoke": a, "b_other_detectors": b, "c_resume": c, "d_timing": d,
              "batch_sizes": batch, "finished_unix": time.time(),
              "all_pass": bool(a and all(x["pass"] for x in a.values()) and (c or {}).get("pass"))}
    (ctx.qa_dir / "phase8_smoke.json").write_text(json.dumps(report, indent=1))
    md = ["# Phase 8 smoke" + (" (QUICK dry run)" if ctx.quick else ""), "",
          "## (a) Xception, %d steps, val on %d videos (stratified)" % (ctx.smoke_steps, ctx.val_videos), "",
          "| variant | loss first | loss last | decreased | val mean AUC | ckpt | img/s | peak VRAM MB | pass |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v, x in a.items():
        ls = x["loss"]
        md.append(f"| {v} | {ls['first']:.4f} | {ls['last']:.4f} | {ls['decreased']} | {(x['val_mean_auc'] or float('nan')):.4f} | "
                  f"{x['ckpt_written']} | {x['img_s'] or 0:.0f} | {x['peak_vram_mb_allocated'] or 0:.0f} | {x['pass']} |")
    md += ["", "## (b) other detectors (3 steps) and batch size", ""]
    for det, x in b.items():
        md.append(f"- {det}: largest batch dcfg={x['largest']['dcfg']}, augft={x['largest']['augft']} -> **{x['batch_size']}**"
                  + (f"; fixed one-step ok={x['fixed_one_step']['ok']}" if "fixed_one_step" in x else "")
                  + (f"; UCF native-loss augft: amp=true ok={x['ucf_augft_native']['amp_true']['ok']} finite="
                     f"{x['ucf_augft_native']['amp_true']['losses_finite']}, amp=false ok="
                     f"{x['ucf_augft_native']['amp_false']['ok']} finite={x['ucf_augft_native']['amp_false']['losses_finite']}"
                     if "ucf_augft_native" in x else ""))
    if c:
        md += ["", "## (c) resume", "", f"- pass: {c['pass']}; killed at ~{c['kill']['kill_lines']} steps, resume markers "
               f"{c['resume_markers']}; steps logged {c['steps_logged']}/{c['target_steps']}; missing {c['missing_steps']}; "
               f"hash conflicts {c['hash_conflicts']}; ref mismatches {c['ref_mismatches']}; repeated hashes "
               f"{c['repeated_batch_hashes']}; re-executed steps (ckpt..kill) {c['reexecuted_steps']}"]
    md += ["", "## (d) timing (img/s, steady state)", ""]
    for k, x in d.items():
        md.append(f"- {k}: bs {x['batch_size']}, {x['img_s'] or 0:.0f} img/s, peak {x['peak_vram_mb_allocated'] or 0:.0f} MB"
                  + (f", val (full, extrapolated) {x['val']['full_total_s'] / 60:.1f} min" if "val" in x else ""))
    md += ["", "ETA table: phase8_eta.md", f"Overall pass (a+c): {report['all_pass']}"]
    (ctx.qa_dir / "phase8_smoke.md").write_text("\n".join(md) + "\n")
    return report


# ============================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="tiny dry run: xception dcfg only, ~20 steps")
    ap.add_argument("--run-dir", default=None, help="markers + heartbeat (default $DCFG_RUN_DIR)")
    ap.add_argument("--qa-dir", default=None, help="output dir (default results/qa)")
    ap.add_argument("--smoke-root", default=None, help="training run dirs (default runs/smoke)")
    ap.add_argument("--only", default="a,b,c,d", help="comma list of sub-step groups")
    a = ap.parse_args(argv)
    if a.quick and not (a.run_dir and a.qa_dir):
        ap.error("--quick requires --run-dir and --qa-dir (scratch dirs; never runs/ or results/)")
    ctx = Ctx(a)
    only = set(a.only.split(","))
    failed: list[str] = []
    done: dict[str, dict[str, Any]] = {}

    def go(name: str, fn: Callable[[], dict[str, Any]]) -> None:
        r = ctx.step(name, fn)
        if r is None:
            failed.append(name)
        else:
            done[name] = r

    variants = ["dcfg"] if ctx.quick else SMOKE_VARIANTS
    dets = ["xception"] if ctx.quick else DETECTORS
    if "a" in only:
        for v in variants:
            go(f"a_{v}", lambda v=v: step_a(ctx, v))
    if "b" in only and not ctx.quick:
        for det in DETECTORS:
            go(f"b_{det}", lambda det=det: step_b(ctx, det))
    if "c" in only:
        go("c_resume", lambda: step_c(ctx))
    counts = None
    if "d" in only:
        ctx.stage = "counts"
        counts = count_frames()
        # batch sizes come from step b; assemble now so step d (and gen_matrix) read phase8_batch.yaml
        assemble(ctx, {**{p.stem: json.loads(p.read_text()) for p in ctx.markers.glob("*.json")}}, None)
        for det in dets:
            classes = ["dcfg", "fixed", "augft"] + (["dcfg_jointft"] if det == "xception" else [])
            for v in (["dcfg"] if ctx.quick else classes):
                go(f"d_{det}_{v}", lambda det=det, v=v: step_d(ctx, det, v, counts, v in ("dcfg", "augft")))
    markers = {p.stem: json.loads(p.read_text()) for p in ctx.markers.glob("*.json")
               if not p.stem.endswith(("phase1",))}
    report = assemble(ctx, markers, counts if counts is not None else (count_frames() if markers else None))
    ctx.hb.beat(0, force=True, stage="finished", failed=failed)
    print(json.dumps({"failed_steps": failed, "all_pass": report["all_pass"], "qa_dir": str(ctx.qa_dir)}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
