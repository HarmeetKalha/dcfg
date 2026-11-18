"""Pilot report (validation split only): AUC table, paired dcfg-vs-baselines deltas, resource summary, wall time.

    .venv/bin/python ops/jobs/pilot_report.py [--eval-dir results/pilot/eval] [--runs-dir runs/pilot]
        [--out results/pilot/pilot_report.md] [--no-notify]

Reads ``<eval-dir>/scores/pilot_<model>_xception__<cond>.npz`` (per-frame scores written by
``dcfg.eval.evaluate --split val``) and ``<runs-dir>/<run>/{resources_summary.json,done.json}``.
Everything is bootstrapped over VIDEOS (as ``dcfg.eval.metrics``); the degraded-mean and the paired deltas
resample the same videos jointly across conditions (and across models for deltas). Writes only the markdown
report and a summary text next to it; then sends ntfy (text, then the sanity grid) unless DCFG_NO_NOTIFY=1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dcfg.eval.metrics import VideoTable, _auc  # noqa: E402

MODELS = ["base", "fixed", "augft", "dcfg"]
DEGRADED = ["noise_s10", "blur_s3", "webp_q20", "mixed_heavy"]
GPU_UTIL_MIN, WAIT_SHARE_MAX = 85.0, 0.15


def run_id(model: str, det: str = "xception") -> str:
    return f"pilot_{model}_{det}"


def load_scores(eval_dir: Path, models: list[str], conds: list[str], det: str = "xception"
                ) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """({(model, cond): frame_prob}, {cond: (frame_key, labels, videos)}). Missing files are skipped. Within a
    condition the frame order must be identical across models (paired analysis); different conditions may cover
    different frames (c40 has its own frames)."""
    probs: dict[tuple[str, str], np.ndarray] = {}
    ref: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for c in conds:
        for m in models:
            p = eval_dir / "scores" / f"{run_id(m, det)}__{c}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=False)
            if c not in ref:
                ref[c] = (z["frame_key"], z["frame_label"], z["frame_video"])
            elif not np.array_equal(ref[c][0], z["frame_key"]):
                raise ValueError(f"{p}: frame order differs from the other models for condition {c}")
            probs[(m, c)] = z["frame_prob"]
    if not ref:
        raise FileNotFoundError(f"no score files under {eval_dir}/scores")
    return probs, ref


def joint_bootstrap(probs: dict[tuple[str, str], np.ndarray], labels: np.ndarray, videos: np.ndarray,
                    deg_conds: list[str], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Video-level bootstrap with ONE resample shared by all models and conditions.

    Returns {"point": {(m,c): auc}, "boot": {(m,c): array}, "deg_point": {m: x}, "deg_boot": {m: array}}.
    ``deg`` = mean over the degraded conditions available for that model (clean excluded)."""
    t = VideoTable.build(labels, videos)
    keys = list(probs)
    point = {k: _auc(labels, probs[k]) for k in keys}
    rng = np.random.default_rng(seed)
    nv = len(t.ids)
    boot = {k: np.full(n_boot, np.nan) for k in keys}
    for b in range(n_boot):
        idx = np.concatenate([t.frame_idx[i] for i in rng.integers(0, nv, nv)])
        yl = labels[idx]
        for k in keys:
            boot[k][b] = _auc(yl, probs[k][idx])
    models = sorted({m for m, _ in keys})
    deg_point, deg_boot = {}, {}
    for m in models:
        cs = [c for c in deg_conds if (m, c) in probs]
        if cs:
            deg_point[m] = float(np.mean([point[(m, c)] for c in cs]))
            deg_boot[m] = np.mean([boot[(m, c)] for c in cs], axis=0)
    return {"point": point, "boot": boot, "deg_point": deg_point, "deg_boot": deg_boot, "n_videos": nv}


def pct(a: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    a = a[~np.isnan(a)]
    lo, hi = np.percentile(a, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_delta(res: dict[str, Any], a: str, b: str, cond: str | None) -> tuple[float, float, float] | None:
    """(delta, lo, hi) of AUC(a) - AUC(b) for one condition or (cond=None) the degraded mean."""
    if cond is None:
        if a not in res["deg_point"] or b not in res["deg_point"]:
            return None
        d, db = res["deg_point"][a] - res["deg_point"][b], res["deg_boot"][a] - res["deg_boot"][b]
    else:
        if (a, cond) not in res["point"] or (b, cond) not in res["point"]:
            return None
        d, db = res["point"][(a, cond)] - res["point"][(b, cond)], res["boot"][(a, cond)] - res["boot"][(b, cond)]
    lo, hi = pct(db)
    return float(d), lo, hi


def cis_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True iff the closed intervals share a point (disjoint if either lies entirely above the other)."""
    return not (a[0] > b[1] or b[0] > a[1])


def verdict(res: dict[str, Any], deg_conds: list[str], top: str = "dcfg", others: tuple[str, ...] = ("augft", "fixed")
            ) -> tuple[str, str, str]:
    """(YES|NO|MIXED, paired-CI line, plain-CI-overlap line).

    YES: degraded-mean delta CI > 0 vs every comparator and no degraded condition has a delta CI < 0.
    NO: degraded-mean delta CI does not exclude 0 on the positive side vs any comparator.
    MIXED: anything else (wins vs some comparators only, or wins on the mean but loses a condition)."""
    wins, losses, lines, overlap = [], [], [], []
    for o in others:
        dm = paired_delta(res, top, o, None)
        if dm is None:
            continue
        wins.append(dm[1] > 0)
        cl = [paired_delta(res, top, o, c) for c in deg_conds]
        n_w = sum(1 for x in cl if x and x[1] > 0)
        n_l = sum(1 for x in cl if x and x[2] < 0)
        losses.append(n_l > 0)
        lines.append(f"{top}-{o}: degraded-mean {dm[0]:+.4f} [{dm[1]:+.4f}, {dm[2]:+.4f}]; per-condition CI>0 in {n_w}, "
                     f"CI<0 in {n_l} of {len(cl)}")
        a_ci, o_ci = pct(res["deg_boot"][top]), pct(res["deg_boot"][o])
        overlap.append(f"{top} vs {o}: degraded-mean CIs {'overlap' if cis_overlap(a_ci, o_ci) else 'do NOT overlap'}")
    if not wins:
        return "NO", "no comparison available", "no comparison available"
    v = "YES" if all(wins) and not any(losses) else ("NO" if not any(wins) else "MIXED")
    return v, "; ".join(lines), "; ".join(overlap)


def fmt_ci(p: float, lo: float, hi: float) -> str:
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def auc_table(res: dict[str, Any], models: list[str], conds: list[str]) -> list[str]:
    head = ["model", *conds, "mean(degraded)"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in models:
        cells = [m]
        for c in conds:
            k = (m, c)
            cells.append(fmt_ci(res["point"][k], *pct(res["boot"][k])) if k in res["point"] else "n/a")
        cells.append(fmt_ci(res["deg_point"][m], *pct(res["deg_boot"][m])) if m in res["deg_point"] else "n/a")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def delta_table(res: dict[str, Any], deg_conds: list[str], extra: tuple[str, ...] = ()) -> list[str]:
    cols = [*deg_conds, "mean(degraded)", *(["c40 (val)"] if "c40" in extra else [])]
    rows = ["| comparison | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for o in ("augft", "fixed"):
        cells = []
        for c in [*deg_conds, None, *extra]:
            d = paired_delta(res, "dcfg", o, c)
            cells.append("n/a" if d is None else f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" + (" *" if d[1] > 0 or d[2] < 0 else ""))
        rows.append(f"| dcfg - {o} | " + " | ".join(cells) + " |")
    return rows


def resource_section(runs_dir: Path, models: list[str]) -> tuple[list[str], bool, list[str]]:
    head = ["run", "GPU util mean/median % (train)", "data-wait share", "CPU util mean %", "peak GPU mem MiB",
            "max temp C", "wall train/val (h)", "CPU-bound?"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    any_flag, why = False, []
    for m in models:
        p = runs_dir / run_id(m) / "resources_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        g, w = s.get("gpu_util_mean_train"), s.get("data_wait_share_mean")
        flag = (g is not None and g < GPU_UTIL_MIN) or (w is not None and w > WAIT_SHARE_MAX)
        any_flag |= flag
        if flag:
            why.append(f"{m}: GPU util {g:.0f}% / data-wait {100 * (w or 0):.0f}%")
        f = lambda x, fmt="{:.1f}": "n/a" if x is None else fmt.format(x)  # noqa: E731
        rows.append(f"| {run_id(m)} | {f(g)} / {f(s.get('gpu_util_median_train'))} | "
                    f"{f(w and 100 * w)}% | {f(s.get('cpu_util_mean_train'))} | {f(s.get('gpu_mem_peak_mib'), '{:.0f}')} | "
                    f"{f(s.get('gpu_temp_max_c'), '{:.0f}')} | {s['wall_train_s'] / 3600:.2f} / {s['wall_val_s'] / 3600:.2f} | "
                    f"{'YES' if flag else 'no'} |")
    return rows, any_flag, why


FIX_TEXT = ("Proposed (not applied): if flagged, the data loader is the bottleneck (JPEG/WebP round trips, blur, noise run on "
            "6 CPU workers). Options: (1) move the degradation augmentation onto the GPU (kornia/torch blur + noise, "
            "differentiable-free JPEG via DCT quantisation) keeping the per-sample seeded RNG; (2) raise `num_workers` "
            "/ `prefetch_factor` if CPU is not saturated; (3) cache decoded uint8 crops in RAM/pinned memory. "
            "Any change must keep the degradation distribution (PLAN 7.2) identical.")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def wall_time(runs_root: Path, pilot_runs_dir: Path, models: list[str]) -> tuple[str, float | None]:
    """Total pilot wall time: runner-log span of the pilot_* jobs; also the sum of done.json total_time_s."""
    parts, total = [], None
    log = runs_root / "_runner.log"
    if log.exists():
        starts, ends = [], []
        for line in log.read_text().splitlines():
            f = line.split()
            if len(f) >= 4 and f[1] == "job" and f[2] == "start" and f[3].startswith("pilot_"):
                starts.append((_ts(f[0]), f[3]))
            elif len(f) >= 7 and f[1] == "job" and f[2].startswith("pilot_") and f[-2] == "exit":
                ends.append(_ts(f[0]))
        if starts:
            t0 = min(t for t, _ in starts)
            rep = [t for t, j in starts if j == "pilot_report"]
            t1 = max(rep) if rep else (max(ends) if ends else None)
            if t1:
                total = (t1 - t0).total_seconds()
                parts.append(f"runner log span (first pilot job start to pilot_report start): {total / 3600:.2f} h")
    tr = 0.0
    for m in models:
        d = pilot_runs_dir / run_id(m) / "done.json"
        if d.exists():
            tr += float(json.loads(d.read_text()).get("total_time_s", 0.0))
    if tr:
        parts.append(f"sum of training total_time_s (done.json): {tr / 3600:.2f} h")
        total = total if total is not None else tr
    return ("; ".join(parts) or "n/a (no runner log / done.json found)"), total


def build_report(eval_dir: Path, runs_dir: Path, runs_root: Path, n_boot: int = 1000, seed: int = 0
                 ) -> tuple[str, dict[str, Any]]:
    conds_seen = sorted({p.stem.split("__", 1)[1] for p in (eval_dir / "scores").glob("pilot_*__*.npz")})
    conds = [c for c in ["clean", *DEGRADED, "c40"] if c in conds_seen]
    deg = [c for c in conds if c not in ("clean", "c40")]
    probs, meta = load_scores(eval_dir, MODELS, conds)
    main_c = [c for c in conds if c != "c40"]
    labels, videos = meta[main_c[0]][1], meta[main_c[0]][2]
    if not all(np.array_equal(meta[main_c[0]][0], meta[c][0]) for c in main_c):
        raise ValueError("synthetic conditions must share the same frames (c23 val)")
    res = joint_bootstrap({k: v for k, v in probs.items() if k[1] != "c40"}, labels, videos, deg, n_boot, seed)
    if "c40" in conds:  # real compression: its own frames/videos, own resample (paired across models); not in the degraded mean
        r40 = joint_bootstrap({k: v for k, v in probs.items() if k[1] == "c40"}, meta["c40"][1], meta["c40"][2],
                              [], n_boot, seed)
        res["point"] |= r40["point"]
        res["boot"] |= r40["boot"]
        res["n_videos_c40"] = r40["n_videos"]
    models = [m for m in MODELS if any((m, c) in probs for c in conds)]
    v, paired_line, overlap_line = verdict(res, deg)
    rrows, flagged, why = resource_section(runs_dir, ["fixed", "augft", "dcfg"])
    wt, total = wall_time(runs_root, runs_dir, ["fixed", "augft", "dcfg"])
    verdict_line = f"dcfg beats augft and fixed beyond the CIs on degraded conditions: {v}"
    L = ["# DCFG pilot report (VALIDATION split only, PILOT - not paper numbers)", "",
         f"{res['n_videos']} val videos, {len(labels)} frames per condition; {n_boot} video-level bootstrap replicates (95% percentile CI). "
         "Models: Xception detector; base = frozen detector, fixed/dcfg = front-end + frozen detector, augft = fine-tuned detector; "
         "5 epochs of the real 30-epoch schedule. Not used for selection of anything on the test split.", "",
         "## (a) Frame AUC [95% CI]", "", *auc_table(res, models, conds), "",
         f"Degraded mean = mean over {', '.join(deg)} (synthetic degradations of the c23 frames; c40 is not part of it); its CI resamples "
         "videos jointly across conditions. c40 (val, real H.264 compression, no synthetic op) has its own video set and resample.", "",
         "## (b) Paired bootstrap delta-AUC (dcfg minus baseline); `*` = CI excludes 0", "", *delta_table(res, deg, ("c40",) if "c40" in conds else ()), "",
         f"**{verdict_line}**", "",
         f"- Paired-CI basis: {paired_line}", f"- Plain CI-overlap view: {overlap_line}",
         "- Rule: YES = degraded-mean delta CI > 0 against both and no degraded condition with delta CI < 0; "
         "NO = degraded-mean delta CI not > 0 against either; MIXED = otherwise.", "",
         "## (c) GPU / CPU summary (training phase only; validation excluded)", "", *rrows, "",
         f"Flag rule: CPU-bound if mean GPU util < {GPU_UTIL_MIN:.0f}% or data-wait share > {100 * WAIT_SHARE_MAX:.0f}%. "
         + (f"**FLAGGED: {'; '.join(why)}.** " + FIX_TEXT if flagged else "No run flagged." ), "",
         "## (d) Pilot wall time", "", wt, "",
         "## (e) c40", "",
         ("c40 val preprocessing failed/missing -> c40 skipped (no usable c40 val frames in FaceForensics++.json; the test "
          "split is never used as a fallback)." if "c40" not in conds else
          f"c40 VAL frames present: included as its own column ({res['n_videos_c40']} videos, real compression, no synthetic op).")]
    summary = {"verdict": v, "verdict_line": verdict_line,
               "deg_mean": {m: res["deg_point"][m] for m in models if m in res["deg_point"]},
               "cpu_bound": flagged, "cpu_bound_why": why, "wall_s": total}
    return "\n".join(L) + "\n", summary


def summary_text(s: dict[str, Any]) -> str:
    deg = ", ".join(f"{m} {a:.4f}" for m, a in s["deg_mean"].items())
    wall = f"{s['wall_s'] / 3600:.1f} h" if s["wall_s"] else "n/a"
    cpu = "CPU-bound FLAG (" + "; ".join(s["cpu_bound_why"]) + ")" if s["cpu_bound"] else "not CPU-bound"
    return f"PILOT done. {s['verdict_line']}\nDegraded-mean val frame AUC: {deg}\n{cpu}\nTotal pilot wall time: {wall}\nReport: results/pilot/pilot_report.md"


def notify(text: str, attach: Path | None) -> None:
    """Text message first, then the sanity grid as a separate attachment (both honour DCFG_NO_NOTIFY / DCFG_NOTIFY_LOG)."""
    ops = REPO / "ops"
    subprocess.run(["bash", str(ops / "notify.sh"), text, "DCFG pilot"], check=False, timeout=60)
    if attach and attach.exists():
        subprocess.run(["bash", str(ops / "notify_file.sh"), str(attach)], check=False, timeout=120)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", default=str(REPO / "results/pilot/eval"))
    ap.add_argument("--runs-dir", default=str(REPO / "runs/pilot"))
    ap.add_argument("--runs-root", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "results/pilot/pilot_report.md"))
    ap.add_argument("--grid", default=str(REPO / "results/pilot/sanity_grid.png"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args(argv)
    if Path(a.out).resolve().is_relative_to((REPO / "results" / "raw").resolve()):
        raise SystemExit("refusing to write into results/raw")
    md, s = build_report(Path(a.eval_dir), Path(a.runs_dir), Path(a.runs_root), a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md)
    txt = summary_text(s)
    print(txt)
    if not a.no_notify:
        notify(txt, Path(a.grid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
