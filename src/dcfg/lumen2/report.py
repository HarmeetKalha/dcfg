"""LUMEN2 evaluation report -> results/lumen2/lumen2_eval.md (+ lumen2_eval.json, read by the pilot report).

    python -m dcfg.lumen2.report [--n-boot 1000] [--out results/lumen2/lumen2_eval.md]

Every pre-declared metric (docs/LUMEN2_BRIEF.md) as paired LUMEN2 - GenD-aug with 95% video-level bootstrap CIs from ONE shared
resample matrix (``dcfg.lumen2.analysis``), win/loss/tie by the pre-declared rule, safety checks, the "worth publishing" rule, exploratory
metrics (AP, EER, every condition), sensor accuracy per condition (+ c40 / WebP / motion), DFD (if evaluated) and cost.
Reads results/lumen2/{gend_aug,lumen2,gend_public} (+ results/dfd, runs/lumen2_train/done.json, results/lumen2/lumen2_cost.json).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dcfg.lumen2 import analysis as A
from dcfg.paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT

R = RESULTS_ROOT / "lumen2"
DFD_DIR = RESULTS_ROOT / "dfd"
DFD_CONDS = ["clean", "mixed_heavy"]
BASE, NEW = "GenD-aug", "LUMEN2"


class Study:
    """Runs + shared bootstrap curves for a set of models on one video universe (FF++ test or DFD)."""

    def __init__(self, models: dict[str, tuple[str, Path]], conds: list[str], n_boot: int, seed: int = 0) -> None:
        self.gf = A.load_group_findings()
        self.conds = conds
        self.runs = {lab: self.gf.Run(lab, rid, d, conds) for lab, (rid, d) in models.items()}
        for lab, r in self.runs.items():
            if not r.complete:
                raise RuntimeError(f"{lab}: incomplete results (missing json {r.missing_json}, npz {r.missing_npz})")
        self.universe = np.array(sorted({v for r in self.runs.values() for s in r.scores.values() for v in set(s["video"])}))
        self.W = A.resample_matrix(len(self.universe), n_boot, seed)
        self.cur = {lab: self.gf.curves(r, self.universe, self.W) for lab, r in self.runs.items()}

    def curve(self, lab: str, metric: str, cond: str) -> np.ndarray:
        return self.cur[lab][(metric, cond)]

    def diff(self, new: str, base: str, metric: str, cond: str) -> tuple[float, float, float]:
        return A.pct(self.curve(new, metric, cond) - self.curve(base, metric, cond))


def f4(v: tuple[float, float, float] | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    s = "+" if signed else ""
    return f"{v[0]:{s}.4f} [{v[1]:{s}.4f}, {v[2]:{s}.4f}]"


def degraded_frames(run: Any, universe: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """probs, labels, video index of all 22 degraded conditions concatenated (fixed order)."""
    uidx = {v: i for i, v in enumerate(universe)}
    p = np.concatenate([run.scores[c]["prob"] for c in A.DEGRADED])
    y = np.concatenate([run.scores[c]["label"] for c in A.DEGRADED])
    v = np.array([uidx[x] for c in A.DEGRADED for x in run.scores[c]["video"]])
    return p, y, v


def check_same_frames(a: Any, b: Any, conds: list[str], raw_a: Path, raw_b: Path, ra: str, rb: str) -> None:
    """Paired comparison is only valid on identical frames: compare the frame_key arrays of every condition."""
    for c in conds:
        ka = np.load(raw_a / f"scores/{ra}__{c}.npz")["frame_key"]
        kb = np.load(raw_b / f"scores/{rb}__{c}.npz")["frame_key"]
        if not np.array_equal(ka, kb):
            raise RuntimeError(f"frame sets differ in {c}: {len(ka)} vs {len(kb)}")


def paired_metrics(st: Study, base: str, new: str, n_boot: int) -> dict[str, Any]:
    """All FF++-based pre-declared metrics: {key: {base, new, diff, outcome}} (values as (point, lo, hi))."""
    agg = {lab: A.aggregate_curves(st.cur[lab]) for lab in (base, new)}
    out: dict[str, Any] = {}
    for k in A.CURVE_METRICS:
        out[k] = {"base": A.pct(agg[base][k]), "new": A.pct(agg[new][k]), "diff": A.pct(agg[new][k] - agg[base][k])}
    pa, ya, va = degraded_frames(st.runs[base], st.universe)
    pn, yn, vn = degraded_frames(st.runs[new], st.universe)
    assert np.array_equal(ya, yn) and np.array_equal(va, vn)
    fb = A.frame_bootstrap(A.FRAME_FNS, pa, pn, ya, va, st.W)
    for k, (cb, cn) in fb.items():
        out[k] = {"base": A.pct(cb), "new": A.pct(cn), "diff": A.pct(cn - cb)}
    out["_n_degraded_frames"] = int(len(ya))
    return out


def dfd_metrics(base: str, new: str, runs: dict[str, str], n_boot: int, d: Path = DFD_DIR) -> dict[str, Any] | None:
    """DFD frame/video AUC with paired LUMEN2 - GenD-aug (None if the DFD results of the two models do not exist yet)."""
    if not (A.exists_all(d, runs[base], DFD_CONDS) and A.exists_all(d, runs[new], DFD_CONDS)):
        return None
    st = Study({base: (runs[base], d), new: (runs[new], d)}, DFD_CONDS, n_boot)
    out: dict[str, Any] = {"n_videos": int(len(st.universe))}
    for metric in ("frame", "video"):
        for c in DFD_CONDS:
            out[f"{metric}_{c}"] = {"base": A.pct(st.curve(base, metric, c)), "new": A.pct(st.curve(new, metric, c)),
                                    "diff": st.diff(new, base, metric, c)}
    out["dfd_clean"], out["dfd_mixed_heavy"] = out["frame_clean"], out["frame_mixed_heavy"]
    return out


# ------------------------------------------------------------------------------------------------ exploratory
def ap_eer(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(average_precision_score(y, p)), float((fpr[i] + fnr[i]) / 2)


def exploratory(st: Study, base: str, new: str) -> dict[str, Any]:
    per, ap = {}, {}
    for c in A.ALL_CONDS:
        per[c] = {"frame_auc": {"base": A.pct(st.curve(base, "frame", c)), "new": A.pct(st.curve(new, "frame", c)),
                                "diff": st.diff(new, base, "frame", c)}, "video_auc_diff": st.diff(new, base, "video", c)}
        res = {}
        for lab in (base, new):
            s = st.runs[lab].scores[c]
            res[lab] = ap_eer(s["prob"], s["label"])
        ap[c] = {"AP": (res[base][0], res[new][0]), "EER": (res[base][1], res[new][1])}
    return {"per_condition": per, "ap_eer": ap}


# ------------------------------------------------------------------------------------------------ sensor
def sensor_section(run: str, out_dir: Path, standalone: Path | None) -> dict[str, Any] | None:
    from dcfg.data.degradations import load_conditions
    from dcfg.degrade.eval_damage_sensor import analyse
    from dcfg.eval.evaluate import CONDITIONS_YAML

    d = out_dir / "sensor"
    if not all((d / f"{run}__{c}.npz").exists() for c in A.ALL_CONDS):
        return None
    conds = load_conditions(CONDITIONS_YAML)
    preds = {}
    for c in A.ALL_CONDS:
        z = np.load(d / f"{run}__{c}.npz")
        preds[c] = (z["presence"], z["strength"])
    res = analyse(preds, conds)
    if standalone and standalone.exists():
        res["standalone_per_op"] = json.loads(standalone.read_text())["per_op"]
    return res


def sensor_md(s: dict[str, Any], gf: Any) -> list[str]:
    from dcfg.degrade.labels import OPS

    L = ["Presence threshold 0.5; labelled conditions pooled (clean + 17 synthetic), strength metrics only where the op is applied "
         "(strength scale 0..1 over the training range). The sensor here is trained jointly with LUMEN2 (loss weight 0.1).", ""]
    sa = s.get("standalone_per_op")
    hdr = ["op", "presence acc", "presence F1", "strength MAE", "Spearman"] + (["standalone F1", "standalone MAE"] if sa else [])
    rows = []
    for op, r in s["per_op"].items():
        sp = r["strength_spearman"]
        row = [op, f"{r['presence_acc']:.3f}", f"{r['presence_f1']:.3f}", f"{r['strength_mae']:.3f}", "n/a" if sp is None else f"{sp:.3f}"]
        if sa:
            row += [f"{sa[op]['presence_f1']:.3f}", f"{sa[op]['strength_mae']:.3f}"]
        rows.append(row)
    L += gf.md_table(hdr, rows)
    L += ["", "Per labelled condition (true ops, exact op-set accuracy, mean P(op) in order jpeg / blur / noise / resize):", ""]
    L += gf.md_table(["condition", "true ops", "exact set acc", "mean P(jpeg/blur/noise/resize)"],
                     [[c, r["true_ops"], f"{r['exact_set_acc']:.3f}", " / ".join(f"{x:.2f}" for x in r["mean_presence"])]
                      for c, r in s["per_condition"].items()])
    L += ["", "Unseen conditions (no ground truth: real compression c40, WebP, motion blur): fraction of frames with any op detected and mean "
          "P(op) per op.", ""]
    L += gf.md_table(["condition", "any op present", *[f"P({o})" for o in OPS]],
                     [[c, f"{r['frac_any_present']:.2f}", *[f"{r['mean_presence'][o]:.2f}" for o in OPS]] for c, r in s["unseen"].items()])
    return L


# ------------------------------------------------------------------------------------------------ report
def verdict(res: dict[str, Any], dfd: dict[str, Any] | None) -> dict[str, Any]:
    """Outcome per metric + the pre-declared "worth publishing" rule (LUMEN-uncond comparison is pending by design)."""
    out: dict[str, str] = {}
    for k in A.METRICS:
        if k in ("sel80", "sel90"):
            out[k] = A.outcome(k, *res[k]["diff"][1:])
        elif k in ("dfd_clean", "dfd_mixed_heavy"):
            out[k] = A.outcome(k, *dfd[k]["diff"][1:]) if dfd else "pending"
        else:
            out[k] = A.outcome(k, *res[k]["diff"][1:])
    selective = A.combine_selective(out["sel80"], out["sel90"])
    wins = [k for k, v in out.items() if v == "win" and A.METRICS[k][1] != "safety" and k not in ("sel80", "sel90")]
    if selective == "win":
        wins.append("selective (80/90)")
    primary = out["synth_mean"] == "win"
    safety = {k: out[k] for k in ("clean", "dfd_clean")}
    safety_ok = all(v == "ok" for v in safety.values())
    pending = [k for k, v in out.items() if v == "pending"]
    rule = (primary or any(w != "synth_mean" for w in wins)) and safety_ok
    return {"outcomes": out, "selective": selective, "wins": wins, "primary_win": primary, "safety": safety, "safety_ok": safety_ok,
            "pending": pending, "rule_passes_before_uncond": bool(rule) if not pending else None}


def build(res: dict[str, Any], dfd: dict[str, Any] | None, ver: dict[str, Any], expl: dict[str, Any], sens: dict[str, Any] | None,
          train: dict[str, Any] | None, cost: dict[str, Any] | None, gf: Any, n_boot: int) -> str:
    L: list[str] = []
    P = L.append
    P("# LUMEN2 evaluation: LUMEN2 minus GenD-aug (pre-declared metrics)\n")
    P(f"All numbers: frame AUC unless stated, FF++ c23 test + c40, GenD crops, degradation code/seeds/frame keys identical to GenD-aug. "
      f"Paired differences LUMEN2 - GenD-aug, 95% video-level bootstrap CI ({n_boot} resamples, ONE shared resample matrix, aggregates bootstrapped "
      f"jointly). Win = CI entirely on the favourable side of 0 (ECE: lower is better); safety checks pass unless the CI is entirely below 0. "
      f"Metric definitions: docs/LUMEN2_BRIEF.md (heavy-damage list amended 2026-11-21). The LUMEN-uncond control is NOT run yet, so this is "
      f"not yet the full publishing rule.\n")
    P("## Pre-declared metrics\n")
    rows = []
    for k, (label, role, _) in A.METRICS.items():
        src = dfd if k.startswith("dfd") else res
        if src is None or k not in src:
            rows.append([label, role, "n/a", "n/a", "n/a", "pending (DFD not evaluated yet)"])
            continue
        r = src[k]
        rows.append([label, role, f4(r["base"]), f4(r["new"]), f4(r["diff"], True), ver["outcomes"][k]])
    L += gf.md_table(["metric", "role", BASE, NEW, f"{NEW} - {BASE}", "outcome"], rows)
    P(f"\nSelective-AUC rule (win if either coverage wins): **{ver['selective']}**. ECE is computed over {res['_n_degraded_frames']} frames "
      f"(22 degraded conditions concatenated: 17 synthetic + 4 held-out + c40).\n")
    P("## Verdict on the pre-declared rule\n")
    P(f"- Primary (synthetic-degraded mean): **{ver['outcomes']['synth_mean']}**")
    P(f"- Secondary wins: {', '.join(ver['wins']) if ver['wins'] else 'none'}")
    P(f"- Safety: clean c23 {ver['safety']['clean']}, clean DFD {ver['safety']['dfd_clean']}")
    if ver["rule_passes_before_uncond"] is None:
        P(f"- Rule status: undecided, pending {', '.join(ver['pending'])}")
    else:
        P(f"- \"Worth publishing\" rule (wins primary or >= 1 secondary, passes both safety checks): "
          f"**{'PASSES' if ver['rule_passes_before_uncond'] else 'FAILS'}** - still requires beating LUMEN-uncond on the won metric (not run).")
    P("\n## Exploratory (never used as a claim)\n")
    P("Per-condition frame AUC, paired difference LUMEN2 - GenD-aug [95% CI], video-AUC difference, and AP / EER (point estimates, no CI):\n")
    rows = []
    for c in A.ALL_CONDS:
        e, x = expl["per_condition"][c], expl["ap_eer"][c]
        rows.append([c, f"{e['frame_auc']['base'][0]:.4f}", f"{e['frame_auc']['new'][0]:.4f}", f4(e["frame_auc"]["diff"], True),
                     f4(e["video_auc_diff"], True), f"{x['AP'][0]:.4f} / {x['AP'][1]:.4f}", f"{x['EER'][0]:.4f} / {x['EER'][1]:.4f}"])
    L += gf.md_table(["condition", f"{BASE} AUC", f"{NEW} AUC", "frame diff [CI]", "video diff [CI]", f"AP ({BASE} / {NEW})", f"EER ({BASE} / {NEW})"], rows)
    if dfd:
        P("\n## DFD (DeepFakeDetection, never used in training)\n")
        rows = [[f"{m} AUC, {c}", f4(dfd[f"{m}_{c}"]["base"]), f4(dfd[f"{m}_{c}"]["new"]), f4(dfd[f"{m}_{c}"]["diff"], True)]
                for m in ("frame", "video") for c in DFD_CONDS]
        L += gf.md_table(["", BASE, NEW, "difference"], rows)
        P(f"\n{dfd['n_videos']} videos; mixed_heavy uses the same seeding rule as the FF++ conditions.")
    P("\n## Damage sensor inside LUMEN2\n")
    L += sensor_md(sens, gf) if sens else ["(sensor outputs not available)"]
    P("\n## Cost\n")
    if cost:
        P(f"- LUMEN parameters: {cost['lumen_params_total']:,} (" + ", ".join(f"{k}: {v:,}" for k, v in cost["lumen_params"].items() if k != "LUMEN parts total") + ")")
        P(f"- trainable-only file: LUMEN2 {cost['trainable_file_mb']['lumen2']:.3f} MB vs GenD-aug {cost['trainable_file_mb']['gend_aug']:.3f} MB")
        P(f"- inference (batch {cost['batch_size']}, {cost['gpu']}, {cost['precision']}): LUMEN2 {cost['lumen2']['median_ms_per_image']:.2f} ms/image vs "
          f"GenD-aug {cost['gend_aug']['median_ms_per_image']:.2f} ms/image (x{cost['overhead_ratio']:.2f})")
    else:
        P("(cost benchmark not run yet)")
    if train:
        P(f"- training: stop reason **{train['stop_reason']}**, best epoch {train['best_epoch']}, stop epoch {train['stop_epoch']} (1-based; cap {train['max_epochs']}, "
          f"patience 6); best val mean AUC {train['best_val_mean_auc']:.4f}; train {train['train_s'] / 3600:.2f} h + val {train['val_s'] / 3600:.2f} h = "
          f"{train['train_wall_time_s'] / 3600:.2f} h; {train['steps']} optimiser steps")
        if train.get("still_improving_at_cap"):
            P("- **FLAG: the run hit the epoch cap while validation was still improving.**")
        P("- val curve: " + ", ".join(f"{h['epoch']}: {h['val_mean_auc']:.4f}" for h in train["val_curve"]))
        if "part_params" in train:
            P("- parameters per part: " + ", ".join(f"{k}: {v:,}" for k, v in train["part_params"].items()))
    return "\n".join(L) + "\n"


def load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.lumen2.report", description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--aug-dir", default=str(R / "gend_aug"))
    ap.add_argument("--lumen-dir", default=str(R / "lumen2"))
    ap.add_argument("--train-dir", default=str(RUNS_ROOT / "lumen2_train"))
    ap.add_argument("--out", default=str(R / "lumen2_eval.md"))
    a = ap.parse_args(argv)
    aug, lum = Path(a.aug_dir), Path(a.lumen_dir)
    st = Study({BASE: ("gend_aug", aug), NEW: ("lumen2", lum)}, A.ALL_CONDS, a.n_boot)
    check_same_frames(None, None, A.ALL_CONDS, aug, lum, "gend_aug", "lumen2")
    res = paired_metrics(st, BASE, NEW, a.n_boot)
    dfd = dfd_metrics(BASE, NEW, {BASE: "gend_aug", NEW: "lumen2"}, a.n_boot)
    ver = verdict(res, dfd)
    expl = exploratory(st, BASE, NEW)
    sens = sensor_section("lumen2", lum, REPO_ROOT / "results/sensor/standalone.json")
    td = Path(a.train_dir)
    train = load_json(td / "done.json")
    if train:
        lum.mkdir(parents=True, exist_ok=True)
        (lum / "train_summary.json").write_text(json.dumps({k: v for k, v in train.items() if k != "train_config"}, indent=2))
        for f in ("metrics.csv", "train_log.csv"):
            if (td / f).exists():
                shutil.copy(td / f, lum / f"train_{f}")
    cost = load_json(R / "lumen2_cost.json")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(res, dfd, ver, expl, sens, train, cost, st.gf, a.n_boot))
    out.with_suffix(".json").write_text(json.dumps({"metrics": res, "dfd": dfd, "verdict": ver, "exploratory": expl, "n_boot": a.n_boot},
                                                    indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
