"""Per-group LUMEN findings: tables (CSV/MD/TeX), figures (PNG/PDF) and FINDINGS_group<N>.md.

    python ops/jobs/group_findings.py --group 1 [--raw-dir results/raw] [--out results] [--n-boot 1000]

All numbers come from results/raw/<run_id>__<cond>.json (single-condition AUC + CI, taken verbatim) and
results/raw/scores/<run_id>__<cond>.npz (aggregate means and all deltas, recomputed here).

Bootstrap: ONE shared video-level resample matrix (multinomial video counts, seed --seed) is applied to every
run, condition and metric. Hence (a) means across conditions are bootstrapped jointly (mean of the per-condition AUCs
in each resample, never an average of CIs), and (b) every comparison between two runs is paired. This is the same
estimator as dcfg.eval.metrics (resample videos with replacement, percentile CI) in a weighted, vectorised form;
point estimates are checked against the result JSONs and flagged on mismatch.

Pilot runs (run id ``pilot_*`` or JSON ``"pilot": true``) are never read. Missing runs are reported, not fatal.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dcfg.analysis.baseline_report import DET_LABEL, DETS, is_pilot, load_conditions  # noqa: E402

KEY = ["clean", "synth_mean", "heldout_mean", "c40"]
ITEM_LABEL = {"clean": "clean", "synth_mean": "synthetic-degraded mean", "heldout_mean": "held-out mean", "c40": "c40"}
FAMILIES = [("JPEG quality", ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "jpeg_q10"],
             ["clean", "90", "70", "50", "30", "10"]),
            ("Blur $\\sigma$", ["clean", "blur_s1", "blur_s2", "blur_s3", "blur_s4"], ["0", "1", "2", "3", "4"]),
            ("Noise $\\sigma$", ["clean", "noise_s5", "noise_s10", "noise_s15", "noise_s20"], ["0", "5", "10", "15", "20"]),
            ("Resize scale", ["clean", "resize_0.5", "resize_0.25"], ["1", "0.5", "0.25"])]
# Okabe-Ito (colour-blind safe), same family as baseline_report; marker + linestyle give a second encoding.
STYLE = {"base": ("#000000", "o", "--"), "fixed": ("#E69F00", "s", "-."), "augft": ("#0072B2", "^", "-"),
         "lumen": ("#D55E00", "D", "-"), "bnft": ("#009E73", "v", ":"), "jointft": ("#CC79A7", "P", ":"),
         "dcfg": ("#56B4E9", "o", "-"), "dcfgcons": ("#999999", "s", "-"), "nocons": ("#E69F00", "^", "--"),
         "nolive": ("#009E73", "v", "--"), "frozensensor": ("#CC79A7", "P", "--")}
SEED_LS = {"s42": "-", "s43": "--", "s44": ":"}
META_KEYS = {"best_epoch": ["best_epoch"], "early_stop_epoch": ["early_stop_epoch", "early_stopped_epoch", "stopped_epoch"],
             "wall_time_s": ["train_wall_time_s", "wall_time_s", "train_time_s", "train_seconds", "training_wall_time_s"],
             "trainable_mb": ["trainable_mb", "trainable_module_mb", "trainable_size_mb", "module_mb"],
             "detector_mb": ["detector_mb", "full_detector_mb", "detector_size_mb"],
             "params": ["trainable_params", "n_params", "params", "params_trainable"],
             "val_mean_auc": ["val_mean_auc"]}


# ----------------------------------------------------------------------------------------- group definitions
def group_spec(n: int) -> dict:
    """{'dets': {det: [(label, run_id)]}, 'cmps': {det: [(a_label, b_label)]}, 'title': str}."""
    def rid(label, det):
        return f"{label}_{det}"
    if n == 1:
        dets = {d: [(m, rid(m, d)) for m in ("base", "fixed", "augft", "lumen")] for d in DETS}
        cmps = {d: [("lumen", "base"), ("lumen", "fixed"), ("lumen", "augft")] for d in DETS}
        return dict(dets=dets, cmps=cmps, title="G1: LUMEN vs fixed vs augft vs base, four detectors")
    x = "xception"
    ref = [("base", f"base_{x}"), ("fixed", f"fixed_{x}"), ("augft", f"augft_{x}"), ("lumen", f"lumen_{x}")]
    if n == 2:
        runs = ref + [("jointft", f"jointft_{x}"), ("bnft", f"bnft_{x}")]
        cm = [("lumen", "base"), ("lumen", "fixed"), ("lumen", "augft"), ("lumen", "jointft"), ("lumen", "bnft"),
              ("jointft", "base"), ("bnft", "base")]
        return dict(dets={x: runs}, cmps={x: cm}, title="G2: joint fine-tuning and BN-only fine-tuning (Xception); "
                    "base/fixed/augft/lumen are G1 reference runs")
    if n == 3:
        runs = [("base", f"base_{x}"), ("augft", f"augft_{x}"), ("lumen", f"lumen_{x}"), ("dcfg", f"dcfg_{x}"),
                ("dcfgcons", f"dcfgcons_{x}"), ("nocons", f"lumen_nocons_{x}"), ("nolive", f"lumen_nolive_{x}"),
                ("frozensensor", f"lumen_frozensensor_{x}")]
        cm = [("dcfgcons", "dcfg"), ("lumen", "dcfgcons"), ("lumen", "dcfg"), ("nocons", "lumen"), ("nolive", "lumen"),
              ("frozensensor", "lumen"), ("lumen", "base"), ("lumen", "augft")]
        return dict(dets={x: runs}, cmps={x: cm}, title="G3: ablation chain dcfg -> dcfgcons -> lumen and LUMEN ablations (Xception)")
    if n == 4:
        runs = [("base", f"base_{x}"), ("lumen_s42", f"lumen_{x}"), ("lumen_s43", f"lumen_{x}_s43"),
                ("lumen_s44", f"lumen_{x}_s44"), ("augft_s42", f"augft_{x}"), ("augft_s43", f"augft_{x}_s43"),
                ("augft_s44", f"augft_{x}_s44")]
        cm = [(f"lumen_{s}", f"augft_{s}") for s in ("s42", "s43", "s44")] + [(f"lumen_{s}", "base") for s in ("s42", "s43", "s44")]
        return dict(dets={x: runs}, cmps={x: cm}, title="G4: seeds 42/43/44, LUMEN vs augft (Xception)")
    raise SystemExit(f"unknown group {n}")


def style(label: str):
    if label in STYLE:
        return STYLE[label]
    fam, _, seed = label.partition("_")
    col, mk, _ = STYLE.get(fam, ("#777777", "o", "-"))
    return col, mk, SEED_LS.get(seed, "-")


def pretty(label: str) -> str:
    return label.replace("_", " ")


# ----------------------------------------------------------------------------------------- loading
class Run:
    """One run id: JSONs per condition (verbatim) plus score arrays."""

    def __init__(self, label, run_id, raw: Path, conds: list[str]):
        self.label, self.run_id, self.raw = label, run_id, raw
        self.docs: dict[str, dict] = {}
        self.sources: list[Path] = []
        self.scores: dict[str, dict] = {}
        self.missing_json: list[str] = []
        self.missing_npz: list[str] = []
        self.pilot_skipped = False
        for c in conds:
            p = raw / f"{run_id}__{c}.json"
            if is_pilot(run_id):
                self.pilot_skipped = True
                continue
            if not p.exists():
                self.missing_json.append(c)
                continue
            try:
                doc = json.load(open(p))
            except ValueError:
                self.missing_json.append(c)
                continue
            if is_pilot(doc=doc):
                self.pilot_skipped = True
                self.missing_json.append(c)
                continue
            self.docs[c] = doc
            self.sources.append(p)
            sp = raw / doc.get("scores_file", f"scores/{run_id}__{c}.npz")
            if not sp.exists():
                sp = raw / "scores" / f"{run_id}__{c}.npz"
            if sp.exists():
                z = np.load(sp, allow_pickle=False)
                self.scores[c] = dict(video=np.asarray(z["frame_video"]).astype(str), label=np.asarray(z["frame_label"]).astype(int),
                                      prob=np.asarray(z["frame_prob"], dtype=np.float64))
                self.sources.append(sp)
            else:
                self.missing_npz.append(c)

    @property
    def complete(self) -> bool:
        return not self.missing_json and not self.missing_npz

    def meta(self) -> dict:
        """Training metadata from the result JSON(s), searched (depth <= 3) under any key spelling in META_KEYS."""
        out = {}
        docs = [self.docs[c] for c in ("clean", *self.docs) if c in self.docs]
        for k, names in META_KEYS.items():
            out[k] = None
            for doc in docs:
                v = _find(doc, names, 3)
                if v is not None:
                    out[k] = v
                    break
        return out


def _find(d, names, depth):
    if not isinstance(d, dict):
        return None
    for n in names:
        if n in d and not isinstance(d[n], (dict, list)):
            return d[n]
    if depth > 0:
        for v in d.values():
            r = _find(v, names, depth - 1)
            if r is not None:
                return r
    return None


# ----------------------------------------------------------------------------------------- bootstrap
def _wauc(prob, label, idx, W, chunk=100):
    """AUC for each row of W (video-count weights, shape (B,V)); ties handled with average ranks."""
    order = np.argsort(prob, kind="stable")
    p, lab, ix = prob[order], label[order].astype(np.float64), idx[order]
    starts = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
    out = np.empty(W.shape[0])
    for s in range(0, W.shape[0], chunk):
        wf = W[s:s + chunk][:, ix]
        pos = wf * lab
        neg = wf - pos
        pg = np.add.reduceat(pos, starts, axis=1)
        ng = np.add.reduceat(neg, starts, axis=1)
        cum = np.cumsum(ng, axis=1) - ng
        num = (pg * (cum + 0.5 * ng)).sum(1)
        den = pos.sum(1) * neg.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + chunk] = np.where(den > 0, num / den, np.nan)
    return out


def curves(run: Run, universe: np.ndarray, W: np.ndarray) -> dict:
    """{(metric, cond): array (B+1,)}: row 0 = point estimate (unit weights), rows 1.. = shared bootstrap."""
    uidx = {v: i for i, v in enumerate(universe)}
    res = {}
    for c, s in run.scores.items():
        vi = np.array([uidx[v] for v in s["video"]])
        res[("frame", c)] = _wauc(s["prob"], s["label"], vi, W)
        uv, inv = np.unique(vi, return_inverse=True)
        cnt = np.bincount(inv)
        vprob = np.bincount(inv, weights=s["prob"]) / cnt
        vlab = np.zeros(len(uv), int)
        vlab[inv] = s["label"]
        res[("video", c)] = _wauc(vprob, vlab, uv, W)
    return res


def pct(arr):
    a = arr[1:]
    a = a[~np.isnan(a)]
    lo, hi = np.percentile(a, [2.5, 97.5]) if len(a) else (np.nan, np.nan)
    return float(arr[0]), float(lo), float(hi)


class Group:
    def __init__(self, n, raw: Path, n_boot=1000, seed=0, conditions=None):
        self.n, self.raw = n, raw
        self.spec = group_spec(n)
        cl = conditions if conditions is not None else load_conditions()
        self.conds = cl
        self.names = [c["name"] for c in cl]
        self.synth = [c["name"] for c in cl if c["name"] != "clean" and not c.get("held_out") and c.get("synthetic", True)]
        self.held = [c["name"] for c in cl if c.get("held_out")]
        self.runs: dict[str, dict[str, Run]] = {}
        for det, lst in self.spec["dets"].items():
            self.runs[det] = {lab: Run(lab, rid, raw, self.names) for lab, rid in lst}
        vids = sorted({v for d in self.runs.values() for r in d.values() for s in r.scores.values() for v in set(s["video"])})
        self.universe = np.array(vids)
        rng = np.random.default_rng(seed)
        V = max(len(vids), 1)
        W = np.vstack([np.ones((1, V)), rng.multinomial(V, np.full(V, 1.0 / V), size=n_boot).astype(np.float64)])
        self.W, self.n_boot, self.seed = W, n_boot, seed
        self.cur = {(det, lab): curves(r, self.universe, W) for det, d in self.runs.items() for lab, r in d.items()}
        self.flags: list[str] = []
        self._flags()

    # -- values
    def arr(self, det, lab, metric, item):
        cur = self.cur[(det, lab)]
        conds = self.synth if item == "synth_mean" else self.held if item == "heldout_mean" else [item]
        if any((metric, c) not in cur for c in conds):
            return None
        return np.mean([cur[(metric, c)] for c in conds], axis=0)

    def val(self, det, lab, metric, item):
        """(auc, lo, hi): single conditions verbatim from the JSON; aggregates from the joint bootstrap."""
        r = self.runs[det][lab]
        if item in ("synth_mean", "heldout_mean"):
            a = self.arr(det, lab, metric, item)
            return None if a is None else pct(a)
        d = r.docs.get(item)
        if d is None:
            return None
        return d[f"{metric}_auc"], d[f"{metric}_auc_ci"][0], d[f"{metric}_auc_ci"][1]

    def delta(self, det, a, b, metric, item):
        x, y = self.arr(det, a, metric, item), self.arr(det, b, metric, item)
        if x is None or y is None:
            return None
        d, lo, hi = pct(x - y)
        return d, lo, hi, bool(lo > 0 or hi < 0)

    # -- consistency
    def _flags(self):
        f = self.flags
        for det, d in self.runs.items():
            for lab, r in d.items():
                if is_pilot(r.run_id) or r.pilot_skipped:
                    f.append(f"PILOT excluded: {r.run_id}")
                if not r.docs:
                    f.append(f"MISSING RUN {r.run_id}: no result JSON found")
                    continue
                if r.missing_json:
                    f.append(f"MISSING CONDITIONS {r.run_id}: {len(r.missing_json)}/{len(self.names)} JSON missing ({', '.join(r.missing_json[:6])}{'...' if len(r.missing_json) > 6 else ''}); aggregates/deltas involving it are n/a")
                if r.missing_npz:
                    f.append(f"MISSING SCORES {r.run_id}: npz missing for {', '.join(r.missing_npz[:6])}")
                for c, doc in r.docs.items():
                    t = f"{r.run_id}/{c}"
                    if doc.get("run_id") not in (None, r.run_id) or doc.get("condition") not in (None, c):
                        f.append(f"{t}: id fields mismatch ({doc.get('run_id')}, {doc.get('condition')})")
                    for m in ("frame", "video"):
                        a, (lo, hi) = doc[f"{m}_auc"], doc[f"{m}_auc_ci"]
                        if not lo <= a <= hi:
                            f.append(f"{t}: {m} AUC {a:.4f} outside its CI [{lo:.4f}, {hi:.4f}]")
                        if a < 0.5:
                            f.append(f"SUSPICIOUS {t}: {m} AUC {a:.4f} below chance")
                        cur = self.cur[(det, lab)].get((m, c))
                        if cur is not None and abs(cur[0] - a) > 1e-6:
                            f.append(f"{t}: {m} AUC from npz {cur[0]:.6f} != JSON {a:.6f}")
                    cl = r.docs.get("clean")
                    if cl and c != "clean" and c != "c40" and doc["frame_auc"] > cl["frame_auc"] + 0.005:
                        f.append(f"SUSPICIOUS {t}: frame AUC {doc['frame_auc']:.4f} above clean {cl['frame_auc']:.4f} by > 0.005")
                    if doc.get("bootstrap") != (cl or doc).get("bootstrap"):
                        f.append(f"{t}: bootstrap settings differ from clean")
                    if doc.get("n_frames") != (cl or doc)["n_frames"] and c != "c40":
                        f.append(f"{t}: n_frames {doc.get('n_frames')} differs from clean {cl['n_frames']}")
        # cross-run checks per condition: same frame / video counts
        for det, d in self.runs.items():
            for c in self.names:
                cnt = {(r.docs[c].get("n_frames"), r.docs[c].get("n_videos")) for r in d.values() if c in r.docs}
                if len(cnt) > 1:
                    f.append(f"{det}/{c}: frame/video counts differ across runs: {sorted(cnt, key=str)}")
        for det, d in self.runs.items():
            for lab, r in d.items():
                m = r.meta()
                if r.docs and lab != "base" and all(v is None for v in m.values()):
                    f.append(f"NO TRAINING METADATA in result JSON of {r.run_id} (best_epoch, wall time, sizes not recorded)")


# ----------------------------------------------------------------------------------------- formatting
def fmt(v, nd=4):
    return "n/a" if v is None else f"{v[0]:.{nd}f} [{v[1]:.{nd}f}, {v[2]:.{nd}f}]"


def fmtd(d, nd=4):
    return "n/a" if d is None else f"{d[0]:+.{nd}f} [{d[1]:+.{nd}f}, {d[2]:+.{nd}f}]{'*' if d[3] else ''}"


def tex_esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def md_table(hdr, rows):
    return ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def tex_table(hdr, rows):
    return [r"\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", r"\toprule", " & ".join(tex_esc(h) for h in hdr) + r" \\", r"\midrule"] + \
           [" & ".join(tex_esc(c) for c in r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        else:
            f.write("")


def item_order(g: Group):
    return KEY + [c for c in g.names if c not in ("clean", "c40")]


def write_tables(g: Group, tdir: Path):
    n, md_main = g.n, {}
    # per-detector AUC tables
    for det, d in g.runs.items():
        labs = list(d)
        rows = []
        for item in item_order(g):
            for lab in labs:
                row = dict(detector=det, item=item, model=lab, run_id=d[lab].run_id,
                           source="bootstrap(npz)" if item.endswith("_mean") else "json")
                for m in ("frame", "video"):
                    v = g.val(det, lab, m, item)
                    row.update({f"{m}_auc": v[0] if v else "", f"{m}_lo": v[1] if v else "", f"{m}_hi": v[2] if v else ""})
                rows.append(row)
        write_csv(tdir / f"group{n}_auc_{det}.csv", rows)
        md = [f"# Group {n}: {DET_LABEL[det]} AUC with 95% CI (FF++ test)", "",
              "Single conditions: CI verbatim from results/raw JSON (video-level bootstrap). Means: joint paired video-level "
              "bootstrap over conditions (npz). n/a = run or condition missing.", ""]
        tx = []
        for m in ("frame", "video"):
            hdr = ["Condition"] + [pretty(l) for l in labs]
            body = [[ITEM_LABEL.get(it, it)] + [fmt(g.val(det, l, m, it), 3) for l in labs] for it in item_order(g)]
            md += [f"## {m.capitalize()} AUC", ""] + md_table(hdr, body) + [""]
            tx += tex_table(hdr, body) + [""]
        (tdir / f"group{n}_auc_{det}.md").write_text("\n".join(md) + "\n")
        (tdir / f"group{n}_auc_{det}.tex").write_text("\n".join(tx) + "\n")
    # deltas
    drows, md, tx = [], [f"# Group {n}: paired bootstrap delta-AUC (a minus b)", "",
                         "Shared video-level resample for both runs; * = 95% CI excludes 0. Key rows first; per-condition rows follow.", ""], []
    for m in ("frame", "video"):
        for det, cm in g.spec["cmps"].items():
            for a, b in cm:
                for item in item_order(g):
                    if a in g.runs[det] and b in g.runs[det]:
                        r = g.delta(det, a, b, m, item)
                        drows.append(dict(detector=det, metric=m, a=a, b=b, item=item,
                                          delta="" if r is None else r[0], lo="" if r is None else r[1],
                                          hi="" if r is None else r[2], ci_excludes_0="" if r is None else int(r[3])))
        for det, cm in g.spec["cmps"].items():
            hdr = ["Comparison (a vs b)"] + [ITEM_LABEL[k] for k in KEY]
            body = [[f"{pretty(a)} vs {pretty(b)}"] + [fmtd(g.delta(det, a, b, m, k)) for k in KEY] for a, b in cm]
            body = [[f"{DET_LABEL[det]}: " + r[0]] + r[1:] for r in body]
            md += [f"## {m.capitalize()} AUC, key aggregates, {DET_LABEL[det]}", ""] + md_table(hdr, body) + [""]
            tx += tex_table(hdr, body) + [""]
    for det, cm in g.spec["cmps"].items():
        hdr = ["Condition"] + [f"{pretty(a)} vs {pretty(b)}" for a, b in cm]
        body = [[it] + [fmtd(g.delta(det, a, b, "frame", it), 3) for a, b in cm] for it in item_order(g)[len(KEY):]]
        md += [f"## Frame AUC, every condition, {DET_LABEL[det]}", ""] + md_table(hdr, body) + [""]
    write_csv(tdir / f"group{n}_deltas.csv", drows)
    (tdir / f"group{n}_deltas.md").write_text("\n".join(md) + "\n")
    (tdir / f"group{n}_deltas.tex").write_text("\n".join(tx) + "\n")
    # training metadata
    hdr = ["run_id", "best_epoch", "early_stop_epoch", "train wall time (h)", "trainable MB", "detector MB", "trainable / detector (%)", "params", "val mean AUC"]
    mrows, body = [], []
    for det, d in g.runs.items():
        for lab, r in d.items():
            if not r.docs:
                body.append([r.run_id, "MISSING"] + [""] * 7)
                continue
            m = r.meta()
            wt = m["wall_time_s"] / 3600 if isinstance(m["wall_time_s"], (int, float)) else None
            ratio = 100 * m["trainable_mb"] / m["detector_mb"] if isinstance(m["trainable_mb"], (int, float)) and isinstance(m["detector_mb"], (int, float)) and m["detector_mb"] else None
            mrows.append(dict(run_id=r.run_id, **{k: ("" if v is None else v) for k, v in m.items()}))
            f = lambda v, p=2: "n/a" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))  # noqa: E731
            body.append([r.run_id, f(m["best_epoch"]), f(m["early_stop_epoch"]), f(wt), f(m["trainable_mb"]), f(m["detector_mb"]),
                         f(ratio, 1), f(m["params"]), f(m["val_mean_auc"], 4)])
    write_csv(tdir / f"group{n}_train_meta.csv", mrows)
    (tdir / f"group{n}_train_meta.md").write_text("\n".join([f"# Group {n}: training metadata (from result JSONs)", ""] + md_table(hdr, body)) + "\n")
    (tdir / f"group{n}_train_meta.tex").write_text("\n".join(tex_table(hdr, body)) + "\n")
    return body, hdr


def seed_stats(g: Group):
    """G4: per key aggregate, mean +- std (ddof=1) of frame AUC over available seeds, and the lumen-augft verdict."""
    det = "xception"
    out = []
    for k in KEY:
        row = dict(item=k)
        for fam in ("lumen", "augft"):
            v = [g.val(det, f"{fam}_{s}", "frame", k) for s in ("s42", "s43", "s44")]
            v = [x[0] for x in v if x is not None]
            row[fam] = v
        L, A = row["lumen"], row["augft"]
        row["n"] = (len(L), len(A))
        row["lm"], row["ls"] = (float(np.mean(L)), float(np.std(L, ddof=1)) if len(L) > 1 else float("nan")) if L else (None, None)
        row["am"], row["as"] = (float(np.mean(A)), float(np.std(A, ddof=1)) if len(A) > 1 else float("nan")) if A else (None, None)
        if L and A:
            row["diff"] = row["lm"] - row["am"]
            spread = np.nanmax([row["ls"], row["as"]]) if len(L) > 1 and len(A) > 1 else float("nan")
            row["spread"] = float(spread)
            row["exceeds"] = bool(abs(row["diff"]) > spread) if not np.isnan(spread) else None
            pairs = []
            for sd in ("s42", "s43", "s44"):
                lv, av = g.val(det, f"lumen_{sd}", "frame", k), g.val(det, f"augft_{sd}", "frame", k)
                if lv and av:
                    pairs.append(lv[0] - av[0])
            row["same_sign"] = bool(len(pairs) > 1 and len({np.sign(x) for x in pairs}) == 1)
        out.append(row)
    return out


def write_seed_tables(g: Group, tdir: Path, st):
    hdr = ["item", "LUMEN mean +- std", "augft mean +- std", "seeds (L/A)", "diff of means", "seed spread (max std)", "diff exceeds spread", "same sign in every paired seed"]
    body, rows = [], []
    for r in st:
        pm = lambda m, s: "n/a" if m is None else f"{m:.4f} +- {s:.4f}"  # noqa: E731
        ex = "n/a" if r.get("exceeds") is None else ("yes" if r["exceeds"] else "NO")
        ss = "n/a" if "same_sign" not in r else ("yes" if r["same_sign"] else "NO")
        body.append([ITEM_LABEL[r["item"]], pm(r["lm"], r["ls"]), pm(r["am"], r["as"]), f"{r['n'][0]}/{r['n'][1]}",
                     "n/a" if "diff" not in r else f"{r['diff']:+.4f}", "n/a" if "spread" not in r else f"{r['spread']:.4f}", ex, ss])
        rows.append(dict(item=r["item"], lumen_mean=r["lm"], lumen_std=r["ls"], augft_mean=r["am"], augft_std=r["as"],
                         n_lumen=r["n"][0], n_augft=r["n"][1], diff=r.get("diff"), spread=r.get("spread"), exceeds=r.get("exceeds"), same_sign=r.get("same_sign")))
    write_csv(tdir / "group4_seeds.csv", rows)
    (tdir / "group4_seeds.md").write_text("\n".join(["# Group 4: seed mean +- std (frame AUC, Xception)", "",
                                                      "std uses ddof=1 over seeds 42/43/44; 'exceeds' = |diff of means| > max(std_lumen, std_augft)."] + [""] + md_table(hdr, body)) + "\n")
    (tdir / "group4_seeds.tex").write_text("\n".join(tex_table(hdr, body)) + "\n")
    return hdr, body


# ----------------------------------------------------------------------------------------- figures
def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    return plt


def make_figures(g: Group, fdir: Path):
    plt = _rc()
    fdir.mkdir(parents=True, exist_ok=True)
    n = g.n
    for det, d in g.runs.items():
        fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.3), sharey=True)
        any_line = False
        for ax, (title, cs, labs) in zip(axes, FAMILIES):
            x = np.arange(len(cs))
            for lab, r in d.items():
                if not all(c in r.docs for c in cs):
                    continue
                col, mk, ls = style(lab)
                y = np.array([r.docs[c]["frame_auc"] for c in cs])
                lo = np.array([r.docs[c]["frame_auc_ci"][0] for c in cs])
                hi = np.array([r.docs[c]["frame_auc_ci"][1] for c in cs])
                ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
                ax.plot(x, y, color=col, marker=mk, ms=3, lw=1.1, ls=ls, label=pretty(lab))
                any_line = True
            ax.set_xticks(x)
            ax.set_xticklabels(labs)
            ax.set_xlabel(title)
            ax.set_ylim(0.4, 1.0)
            ax.axhline(0.5, color="0.6", lw=0.5, ls=(0, (2, 2)))
            ax.grid(axis="y", lw=0.3, color="0.85")
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Frame AUC")
        if any_line:
            h, l = axes[0].get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", ncol=min(len(l), 7), frameon=False, bbox_to_anchor=(0.5, 1.0))
        else:
            fig.text(0.5, 0.5, "no complete runs", ha="center")
        fig.suptitle(f"{DET_LABEL[det]}", y=0.0, va="bottom", fontsize=7)
        fig.subplots_adjust(top=0.82, bottom=0.27, left=0.07, right=0.99, wspace=0.08)
        fig.savefig(fdir / f"group{n}_curves_{det}.png", dpi=300)
        fig.savefig(fdir / f"group{n}_curves_{det}.pdf")
        plt.close(fig)
    # main bars
    items = ["synth_mean", "heldout_mean", "c40"]
    titles = ["Synthetic-degraded mean", "Held-out mean", "c40 (real compression)"]
    dets = list(g.runs)
    models = list(dict.fromkeys(l for d in g.runs.values() for l in d))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)
    for ax, it, ti in zip(axes, items, titles):
        if len(dets) == 1:
            det = dets[0]
            for i, lab in enumerate(models):
                v = g.val(det, lab, "frame", it)
                if v is None:
                    continue
                ax.bar(i, v[0], 0.7, color=style(lab)[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], error_kw=dict(lw=0.6, capsize=1.2, ecolor="0.25"))
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([pretty(m) for m in models], rotation=60, ha="right")
        else:
            w = 0.8 / len(models)
            for i, lab in enumerate(models):
                xs, ys, es = [], [], [[], []]
                for j, det in enumerate(dets):
                    v = g.val(det, lab, "frame", it) if lab in g.runs[det] else None
                    if v is None:
                        continue
                    xs.append(j + (i - (len(models) - 1) / 2) * w)
                    ys.append(v[0])
                    es[0].append(v[0] - v[1])
                    es[1].append(v[2] - v[0])
                if xs:
                    ax.bar(xs, ys, w * 0.92, color=style(lab)[0], yerr=es, error_kw=dict(lw=0.6, capsize=1.0, ecolor="0.25"), label=pretty(lab))
            ax.set_xticks(range(len(dets)))
            ax.set_xticklabels([DET_LABEL[d] for d in dets], rotation=20, ha="right")
        ax.set_title(ti)
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="0.6", lw=0.5, ls=(0, (2, 2)))
        ax.grid(axis="y", lw=0.3, color="0.85")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Frame AUC")
    if len(dets) > 1:
        h, l = axes[0].get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=0.78, bottom=0.2, left=0.07, right=0.99, wspace=0.08)
    else:
        fig.subplots_adjust(top=0.9, bottom=0.3, left=0.07, right=0.99, wspace=0.08)
    fig.savefig(fdir / f"group{n}_main_bars.png", dpi=300)
    fig.savefig(fdir / f"group{n}_main_bars.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------- findings
def verdicts(g: Group):
    """Honest per-comparison verdicts on frame AUC for the key items; returns (gains, no_gain) bullet lists."""
    gains, none = [], []
    for det, cm in g.spec["cmps"].items():
        for a, b in cm:
            for k in KEY:
                r = g.delta(det, a, b, "frame", k)
                if r is None:
                    continue
                txt = f"{DET_LABEL[det]}: {pretty(a)} vs {pretty(b)} on {ITEM_LABEL[k]}: {fmtd(r)}"
                (gains if r[0] > 0 and r[3] else none).append(txt + ("" if r[3] else " (CI includes 0)" if r[0] >= 0 else " (CI includes 0, point estimate negative)") + (" WORSE (CI < 0)" if r[0] < 0 and r[3] else ""))
    return gains, none


def write_findings(g: Group, out: Path, body_meta, hdr_meta, seed=None, nboot=1000, summary=None):
    n = g.n
    L = [f"# LUMEN findings, group {n}", "", f"{g.spec['title']}.", "",
         f"All numbers from results/raw/*.json (single conditions) and results/raw/scores/*.npz (means, deltas; {nboot} shared video-level "
         f"bootstrap resamples, seed {g.seed}). Pilot results excluded. Generated by ops/jobs/group_findings.py --group {n}.", ""]
    L += ["## Run status", ""]
    for det, d in g.runs.items():
        for lab, r in d.items():
            st = "complete" if r.complete else ("MISSING" if not r.docs else f"INCOMPLETE ({len(r.docs)}/{len(g.names)} conditions)")
            L.append(f"- {r.run_id}: {st}")
    L += ["", "## Flags (inconsistencies, missing runs, suspicious values)", ""]
    L += [f"- {x}" for x in g.flags] or ["- none"]
    L += ["", "## Main comparison: key aggregates (frame AUC [95% CI]; * = paired delta CI excludes 0)", ""]
    for det, d in g.runs.items():
        hdr = ["Model"] + [ITEM_LABEL[k] for k in KEY]
        L += [f"### {DET_LABEL[det]}", ""] + md_table(hdr, [[pretty(l)] + [fmt(g.val(det, l, "frame", k), 4) for k in KEY] for l in d]) + [""]
        for m in ("frame", "video"):
            cm = g.spec["cmps"][det]
            L += [f"Paired delta, {m} AUC:", ""] + md_table(["a vs b"] + [ITEM_LABEL[k] for k in KEY],
                                                            [[f"{pretty(a)} vs {pretty(b)}"] + [fmtd(g.delta(det, a, b, m, k)) for k in KEY] for a, b in cm]) + [""]
    gains, none = verdicts(g)
    L += ["## Where the method helps (frame AUC, paired CI excludes 0, positive)", ""] + [f"- {x}" for x in gains] + ([] if gains else ["- none"]) + [""]
    L += ["## Where it does not help (CI includes 0, or negative)", ""] + [f"- {x}" for x in none] + ([] if none else ["- none"]) + [""]
    if n == 3:
        L += ["## Ablation chain and ablations vs lumen (frame AUC delta, key aggregates)", ""]
        cm = g.spec["cmps"]["xception"][:6]
        L += md_table(["a vs b"] + [ITEM_LABEL[k] for k in KEY], [[f"{pretty(a)} vs {pretty(b)}"] + [fmtd(g.delta("xception", a, b, "frame", k)) for k in KEY] for a, b in cm]) + [""]
    if n == 4 and seed is not None:
        hdr, body = seed
        L += ["## Seed spread (Xception, frame AUC)", ""] + md_table(hdr, body) + [""]
        ex = [r for r in body if r[-2] == "yes"]
        L += [f"Lumen-vs-augft difference exceeds the seed spread on {len(ex)}/{len(body)} key aggregates "
              f"(definition: |diff of means| > max of the two seed std; only 3 seeds, so std is itself noisy).", ""]
    L += ["## Training metadata", ""] + md_table(hdr_meta, body_meta) + [""]
    if summary:
        L += ["## Notification summary", ""] + [f"- {s}" for s in summary] + [""]
    L += ["## Source files", "", "Tables: results/tables/group%d_*; figures: results/figures/group%d_*." % (n, n), ""]
    srcs = sorted({str(p) for d in g.runs.values() for r in d.values() for p in r.sources})
    L += [f"- {s}" for s in srcs] or ["- (no source files found)"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")


def build_summary(g: Group) -> list[str]:
    x = "xception"
    s = []

    def line(a, b):
        if x not in g.runs or a not in g.runs[x] or b not in g.runs[x]:
            return None
        ds, dc = g.delta(x, a, b, "frame", "synth_mean"), g.delta(x, a, b, "frame", "c40")
        return f"Xception {a} vs {b}: synth-degraded {fmtd(ds, 3)}, c40 {fmtd(dc, 3)}"
    pairs = {1: [("lumen", "augft"), ("lumen", "base")], 2: [("jointft", "base"), ("bnft", "base"), ("lumen", "jointft")],
             3: [("lumen", "dcfg"), ("lumen", "dcfgcons"), ("nocons", "lumen")], 4: [("lumen_s42", "augft_s42"), ("lumen_s43", "augft_s43"), ("lumen_s44", "augft_s44")]}[g.n]
    s += [t for t in (line(a, b) for a, b in pairs) if t]
    if g.n == 1:
        cnt = {k: 0 for k in ("synth_mean", "c40")}
        have = 0
        for det in g.runs:
            for k in cnt:
                r = g.delta(det, "lumen", "fixed", "frame", k)
                if r is not None:
                    have += 1 if k == "synth_mean" else 0
                    cnt[k] += int(r[3] and r[0] > 0)
        s.append(f"LUMEN beats fixed beyond CI (frame): synth-degraded mean in {cnt['synth_mean']}/{len(g.runs)} detectors, c40 in {cnt['c40']}/{len(g.runs)}")
    n_miss = sum(1 for d in g.runs.values() for r in d.values() if not r.complete)
    s.append(f"{n_miss} run(s) missing/incomplete; {len(g.flags)} flag(s)")
    return s[:5]


def notify(g: Group, summary: list[str]):
    sh = REPO / "ops" / "notify.sh"
    if os.environ.get("DCFG_NO_NOTIFY") == "1" and not os.environ.get("DCFG_NOTIFY_LOG"):
        return
    if sh.exists():
        subprocess.run(["bash", str(sh), "\n".join(summary), f"LUMEN group {g.n} done"], check=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--raw-dir", default=str(REPO / "results" / "raw"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    g = Group(a.group, Path(a.raw_dir), a.n_boot, a.seed)
    body, hdr = write_tables(g, out / "tables")
    seed = None
    if a.group == 4:
        seed = write_seed_tables(g, out / "tables", seed_stats(g))
    make_figures(g, out / "figures")
    summary = build_summary(g)
    write_findings(g, out / f"FINDINGS_group{a.group}.md", body, hdr, seed, a.n_boot, summary)
    print(f"wrote {out / f'FINDINGS_group{a.group}.md'}; {len(g.flags)} flag(s)")
    notify(g, summary)
    return g


if __name__ == "__main__":
    main()
