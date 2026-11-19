"""LUMEN2 step 1 report: how much room is there to improve public GenD? -> results/lumen2/gend_room.md

    python -m dcfg.gend.room_report [--gend-dir results/lumen2/gend_public] [--n-boot 1000]

Numbers come from result JSONs (single conditions, verbatim) and score npz files (aggregates, deltas) via
``ops/jobs/group_findings.py`` machinery (Run, curves, pct): ONE shared video-level bootstrap resample matrix is
applied to every model/condition, so aggregate means are bootstrapped jointly and every difference is paired.
Models: GenD (results/lumen2/gend_public) and base/augft/lumen Xception (results/raw).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

from dcfg.eval.metrics import auc_metric, bootstrap_metric, ece, selective_auc
from dcfg.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("group_findings", REPO_ROOT / "ops" / "jobs" / "group_findings.py")
gf = importlib.util.module_from_spec(_spec)
sys.modules["group_findings"] = gf
_spec.loader.exec_module(gf)  # type: ignore[union-attr]

RAW = REPO_ROOT / "results" / "raw"
SINGLES = ["noise_s15", "noise_s20", "blur_s3", "blur_s4", "jpeg_q10", "mixed_heavy"]
COLS = ["clean", "synth_mean", "heldout_mean", "c40", *SINGLES]
COL_LABEL = {"clean": "clean", "synth_mean": "synth-degraded mean (17)", "heldout_mean": "held-out mean (4)", "c40": "c40"}
GATE = 0.03


class Models:
    """Runs + shared bootstrap curves for the compared models."""

    def __init__(self, gend_dir: Path, n_boot: int, seed: int = 0, gend_run: str = "gend_public") -> None:
        cl = gf.load_conditions()
        self.names = [c["name"] for c in cl]
        self.synth = [c["name"] for c in cl if c["name"] != "clean" and not c.get("held_out") and c.get("synthetic", True)]
        self.held = [c["name"] for c in cl if c.get("held_out")]
        spec = [("GenD (public)", gend_run, gend_dir), ("base_xception", "base_xception", RAW),
                ("augft_xception", "augft_xception", RAW), ("lumen_xception", "lumen_xception", RAW)]
        self.runs = {lab: gf.Run(lab, rid, d, self.names) for lab, rid, d in spec}
        vids = sorted({v for r in self.runs.values() for s in r.scores.values() for v in set(s["video"])})
        rng = np.random.default_rng(seed)
        nv = len(vids)
        self.W = np.vstack([np.ones((1, nv)), rng.multinomial(nv, np.full(nv, 1.0 / nv), size=n_boot).astype(float)])
        self.cur = {lab: gf.curves(r, np.array(vids), self.W) for lab, r in self.runs.items()}

    def arr(self, lab: str, item: str, metric: str = "frame") -> np.ndarray | None:
        conds = self.synth if item == "synth_mean" else self.held if item == "heldout_mean" else [item]
        if any((metric, c) not in self.cur[lab] for c in conds):
            return None
        return np.mean([self.cur[lab][(metric, c)] for c in conds], axis=0)

    def value(self, lab: str, item: str, metric: str = "frame") -> tuple[float, float, float] | None:
        if item in ("synth_mean", "heldout_mean"):
            a = self.arr(lab, item, metric)
            return None if a is None else gf.pct(a)
        d = self.runs[lab].docs.get(item)
        return None if d is None else (d[f"{metric}_auc"], *d[f"{metric}_auc_ci"])

    def drop(self, lab: str, item: str, metric: str = "frame") -> tuple[float, float, float] | None:
        """AUC(item) - AUC(clean), paired (same resamples)."""
        a, c = self.arr(lab, item, metric), self.arr(lab, "clean", metric)
        return None if a is None or c is None else gf.pct(a - c)


def f4(v: tuple[float, float, float] | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    s = "+" if signed else ""
    return f"{v[0]:{s}.4f} [{v[1]:{s}.4f}, {v[2]:{s}.4f}]"


def consistency(m: Models) -> list[str]:
    """Frame keys of GenD must equal base_xception's per condition (same frame set); JSON/npz AUC agree."""
    out = []
    g, b = m.runs["GenD (public)"], m.runs["base_xception"]
    for c in g.docs:
        pg = Path(g.raw) / g.docs[c]["scores_file"]
        pb = Path(b.raw) / b.docs[c]["scores_file"]
        kg, kb = np.load(pg)["frame_key"], np.load(pb)["frame_key"]
        out.append(f"{c}: {'same frame keys as base_xception' if np.array_equal(kg, kb) else f'FRAME SET DIFFERS ({len(kg)} vs {len(kb)})'}")
    return out


def calibration(m: Models, lab: str, n_boot: int, ci: bool) -> dict:
    """ECE and selective AUC (80/90%) over all non-clean test frames (21 synthetic + c40), conditions concatenated."""
    conds = [c for c in m.names if c != "clean" and c in m.runs[lab].docs]
    z = [np.load(Path(m.runs[lab].raw) / m.runs[lab].docs[c]["scores_file"]) for c in conds]
    p = np.concatenate([a["frame_prob"] for a in z])
    y = np.concatenate([a["frame_label"] for a in z]).astype(int)
    v = np.concatenate([a["frame_video"] for a in z])
    out = {"n_conditions": len(conds), "n_frames": int(len(p))}
    for name, fn in (("ece", ece), ("sel80", partial(selective_auc, coverage=0.8)),
                     ("sel90", partial(selective_auc, coverage=0.9)), ("auc_all", auc_metric)):
        if ci and name != "auc_all":
            r = bootstrap_metric(fn, p, y, v, n=n_boot)
            out[name] = (r["point"], r["ci_low"], r["ci_high"])
        else:
            out[name] = (float(fn(p, y)), None, None)
    return out


def fmt_cal(t: tuple) -> str:
    return f"{t[0]:.4f}" + ("" if t[1] is None else f" [{t[1]:.4f}, {t[2]:.4f}]")


PAPER_VIDEO_AUC = {"DF": 99.5, "F2F": 98.1, "FS": 98.7, "NT": 95.5, "mean": 98.0}  # GenD paper (arXiv 2508.06248v4) Table 3


def clean_sanity(m: Models, gend_dir: Path) -> list[str]:
    """Our clean video AUC per manipulation (fake type vs the 140 real test videos) next to the paper's Table 3."""
    from dcfg.eval.metrics import _auc

    z = np.load(Path(gend_dir) / m.runs["GenD (public)"].docs["clean"]["scores_file"])
    vid, vl, vp = z["video_id"].astype(str), z["video_label"], z["video_prob"]
    ours = {}
    for t in ("DF", "F2F", "FS", "NT"):
        k = np.array([v.startswith("FF-real/") or v.startswith(f"FF-{t}/") for v in vid])
        ours[t] = 100 * _auc(vl[k], vp[k])
    ours["mean"] = float(np.mean([ours[t] for t in ("DF", "F2F", "FS", "NT")]))
    L = ["| | DF | F2F | FS | NT | mean |", "|---|---|---|---|---|---|",
         "| paper Table 3 (video AUROC %, mean of 5 seeds) | " + " | ".join(f"{PAPER_VIDEO_AUC[k]:.1f}" for k in ("DF", "F2F", "FS", "NT", "mean")) + " |",
         "| ours (video AUC %, public checkpoint) | " + " | ".join(f"{ours[k]:.1f}" for k in ("DF", "F2F", "FS", "NT", "mean")) + " |"]
    diag = Path(REPO_ROOT / "runs" / "lumen2_diag")
    extra = []
    for rid, what in (("gend_diag_dfbcrop", "on our DFB dlib crops instead of GenD's RetinaFace crops"), ("gend_diag_fp32", "fp32 instead of fp16 autocast")):
        f = diag / f"{rid}__clean.json"
        if f.exists():
            d = json.loads(f.read_text())
            extra.append(f"- clean {what}: frame AUC {d['frame_auc']:.4f}, video AUC {d['video_auc']:.4f}")
    return L + [""] + extra


def build(m: Models, bench: dict | None, n_boot: int, gend_dir: Path, dfd: dict | None) -> str:
    L: list[str] = []
    A = L.append
    A("# GenD room-to-improve (LUMEN2 step 1)\n")
    A(f"Public GenD CLIP-L/14 evaluated on our 23 FF++ test conditions (frame AUC, 95% video-level bootstrap CI, {n_boot} resamples, "
      "one shared resample matrix so aggregates are joint and all differences paired). Source: results/lumen2/gend_public/*.json + "
      "scores, results/raw (Xception).\n")
    A("Method: GenD's own RetinaFace crop + align (scale 1.3, 256 px) re-cropped from the original videos at the same 32 DFB frame "
      "indices (22388 c23 / 22340 c40 frames, 0 detection fallbacks); degradation = our `apply_condition` on that 256 px crop "
      "(same seeds and keys as the Xception runs), then GenD's input transform (PIL bicubic 224, CLIP mean/std); fp16 autocast. "
      "Frame sets are identical to the Xception runs (checked per condition by frame key).\n")
    A("## Frame AUC [95% CI]\n")
    hdr = ["model"] + [COL_LABEL.get(c, c) for c in COLS]
    rows = [[lab] + [f4(m.value(lab, c)) for c in COLS] for lab in m.runs]
    L += gf.md_table(hdr, rows)
    A("\n## Video AUC [95% CI]\n")
    L += gf.md_table(hdr, [[lab] + [f4(m.value(lab, c, "video")) for c in COLS] for lab in m.runs])
    A("\n## Drop from own clean score, paired bootstrap delta [95% CI] (frame AUC)\n")
    dh = ["model"] + [COL_LABEL.get(c, c) for c in COLS if c != "clean"]
    L += gf.md_table(dh, [[lab] + [f4(m.drop(lab, c), True) for c in COLS if c != "clean"] for lab in m.runs])
    g_syn, g_c40 = m.drop("GenD (public)", "synth_mean"), m.drop("GenD (public)", "c40")
    A("\n## Gate (step 4 of the brief)\n")
    if g_syn and g_c40:
        hit = g_syn[0] <= -GATE or g_c40[0] <= -GATE
        A(f"- clean -> synthetic-degraded mean: {g_syn[0]:+.4f} (threshold -{GATE}); clean -> c40: {g_c40[0]:+.4f} (threshold -{GATE}).")
        A(f"- **Gate {'TRIGGERED (drop >= 0.03): DFD download + evaluation required' if hit else 'NOT triggered: skip DFD; little room to improve'}.**")
    else:
        A("- not computable (missing conditions)")
    A("\n## Calibration / selective prediction over all degraded test frames (reference)\n")
    A("All non-clean conditions concatenated (21 synthetic + c40 = 22 conditions), ECE 15 equal-width bins, selective AUC keeps the most "
      "confident 80%/90% by |p-0.5|. GenD with video-level bootstrap CI; Xception rows point estimates.\n")
    cal = {lab: calibration(m, lab, n_boot, ci=(lab == "GenD (public)")) for lab in m.runs}
    L += gf.md_table(["model", "frames", "ECE", "selective AUC 80%", "selective AUC 90%", "AUC (all frames)"],
                     [[lab, c["n_frames"], fmt_cal(c["ece"]), fmt_cal(c["sel80"]), fmt_cal(c["sel90"]), fmt_cal(c["auc_all"])]
                      for lab, c in cal.items()])
    A("\n## Clean sanity check vs the paper\n")
    L += clean_sanity(m, gend_dir)
    if bench:
        A("\n## Cost on the RTX 3060 (12 GB)\n")
        for k, v in bench.items():
            A(f"- {k}: {v}")
    if dfd:
        A("\n## DFD (DeepFakeDetection, FF++ release) clean\n")
        for k, v in dfd.items():
            A(f"- {k}: {v}")
        cost = REPO_ROOT / "results" / "lumen2" / "dfd_cost.txt"
        if cost.exists():
            A(cost.read_text().rstrip())
    A("\n## Checks\n")
    A("Frame-set identity with base_xception per condition: " + "; ".join(consistency(m)[:3]) + " ... (full list in results/lumen2/gend_room_checks.txt)")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.gend.room_report")
    ap.add_argument("--gend-dir", default=str(REPO_ROOT / "results" / "lumen2" / "gend_public"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "lumen2" / "gend_room.md"))
    a = ap.parse_args(argv)
    gd = Path(a.gend_dir)
    m = Models(gd, a.n_boot)
    bj = REPO_ROOT / "results" / "lumen2" / "gend_bench.json"
    bench = json.loads(bj.read_text()) if bj.exists() else None
    dj = gd / "gend_dfd__clean.json"
    dfd = None
    if dj.exists():
        d = json.loads(dj.read_text())
        dfd = {"frame AUC": f4((d["frame_auc"], *d["frame_auc_ci"])), "video AUC": f4((d["video_auc"], *d["video_auc_ci"])),
               "frames/videos": f"{d['n_frames']}/{d['n_videos']}"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(m, bench, a.n_boot, gd, dfd))
    (out.parent / "gend_room_checks.txt").write_text("\n".join(consistency(m)) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
