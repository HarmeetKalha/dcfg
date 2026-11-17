"""Phase 5.4 baseline robustness report (GATE 5). All numbers come from results/raw/base_<det>__<cond>.json.

    python -m dcfg.analysis.baseline_report

Outputs: results/tables/baseline_robustness.{csv,md,tex}, results/figures/baseline_curves.{png,pdf},
a consistency report on stdout (exit code 1 if any check fails), and a frequency-damage summary.
Reusable: ``load_runs(prefix)`` / ``check_consistency`` work for any run id (e.g. dcfg_xception).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "results" / "raw"
DETS = ["xception", "efficientnetb4", "f3net", "ucf"]
DET_LABEL = {"xception": "Xception", "efficientnetb4": "EfficientNet-B4", "f3net": "F3Net", "ucf": "UCF"}
# Okabe-Ito (colorblind-safe), fixed per detector; markers/linestyles give a second encoding.
DET_STYLE = {"xception": ("#0072B2", "o", "-"), "efficientnetb4": ("#D55E00", "s", "--"),
             "f3net": ("#009E73", "^", "-."), "ucf": ("#CC79A7", "D", ":")}
REPRO = {"xception": 0.9761, "efficientnetb4": 0.9702, "f3net": 0.9782, "ucf": 0.9813}  # PROGRESS.md GATE 4
GROUP_TITLE = {"clean": "Clean", "jpeg": "JPEG", "blur": "Gaussian blur", "noise": "Gaussian noise",
               "resize": "Resize", "mixed": "Mixed", "held_out": "Held-out types", "real_compression": "Real compression"}


def load_conditions() -> list[dict]:
    return yaml.safe_load(open(REPO / "configs" / "eval" / "conditions.yaml"))["conditions"]


def is_pilot(prefix_or_id: str = "", doc: dict | None = None) -> bool:
    """Pilot results (run id ``pilot_*`` or JSON with ``"pilot": true``) never enter paper tables/reports."""
    return prefix_or_id.startswith("pilot_") or bool(doc and (doc.get("pilot") or str(doc.get("run_id", "")).startswith("pilot_")))


def iter_raw_jsons(raw_dir: Path | None = None):
    """Yield ``(path, doc)`` for every results/raw/*.json that is not a pilot result. Use this instead of globbing."""
    for p in sorted((raw_dir or RAW).glob("*.json")):
        if is_pilot(p.name):
            continue
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if isinstance(doc, dict) and is_pilot(doc=doc):
            continue
        yield p, doc


def load_runs(prefix: str = "base", dets=DETS, conds=None) -> dict:
    """{(det, cond): json dict or None if missing}. Pilot runs are rejected / skipped."""
    if is_pilot(prefix):
        raise ValueError(f"pilot run prefix {prefix!r} must not be used for paper reports")
    conds = conds or [c["name"] for c in load_conditions()]
    out = {}
    for d in dets:
        for c in conds:
            p = RAW / f"{prefix}_{d}__{c}.json"
            doc = json.load(open(p)) if p.exists() else None
            out[(d, c)] = None if doc is not None and is_pilot(doc=doc) else doc
    return out


def check_consistency(runs: dict, conds: list[str], dets=DETS, prefix="base") -> list[str]:
    """Return a list of problem strings (empty = all good)."""
    bad = []
    for (d, c), r in runs.items():
        if r is None:
            bad.append(f"MISSING {prefix}_{d}__{c}")
    ok = {k: r for k, r in runs.items() if r is not None}
    for (d, c), r in ok.items():
        tag = f"{d}/{c}"
        if r["condition"] != c or r["detector"] != d:
            bad.append(f"{tag}: id fields mismatch ({r['detector']},{r['condition']})")
        for m in ("frame", "video"):
            a, (lo, hi) = r[f"{m}_auc"], r[f"{m}_auc_ci"]
            if not (lo <= a <= hi):
                bad.append(f"{tag}: {m} AUC {a:.4f} outside CI [{lo:.4f},{hi:.4f}]")
            if not (0.0 <= lo <= hi <= 1.0):
                bad.append(f"{tag}: {m} CI out of [0,1]")
        if r["bootstrap"] != ok[(d, "clean")]["bootstrap"]:
            bad.append(f"{tag}: bootstrap settings differ from clean")
        if r["dfb_commit"] != ok[(d, "clean")]["dfb_commit"] or r["fp16"] != ok[(d, "clean")]["fp16"]:
            bad.append(f"{tag}: dfb_commit/fp16 differ from clean")
    # counts: same per (split, compression) across every json
    cnt = {}
    for (d, c), r in ok.items():
        cnt.setdefault((r["split"], r["compression"]), set()).add((r["n_frames"], r["n_videos"]))
    for k, v in cnt.items():
        if len(v) != 1:
            bad.append(f"counts differ within split/comp {k}: {sorted(v)}")
    if ("test", "c23") in cnt and cnt[("test", "c23")] != {(22388, 700)}:
        bad.append(f"c23 test counts {cnt[('test','c23')]} != (22388, 700)")
    # clean reproduction
    for d in dets:
        r = ok.get((d, "clean"))
        if r and round(r["frame_auc"], 4) != REPRO[d]:
            bad.append(f"{d}: clean frame AUC {r['frame_auc']:.4f} != reproduction {REPRO[d]}")
    # suspicious values: non-monotonic severity, degraded > clean, AUC < 0.5
    for d in dets:
        cl = ok.get((d, "clean"))
        if not cl:
            continue
        for c in conds:
            r = ok.get((d, c))
            if r and c != "clean":
                if r["frame_auc"] > cl["frame_auc"] + 1e-9:
                    bad.append(f"NOTE {d}/{c}: frame AUC above clean ({r['frame_auc']:.4f} > {cl['frame_auc']:.4f})")
                if r["frame_auc"] < 0.5:
                    bad.append(f"NOTE {d}/{c}: frame AUC below chance ({r['frame_auc']:.4f})")
    for d in dets:
        for fam in (["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "jpeg_q10"],
                    ["blur_s1", "blur_s2", "blur_s3", "blur_s4"], ["noise_s5", "noise_s10", "noise_s15", "noise_s20"],
                    ["resize_0.5", "resize_0.25"]):
            v = [ok[(d, c)]["frame_auc"] for c in fam if (d, c) in ok]
            if len(v) == len(fam) and any(b > a + 1e-9 for a, b in zip(v, v[1:])):
                bad.append(f"NOTE {d}: non-monotonic frame AUC along {fam[0].split('_')[0]}: {np.round(v, 4).tolist()}")
    return bad


def fmt_ci(a, ci):
    return f"{a:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def build_tables(runs, conds, out_dir: Path):
    rows = []
    for c in conds:
        for d in DETS:
            r, cl = runs[(d, c["name"])], runs[(d, "clean")]
            rows.append(dict(group=c["group"], condition=c["name"], detector=d,
                             n_frames=r["n_frames"], n_videos=r["n_videos"],
                             frame_auc=r["frame_auc"], frame_lo=r["frame_auc_ci"][0], frame_hi=r["frame_auc_ci"][1],
                             video_auc=r["video_auc"], video_lo=r["video_auc_ci"][0], video_hi=r["video_auc_ci"][1],
                             drop_frame=cl["frame_auc"] - r["frame_auc"], drop_video=cl["video_auc"] - r["video_auc"]))
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = {(r["condition"], r["detector"]): r for r in rows}
    # Markdown: one block per detector-wide metric
    md = ["# Baseline robustness (frozen detectors, no module), FF++ c23 test unless noted",
          "", "Frame AUC with 95% video-level bootstrap CI (1000 resamples); video AUC; Delta = condition minus clean "
          "(frame AUC; negative = worse). Source: results/raw/base_<det>__<cond>.json. Drops are differences of point "
          "estimates; CIs of the drop are not computed here (needs paired resampling from the score files).", ""]
    hdr = ["Group", "Condition"]
    for d in DETS:
        hdr += [f"{DET_LABEL[d]} frame AUC [95% CI]", "video", "Δ frame"]
    md.append("| " + " | ".join(hdr) + " |"); md.append("|" + "---|" * len(hdr))
    tex = [r"\begin{tabular}{ll" + "rrr" * 4 + "}", r"\toprule",
           "Group & Condition & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{DET_LABEL[d]}}}" for d in DETS) + r" \\",
           " & & " + " & ".join("Frame [CI] & Video & $\\Delta$" for _ in DETS) + r" \\", r"\midrule"]
    last = None
    for c in conds:
        cells = [GROUP_TITLE[c["group"]] if c["group"] != last else "", c["name"]]
        tcells = [cells[0], c["name"].replace("_", r"\_")]
        for d in DETS:
            r = by[(c["name"], d)]
            cells += [f"{r['frame_auc']:.4f} [{r['frame_lo']:.3f}, {r['frame_hi']:.3f}]", f"{r['video_auc']:.4f}",
                      "-" if c["name"] == "clean" else f"{-r['drop_frame']:+.4f}"]
            tcells += [f"{r['frame_auc']:.3f} [{r['frame_lo']:.3f}, {r['frame_hi']:.3f}]", f"{r['video_auc']:.3f}",
                       "--" if c["name"] == "clean" else f"{-r['drop_frame']:+.3f}"]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(tcells) + r" \\")
        if c["group"] != last and last is not None:
            pass
        last = c["group"]
    tex += [r"\bottomrule", r"\end{tabular}"]
    n23 = runs[("xception", "clean")]; n40 = runs[("xception", "c40")]
    md += ["", f"Counts: c23 test {n23['n_frames']} frames / {n23['n_videos']} videos; c40 test {n40['n_frames']} frames / "
           f"{n40['n_videos']} videos. Δ = frame AUC change vs clean.", ""]
    (out_dir / "baseline_robustness.md").write_text("\n".join(md) + "\n")
    (out_dir / "baseline_robustness.tex").write_text("\n".join(tex) + "\n")
    return by


def make_figure(by, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fams = [("JPEG quality", ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "jpeg_q10"],
             ["clean", "90", "70", "50", "30", "10"]),
            ("Gaussian blur $\\sigma$", ["clean", "blur_s1", "blur_s2", "blur_s3", "blur_s4"], ["0", "1", "2", "3", "4"]),
            ("Gaussian noise $\\sigma$", ["clean", "noise_s5", "noise_s10", "noise_s15", "noise_s20"], ["0", "5", "10", "15", "20"]),
            ("Resize scale", ["clean", "resize_0.5", "resize_0.25"], ["1", "0.5", "0.25"])]
    fig = plt.figure(figsize=(7.16, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, (title, cs, labs) in zip(axes, fams):
        x = np.arange(len(cs))
        for d in DETS:
            col, mk, ls = DET_STYLE[d]
            y = np.array([by[(c, d)]["frame_auc"] for c in cs])
            lo = np.array([by[(c, d)]["frame_lo"] for c in cs]); hi = np.array([by[(c, d)]["frame_hi"] for c in cs])
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(x, y, color=col, marker=mk, ms=3.2, lw=1.2, ls=ls, label=DET_LABEL[d])
        ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel(title); ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="0.6", lw=0.5, ls=(0, (2, 2)))
        ax.grid(axis="y", lw=0.3, color="0.85"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Frame AUC")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axb = fig.add_subplot(gs[1, :])
    bc = ["clean", "mixed_light", "mixed_heavy", "webp_q50", "webp_q20", "motion_k7", "motion_k11", "c40"]
    bl = ["clean", "mixed\nlight", "mixed\nheavy", "WebP\nq50", "WebP\nq20", "motion\nk7", "motion\nk11", "c40\n(real)"]
    w = 0.19
    for i, d in enumerate(DETS):
        col = DET_STYLE[d][0]
        y = np.array([by[(c, d)]["frame_auc"] for c in bc])
        err = np.array([[by[(c, d)]["frame_auc"] - by[(c, d)]["frame_lo"], by[(c, d)]["frame_hi"] - by[(c, d)]["frame_auc"]] for c in bc]).T
        axb.bar(np.arange(len(bc)) + (i - 1.5) * w, y, w * 0.92, color=col, label=DET_LABEL[d], yerr=err,
                error_kw=dict(lw=0.6, capsize=1.2, ecolor="0.25"))
    axb.axhline(0.5, color="0.6", lw=0.5, ls=(0, (2, 2)))
    axb.set_xticks(np.arange(len(bc))); axb.set_xticklabels(bl); axb.set_ylim(0.4, 1.0); axb.set_ylabel("Frame AUC")
    axb.grid(axis="y", lw=0.3, color="0.85"); axb.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.9, bottom=0.09, left=0.07, right=0.99)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "baseline_curves.png", dpi=300); fig.savefig(fig_dir / "baseline_curves.pdf")
    plt.close(fig)


def freq_summary():
    """Print per-condition mean relative DCT-energy change by channel and by radial bin groups (descriptive only)."""
    z = np.load(RAW / "freq_study.npz")
    conds = list(z["conditions"]); rc = z["rel_change"]; rad = z["radial_rel_change"]
    print(f"\nfreq_study: n_images={int(z['n_images'])}, seed={int(z['seed'])}")
    print(f"{'cond':12s} {'Ymean':>8s} {'Cbmean':>8s} {'Crmean':>8s} | radial bins 0,1-3,4-6,7-9 (Y)        | Cb r7-9 | Cr r7-9")
    for i, c in enumerate(conds):
        m = rc[i].reshape(3, -1).mean(1)
        r = rad[i]
        print(f"{c:12s} {m[0]:+8.3f} {m[1]:+8.3f} {m[2]:+8.3f} | {r[0,0]:+.3f} {r[0,1:4].mean():+.3f} {r[0,4:7].mean():+.3f} {r[0,7:].mean():+.3f} | {r[1,7:].mean():+.3f} | {r[2,7:].mean():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="base")
    a = ap.parse_args()
    conds = load_conditions(); names = [c["name"] for c in conds]
    runs = load_runs(a.prefix, conds=names)
    problems = check_consistency(runs, names, prefix=a.prefix)
    print(f"{len([r for r in runs.values() if r])}/{len(runs)} runs found")
    print("\n".join(problems) if problems else "consistency: all checks passed")
    if any(r is None for r in runs.values()):
        sys.exit(1)
    by = build_tables(runs, conds, REPO / "results" / "tables")
    make_figure(by, REPO / "results" / "figures")
    freq_summary()
    sys.exit(1 if any(not p.startswith("NOTE") for p in problems) else 0)


if __name__ == "__main__":
    main()
