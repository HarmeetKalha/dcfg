#!/usr/bin/env python
"""Regenerate results/STATUS_FOR_MEETING.md from what exists on disk (robust to missing files). All times in IST.

    python ops/jobs/meeting_status.py

Sections: current jobs (both runners: current epoch, best epoch so far, ETA), queue ETA, finished results (one short block per
finished step). Read-only except for the output file. ETAs: measured epoch time from the training heartbeat/metrics where
available, otherwise fixed planning estimates (``PLAN_MIN``) for jobs that have not started."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IST = timezone(timedelta(hours=5, minutes=30))
#: (runner label, runs root, queue file)
RUNNERS = [("meeting", REPO / "runs", REPO / "ops/queue_meeting.txt"), ("background (DFD)", REPO / "runs_bg", REPO / "ops/queue_dfd_bg.txt")]
#: planning estimate in minutes for jobs that have not started (training ones are refined from measurements)
PLAN_MIN = {"gate_heatmaps_lumen1": 20, "sensor_train": 95, "sensor_eval": 25, "lumen_nogate_xception": 11 * 60, "eval_lumen_nogate_xception": 35,
            "gate_ablation": 5, "lumen2_smoke": 45, "lumen2_train": 27 * 60, "lumen2_eval": 120, "lumen2_bench": 5,
            "lumen2_report": 30, "gate_heatmaps_lumen2": 25}
NOGATE_EPOCH_MIN, NOGATE_MAX_EPOCHS = 22.5, 30  # from lumen_xception: 33,393 s / 25 epochs = 22.3 min


def ist(ts: float | datetime | None) -> str:
    if ts is None:
        return "n/a"
    d = ts if isinstance(ts, datetime) else datetime.fromtimestamp(ts, timezone.utc)
    return d.astimezone(IST).strftime("%a %d %b %H:%M IST")


def read(p: Path) -> str | None:
    try:
        return p.read_text()
    except Exception:  # noqa: BLE001
        return None


def jload(p: Path) -> dict | None:
    t = read(p)
    try:
        return json.loads(t) if t else None
    except Exception:  # noqa: BLE001
        return None


def queue_ids(q: Path) -> list[tuple[str, str]]:
    out = []
    for line in (read(q) or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "|" in line:
            jid, cmd = line.split("|", 1)
            out.append((jid.strip(), cmd.strip()))
    return out


def last_heartbeat(d: Path) -> dict | None:
    t = read(d / "heartbeat")
    try:
        return json.loads(t.strip().splitlines()[-1]) if t else None
    except Exception:  # noqa: BLE001
        return None


def metrics_rows(p: Path) -> list[dict]:
    try:
        with open(p) as f:
            return list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        return []


def job_progress(runs: Path, jid: str) -> tuple[str, float | None]:
    """(description, remaining minutes or None) of a running job."""
    d = runs / jid
    hb = last_heartbeat(d)
    desc, rem = "running", None
    if jid == "sensor_train":
        rows = metrics_rows(REPO / "runs/damage_sensor/metrics.csv")
        if rows:
            r = rows[-1]
            best = min(rows, key=lambda x: float(x["val_loss"]))
            desc = f"mini-epoch {r['epoch']} done, train {int(float(r['train_s'])) // 60} min of 90, val loss {r['val_loss']}, best epoch {best['epoch']} ({best['val_loss']})"
            rem = max(0.0, 90 - float(r["train_s"]) / 60) + 2
    elif jid == "lumen_nogate_xception":
        rows = metrics_rows(d / "metrics.csv")
        if rows:
            best = max(rows, key=lambda x: float(x["val_mean_auc"]))
            ep = int(rows[-1]["epoch"]) + 1
            bad = rows[-1].get("bad_epochs")
            desc = f"{ep} of max {NOGATE_MAX_EPOCHS} epochs done, val mean AUC {rows[-1]['val_mean_auc']}, best epoch {best['epoch']} ({best['val_mean_auc']}), patience counter {bad}/6"
            rem = (NOGATE_MAX_EPOCHS - ep) * NOGATE_EPOCH_MIN  # upper bound; early stopping may end it sooner
        else:
            desc = "epoch 0 in progress"
            rem = NOGATE_MAX_EPOCHS * NOGATE_EPOCH_MIN
        if hb and hb.get("eta"):
            desc += f"; heartbeat eta {hb['eta']}"
    elif hb:
        desc = "running (" + ", ".join(f"{k}={hb[k]}" for k in ("condition", "epoch", "step") if k in hb) + ")"
    return desc, rem


def queue_section() -> tuple[list[str], float | None]:
    L, now = [], time.time()
    final_eta = None
    for label, runs, q in RUNNERS:
        ids = queue_ids(q)
        if not ids:
            continue
        alive = False
        pid = read(runs / ".runner.pid")
        if pid:
            try:
                alive = (Path("/proc") / pid.strip() / "cmdline").exists()
            except Exception:  # noqa: BLE001
                alive = False
        cur = (read(runs / "_current_job") or "").strip()
        L.append(f"### Runner: {label} (alive: {'yes' if alive else 'NO - restart it if jobs are pending'})")
        t = now
        for jid, _ in ids:
            if (runs / jid / "DONE").exists():
                st = "done"
            elif (runs / jid / "FAILED").exists():
                st = "FAILED"
            elif jid == cur:
                desc, rem = job_progress(runs, jid)
                t += (rem if rem is not None else PLAN_MIN.get(jid, 30)) * 60
                st = f"RUNNING: {desc}; ETA {ist(t)}"
            else:
                t += PLAN_MIN.get(jid, 30) * 60
                st = f"pending, ETA {ist(t)}"
            if jid.startswith("after_"):
                continue
            L.append(f"- `{jid}`: {st}")
        if label == "meeting":
            final_eta = t
        L.append("")
    return L, final_eta


def block(title: str, body: list[str]) -> list[str]:
    return [f"### {title}", *body, ""]


def head(p: Path, n: int, start: str | None = None) -> list[str]:
    t = read(p)
    if t is None:
        return ["(not available yet)"]
    lines = t.splitlines()
    if start:
        for i, ln in enumerate(lines):
            if ln.startswith(start):
                lines = lines[i:]
                break
    return [ln for ln in lines if ln.strip()][:n]


def results_section() -> list[str]:
    import sys
    sys.path.insert(0, str(REPO / "ops/jobs"))
    L: list[str] = []
    # 1 gate heatmaps
    if (REPO / "results/gate/lumen1/summary.md").exists():
        L += block("Step 1: LUMEN1 gate heatmaps (Xception) - done", ["Per-damage change of the gate weights vs clean (`results/gate/lumen1/`):"] +
                   head(REPO / "results/gate/lumen1/summary.md", 13, "## Headline")[2:13])
    # 2 sensor
    sd = jload(REPO / "results/sensor/standalone.json")
    if sd:
        po = sd["per_op"]
        L += block("Step 2: standalone damage sensor - done", [f"{sd['meta'].get('params')} params. Presence F1 / strength Spearman per op: " +
                   "; ".join(f"{o} {r['presence_f1']:.2f} / {r['strength_spearman'] if r['strength_spearman'] is None else round(r['strength_spearman'], 2)}" for o, r in po.items()),
                   "Full report: `results/sensor/standalone.md`."])
    elif (REPO / "runs/damage_sensor/done.json").exists():
        L += block("Step 2: damage sensor trained, eval pending", [json.dumps(jload(REPO / "runs/damage_sensor/done.json"))[:300]])
    # LIVE sensor
    lv = jload(REPO / "results/raw/sensor_live__test.json")
    if lv:
        L += block("Earlier: LIVE-pretrained sensor (quality regression)", [json.dumps({k: lv[k] for k in list(lv)[:8] if not isinstance(lv[k], (dict, list))})[:300]])
    # 3 ablation
    if (REPO / "results/gate/ablation_xception.md").exists():
        L += block("Step 3: gate ablation (Xception) - done", head(REPO / "results/gate/ablation_xception.md", 12, "## Frame AUC")[:12] +
                   ["Full table + paired differences: `results/gate/ablation_xception.md`."])
    # LUMEN1 headline numbers
    try:
        from job_summary import raw_aggregates
        rows = []
        for run in ("base_xception", "fixed_xception", "lumen_nogate_xception", "lumen_xception", "augft_xception"):
            a = raw_aggregates(run)
            if a:
                rows.append(f"| {run} | " + " | ".join(f"{a[k]:.3f}" for k in ("clean", "synth", "heldout", "c40", "heavy")) + " |")
        if rows:
            L += block("Xception frame AUC, point estimates", ["| run | clean | synthetic mean | held-out | c40 | heavy (5) |", "|---|---|---|---|---|---|"] + rows)
    except Exception as e:  # noqa: BLE001
        L += [f"(aggregate table failed: {e})", ""]
    # GenD family
    L += block("LUMEN2 track: GenD (public) / GenD-aug", head(REPO / "results/lumen2/gend_aug.md", 8))
    lum2 = sorted(p for p in (REPO / "results/lumen2").glob("lumen2*.md") if "smoke" not in p.name)
    if lum2:
        L += block("LUMEN2", head(lum2[-1], 12))
    # LUMEN2 interim (Monday backup) + smoke
    part = sorted((REPO / "results/lumen2").glob("partial_epoch*.json"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    if part:
        d = jload(part[-1]) or {}
        b = d.get("gend_aug_same_subset") or {}
        L += block(f"LUMEN2 INTERIM ({d.get('label')}; test videos, fixed frame subset, not a paper number)",
                   [f"clean {d.get('clean', float('nan')):.4f}, synthetic mean {d.get('synth_mean', float('nan')):.4f}, c40 {d.get('c40', float('nan')):.4f}"
                    + (f"; GenD-aug on the same frames: {b.get('clean', float('nan')):.4f} / {b.get('synth_mean', float('nan')):.4f} / {b.get('c40', float('nan')):.4f}" if b else ""),
                    "sensor presence F1: " + ", ".join(f"{o} {r['presence_f1']:.2f}" for o, r in (d.get("sensor", {}).get("per_op", {})).items()),
                    "strength MAE: " + ", ".join(f"{o} {r['strength_mae']:.3f}" for o, r in (d.get("sensor", {}).get("per_op", {})).items())])
    if (REPO / "results/lumen2/lumen2_smoke.md").exists():
        L += block("LUMEN2 smoke test", head(REPO / "results/lumen2/lumen2_smoke.md", 12))
    # DFD
    dfd = jload(REPO / "results/dfd/gend_public__clean.json")
    if dfd:
        L += block("DFD (GenD public, clean)", [f"frame AUC {dfd['frame_auc']:.4f}, video AUC {dfd['video_auc']:.4f}"])
    else:
        bg, state = REPO / "runs_bg", []
        for j in ("bg_dl_dfd_original", "bg_dl_dfd_fakes", "bg_recrop_dfd", "bg_dfb_pre_dfd"):
            state.append(f"{j}: " + ("done" if (bg / j / "DONE").exists() else "FAILED" if (bg / j / "FAILED").exists() else "pending/running"))
        L += block("DFD prep (background, CPU)", state)
    return L


def main() -> int:
    now = datetime.now(timezone.utc)
    L = ["# STATUS FOR MEETING (auto-generated)", "", f"Generated {ist(now)} by `ops/jobs/meeting_status.py`. Meeting: Tue 24 Nov 08:30 IST. All times IST.", ""]
    try:
        gitl = subprocess.check_output(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"], text=True).strip()
        L += [f"Last commit: {gitl[:100]}", ""]
    except Exception:  # noqa: BLE001
        pass
    q, final = queue_section()
    L += ["## Queue and ETA", ""] + q
    if final:
        L += [f"Meeting-queue estimated finish (upper bound for LUMEN1-no-gate: all 30 epochs, early stopping may end sooner): {ist(final)}", ""]
    L += ["## Finished results", ""] + results_section()
    out = REPO / "results" / "STATUS_FOR_MEETING.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
