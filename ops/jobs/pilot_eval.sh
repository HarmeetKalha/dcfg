#!/usr/bin/env bash
# Pilot evaluation on the VALIDATION split only (never test, never results/raw).
#   bash ops/jobs/pilot_eval.sh <run_id>      e.g. pilot_base_xception | pilot_{dcfg,fixed,augft}_xception | design-selection runs
#   pilot_{jointft,dcfgcons,dcfgfilm,dcfgfilmlite}_xception (frontend.pt incl. FiLM weights; jointft also detector.pt)
# Conditions: clean, noise_s10, blur_s3, webp_q20, mixed_heavy (+ c40 only if c40 VAL frames exist in the JSON, else skipped).
# Output: results/pilot/eval/<run_id>__<cond>.json + scores/<run_id>__<cond>.npz (per-frame scores for paired comparisons).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
run="${1:?usage: pilot_eval.sh <run_id>}"
[[ "$run" == pilot_* ]] || { echo "run id must start with pilot_" >&2; exit 2; }
PY="${PY:-.venv/bin/python}"
det="${run#pilot_*_}"
conds="clean,noise_s10,blur_s3,webp_q20,mixed_heavy"
# c40 only if c40 VAL frames exist and cover (nearly) all val videos (a failed/partial preprocess -> skipped)
ok_c40=$("$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "src")
from dcfg.data.ffpp import DFBJsonSource
s = DFBJsonSource()
n40 = len({r.video for r in s.records("val", "c40")}); n23 = len({r.video for r in s.records("val", "c23")})
print(int(n40 > 0 and n40 >= 0.98 * n23))
PYEOF
)
if [ "$ok_c40" = 1 ]; then conds="$conds,c40"; else echo "c40 val preprocessing failed/missing -> c40 skipped"; fi
args=(--run "$run" --detector "$det" --conditions "$conds" --split val --out-dir results/pilot/eval)
if [[ "$run" != pilot_base_* ]]; then
  d="runs/pilot/$run"
  [ -f "$d/done.json" ] || { echo "missing $d/done.json (training not finished)" >&2; exit 3; }
  [ -f "$d/frontend.pt" ] && args+=(--frontend "$d/frontend.pt")
  [ -f "$d/detector.pt" ] && args+=(--detector-ckpt "$d/detector.pt")
fi
exec "$PY" -m dcfg.eval.evaluate "${args[@]}"
