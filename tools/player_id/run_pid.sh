#!/bin/bash
# 4-player ID tracker measurement: detect -> vision -> eval (one job, resumable per point).
# Private outputs (crops, truth, models trained on this match) stay under $OUT and are never committed.
# Point PID_VIDEO, PID_PRELABELS and PID_COURT at your own local files (see pid_run.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-$PWD/pid_out}"; PY="${PY:-python3}"; CODE="${CODE:-$ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 PYTHONPATH="$CODE:$CODE/tools/player_id"
mkdir -p "$OUT"
for step in ${STEPS:-detect vision eval}; do
  [ -f "$OUT/.$step.done" ] && { echo "skip $step"; continue; }
  echo "$(date '+%F %T') $step"; nice -n 20 "$PY" -W ignore "$CODE/tools/player_id/pid_run.py" "$step" --out "$OUT"; date '+%F %T' > "$OUT/.$step.done"
done
