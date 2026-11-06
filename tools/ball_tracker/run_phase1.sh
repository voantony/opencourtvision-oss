#!/bin/bash
# Own ball tracker, phase 1, on Antony's Mac (one queue job; resumable: each step leaves a marker and is skipped next time).
#   data (frames + teacher + physics pseudo-labels) -> train r1 -> self-training label refresh -> train r2 -> held-out eval
#   + point-winner re-run. Everything it writes stays under $BB (private; never committed).
# Mac only (no cloud GPU): every step under nice 20; training = crops + fp16 + curated subset + early stopping (train.py).
# Env: BB (default ~/VamosFootage/ballbench), PY (default ~/VamosFootage/venv/bin/python),
#      SMOKE=1: small data subset + a 10-minute training run + eval, to check the whole chain before the long run.
set -euo pipefail
BB="${BB:-$HOME/VamosFootage/ballbench}"; PY="${PY:-$HOME/VamosFootage/venv/bin/python}"
CODE="$BB/ocv"; T="$CODE/tools/ball_tracker"
V="$HOME/Videos/Video Projects/2026-10-17 Doubles Match/source/full_match_video 4 - doubles.MP4"
LAB="${V%.MP4}.labels.json"; TR="$HOME/VamosFootage/dev_match_tracks_v2/seg_r"; PRE="$HOME/VamosFootage/dev_match_tracks/prelabel/prelabels.json"
SVJ="$BB/sv_events_dev_match.json"; AUD="$BB/audio_env_dev_match.npy"
export PYTHONPATH="$CODE" OCV_FFMPEG="$HOME/bin/ffmpeg" PYTHONUNBUFFERED=1
if [ "${SMOKE:-0}" = 1 ]; then OUT="$BB/smoke"; LIM="--limit17 6 --cc_max_videos 6 --cc_windows 2"; M1=10; M2=2; EP=30
else OUT="$BB/run1"; LIM=""; M1=${M1:-45}; M2=${M2:-25}; EP=30; fi
MALL=${MALL:-$M1}; [ "${SMOKE:-0}" = 1 ] && MALL=2
mkdir -p "$OUT/models" "$OUT/eval"
step(){ local n=$1; shift; if [ -f "$OUT/.$n.done" ]; then echo "skip $n"; return; fi
  echo "$(date '+%F %T') step $n"; "$@"; date '+%F %T' > "$OUT/.$n.done"; }
step data   nice -n 20 "$PY" "$T/make_data.py" --video17 "$V" --labels17 "$LAB" --tracks17 "$TR" --prelabels17 "$PRE" --out "$OUT/data" $LIM
step train1 nice -n 20 "$PY" "$T/train.py" --data "$OUT/data" --out "$OUT/models/heatnet_r1.pt" --minutes "$M1" --epochs "$EP" --workers 3
# A/B arm: same minutes on all labelled frames (no hard-example curation)
step train1all nice -n 20 "$PY" "$T/train.py" --data "$OUT/data" --out "$OUT/models/heatnet_r1_all.pt" --minutes "$MALL" --epochs "$EP" --workers 3 --per_clip 0
step refresh nice -n 20 "$PY" "$T/refresh_labels.py" --data "$OUT/data" --weights "$OUT/models/heatnet_r1.pt"
step train2 nice -n 20 "$PY" "$T/train.py" --data "$OUT/data" --out "$OUT/models/heatnet_r2.pt" --labels r2 --init "$OUT/models/heatnet_r1.pt" \
     --minutes "$M2" --epochs "$EP" --lr 1e-3 --workers 3
step eval   nice -n 20 "$PY" "$T/eval_heldout.py" --data "$OUT/data" \
     --weights "r2=$OUT/models/heatnet_r2.pt" "r1_cur=$OUT/models/heatnet_r1.pt" "r1_all=$OUT/models/heatnet_r1_all.pt" --tracks17 "$TR" --labels17 "$LAB" --prelabels17 "$PRE" \
     --sv17 "$SVJ" --audio "$AUD" --pw_stages "$CODE/tools/point_winner/pw_stages.py" --out "$OUT/eval"
echo "$(date '+%F %T') phase 1 done: $OUT/eval/ball_metrics.json"
