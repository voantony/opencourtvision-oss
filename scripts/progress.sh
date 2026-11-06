#!/usr/bin/env bash
# Live progress view for long OpenCourtVision jobs (STT logs, render log, exports, active processes).
# Usage: scripts/progress.sh WORK_DIR [EXPORTS_DIR]   (refreshes every 3 s; Ctrl-C to quit, jobs keep running)
W=${1:?usage: progress.sh WORK_DIR [EXPORTS_DIR]}
E=${2:-$W/../exports}
while true; do
  clear
  echo "=== OpenCourtVision job progress ===   $(date '+%H:%M:%S')"
  echo
  echo "Speech-to-text (score callouts):"
  for f in "$W"/stt*.log; do [ -f "$f" ] && echo "  $(basename "$f"): $(tail -c 300 "$f" | tr '\r' '\n' | grep -E '%|done' | tail -1)"; done
  echo
  echo "Render log:"
  [ -f "$W/render.log" ] && tail -n 3 "$W/render.log" | sed 's/^/  /'
  echo
  echo "Exports:"
  ls -lh "$E" 2>/dev/null | awk 'NR>1{print "  "$5"  "$6" "$7" "$8"  "substr($0, index($0,$9))}'
  echo
  echo "Active jobs:"
  ps -Ao pcpu,etime,command | grep -E 'ffmpeg|stt|whisper|swingcut|ocv-' | grep -v grep | awk '{printf "  %5s%% CPU  %s  ", $1, $2; for(i=3;i<=NF&&i<6;i++) printf "%s ", $i; print ""}' | cut -c1-110
  echo
  echo "(This window refreshes every 3 s. Close it any time; the job keeps running.)"
  sleep 3
done
