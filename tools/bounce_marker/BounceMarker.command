#!/bin/bash
# Double-click to open the last-bounce marker. Runs in the background: closing this window does not stop it.
cd "$(dirname "$0")"
if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8769/items.json; then
  open "http://127.0.0.1:8769/"; exit 0
fi
nohup /usr/bin/python3 serve.py --no-open >> serve.log 2>&1 &
disown
for i in 1 2 3 4 5 6 7 8; do sleep 0.4; curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8769/items.json && break; done
open "http://127.0.0.1:8769/"
