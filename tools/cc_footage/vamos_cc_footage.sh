#!/bin/bash
# Vamos CC BY bootstrap footage: download + pre-label pass, for Antony's Mac (Apple Silicon, macOS).
# One self-contained, resumable script. Re-run it to resume. Everything lives under ~/VamosFootage.
#   Launch:  nohup bash vamos_cc_footage.sh >/dev/null 2>&1 &
#   Options: --dry-run (setup + plan, no downloads), --no-download, --no-prelabel, --limit N
#   Stop:    kill "$(cat ~/VamosFootage/worker.pid)"  (SIGTERM: stops cleanly at the next checkpoint; re-run resumes)
#   Watch:   tail -f ~/VamosFootage/run.log ; column -t -s $'\t' ~/VamosFootage/progress.tsv
# Rules baked in: only the verified CC BY list from rallycut/footage_sources.md minus red flags (re-uploads/content farms,
# broadcasts, minors); licence re-checked per video from yt-dlp metadata; sequential polite downloads with sleeps,
# backoff and --download-archive; nice -n 10 + caffeinate; stops cleanly below 30 GB free; waits while another
# heavy ffmpeg/python job is running.
set -u
BASE="${VAMOS_FOOTAGE_DIR:-$HOME/VamosFootage}"
REPO_URL="https://github.com/voantony/OpenCourtVision.git"
BRANCH="feature/cc-prelabel"
mkdir -p "$BASE/tools" "$BASE/prelabels"
LOG="$BASE/run.log"
log() { echo "$(TZ=Europe/Madrid date '+%Y-%m-%d %H:%M:%S %Z') [setup] $*" >> "$LOG"; }

if [ -f "$BASE/run.pid" ] && kill -0 "$(cat "$BASE/run.pid")" 2>/dev/null; then
  log "another run is active (pid $(cat "$BASE/run.pid")); exiting"; exit 0
fi
echo $$ > "$BASE/run.pid"
log "start (args: $*)"
ARCH=$(uname -m)

# --- ffmpeg: prefer an existing install (~/bin static build, Homebrew, PATH); never installed by this script
for c in "${OCV_FFMPEG:-}" "$HOME/bin/ffmpeg" /opt/homebrew/bin/ffmpeg /usr/local/bin/ffmpeg "$(command -v ffmpeg 2>/dev/null)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then export OCV_FFMPEG="$c"; break; fi; done
for c in "${OCV_FFPROBE:-}" "$HOME/bin/ffprobe" /opt/homebrew/bin/ffprobe /usr/local/bin/ffprobe "$(command -v ffprobe 2>/dev/null)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then export OCV_FFPROBE="$c"; break; fi; done
if [ -z "${OCV_FFMPEG:-}" ] || [ -z "${OCV_FFPROBE:-}" ]; then log "ERROR: ffmpeg/ffprobe not found"; rm -f "$BASE/run.pid"; exit 1; fi
export PATH="$(dirname "$OCV_FFMPEG"):$BASE/tools:$PATH"
log "ffmpeg: $OCV_FFMPEG"

# --- uv (standalone, for a Python >= 3.10; macOS ships 3.9) and deno (JS runtime yt-dlp needs for YouTube)
if [ ! -x "$BASE/tools/uv" ]; then
  log "installing uv into $BASE/tools"
  curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-${ARCH/arm64/aarch64}-apple-darwin.tar.gz" | tar -xz -C "$BASE/tools" --strip-components 1 || { log "ERROR: uv download failed"; rm -f "$BASE/run.pid"; exit 1; }
fi
if [ ! -x "$BASE/tools/deno" ]; then
  log "installing deno into $BASE/tools"
  curl -fsSL -o "$BASE/tools/deno.zip" "https://github.com/denoland/deno/releases/latest/download/deno-${ARCH/arm64/aarch64}-apple-darwin.zip" \
    && (cd "$BASE/tools" && unzip -oq deno.zip && rm -f deno.zip) || log "WARN: deno download failed; yt-dlp may miss formats"
fi
export UV_PYTHON_INSTALL_DIR="$BASE/tools/python" UV_CACHE_DIR="$BASE/tools/uv-cache"

# --- code: clone (or update) the repo branch; fall back to a git bundle at $BASE/ocv.bundle if GitHub auth is missing
if [ ! -d "$BASE/ocv/.git" ]; then
  if git clone -q -b "$BRANCH" "$REPO_URL" "$BASE/ocv" 2>>"$LOG"; then log "cloned $BRANCH";
  elif [ -f "$BASE/ocv.bundle" ]; then git clone -q -b "$BRANCH" "$BASE/ocv.bundle" "$BASE/ocv" && git -C "$BASE/ocv" remote set-url origin "$REPO_URL" && log "cloned $BRANCH from ocv.bundle";
  else log "ERROR: cannot clone $REPO_URL (no GitHub auth?) and no $BASE/ocv.bundle"; rm -f "$BASE/run.pid"; exit 1; fi
else
  git -C "$BASE/ocv" pull -q --ff-only 2>/dev/null && log "repo updated" || log "repo pull skipped (offline/no auth); using $(git -C "$BASE/ocv" rev-parse --short HEAD)"
fi

# --- venv
if [ ! -x "$BASE/venv/bin/python" ]; then
  log "creating venv (Python 3.12)"
  "$BASE/tools/uv" venv -q --python 3.12 "$BASE/venv" >>"$LOG" 2>&1 || { log "ERROR: venv failed"; rm -f "$BASE/run.pid"; exit 1; }
fi
DEPS_VER=2   # v2: torch (MPS) + gdown for engine-v2 TrackNetV3
if [ "$(cat "$BASE/venv/.deps_ok" 2>/dev/null)" != "$DEPS_VER" ] || [ "$BASE/ocv/pyproject.toml" -nt "$BASE/venv/.deps_ok" ]; then
  log "installing deps (v$DEPS_VER)"
  "$BASE/tools/uv" pip install -q --python "$BASE/venv/bin/python" -U "yt-dlp[default]" numpy scipy opencv-python-headless torch gdown >>"$LOG" 2>&1 \
   && "$BASE/tools/uv" pip install -q --python "$BASE/venv/bin/python" -e "$BASE/ocv" >>"$LOG" 2>&1 \
   && echo "$DEPS_VER" > "$BASE/venv/.deps_ok" || { log "ERROR: pip install failed"; rm -f "$BASE/run.pid"; exit 1; }
else
  "$BASE/tools/uv" pip install -q --python "$BASE/venv/bin/python" -U "yt-dlp[default]" >>"$LOG" 2>&1 || true
fi

PY="$BASE/ocv/tools/cc_footage/cc_footage.py"
log "launching $PY under caffeinate + nice -n 10"
CAF=""; command -v caffeinate >/dev/null && CAF="caffeinate -i"
export OCV_CACHE="$BASE/tools/cache"   # TrackNetV3 weights (SHA-256 checked) live here
$CAF nice -n 10 "$BASE/venv/bin/python" -u "$PY" --base "$BASE" "$@" >> "$LOG" 2>&1
rc=$?
log "exit $rc"
rm -f "$BASE/run.pid"
exit $rc
