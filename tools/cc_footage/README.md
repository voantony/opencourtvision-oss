# CC BY bootstrap footage: download + pre-label (Mac)

`vamos_cc_footage.sh` is the single entry point. It sets up `~/VamosFootage` (uv-managed Python 3.12 venv, yt-dlp, deno, numpy/scipy/opencv, this repo in editable mode, existing ffmpeg from `~/bin` or Homebrew), then runs `cc_footage.py` under `caffeinate -i nice -n 10`.

- Sources: the 72 verified CC BY videos from `rallycut/footage_sources.md`, minus the red flags (content-farm/re-uploads, broadcasts, minors), in pickleball, padel, tennis order. The licence is re-checked per video from yt-dlp metadata before download; anything that doesn't read "Creative Commons Attribution" is skipped and logged.
- Download: sequential, 45-100 s between videos, `--sleep-requests`, retries, backoff of 10-60 min on rate limits (it processes already-downloaded videos while it waits), `--download-archive`. Source resolution kept up to 2160p (VP9 preferred over AV1).
- Video kind: every pre-label is tagged `video_kind` = `full_match` or `partial_or_practice` (app mode suggestion on the audio rallies, < 20 min, or edited/moving camera) and `label_mode` = `match` or `rallies`; partial/practice videos are labelled rallies-only (no winners or score). `cc_footage.py --retag` adds the tags to existing pre-labels without decoding.
- Pre-labels (`prelabels/<sport>/<id>/`): `contact_sheet.jpg`, `trim.json` (warm-up/non-play trim, no re-encode), `court_check.jpg`, `prelabels.json` (court homography, rallies with audio contacts and track bounces), `ball.csv`, `players.csv`, `qa.json`; `prelabels/qa_summary.csv` over all videos. All machine guesses, `verified=false`.
- Speed (M2 Pro): the downloader runs in its own thread (polite sleeps/backoff only there) and feeds a queue, so the next video downloads while the current one is labelled. Rallies fan out over a process pool sized to the performance cores (`sysctl hw.perflevel0.physicalcpu`, override `VAMOS_WORKERS`), each decoding with VideoToolbox at 1280 px and <= 30 fps (originals untouched). Ball tracking is engine-v2's BallEngine (ROI/Kalman HSV, size and court gates, live-ball filter, bounces) with HSV re-acquire by default. TrackNetV3 re-acquire on PyTorch MPS is opt-in (`VAMOS_TN=1`): with badminton weights it was GPU-bound at ~75 frames/s total on the M2 Pro (6 workers sharing MPS) and didn't improve the live-ball rate, so it waits for fine-tuned weights. No Core ML/ANE path yet.
- Guards: stops cleanly under 30 GB free; waits while another heavy (>80 % CPU) ffmpeg/python job is running; single instance; re-run to resume.

```bash
nohup bash ~/VamosFootage/ocv/tools/cc_footage/vamos_cc_footage.sh >/dev/null 2>&1 &   # start / resume
tail -f ~/VamosFootage/run.log; column -t -s $'\t' ~/VamosFootage/progress.tsv          # progress
kill "$(cat ~/VamosFootage/worker.pid)"                                                   # stop cleanly
```
