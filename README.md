# OpenCourtVision

Multimodal tools that fuse **audio and vision** to turn a fixed-camera tennis match recording (phone on the fence,
SwingVision-style setups, or any 4K camera) into the outputs below. No single signal is trusted on its own: hits, bounces,
point boundaries and labels are only *verified* when independent audio and visual evidence agree (see the verify gate).

- **Point detection (audio + vision)**: ball-hit transients in the 1.5–6 kHz band fused with ball tracking, wrist-speed peaks and player movement; a point ends only when the signals agree it is over.
- **Automatic warmup / match-start detection**: finds where the continuous warmup hitting ends and the first serve starts.
- **Lossless 4K cuts**: keyframe-snapped point segments, stream-copied (no video re-encode). You get a "points only" cut and a highlights cut.
- **Highlights**: long rallies and loud crowd/player reactions, or story beats (see below).
- **Score tracking (multi-signal)**: spoken score callouts (English and Spanish) are one input alongside serve order, side and the point winner from video; disagreements are flagged, never silently fixed.
- **Ball-follow 9:16 reel**: tracks the ball and pans a vertical crop over the 4K frame, with slow-mo ramps at contact and captions. *(work in progress)*
- **Doubles label schema, verify gate and `ocv eval`** *(new)*: typed shot labels with handedness rules, captions only for verified events, and per-event/per-label precision/recall against JSON ground truth.
- **Story / shot finder**: per-point excitement, a momentum curve, story beats (opener, turning point, longest rally, peak, close) and difficult-shot candidates. *(work in progress)*
- **Pose and stroke analysis** *(WIP)*: serve/forehand/backhand at each hit (audio onset confirmed by a wrist-speed peak), stance angle, contact point, finish height, serve lean, footwork and court coverage, unsupervised movement "syllables", and era/group comparisons. See [docs/pose.md](docs/pose.md).
- **Player identification** *(WIP)*: four stable identities in doubles from court position, motion, clothing colour and an optional re-ID embedding (no faces). See [docs/player_id.md](docs/player_id.md).

Everything runs locally. The primary target is Apple Silicon (developed on an M2 Pro); the core pipeline also runs on Linux.

## Quickstart

```bash
git clone https://github.com/voantony/opencourtvision-oss.git && cd opencourtvision-oss
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[reel,dev]"          # add ".[stt]" on Apple Silicon for callout transcription
python benchmarks/run_eval.py         # synthetic clips only; no footage in the repo
python benchmarks/run_pose_eval.py                 # stroke labelling on synthetic stick figures
python benchmarks/public_eval/measure.py           # far-player recall and p(A) calibration
python benchmarks/public_eval/run_eval.py          # point-winner harness, synthetic bundle
python benchmarks/detector_bench.py                # in-memory frames, no images written
ocv-pose demo                                      # synthetic player, no video needed (needs ".[pose]")
```

The code is AGPL-3.0-only. If you run a modified version as a network service that users interact with, you must offer those users the source of that version (AGPL section 13). Names and logos are not covered by the licence; see NOTICE.

This public snapshot is a **work in progress**. See [Status: work in progress](#status-work-in-progress).

Demo data in this tree is synthetic. Public clips, if you add them, must be CC0 or CC BY with a checksum and an attribution line (`docs/LABELING.md`, `scripts/fetch_public_clips.py`). Do not commit footage, labels from private matches, or weight files.

`work/` then holds `segments.csv`, `score_timeline.csv`, `story.json` and `swingcut_report.json`, and
`exports/` holds `Match - points only 4K.mp4` and `Match - highlights 4K.mp4`.

## Requirements

- Python 3.10+ with `numpy` and `scipy` (core). Add `opencv-python` for the reel and `mlx-whisper` for `--stt`.
- **ffmpeg / ffprobe.** We recommend a **static build** (for example from evermeet.cx or osxexperts.net for macOS arm64, or johnvansickle.com for Linux), dropped into `~/bin/ffmpeg` and `~/bin/ffprobe`. Lookup order is `$OCV_FFMPEG`/`$OCV_FFPROBE`, then `~/bin/`, then `PATH`. A static build avoids Homebrew dependency churn and gives the same filters everywhere. The pipeline needs `drawtext` (libfreetype), `loudnorm`, `minterpolate`, `sendcmd` and, on macOS, `h264_videotoolbox`.
- **Apple Silicon target (M2 Pro).** `--stt` uses mlx-whisper on the Apple GPU. The reel decodes with `-hwaccel videotoolbox` and encodes with `h264_videotoolbox`, and the optional TrackNet runs on `mps`. On Linux the reel falls back to software decode and `libx264`, and `--stt` is unavailable.
- Fonts: `$OCV_FONT`, else macOS Arial Bold or DejaVu Sans Bold, else the fontconfig default.

See `requirements.txt` (pip) or `pyproject.toml` extras: `reel`, `stt`, `tracknet`, `pose`/`pose-mac`, `players`, `dev`.

## Usage

### `ocv-swingcut`: full match pipeline

```bash
ocv-swingcut match.mp4 --final "6-4 6-3"                  # analysis only (cached in <video dir>/../work)
ocv-swingcut match.mp4 --final "6-4 6-3" --stt            # + score callouts (Apple Silicon)
ocv-swingcut match.mp4 --final "6-4 6-3" --story --hl 12 --hl_max 240 --render exports/ --name "Club final"
ocv-swingcut match.mp4 --final "6-4 6-3" --start 512.3    # override a wrong match-start guess
```

Every stage caches in `--work`; delete a cache file (e.g. `rallies.json`) to recompute that stage.
`--model` takes an mlx-whisper model dir or Hugging Face repo (default `mlx-community/whisper-large-v3-turbo`, or `$OCV_WHISPER_MODEL`).

### `ocv-match-start`: standalone warmup → match detector

```bash
ocv-match-start match.mp4 --out start.json --sheet start_sheet.jpg
ocv-match-start match.wav --transcript whisper.json      # fuse with the first cluster of spoken score calls
```

### `ocv-reel`: ball-follow 9:16 reel *(WIP)*

```bash
cp examples/reel_moments.example.json my_reel.json      # list moments (start/end s, slow-mo ramps, caption triggers)
ocv-reel my_reel.json --work work/reel --out exports/reel.mp4 [--tracknet /path/to/TrackNet]
```

Moments can `snap` to the first detected hit (this needs `rallies.json` from `ocv-swingcut`). `ball_stats.json` and
`ball_debug_sheet.png` in the work dir show detection coverage, the share of frames framed by players (ball lost > 6
frames or low confidence), the crop-height range, wide-mode use and `jerk_p95` (camera smoothness on the float crop path
before rounding, with the rounded value as a debug field; pass/fail against `camera.jerk_baseline`), and the share of
frames where players were clipped or in wide mode. The camera (`opencourtvision/reel/camera.py`) shows 75 % of the
source height by default, widens to 100 % when framing players or when they spread out, tightens to 65 % only for
verified big shots (the ease starts ~1.3 s before contact and lands on the shot), and obeys hard caps: pan and tilt ≤ 0.5
crop-sizes/s (pan up to 0.8 only while the ball is in flight between two verified contacts), accel ≤ 0.6/s², zoom ≤ 15 %/s
(25 %/s, with the pan cap lifted, only on a verified big shot of a `punch_in` moment), no direction reversal within
0.5 s. The pan is planned offline: it eases from contact to contact and reads the target 0.5 s ahead, so it is moving
before the hit; when the ball or players would leave the crop within 0.5 s it widens first and pans only for the rest,
never above 90 % of the source height unless the players' spread needs it or the ball is in a verified fast flight.
`mean_crop_height` and `pct_frames_ge95` are soft-checked (warning only) against a mean of 0.75-0.85 with >= 95 % on
under 25 % of frames. Smoothness wins over coverage. Vertically it follows the players' box (near feet to far heads) with a 15 % dead zone and
widens rather than clip anyone; `camera.center_y` is only the fallback when there are no player boxes. At most one moment per reel (`camera.max_wide`) may use wide mode (whole 16:9 frame
over a blurred fill) when players don't fit even at full height during a fast net exchange. Captions are never typed in: a moment names a
`caption_trigger` (`shot`, `net_exchange`, `rally_of_match`, `ace`, `running_get`, `break_point`, `set_point`,
`match_point`; rules in `opencourtvision/captions.py`) evaluated on the verified Events in the config's `events` file,
and gets no caption when the trigger does not pass. TrackNet weights are **not** shipped; point
`--tracknet` at a checkout that provides `model.BallTrackerNet` and `model_best.pt`.

### Ball engine v2 *(Step 1 of the engine rewrite; `opencourtvision/ball/`)*

```bash
pip install -e ".[ball]"                     # torch + opencv + gdown
ocv-fetch-weights tracknetv3                 # TrackNetV3 checkpoints (MIT) -> ~/.cache/opencourtvision/tracknetv3, SHA-256 checked
ocv-bench-ball --video match.mp4 --sv shots.xlsx --clips clips.json --calib court_calib.json --out work/bench
ocv-label-ball template --video match.mp4 --start 139.2 --end 149 --clip 16 --prefill work/bench/engine_clip16.csv \
    --out labels/ball/clip16.csv             # then fill/check it (contact sheets, or `ocv-label-ball click` on a Mac)
```

`BallEngine` (`ball/tracker.py`) searches HSV ball candidates only inside a Kalman-predicted window, size-gates them
(6.54–6.86 cm at that depth, minor axis), runs full-frame TrackNetV3 (MPS on Apple silicon, CUDA, else CPU) only to
re-acquire a lost ball, fuses both with the reel camera's per-frame confidence (`reel.ball.frame_confidence`), then
keeps only **live** balls (`ball/live.py`): ballistic or bouncing motion with low fit residuals, a contact (audio hit
within ±80 ms near a player, or a velocity reversal), above our court plus margin, not static, not rolling. Bounces are
cleaned with tennis-vision's rules (≤ 1 per 400 ms, ≤ 2 m off court). Every rejected tracklet carries its reasons.
`ocv-bench-ball` reports proxy metrics (event-time agreement with SwingVision and audio hits, false-track rate, fps)
until labelled frames exist, then precision/recall against `labels/ball/*.csv`. Weights are never committed.


### `ocv eval`: score predictions against labelled ground truth

```bash
ocv eval --pred examples/pred_synthetic_doubles.json --labels labels/
ocv eval --synthetic
```

This reports precision, recall and F1 per event type and per label value (±80 ms for contact/bounce, ±150 ms otherwise),
a stroke confusion matrix, and a confidence calibration table. The label format is described in [labels/README.md](labels/README.md).

### `ocv-sv-audit`: import and audit a SwingVision shot export

```bash
pip install -e ".[swingvision]"
ocv-sv-audit match.xlsx --out audit.md \
  --segments "work/segments.csv" --hits work/rallies.json --score "work/score_timeline.csv"   # optional: compare with our detection
```

`opencourtvision.importers.swingvision` turns SwingVision's "Export Shots" workbook into Events (one `contact` per shot,
`point_start`/`point_end` per point), all with `source="swingvision"` and `verified=False`: a reference to compare
against, never ground truth. The audit flags implausible speeds (serves over 90 mph, repeated cap values, second serves
faster than first), coordinates outside the court, sequence and Detail contradictions, Stats that do not follow from the
points, and points that end too soon after the last contact. Player names are replaced by H1/H2/G1/G2 in all output.
Never commit a real export.

### `ocv-pose`: pose, strokes and mechanics *(WIP)*

```bash
pip install -e ".[pose]"                                   # MediaPipe (Apache-2.0); model downloads on first use
ocv-pose demo                                              # synthetic player, no video needed
ocv-pose extract match.mp4 --rallies work/rallies.json --out work/kp_near.npz     # pose only near audio hits
ocv-pose analyze work/kp_near.npz --rallies work/rallies.json \
    --court 412,1820,3420,1830,2610,640,1240,636 --out work/pose.json --work work   # corners: near-L, near-R, far-R, far-L
ocv-pose compare "spring/*.json:spring" "autumn/*.json:autumn" --kind forehand    # did my forehand change?
```

`pose.json` holds the strokes (kind, confidence, evidence, stance_deg, contact_cm, finish_cm, lean_cm, court position),
contact `events` (verified only when audio and pose agree and the label passes the schema rules), `footwork`, and
`syllables` (movement pieces with a 2-D map position). On macOS you can use `--backend apple-vision` instead. Details,
assumptions and sources are in [docs/pose.md](docs/pose.md).

### Player identification *(WIP)*

See [docs/player_id.md](docs/player_id.md) for the 4-player identity tracker (`opencourtvision/players/`, optional extra `.[players]`). It uses court position, motion, clothing colour and an optional OSNet re-ID embedding. No faces. The Mac measurement tool is `tools/player_id/`. Pose and player-ID weight files are not in this repository; public pretrained weights are downloaded at runtime.

Users are responsible for complying with the privacy and biometric laws that apply where they run this software, including getting consent from the people they film.

### Shell helpers

```bash
scripts/static_reel.sh match.mp4 examples/static_reel_clips.example.txt reel.mp4 cover.jpg   # fixed-crop reel, no tracking
TITLE="FINAL · 6-4 6-3" WINNERS="Team A" LOSERS="Team B" SCORE="6-4  6-3" scripts/static_reel.sh ...
scripts/progress.sh work/ exports/      # live view of long STT/render jobs
```

## Architecture

```mermaid
flowchart LR
    V[match video] --> A[detection.audio<br/>16 kHz wav, band loudness,<br/>hit onsets, rallies]
    A --> M[detection.match_start<br/>warmup cutoff]
    A --> S[scoring.callouts<br/>STT before each rally,<br/>parse + score timeline]
    M --> R[render.cuts<br/>keyframe-snapped segments]
    R --> G[scoring.games<br/>changeovers from final score]
    R --> ST[story.ranking<br/>excitement, beats,<br/>difficult shots]
    G --> H[highlight selection<br/>pipeline.py]
    ST --> H
    S --> REP[reports<br/>segments.csv, score_timeline.csv,<br/>swingcut_report.json]
    H --> REP
    H --> L[render.cuts<br/>lossless points-only + highlights]
    A -. rallies.json .-> REEL[reel.ball + reel.reel<br/>ball track, crop path,<br/>9:16 render]
```

| Package | Responsibility |
|---|---|
| `opencourtvision/detection/` | `audio.py`: hit onsets and rally grouping. `match_start.py`: warmup cutoff and rhythm/callout estimator |
| `opencourtvision/scoring/` | `games.py`: sets and changeovers. `callouts.py`: STT, callout parsing, score timeline |
| `opencourtvision/story/` | `ranking.py`: excitement, momentum, story beats, difficult shots |
| `opencourtvision/render/` | `cuts.py`: segments (end never < last contact + 2 s), lossless cut and concat |
| `opencourtvision/labels/`, `events.py`, `eval/` | shot label schema and rules, Event + verify gate, `ocv eval` |
| `opencourtvision/importers/` | `swingvision.py`: SwingVision export → reference Events, plausibility audit (`ocv-sv-audit`) |
| `windows.py`, `stages.py` | hit windows, stage layout and content-hash stage cache |
| `opencourtvision/reel/` | `ball.py`: detection, tracking, crop path, zoom, optional TrackNet. `reel.py`: moments → 9:16 reel |
| `opencourtvision/ball/` | ball engine v2: TrackNetV3 (vendored MIT model), ROI/Kalman HSV tracker, size and court gates, live-ball filter, bounces, proxy metrics |
| `opencourtvision/pose/` | keypoints, strokes, mechanics, footwork, syllables, comparisons (`ocv-pose`); see [docs/pose.md](docs/pose.md) |
| `opencourtvision/players/` | 4-player identity tracking (court/motion/colour/re-ID cues); see [docs/player_id.md](docs/player_id.md) |
| `opencourtvision/pipeline.py` | wires the stages for `ocv-swingcut` |
| `opencourtvision/cli/` | thin argparse entry points |
| `opencourtvision/synthetic.py` | synthetic clips for tests and benchmarks |
| `benchmarks/` | `make_samples.py`, `run_eval.py`, committed ground truth |

## Benchmark

`python benchmarks/run_eval.py` generates synthetic clips (ffmpeg `testsrc` video plus decaying 2–5 kHz "pops"
at known times over noise, with distractor thumps and chatter in the noisy clip) and reports hit precision, recall
and F1 at ±50 ms, timing error, rally count, and match-start error. No real footage is used. `python benchmarks/run_pose_eval.py` benchmarks stroke labelling and mechanics on synthetic stick figures. Current output:

```
clip | gt_hits | detected | precision | recall | f1 | timing_err_ms_mean | timing_err_ms_max | gt_rallies | detected_rallies | detect_runtime_s
hits_clean | 36 | 36 | 1.0 | 1.0 | 1.0 | 3.3 | 8.0 | 5 | 5 | 0.14
hits_noisy | 36 | 15 | 1.0 | 0.417 | 0.588 | 3.4 | 8.0 | 5 | 5 | 0.13
match_sim | 234 | 234 | 1.0 | 1.0 | 1.0 | 3.5 | 8.0 | 23 | 23 | 0.64

match start (match_sim): truth 281.48 s | warmup_cutoff 281.48 s (err +0.0 s, conf high) | rhythm estimate 171.41 s (err -110.07 s)
```

Known weaknesses that this exposes: recall drops sharply at low SNR (`hits_noisy`), and the standalone rhythm
estimator fires early when its 5-minute window still overlaps the warmup. Improving either is a good first task.

### Public benches (synthetic only)

`python benchmarks/public_eval/run_eval.py` scores the point-winner decoder on a seeded synthetic bundle and always
prints trivial baselines. `python benchmarks/detector_bench.py` draws frames in memory and reports far-ball, fast-ball,
far-player, bounce and point-end recall with Wilson intervals. Neither command reads a match file. Numbers from private
matches are not part of this tree.

## Status: work in progress

This snapshot is published so people can help. It is not a finished product.

**Point-winner scoring does not yet beat simple baselines** (for example “the server wins”) on the held-out match. That is the main place help is wanted. The public harness is `python benchmarks/public_eval/run_eval.py`: it always prints always-A, always-B, server-wins and previous-winner, and exits 3 when the model does not beat the best of those. Please do not commit private match labels, footage, or faces.

Other known weak spots include far-side and fast/blurry ball tracking, court calibration / homography after the camera is bumped, far-side player detection, and p(A) calibration. See the open issues labeled `help wanted`.

How to film a match, including consent and the 18+ rule: [docs/filming.md](docs/filming.md).

Contact: antonyvo@mail.grokbot.com

| Part | Status |
|---|---|
| Audio hit/rally detection, warmup cutoff, lossless cuts, highlights | Stable core on synthetic clips; real-match numbers are not published here |
| Callout score tracking | Works for clear English/Spanish calls; mixed-language and noisy speech need work |
| Story/shot finder | WIP: multimodal ranking in progress |
| Ball-follow reel | WIP: HSV+motion tracker (+ optional TrackNet); coverage varies with lighting and ball colour |
| Ball engine v2 | ROI tracker + TrackNetV3 re-acquire + live-ball filter. Public numbers are the synthetic detector bench only |
| Point-winner scoring | Does not yet beat simple baselines on the held-out match; help wanted |
| Label schema and `ocv eval` | Present; tested on synthetic data |
| Pose analysis | Included (`ocv-pose`); tested on synthetic data. See [docs/pose.md](docs/pose.md) |
| Player-ID | Included (`opencourtvision/players/`); see [docs/player_id.md](docs/player_id.md). Weight files are not in the repo |
| Padel, 4-player doubles tracking | Not started (see `ROADMAP.md`) |

## Roadmap

See [ROADMAP.md](ROADMAP.md). Third-party licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md). Security issues (please do not file them in public): [SECURITY.md](SECURITY.md).

## Contributors and credits

Code and labels are credited by the handle the person chooses, and only if they opt in. A GitLab username in a merge request or a `labels.json` is how we know who sent the work. It is not automatic public credit. We never list a legal name, email, or face.

**Code.** Accepted merge requests show up in the git history (DCO sign-off, see [CONTRIBUTING.md](CONTRIBUTING.md)). Opt-in code contributors are listed below, by GitLab handle.

<!-- Maintainers: add a handle here only after the contributor has said yes, in the MR or by email. -->

**Labelers.** Public CC0 and CC BY points are labelled by volunteers and by the maintainers. To be named, say so when you send the file (a line in the issue comment, the merge request, or the JSON, such as `"credit_as": "your-handle"`). If you say nothing, we still use the labels and we do not put your handle on a credits list.

**Dataset releases.** Each public label set ships with:

- the CC0 / CC BY source blocks already kept in [docs/CC_ATTRIBUTION.md](docs/CC_ATTRIBUTION.md) (title, author, URL, licence, sha256, access date — whatever the licence asks for, and a source line even for CC0)
- a short labeler list of opt-in handles and how many accepted points they contributed
- no private-match labels, no footage, and no real names of people on court

How to label, and what we collect (username and A/B/S answers only): [docs/LABELING.md](docs/LABELING.md). To have your labels or credit line removed, open a confidential issue with the Delete my data template or email antonyvo@mail.grokbot.com.

## Contact

Antony Vo, antonyvo@mail.grokbot.com

## License

GNU AGPL-3.0-only. Copyright (c) 2026 Antony Vo. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Network use: a modified version that users interact with over a network must offer those users the corresponding source.
Contact: antonyvo@mail.grokbot.com.
