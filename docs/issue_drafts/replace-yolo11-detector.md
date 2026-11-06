# Replace YOLO11 with a stronger player/ball detector

Status: filed or ready to file as a GitLab `help wanted` issue. Measure with `python benchmarks/public_eval/measure.py`.

Labels: `help wanted`

## Goal

Replace the Ultralytics YOLO11-pose call with a stronger detector whose licence can sit inside an AGPL-3.0-only project. RF-DETR (Apache-2.0) is an example, not a requirement. AGPL-3.0 itself, MIT, BSD and Apache-2.0 are compatible. GPL-2.0-only, non-commercial, research-only and proprietary terms are not.

## Where in the code

`tools/point_winner/pw_tracks.py`: `from ultralytics import YOLO` and `pose = YOLO(a.pose)`. Ball candidates that do not need that import already live in `opencourtvision.ball.hsv.candidates`. TrackNetV3 re-acquire is `opencourtvision.ball.tracknet3.TrackNetV3`. A replacement should expose the same boxes and, if it does pose, the same keypoint array the rest of `pw_tracks.py` writes. Do not start the swap inside this snapshot; the import stays until a contribution lands.

## Baseline (synthetic, reproducible)

`python benchmarks/detector_bench.py` (seed 3, 80 frames, 640×360, drawn in memory on the machine that prepared this draft). Not a private match.

| Metric | Role | Estimate | 95% Wilson CI | k/n |
|---|---|---|---|---|
| Ball recall, far half | primary | 0.800 | [0.652, 0.895] | 32/40 |
| Ball recall, fast/blurry | primary | 0.000 | [0.000, 0.194] | 0/16 |
| Far-player recall | primary | 0.000 | [0.000, 0.046] | 0/80 |
| Bounce recall | primary | 0.000 | [0.000, 0.161] | 0/20 |
| Point-end recall | primary | 1.000 | [0.839, 1.000] | 20/20 |
| Ball average precision | secondary | 1.000 on emitted boxes | n = 64 boxes | |
| CPU fps | secondary | ~292 | this Linux host, not Apple Silicon | 80 frames |

## How a contribution is scored

Re-run the same script. Lead with the five primary recalls. mAP and fps are secondary. A primary metric regresses when the new point estimate is below `baseline_p - baseline_half` (the half-width of the CI above). The contribution must not regress any primary metric. Licence of the new code and weights must be AGPL-compatible, and weights are downloaded with a checksum, not committed.

## Acceptance

- YOLO11 is no longer imported on the default path.
- Primary metrics do not regress beyond the rule above, and far-player recall and fast-ball recall are the ones this bench shows as empty.
- `pytest` and `scripts/check_no_media.py` pass. No image or video file is added.
