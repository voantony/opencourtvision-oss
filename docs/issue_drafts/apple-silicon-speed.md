# Apple Silicon speed

Status: filed or ready to file as a GitLab `help wanted` issue. Measure with `python benchmarks/public_eval/measure.py`.

Labels: `help wanted`, `good first issue`

## Goal

Record detector-bench frames per second on an Apple Silicon Mac, at the same 640×360 in-memory frames, and keep the primary recalls intact.

## Baseline

On the Linux host that prepared this draft, `benchmarks/detector_bench.py` reported about **292 CPU fps** over 80 frames. That number is not an M-series measurement and is not a 4K match. Primary recalls are in `replace-yolo11-detector.md`.

## Where in the code

`benchmarks/detector_bench.py` (the timed loop). Device choice for the optional TrackNet path is `opencourtvision/ball/tracknet3.py` `pick_device` (CUDA, then MPS, then CPU). This issue is the HSV bench first, not a new model.

## How it's scored

Paste the JSON from `python benchmarks/detector_bench.py` run on the Mac. Primary recalls must not regress beyond their CI half-widths versus the committed baseline table. fps is secondary: report it; a drop larger than 10% needs a sentence on why.

## Acceptance

- A short note in the pull request with the Mac's fps, the five primary recalls with n, and the machine (chip, not a person's name).
- No video file added.
