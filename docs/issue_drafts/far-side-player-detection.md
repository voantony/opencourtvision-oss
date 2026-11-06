# Far-side player detection

Status: filed or ready to file as a GitLab `help wanted` issue. Measure with `python benchmarks/public_eval/measure.py`.

Labels: `help wanted`

## Goal

The synthetic bench does not see the far player at all. A contribution should raise far-player recall without losing the other primary recalls.

## Baseline

`benchmarks/detector_bench.py`, seed 3, 80 frames. Far-player recall **0.000**, 95% CI **[0.000, 0.046]**, **0/80**. The stand-in detector only keeps saturated blue blobs; the far player is drawn dull on purpose. This is not a measurement from a private match.

## Where in the code

`benchmarks/detector_bench.py` `detect_players`. The product path that still calls YOLO11 for people is `tools/point_winner/pw_tracks.py` (`YOLO(a.pose)`). Player-ID weights are not v1 release assets (`weights/MANIFEST.json`).

## How it's scored

Far-player recall with a Wilson interval and the sample count, from the same script. It must not regress far-half ball recall, fast-ball recall, bounce recall, or point-end recall beyond each metric's CI half-width (see `replace-yolo11-detector.md` for the table).

## Acceptance

- Far-player recall point estimate above 0.046 (the current CI upper bound) on the default seed, with `n` reported.
- No primary regression.
- AGPL-compatible code only.
