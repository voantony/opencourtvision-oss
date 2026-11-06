# Ball recall on fast, blurry frames

Status: filed or ready to file as a GitLab `help wanted` issue. Measure with `python benchmarks/public_eval/measure.py`.

Labels: `help wanted`, `good first issue`

## Goal

First serves and overheads are streaks. On the synthetic bench the HSV candidate function scores none of those frames.

## Baseline

`benchmarks/detector_bench.py`, seed 3. Fast/blurry ball recall **0.000**, 95% CI **[0.000, 0.194]**, **0/16**. Far-half ball recall on sharp frames is 0.800 [0.652, 0.895] (32/40), so the hole is the streak, not "any ball".

## Where in the code

`opencourtvision/ball/hsv.py` `candidates` (HSV range and the motion mask). The bench draws the streak in `benchmarks/detector_bench.py` `draw`.

## How it's scored

Fast/blurry ball recall, Wilson CI, sample count. Do not regress far-half ball recall, far-player recall, bounce recall, or point-end recall beyond their CI half-widths.

## Acceptance

- Fast/blurry recall point estimate above 0.194 on the default invocation, `n` at least 16.
- No primary regression. No new media files.
