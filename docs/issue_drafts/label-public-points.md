# Label 400+ public points

Status: filed or ready to file as a GitLab `help wanted` issue. Measure with `python benchmarks/public_eval/measure.py`.

Labels: `help wanted`, `good first issue`

## Goal

The synthetic report bundle has 48 points. A Wilson interval around a 60% accuracy is about ±5 percentage points only once n is a few hundred. Collect CC0 or CC BY points until the report set has at least 400 scored points.

## Baseline

`statsutil.n_for_wilson_halfwidth(0.6, 0.05)` is **369** trials for a 5-point half-width at p=0.6. The shipped report bundle has **n=48**, where decode's interval is [0.926, 1.000] only because the synthetic p(A) is aimed at the truth. That interval will widen on real clips. Current pinned public clips: **none** (`benchmarks/public_data/clips_manifest.json`).

## Where in the code

`docs/LABELING.md`, `docs/label_schema.json`, `docs/CC_ATTRIBUTION.md`, `scripts/fetch_public_clips.py`.

## How it's scored

`validate_labels.py` must pass on every file (no duplicate idx, no idx collision, no overlap over 2s, score replays). Each clip's licence is CC0 or CC BY, with sha256 and an attribution block. Count scored points with `winner` in {A, B}.

## Acceptance

- At least 400 scored points across the report split, or a written split update in `benchmarks/public_eval/split.json` that keeps tune and report disjoint.
- No media committed. `scripts/check_no_media.py` passes.
