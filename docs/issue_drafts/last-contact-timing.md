# Last-contact timing

Status: filed or ready to file as a GitLab `help wanted` issue. Measure with `python benchmarks/public_eval/measure.py`.

Labels: `help wanted`

## Goal

Point-winner logic needs the time of the last contact, the hitter as a court position, and the bounce. The public tree has no ground truth for those on a real clip yet. The labeling schema now asks for them so contributors can create it.

## Baseline

There is no public last-contact recall yet, and this draft does not invent one. What the synthetic bench does measure downstream:

- bounce recall **0.000** [0.000, 0.161], **0/20** (`benchmarks/detector_bench.py`)
- point-end recall **1.000** [0.839, 1.000], **20/20**

The honest p(A) bundle is not a last-contact test. Do not cite a private-match timing error here.

## Where in the code

Label fields: `docs/label_schema.json` `shots[].t` and `bounces[]`. Validator: `benchmarks/public_eval/validate_labels.py`. The research reader that consumes a last-contact time is `tools/point_winner/pw_stages.py` (audio contact window). Hitter positions allowed by the schema are `near_deuce`, `near_ad`, `far_deuce`, `far_ad` only.

## How it's scored

Once a CC label file has per-shot times, report last-contact recall at ±80 ms with a Wilson interval and n, plus hitter-slot accuracy and bounce recall. A contribution must not regress detector-bench primary metrics beyond their CI half-widths, and must not weaken the point-winner baseline gate.

## Acceptance

- A scorer in `benchmarks/public_eval/` that reads `shots` and `bounces` and prints those intervals.
- At least one synthetic fixture in the test suite where a known contact time is recovered or a miss is counted. No private labels.
