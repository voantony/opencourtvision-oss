# Contributing

This repository is a public work-in-progress snapshot of OpenCourtVision (Vamos). Help is wanted, especially on point-winner scoring, which does not yet beat simple baselines on the held-out match.

There is no CLA. A Developer Certificate of Origin sign-off is required. The code is AGPL-3.0-only; contributions are accepted under that licence.

Questions: antonyvo@mail.grokbot.com.

## Privacy

- Do not commit footage, audio, images, weight files, transcripts, or labels from a private match.
- Do not commit faces or anything that identifies other players.
- Everyone on court agrees, and everyone is 18 or older. Ask the other players before you film. Never film or upload a match with minors in it.
- How to film: [docs/filming.md](docs/filming.md).
- Player names in tool output should stay as roles such as H1/H2 or near/far, not real names.

`python3 scripts/check_no_media.py` must pass before you open a merge request.

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[reel,dev]"
python3 scripts/check_no_media.py
python -m pytest -q
python -m ruff check .
```

Python 3.10+. ffmpeg is needed for clip tests and the labeler browser test.

## Measure a fix

```bash
python benchmarks/public_eval/measure.py            # far-player recall and p(A) ECE, with intervals
python benchmarks/public_eval/measure.py --check    # fail only if this run is worse than baseline.json
python benchmarks/public_eval/run_eval.py           # point-winner harness; prints trivial baselines
```

`measure.py` is the scoreboard for far-side player detection and p(A) calibration. Under it:

- `benchmarks/detector_bench.py` draws frames in memory. Far-player recall is the player number. The other primary recalls (far-half ball, fast/blurry ball, bounce, point end) must not fall by more than their own 95% CI half-width.
- `benchmarks/public_eval/run_eval.py` validates labels, always prints always-A, always-B, server-wins and previous-winner, and exits 3 when the model does not beat the best of those. Bootstrap CIs and McNemar are marked DISTINGUISHABLE or NOT DISTINGUISHABLE. `synth_report` is the held-out synthetic bundle. `synth_biased` is the calibration tripwire: p(A) is 0.95 on every point, and that run must keep failing the baseline gate.

Do not tune on `synth_report` and then call the number held-out (`benchmarks/public_eval/split.json`). `python benchmarks/run_eval.py` is the separate audio-hit bench on synthetic clips.

A detector change must not regress a primary metric in `detector_bench.py` by more than that metric's 95% CI half-width. Primary metrics are far-half ball recall, fast/blurry ball recall, far-player recall, bounce recall and point-end recall. mAP and fps are secondary.

## Labels

Read `docs/LABELING.md`. Only CC0 and CC BY clips, with a line in `docs/CC_ATTRIBUTION.md` and a sha256 in `benchmarks/public_data/clips_manifest.json`. `scripts/fetch_public_clips.py` refuses a download that has no checksum. Submit the label JSON in a merge request. Do not commit the media.

## Code of conduct

See `CODE_OF_CONDUCT.md`.

## DCO sign-off

This project does not use a contributor licence agreement. Every commit must end with:

```
Signed-off-by: Your Name <you@example.com>
```

That certifies the [Developer Certificate of Origin](https://developercertificate.org/) version 1.1: you wrote the change, or you have the right to submit it under the AGPL-3.0-only, and you understand the project will redistribute it under that licence.

```bash
git commit -s
```

## Merge requests

1. Branch from `main`, keep the change focused, and include the sign-off.
2. `ruff check .` and `pytest` must pass. Detection changes should paste `benchmarks/public_eval/run_eval.py` and, if you touched a detector, `benchmarks/detector_bench.py`.
3. This project lives on GitLab. Open a merge request there.

Known problems ready for help are filed as GitLab issues with the `help wanted` label. Draft write-ups also live in `docs/issue_drafts/`.
