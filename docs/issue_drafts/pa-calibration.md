# Calibrate p(A)

Status: filed or ready to file as a GitLab `help wanted` issue. Measure with `python benchmarks/public_eval/measure.py`.

Labels: `help wanted`

## Goal

A constant, over-confident p(A) must not look like a model. The public harness already fails that probe. Keep the gate, and make a real estimator that beats it once CC labels exist.

## Baseline

Synthetic, `python benchmarks/public_eval/run_eval.py`.

Honest bundle `synth_report` (n=48), p(A) drawn toward the true winner by construction (this is not a claim about real matches):

- decode accuracy 1.000, 95% CI [0.926, 1.000] (48/48)
- best trivial baseline always-A 0.646 [0.504, 0.766] (31/48)
- difference +0.352, bootstrap CI [+0.229, +0.500], **DISTINGUISHABLE**
- Brier 0.0327, 10-bin ECE 0.1789

Biased probe `synth_biased` (same winners, p(A)=0.95 on every point):

- decode ties always-A at 0.646, bootstrap CI [0, 0], **NOT DISTINGUISHABLE**
- the process exits 3 unless `--allow-below-baseline` is set
- Brier 0.3212, ECE 0.3042
- every argmax error predicts A

## Where in the code

`opencourtvision/scoring/decoder.py` `decode` (consumes `PointEvidence.p_a`). The report is `benchmarks/public_eval/run_eval.py` (`_calibrate`, baseline gate).

## How it's scored

On `synth_report`, still beat the best trivial baseline with a distinguishable interval. On `synth_biased`, the gate must keep failing if someone wires that vector in as the model. ECE and Brier are printed every run. A change that only moves accuracy inside the CI is not a result (`NOT DISTINGUISHABLE`).

## Acceptance

- `run_eval.py` with no arguments still exits 0.
- `run_eval.py --bundle synth_biased` still exits 3.
- Any new estimator reports Brier, ECE, n, and the argmax error bias, and does not drop `synth_report` below the baseline gate.
