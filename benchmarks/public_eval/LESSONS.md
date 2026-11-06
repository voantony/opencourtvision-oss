# How the public harness stays honest

These are structural guards. They are not measurements of any private match. Evidence for each guard is a synthetic bundle this tree can regenerate.

| Guard | What it prevents | Where |
|---|---|---|
| Trivial baselines on every run | Quoting a model accuracy that does not beat always-A, always-B, server-wins, or previous-winner | `run_eval.py` exits 3 |
| Bootstrap CI and McNemar | Calling a small gap a win when the interval covers zero (`NOT DISTINGUISHABLE`) | `statsutil.py` |
| Frozen tune / report split | Tuning a threshold on the same bundle you then call held-out | `split.json`, `heldout.py` |
| Validator before scores | Duplicate idxs, idx collisions, long overlaps, p(A) length mismatches, score drift | `validate_labels.py` |
| Calibration block | Treating a constant p(A) as information. The `synth_biased` probe is confident and useless | `run_eval.py` prints Brier, ECE, and argmax error bias |
| Component metrics before end-to-end claims | A point-winner number with no far-player or fast-ball recall beside it | `benchmarks/detector_bench.py` |
| One scoreboard for the current fixes | Far-player recall and p(A) ECE, plus a check that fails only when a number gets worse | `measure.py` against `baseline.json` |

```bash
python benchmarks/public_eval/run_eval.py
python benchmarks/public_eval/run_eval.py --bundle synth_biased --allow-below-baseline
python benchmarks/detector_bench.py
```
