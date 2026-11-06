"""Paired bootstrap, exact McNemar, and Wilson intervals for public point-winner eval.

No recorded match is referenced. Callers pass per-point correct/incorrect vectors.
"""
from __future__ import annotations

import math
import random
from math import comb
from typing import Sequence


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    """95% Wilson score interval for k successes in n trials."""
    if n <= 0:
        return {"p": 0.0, "lo": 0.0, "hi": 0.0, "k": 0, "n": 0, "half": 0.0}
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return {"p": p, "lo": lo, "hi": hi, "k": int(k), "n": int(n), "half": (hi - lo) / 2}


def bootstrap_diff(a_correct: Sequence[int], b_correct: Sequence[int], n_boot: int = 4000, seed: int = 42) -> dict:
    """Mean of (acc_a - acc_b) with a percentile CI. Positive means a is better than b."""
    n = len(a_correct)
    if n != len(b_correct) or n <= 0:
        raise ValueError("bootstrap_diff needs two equal non-empty vectors")
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_correct[i] for i in idx) / n
        db = sum(b_correct[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "mean": sum(diffs) / n_boot,
        "lo": lo,
        "hi": hi,
        "distinguishable_at_95": not (lo <= 0.0 <= hi),
        "n_boot": n_boot,
        "seed": seed,
    }


def mcnemar(base_correct: Sequence[int], model_correct: Sequence[int]) -> dict:
    """Exact two-sided McNemar. n01 = base wrong and model right; n10 = base right and model wrong."""
    n01 = sum(1 for b, m in zip(base_correct, model_correct) if b == 0 and m == 1)
    n10 = sum(1 for b, m in zip(base_correct, model_correct) if b == 1 and m == 0)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0, "p_value": 1.0}
    k = min(n01, n10)
    cdf = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n01": n01, "n10": n10, "n_discordant": n, "p_value": min(1.0, 2 * cdf)}


def n_for_wilson_halfwidth(p: float = 0.6, halfwidth: float = 0.05, z: float = 1.96) -> int:
    return math.ceil((z ** 2) * p * (1 - p) / (halfwidth ** 2))


def ece_10bin(p_a: Sequence[float], winners: Sequence[str]) -> float:
    """10-bin expected calibration error of p(A). Lower is better."""
    n = len(p_a)
    if n == 0 or n != len(winners):
        raise ValueError("ece_10bin needs p_a and winners of the same non-zero length")
    y = [1.0 if w == "A" else 0.0 for w in winners]
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i, p in enumerate(p_a) if (lo <= p < hi) or (hi == 1.0 and p == 1.0)]
        if not idx:
            continue
        mp = sum(p_a[i] for i in idx) / len(idx)
        fa = sum(y[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(mp - fa)
    return ece


def brier(p_a: Sequence[float], winners: Sequence[str]) -> float:
    y = [1.0 if w == "A" else 0.0 for w in winners]
    n = len(p_a)
    return sum((p_a[i] - y[i]) ** 2 for i in range(n)) / n


def bootstrap_ece(p_a: Sequence[float], winners: Sequence[str], n_boot: int = 2000, seed: int = 42) -> dict:
    """Percentile CI for 10-bin ECE. half is what a later run may not exceed."""
    n = len(p_a)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(ece_10bin([p_a[i] for i in idx], [winners[i] for i in idx]))
    vals.sort()
    lo = vals[int(0.025 * n_boot)]
    hi = vals[min(n_boot - 1, int(0.975 * n_boot))]
    point = ece_10bin(p_a, winners)
    return {
        "ece": point,
        "brier": brier(p_a, winners),
        "lo": lo,
        "hi": hi,
        "half": (hi - lo) / 2,
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
    }


def recall_regressed(new: dict, baseline: dict) -> bool:
    """True when the new point estimate falls below baseline_p - baseline half-width."""
    return float(new["p"]) < float(baseline["p"]) - float(baseline["half"]) - 1e-12


def ece_regressed(new_ece: float, baseline: dict) -> bool:
    """True when ECE rises by more than the baseline interval's half-width."""
    return float(new_ece) > float(baseline["ece"]) + float(baseline["half"]) + 1e-12
