#!/usr/bin/env python3
"""Public point-winner eval.

Gates, in order:
  1. validate_labels must pass
  2. trivial baselines are always printed (always-A, always-B, server-wins, previous-winner)
  3. exit 3 if the model does not beat the best trivial baseline on point accuracy
  4. every comparison prints a paired bootstrap CI, McNemar, and DISTINGUISHABLE or NOT DISTINGUISHABLE
  5. the bundle's tune/report role is printed (IN-SAMPLE vs HELD-OUT)

Default input is a synthetic bundle. Pass --labels and --p for a CC label file you produced
under docs/LABELING.md. This script never reads a private match path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from heldout import claim_label, roles  # noqa: E402
from statsutil import bootstrap_diff, mcnemar, n_for_wilson_halfwidth, wilson  # noqa: E402
from synthetic import BUNDLES  # noqa: E402
from validate_labels import validate  # noqa: E402

from opencourtvision.scoring.decoder import PointEvidence, decode  # noqa: E402
from opencourtvision.scoring.formats import get_format  # noqa: E402


def _corr(preds, winners):
    return [1 if a == b else 0 for a, b in zip(preds, winners)]


def _calibrate(p_a, winners):
    y = [1 if w == "A" else 0 for w in winners]
    n = len(y)
    bins = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i, p in enumerate(p_a) if (lo <= p < hi) or (hi == 1.0 and p == 1.0)]
        if not idx:
            bins.append({"bin": [lo, hi], "n": 0})
            continue
        mp = sum(p_a[i] for i in idx) / len(idx)
        fa = sum(y[i] for i in idx) / len(idx)
        bins.append({"bin": [lo, hi], "n": len(idx), "mean_p": mp, "frac_A": fa, "gap": mp - fa})
    brier = sum((p_a[i] - y[i]) ** 2 for i in range(n)) / n
    ece = sum((r["n"] / n) * abs(r["gap"]) for r in bins if r["n"])
    strong = sum(1 for p in p_a if p <= 0.2 or p >= 0.8)
    return {"brier": brier, "ece_10bin": ece, "reliability": bins, "n_strong": strong, "frac_strong": strong / n, "n": n}


def evaluate(bundle_id, labels, p_a, allow_below_baseline=False):
    errors, warnings, pts = validate(labels, p_a, name=bundle_id)
    if errors:
        print(f"[{bundle_id}] VALIDATOR FAIL: {errors}", file=sys.stderr)
        raise SystemExit(2)
    pts = sorted(pts, key=lambda p: float(p["start"]))
    winners = [p["winner"] for p in pts]
    n = len(winners)
    first = (labels.get("match") or {}).get("first_server") or "A"
    fmt = get_format((labels.get("match") or {}).get("sport") or "tennis", (labels.get("match") or {}).get("preset"))
    servers = [p.get("server") if p.get("server") in ("A", "B") else "A" for p in pts]
    ev = [PointEvidence(p_a=float(p), p_point=1.0) for p in p_a]
    decoded = list(decode(ev, fmt, first_server=first).winners)
    argmax = ["A" if p >= 0.5 else "B" for p in p_a]
    baselines = {
        "model_decode": decoded,
        "argmax": argmax,
        "always_A": ["A"] * n,
        "always_B": ["B"] * n,
        "server_wins": servers,
        "prev_winner": ["A"] + winners[:-1],
    }
    vecs = {name: _corr(pred, winners) for name, pred in baselines.items()}
    rows = {name: wilson(sum(v), n) for name, v in vecs.items()}
    trivial = ["always_A", "always_B", "server_wins", "prev_winner"]
    best_name = max(trivial, key=lambda k: (rows[k]["p"], k))
    model_name = "model_decode"
    beats = rows[model_name]["p"] > rows[best_name]["p"] + 1e-12
    vs = bootstrap_diff(vecs[model_name], vecs[best_name])
    pairs = [("model_decode", "argmax"), ("model_decode", "always_A"), ("model_decode", "always_B"),
             ("model_decode", "server_wins"), ("model_decode", "prev_winner"), ("argmax", "always_A")]
    comparisons = {}
    for a, b in pairs:
        bd = bootstrap_diff(vecs[a], vecs[b])
        mc = mcnemar(vecs[b], vecs[a])
        comparisons[f"{a}_minus_{b}"] = {
            "boot": bd,
            "mcnemar": mc,
            "verdict": "DISTINGUISHABLE" if bd["distinguishable_at_95"] else "NOT DISTINGUISHABLE",
        }
    wrong = [i for i in range(n) if argmax[i] != winners[i]]
    try:
        claim = claim_label(bundle_id)
    except ValueError:
        claim = "UNASSIGNED (not in the frozen split)"
    report = {
        "bundle": bundle_id,
        "claim_role": claim,
        "heldout_roles": roles(),
        "n": n,
        "validator_warnings": warnings,
        "baselines": rows,
        "best_trivial": {"name": best_name, **rows[best_name]},
        "model": {"name": model_name, **rows[model_name]},
        "beats_best_trivial": beats,
        "model_minus_best_trivial": {**vs, "verdict": "DISTINGUISHABLE" if vs["distinguishable_at_95"] else "NOT DISTINGUISHABLE"},
        "comparisons": comparisons,
        "calibration": _calibrate(p_a, winners),
        "argmax_error_bias": {
            "n_wrong": len(wrong),
            "pred_A": sum(argmax[i] == "A" for i in wrong),
            "pred_B": sum(argmax[i] == "B" for i in wrong),
        },
        "A_rate_truth": wilson(sum(w == "A" for w in winners), n),
        "power_hints": {"n_for_CI_halfwidth_5pp_at_p60": n_for_wilson_halfwidth(0.6, 0.05)},
        "licence": (labels.get("video") or {}).get("licence", "unspecified"),
    }
    print(f"\n===== {bundle_id} [{claim}] n={n} =====")
    for k, v in rows.items():
        print(f"  {k:16s} {v['p']:.3f}  95% CI [{v['lo']:.3f}, {v['hi']:.3f}]  ({v['k']}/{v['n']})")
    print(f"best trivial: {best_name}={rows[best_name]['p']:.3f}")
    print(f"model beats best trivial: {beats}  "
          f"boot Δ={vs['mean']:+.3f} CI [{vs['lo']:+.3f}, {vs['hi']:+.3f}] "
          f"{'DISTINGUISHABLE' if vs['distinguishable_at_95'] else 'NOT DISTINGUISHABLE'}")
    cal = report["calibration"]
    print(f"calibration Brier={cal['brier']:.4f} ECE={cal['ece_10bin']:.4f} n={cal['n']}")
    for ck, c in comparisons.items():
        print(f"  {ck}: Δ={c['boot']['mean']:+.3f} CI [{c['boot']['lo']:+.3f}, {c['boot']['hi']:+.3f}] "
              f"{c['verdict']} McNemar p={c['mcnemar']['p_value']:.3f}")
    if not beats and not allow_below_baseline:
        print(f"BASELINE GATE FAIL: {model_name} does not beat {best_name}", file=sys.stderr)
        raise SystemExit(3)
    if beats and not vs["distinguishable_at_95"]:
        print("NOTE: beats the trivial baseline on the point estimate but NOT DISTINGUISHABLE at 95%", file=sys.stderr)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="synth_report", choices=sorted(BUNDLES))
    ap.add_argument("--labels", help="CC or synthetic label JSON. Default: the named synthetic bundle")
    ap.add_argument("--p", help="p(A) JSON. Default: the synthetic bundle's vector")
    ap.add_argument("--allow-below-baseline", action="store_true")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    if a.labels:
        labels = json.load(open(a.labels))
        raw = json.load(open(a.p))
        p_a = raw["p_a"] if isinstance(raw, dict) else raw
        bundle_id = Path(a.labels).stem
    else:
        built = BUNDLES[a.bundle]()
        labels, p_a, bundle_id = built["labels"], built["p_a"], built["id"]
    report = evaluate(bundle_id, labels, p_a, allow_below_baseline=a.allow_below_baseline)
    if a.json_out:
        json.dump(report, open(a.json_out, "w"), indent=2)
        print(f"wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
