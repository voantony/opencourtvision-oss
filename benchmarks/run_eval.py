#!/usr/bin/env python3
"""Detection benchmark on synthetic clips: hit precision/recall/F1 and timing error, rally count,
and match-start error for both detectors. Generates the clips first if missing.

  python benchmarks/run_eval.py [--tol 0.05] [--json results.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from opencourtvision.detection.audio import detect_hits, group_rallies, load_audio, loudness  # noqa: E402
from opencourtvision.detection.match_start import estimate_match_start, warmup_cutoff  # noqa: E402
from opencourtvision.synthetic import SCENARIOS, make_clip, scenario  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def match_hits(pred, truth, tol):
    """Greedy one-to-one matching within tol seconds; returns tp, fp, fn, abs errors."""
    pred, truth = sorted(pred), sorted(truth)
    used, errs = set(), []
    for t in truth:
        best = None
        for j, p in enumerate(pred):
            if j in used or abs(p - t) > tol:
                continue
            if best is None or abs(p - t) < abs(pred[best] - t):
                best = j
        if best is not None:
            used.add(best)
            errs.append(abs(pred[best] - t))
    tp = len(errs)
    return tp, len(pred) - tp, len(truth) - tp, errs


def evaluate(name, data_dir, tol):
    src = os.path.join(data_dir, f"{name}.mp4")
    if not os.path.exists(src):
        make_clip(scenario(name), data_dir)
    gt = json.load(open(os.path.join(data_dir, f"{name}.json")))
    t0 = time.time()
    sr, x = load_audio(src)
    le, bg, _ = loudness(sr, x)
    t, amp = detect_hits(le, bg)
    R = group_rallies(t, amp)
    rt = time.time() - t0
    tp, fp, fn, errs = match_hits(list(t), gt["hits"], tol)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    res = dict(clip=name, audio_s=round(len(x) / sr, 1), gt_hits=len(gt["hits"]), detected=len(t), tp=tp, fp=fp, fn=fn,
               precision=round(prec, 3), recall=round(rec, 3), f1=round(2 * prec * rec / max(1e-9, prec + rec), 3),
               timing_err_ms_mean=round(1000 * float(np.mean(errs)), 1) if errs else None,
               timing_err_ms_max=round(1000 * float(np.max(errs)), 1) if errs else None,
               gt_rallies=len(gt["rallies"]), detected_rallies=len(R), detect_runtime_s=round(rt, 2))
    if "match_start_s" in gt:
        start, ev = warmup_cutoff(R)
        res["match_start_gt_s"] = gt["match_start_s"]
        res["warmup_cutoff_s"] = round(start, 2)
        res["warmup_cutoff_err_s"] = round(start - gt["match_start_s"], 2)
        res["warmup_cutoff_conf"] = ev.get("confidence")
        rs = estimate_match_start(src)
        est = rs.get("estimate_s")
        res["rhythm_estimate_s"] = None if est is None else round(est, 2)
        res["rhythm_estimate_err_s"] = None if est is None else round(est - gt["match_start_s"], 2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--tol", type=float, default=0.05, help="hit match tolerance (s)")
    ap.add_argument("--json", help="write results JSON here")
    A = ap.parse_args(argv)
    rows = [evaluate(n, A.data, A.tol) for n in SCENARIOS]
    cols = ["clip", "gt_hits", "detected", "precision", "recall", "f1", "timing_err_ms_mean", "timing_err_ms_max", "gt_rallies", "detected_rallies", "detect_runtime_s"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    for r in rows:
        if "match_start_gt_s" in r:
            print(f"\nmatch start ({r['clip']}): truth {r['match_start_gt_s']} s | warmup_cutoff {r['warmup_cutoff_s']} s "
                  f"(err {r['warmup_cutoff_err_s']:+} s, conf {r['warmup_cutoff_conf']}) | rhythm estimate {r['rhythm_estimate_s']} s "
                  f"(err {r['rhythm_estimate_err_s']} s)")
    if A.json:
        json.dump(rows, open(A.json, "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
