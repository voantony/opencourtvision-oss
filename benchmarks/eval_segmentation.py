#!/usr/bin/env python3
"""Point segmentation vs human point labels (labeler `ocv.point_labels.v1`), step by step: E1, E1+E3, fault attach,
optional motion cues. Labels are read only and never written or copied; pass their path explicitly.

  python benchmarks/eval_segmentation.py --labels PRIVATE.labels.json --features work/features.npz --start 879.45 \
      --final "6-1 6-2" [--baseline work/segments.csv] [--motion motion.npz]

Metric (greedy 1:1 by overlap; a match needs >= 50% of the labelled segment or of the proposal):
recall = labelled points matched / points; precision = proposals matched to a point / proposals;
x = proposals matched to a labelled non-point (fault/let); frag = other proposals inside labelled segments; junk = the rest.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from opencourtvision.editloop import segment as SG  # noqa: E402
from opencourtvision.scoring.games import parse_final  # noqa: E402


def _ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def score(props, L):
    P = [q for q in L if q["kind"] == "point"]
    lab = [q for q in L if q["kind"] in ("point", "non_point")]
    pairs = []
    for i, (a, b) in enumerate(props):
        for q in lab:
            o = _ov(a, b, q["start"], q["end"])
            if o > 0 and (o >= 0.5 * (q["end"] - q["start"]) or o >= 0.5 * (b - a)):
                pairs.append((o, i, q["idx"], q["kind"]))
    pairs.sort(reverse=True); up, uq, M = set(), set(), {}
    for o, i, j, k in pairs:
        if i in up or j in uq: continue
        up.add(i); uq.add(j); M[i] = k
    n = len(props); pts = sum(k == "point" for k in M.values())
    rest = [i for i in range(n) if i not in M]
    junk = sum(1 for i in rest if not any(_ov(props[i][0], props[i][1], q["start"], q["end"]) > 0 for q in lab))
    return dict(n=n, recall=round(pts / len(P), 3), precision=round(pts / n, 3) if n else 0.0, points=pts,
                x=sum(k == "non_point" for k in M.values()), frag=len(rest) - junk, junk=junk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--features", required=True)
    ap.add_argument("--start", type=float, required=True); ap.add_argument("--final", required=True)
    ap.add_argument("--baseline"); ap.add_argument("--motion"); ap.add_argument("--json")
    a = ap.parse_args()
    L = json.load(open(a.labels))["points"]
    z = np.load(a.features); le, bg = z["le"], z["bg"]; sets = parse_final(a.final)
    rows = []
    if a.baseline:
        rows.append(("S0 shipped segmenter", score([(float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(open(a.baseline))], L)))
    P, R = SG.audio_points(le, bg, a.start)
    clips = lambda Q: [(s["A"], s["B"]) for s in SG.clip_bounds(Q)]  # noqa: E731
    rows.append(("S1 E1 audio points", score(clips(P), L)))
    P3, _ = SG.gap_fill(P, R, a.start, sets)
    rows.append(("S2 E1+E3 gap fill", score(clips(P3), L)))
    rows.append(("S3 + fault attach (audio)", score(clips(SG.attach_faults(P3)), L)))
    if a.motion:
        from opencourtvision.editloop.motion import MotionCues, apply_cues, live_gate
        cues = MotionCues.load(a.motion)
        rows.append(("S4 EU + motion cues (flag on)", score(clips(apply_cues(P3, cues, R, a.start, sets)), L)))
        rows.append(("S4-US motion live gate only", score(clips(live_gate(SG.attach_faults(P3), cues)), L)))
    for name, r in rows:
        print(f"{name:28s} " + " ".join(f"{k}={v}" for k, v in r.items()))
    if a.json: json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
