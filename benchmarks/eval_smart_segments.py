#!/usr/bin/env python3
"""Smart dead space (editloop/smart.py) vs the shipped / PR #12 segmenters, per build, on a labelled match, with a
tune/test split by time (first half of the labelled points = tune, second half = test). Labels are read only; outputs
(segment CSVs, JSON) go to --out, which must be outside the repo for private matches.

  python benchmarks/eval_smart_segments.py --labels PRIVATE.labels.json --features features.npz --hits rallies.json \
      --sv-fixture points.json --sv-xlsx shots.xlsx --final "6-4 6-3" \
      [--motion motion.npz] [--baseline segments.csv] --out /tmp/smartseg [--set KEY=VALUE ...]
"""
import argparse
import copy
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from benchmarks import scoreboard as SB  # noqa: E402
from opencourtvision.editloop import segment as SG  # noqa: E402
from opencourtvision.editloop.smart import smart_points  # noqa: E402
from opencourtvision.labels.points import load_point_labels  # noqa: E402
from opencourtvision.scoring.games import parse_final  # noqa: E402


def write_csv(path, clips):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "parts", "hits", "fault_first", "quiet_serve", "reaction_db", "reaction_ext"])
        for c in clips:
            parts = c.get("parts") or [(c["A"], c["B"])]
            w.writerow([f"{parts[0][0]:.2f}", f"{c['B']:.2f}", ";".join(f"{a:.2f}_{b:.2f}" for a, b in parts), c["hits"],
                        int(bool(c.get("fault_first"))), int(bool(c.get("quiet_serve"))), c.get("reaction_db", ""),
                        int(bool(c.get("reaction_ext")))])


def variants(le, bg, lb, start, sets, cues=None, over=None):
    P, R = SG.audio_points(le, bg, start)
    P3, _ = SG.gap_fill(P, R, start, sets)
    out = {"pr12_us": SG.clip_bounds(SG.attach_faults(P3))}
    out["smart_us"], _ = smart_points(P3, le, bg, lb, sets, over)
    if cues is not None:
        from opencourtvision.editloop.motion import apply_cues, live_gate
        out["pr12_eu"] = SG.clip_bounds(apply_cues(P3, cues, R, start, sets))
        out["smart_us_gate"] = live_gate(out["smart_us"], cues)
        from opencourtvision.editloop.motion import admit_candidates
        cand = admit_candidates(P3, cues, R, start, sets)     # EU: motion admission; boundaries still from audio here
        out["smart_eu"], _ = smart_points(cand, le, bg, lb, sets, over)
    return out


def sub_labels(L, t0, t1):
    M = copy.copy(L)
    M.segments = [s for s in L.segments if t0 <= s.start < t1]
    return M


def evaluate(L, path, hits, shots, t0=None, t1=None, matcher="window"):
    M = L if t0 is None else sub_labels(L, t0, t1)
    preds, parts = SB.read_intervals(path), SB.read_parts(path)
    if t0 is not None:
        keep = [k for k, p in enumerate(preds) if t0 <= p[0] < t1]
        preds, parts = [preds[k] for k in keep], ([parts[k] for k in keep] if parts else None)
    ct = SB.contacts(M, hits, shots)
    return SB.stage_metrics(M, preds, hits, ct, parts, matcher=matcher)


def row(name, m):
    return (f"{name:18s} rec {m['found']}/{m['n_points']}={100*m['recall']:.1f}% prec {100*m['precision']:.1f}% "
            f"non {m['non_points']}/{m['n_pred']} dead {m['dead_s']:.2f}s clip {m['clip_len']:.1f}s "
            f"cut serve {m['cut_serve']} celeb {m['cut_celebration']}")


def main():
    ap = argparse.ArgumentParser()
    for k in ("labels", "features", "hits", "sv_fixture", "sv_xlsx", "final", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--motion"); ap.add_argument("--baseline"); ap.add_argument("--sv-clips")
    ap.add_argument("--matcher", choices=("window", "label"), default="window",
                    help="match clips to points by serve..last-contact window (default) or by labelled span (pre-18 Oct)")
    ap.add_argument("--set", action="append", default=[], help="override a SMART parameter, KEY=VALUE")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import ast
    over = {k: ast.literal_eval(v) for k, v in (s.split("=", 1) for s in a.set)}
    L = load_point_labels(a.labels)
    z = np.load(a.features); le, bg, lb = z["le"], z["bg"], z["lb"]
    hits = sorted(t for r in json.load(open(a.hits)) for t in r.get("times", []))
    shots = SB.read_sv_shots(a.sv_xlsx)
    cues = None
    if a.motion:
        from opencourtvision.editloop.motion import MotionCues
        cues = MotionCues.load(a.motion)
    V = variants(le, bg, lb, a.start, parse_final(a.final), cues, over)
    paths = {}
    if a.baseline: paths["shipped"] = a.baseline
    if a.sv_clips: paths["swingvision"] = a.sv_clips
    for k, clips in V.items():
        paths[k] = os.path.join(a.out, f"{k}.csv"); write_csv(paths[k], clips)
    pts = [s for s in L.segments if s.kind == "point"]
    h = len(pts) // 2
    split = (pts[h - 1].end + pts[h].start) / 2
    res = {}
    for part, (t0, t1) in {"all": (None, None), "tune (1st half)": (0, split), "test (2nd half)": (split, 1e9)}.items():
        print(f"== {part}" + (f"  split at {split:.1f} s" if t0 is not None else ""))
        for k, pth in paths.items():
            m = evaluate(L, pth, hits, shots, t0, t1, a.matcher); res.setdefault(part, {})[k] = m
            print(row(k, m))
    json.dump(dict(split=split, results=res), open(os.path.join(a.out, "eval_smart.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
