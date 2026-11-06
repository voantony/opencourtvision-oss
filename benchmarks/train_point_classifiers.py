#!/usr/bin/env python3
"""Train / validate the learned onset + stretch classifiers (editloop/learned.py) on a labelled match, split-half.

Labels are read only; models and segment CSVs go to --out (keep it outside the repo: the weights are derived from
private labels and are not committed). Truth per onset comes from the labelled points' played rallies (serve contact
and last contact; SwingVision shot times snapped to audio hits, see benchmarks/scoreboard.contacts):
  in-play = between serve - TOL and last contact + TOL;  serve = the onset nearest the serve contact (<= TOL).
Truth per stretch: point (contains a labelled serve), fault/let (ends <= 25 s before a labelled serve and overlaps a
labelled point or non-point segment), junk (the rest).

  python benchmarks/train_point_classifiers.py --labels PRIVATE.labels.json --features features.npz --wav audio_16k.wav \
      --hits rallies.json --sv-xlsx shots.xlsx --out /private/dir
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from benchmarks import eval_smart_segments as E  # noqa: E402
from benchmarks import scoreboard as SB  # noqa: E402
from opencourtvision.editloop import learned as LN  # noqa: E402
from opencourtvision.labels.points import load_point_labels  # noqa: E402

TOL = 0.35


def onset_truth(t, L, ct):
    segs = L.segments
    rallies = [ct[i] for i, s in enumerate(segs) if s.kind == "point" and i in ct]
    inplay = np.zeros(len(t)); serve = np.zeros(len(t))
    for fh, lh in rallies:
        inplay[(t >= fh - TOL) & (t <= lh + TOL)] = 1
        k = int(np.argmin(np.abs(t - fh)))
        if abs(t[k] - fh) <= TOL: serve[k] = 1
    return inplay, serve, rallies


def stretch_truth(S, rallies, L):
    fhs = sorted(fh for fh, _ in rallies)
    used, y_point, y_fault = set(), [], []
    labelled = [(s.start, s.end) for s in L.segments if s.kind in ("point", "non_point")]
    for st in S:
        hit = [fh for fh in fhs if st["t0"] - 0.5 <= fh <= st["t1"] + 0.3 and fh not in used]
        if hit:
            used.add(hit[0]); y_point.append(1); y_fault.append(0); continue
        y_point.append(0)
        nxt = [fh for fh in fhs if fh > st["t1"]]
        ov = any(SB.ov((st["t0"], st["t1"] + 0.01), x) > 0 for x in labelled)
        y_fault.append(int(bool(nxt) and nxt[0] - st["t1"] <= 25.0 and ov))
    return np.array(y_point), np.array(y_fault)


MODEL = LN.GBDT


def fit(t, a, X, inplay, serve, rallies, L, mask, s=None):
    m_in = MODEL().fit(X[mask], inplay[mask])
    m_sv = MODEL().fit(X[mask], serve[mask])
    p_in, p_sv = m_in.predict(X), m_sv.predict(X)
    S = LN.stretches(t, a, X, p_in, p_sv, s)
    SF = LN.stretch_features(S, t, a, X, p_in, p_sv)
    yp, yf = stretch_truth(S, rallies, L)
    sm = np.array([mask[st["idx"][0]] for st in S])
    return LN.Models(m_in, m_sv, MODEL(n=40, depth=2).fit(SF[sm], yp[sm]), MODEL(n=40, depth=2).fit(SF[sm], yf[sm]))


def sub(L, t0, t1):
    M = copy.copy(L); M.segments = [x for x in L.segments if t0 <= x.start < t1]; return M


def main():
    ap = argparse.ArgumentParser()
    for k in ("labels", "features", "wav", "hits", "sv_xlsx", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--no-fault-part", action="store_true", help="drop the first-serve fault lead-in from clips")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import scipy.io.wavfile as wf
    L = load_point_labels(a.labels)
    z = np.load(a.features); le, bg, lb = z["le"], z["bg"], z["lb"]
    sr, x = wf.read(a.wav); x = x.astype(np.float32) / 32768
    hits = sorted(v for r in json.load(open(a.hits)) for v in r.get("times", []))
    shots = SB.read_sv_shots(a.sv_xlsx)
    ct = SB.contacts(L, hits, shots)
    t, amp, X = LN.onset_features(le, bg, lb, x)
    keep = t >= a.start; t, amp, X = t[keep], amp[keep], X[keep]
    inplay, serve, rallies = onset_truth(t, L, ct)
    pts = [s for s in L.segments if s.kind == "point"]; h = len(pts) // 2
    split = (pts[h - 1].end + pts[h].start) / 2
    A, B = t < split, t >= split
    s = {"fault_part": None} if a.no_fault_part else None
    res = {}
    clips = {}
    for name, mask in (("trainA", A), ("trainB", B)):
        M = fit(t, amp, X, inplay, serve, rallies, L, mask, s)
        M.save(os.path.join(a.out, f"models_{name}.json"))
        out, dbg = LN.learned_points(le, bg, lb, x, M, s, feats=(t, amp, X))
        clips[name] = out
        p_in = dbg["p_in"]
        for part, m in (("A", A), ("B", B)):
            acc = float(((p_in[m] >= 0.5) == (inplay[m] > 0)).mean())
            res[f"{name}/onset_acc_{part}"] = round(acc, 3)
    # cross-fitted: B from the A-trained models, A from the B-trained models
    cross = [c for c in clips["trainA"] if c["t0"] >= split] + [c for c in clips["trainB"] if c["t0"] < split]
    cross.sort(key=lambda c: c["t0"])
    fit_all = fit(t, amp, X, inplay, serve, rallies, L, np.ones(len(t), bool), s)
    fit_all.save(os.path.join(a.out, "models_all.json"))
    for name, cl in (("learned_trainA", clips["trainA"]), ("learned_trainB", clips["trainB"]), ("learned_cross", cross)):
        p = os.path.join(a.out, f"{name}.csv"); E.write_csv(p, cl)
        for part, (t0, t1) in (("all", (None, None)), ("A", (0, split)), ("B", (split, 1e9))):
            m = E.evaluate(L, p, hits, shots, t0, t1); res[f"{name}/{part}"] = m
            print(E.row(f"{name}/{part}", m))
    print({k: v for k, v in res.items() if "onset_acc" in k})
    json.dump(dict(split=split, results=res), open(os.path.join(a.out, "train_eval.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
