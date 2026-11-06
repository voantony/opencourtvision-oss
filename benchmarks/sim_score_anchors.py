"""SIMULATION: score anchors (MD moment_search_spec v0.6) on the dev_match match. True winners are real; evidence is simulated
exactly as in sim_scoring_decoder.py. For each seed: decode with no anchor, with the final result only ("6-1 6-2" = set
scores + sets), and with the final result plus k full anchors (sets, games, points, server) at random points. Reports the
per-point accuracy of the decoded path (no other questions asked).

  python benchmarks/sim_score_anchors.py --seeds 20
"""
import argparse
import os
import random
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim_scoring_decoder import CALLS60, FMT, PARITY, WEAK, make_evidence, truth  # noqa: E402

from opencourtvision.scoring.decoder import Anchor, confirm_loop, decode, replay  # noqa: E402

SCEN = [("a=0.90 + parity cues", dict(a=0.90, **PARITY, asks=True)),
        ("a=0.80 + parity cues", dict(a=0.80, **PARITY)),
        ("a=0.90 + parity + calls 60%", dict(a=0.90, **PARITY, **CALLS60, asks=True)),
        ("WEAK GEOMETRY a=0.75 + weak parity", dict(a=0.75, **WEAK))]
KS = ("none", "final", 1, 2, 8, 16, 32)
ASK_KS = ("none", "final", 1, 2)
SET_SCORES = [(6, 1), (6, 2)]


def anchor_of(st, t):
    if st.tb:
        pts = st.pts
    elif min(st.pts) >= 3:
        pts = "deuce" if st.pts[0] == st.pts[1] else ("ad us" if st.pts[0] > st.pts[1] else "ad them")
    else:
        pts = tuple(["0", "15", "30", "40"][x] for x in st.pts)
    return Anchor(t=t, sets=st.sets, games=st.games, pts=pts, server="us" if st.server == "A" else "them")


def run(args):
    name, sc, seed = args
    W, S, first = truth()
    rng = random.Random(zlib.crc32(f"{name}|{seed}".encode()))
    stats = dict(ours=0, ours_acc=0, foreign=0, foreign_acc=0, foreign_weight=0.0, foreign_strong=0)
    ev, _ = make_evidence(W, S, sc, rng, stats)
    out = {}
    for k in KS:
        kw = {}
        if k != "none":
            kw = dict(final=S[-1].sets, set_scores=SET_SCORES)
        if isinstance(k, int):
            idx = rng.sample(range(1, len(W)), k)
            kw["anchors"] = {i: anchor_of(S[i], float(i)) for i in idx}
        d = decode(ev, FMT, **kw)
        out[k] = sum(a != b for a, b in zip(d.winners, W))
        st = replay(d.winners, FMT, first_server=d.states[0].server, ends0=d.states[0].ends)
        out[f"state_{k}"] = sum((a.sets, a.games, a.pts) != (b.sets, b.games, b.pts) for a, b in zip(st[:len(W)], S[:len(W)]))
        if k in ASK_KS and sc.get("asks"):
            dd, asked = confirm_loop(ev, FMT, W, 0.999, batch=3, **kw)
            out[f"ask_{k}"] = len(asked); out[f"askerr_{k}"] = sum(a != b for a, b in zip(dd.winners, W))
    return name, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    A = ap.parse_args(argv)
    res = {n: [] for n, _ in SCEN}
    with ProcessPoolExecutor(A.workers) as ex:
        for name, out in ex.map(run, [(n, sc, s) for n, sc in SCEN for s in range(A.seeds)]):
            res[name].append(out)
    print(f"SIMULATION, score anchors, {A.seeds} seeds x 101 points. Cell = point-winner accuracy / score-shown accuracy of "
          "the decoded path with no other questions (anchors are full states at random points)\n")
    print("| scenario | no anchor | final result only | final + 1 | final + 2 | final + 8 | final + 16 | final + 32 |")
    print("|---|---|---|---|---|---|---|---|")
    for n, _ in SCEN:
        R = res[n]; N = 101 * len(R)
        cells = [f"{1 - sum(r[k] for r in R) / N:.3f} / {1 - sum(r[f'state_{k}'] for r in R) / N:.3f}" for k in KS]
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\nAsks to reach 99.9% (tau = 0.999) with anchors: ask rate (errors left)\n")
    print("| scenario | no anchor | final result only | final + 1 | final + 2 |")
    print("|---|---|---|---|---|")
    for n, sc in SCEN:
        if sc.get("asks"):
            R = res[n]; N = 101 * len(R)
            print(f"| {n} | " + " | ".join(f"{sum(r[f'ask_{k}'] for r in R) / N:.1%} ({sum(r[f'askerr_{k}'] for r in R)})" for k in ASK_KS) + " |")


if __name__ == "__main__":
    main()
