"""SIMULATION: forward-only scoring (D1, Legal 18 Oct 2026) vs the full-app global decoder, on the dev_match match.

Same truth and the same simulated evidence as benchmarks/sim_scoring_decoder.py (identical seeds -> identical evidence).
Configurations:
  global          full-app whole-match decoder (Viterbi + forward-backward; compiled out of the US app, off by default in full)
  fwd point_end   D1 default: each point decided once when it ends, from its own evidence; frozen afterwards
  fwd +voice      + players' gated score calls treated as voice corrections of the last point / score prompts (Legal Q2)
  fwd next_serve  point held as pending "?" through the between-point window, committed once at the next serve (Legal Q1)
Metrics pooled over seeds: first-call point accuracy (no taps), score shown right (state before each point == truth),
confidence ECE, then prompts: sweep tau, the user answers each prompt (point prompts and score prompts both count as taps);
report the smallest tap rate whose pooled point accuracy >= 99.9%. Global: tap rate by posterior order (oracle sort, no
re-decode) and confirm_loop at tau 0.999 (--loop).
Anchor: one user-typed score before point 50 (truth), point_end policy: share of matches flagged, points in the flagged
span, review taps (the user checks suggestions in stored first-call confidence order until the span adds up), and the tap
rate for 99.9% with anchor + prompts.

  python benchmarks/sim_forward_scoring.py --seeds 40 --json /tmp/fwd.json
"""
import argparse
import json
import os
import random
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.sim_scoring_decoder import CALLS60, COURTS, FMT, PARITY, WEAK, ece, make_evidence, needed_ask_rate, truth  # noqa: E402
from opencourtvision.scoring.forward import ForwardScorer, same_score  # noqa: E402

SCENARIOS = [
    ("a=0.80 + parity cues", dict(a=0.80, **PARITY)),
    ("a=0.90 + parity cues", dict(a=0.90, **PARITY)),
    ("a=0.90 + parity + calls 60%", dict(a=0.90, **PARITY, **CALLS60)),
    ("a=0.95 + parity + calls 60%", dict(a=0.95, **PARITY, **CALLS60)),
    ("a=0.90 + parity + calls 60% + OTHER COURTS, gated", dict(a=0.90, **PARITY, **CALLS60, **COURTS, gate=True)),
    ("WEAK GEOMETRY a=0.75 + weak parity + calls 60%", dict(a=0.75, **WEAK, **CALLS60)),
    ("NO BALL CURVE: audio winner a=0.60 + calls 60%", dict(a=0.60, **CALLS60)),
    ("NO BALL CURVE: audio winner a=0.70 + calls 80%", dict(a=0.70, r_call=0.8, q_call=0.95)),
]
TAUS = (0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.01)
ANCHOR_AT = 50
VARIANTS = {"point_end": dict(policy="point_end"), "checks": dict(policy="point_end", score_checks=0.1),
            "voice": dict(policy="point_end", voice_edits=True, score_checks=0.1),
            "next_serve": dict(policy="next_serve"), "suggest": dict(policy="point_end", suggestions=True),
            "shipped": dict(policy="point_end", voice_edits=True, score_checks=0.1, suggestions=True),
            "shipped_strict_q2": dict(policy="point_end", voice_edits=True, score_checks=0.1, suggestions=True, min_src=0.8,
                                      min_call_acc=0.9)}


def _state_ok(states, S):
    return sum(same_score(a, b) for a, b in zip(states[:len(S)], S))


def run_fwd(ev, W, S, kw, tau=None, anchor=False):
    def ask(i, kind):
        if kind == "suggestion":
            return all(W[j] == w for j, w in i.fix)
        if kind == "score_check":
            return ("fix",) if i.fix is not None and all(W[j] == w for j, w in i.fix.fix) else ("edit", S[i.at])
        return W[i] if kind == "point" else S[i]
    s = ForwardScorer(FMT, tau=tau, ask=ask, **kw)
    flagged = span = review = 0
    for i, e in enumerate(ev):
        if anchor and i == ANCHOR_AT:
            n0 = len(s.prompts)
            f = s.anchor_and_review(S[i])                 # user types the score (1 tap) + checks suggestions
            review = len(s.prompts) - n0 + 1
            if f:
                flagged, span = 1, f.last - f.first + 1
        s.point_ended(e)
    s.finish()
    err = sum(x != y for x, y in zip(s.winners, W))
    return dict(err=err, state=_state_ok(s.states, S), taps=len(s.prompts) + (1 if anchor else 0), prompts=len(s.prompts),
                review=review, flagged=flagged, span=span, voice=sum(x.source == "voice" for x in s.edits),
                score_prompts=sum(k == "score" for _, k in s.prompts),
                conf_err=[(c.confidence, x != y) for c, x, y in zip(s.commits, s.winners, W)])


def run_seed(args):
    name, sc, seed, loop = args
    W, S, first = truth()
    S = S[:len(W)]
    rng = random.Random(zlib.crc32(f"{name}|{seed}".encode()))
    stats = dict(ours=0, ours_acc=0, foreign=0, foreign_acc=0, foreign_weight=0.0, foreign_strong=0)
    ev, _ = make_evidence(W, S, sc, rng, stats)
    from opencourtvision.variants.full.scoring_global import confirm_loop, decode  # full-app module (benchmark only)
    d = decode(ev, FMT)
    out = dict(n=len(W), g_err=sum(x != y for x, y in zip(d.winners, W)), g_state=_state_ok(d.states, S),
               g_conf_err=[(c, x != y) for c, x, y in zip(d.confidence, d.winners, W)])
    if loop:
        dd, asked = confirm_loop(ev, FMT, W, 0.999, batch=3)
        out["g_loop_ask"] = len(asked); out["g_loop_err"] = sum(x != y for x, y in zip(dd.winners, W))
    for v, kw in VARIANTS.items():
        out[v] = run_fwd(ev, W, S, kw)
        out[v + "_tau"] = {t: {k: x for k, x in run_fwd(ev, W, S, kw, tau=t).items() if k != "conf_err"} for t in TAUS}
    out["anchor"] = run_fwd(ev, W, S, VARIANTS["point_end"], anchor=True)
    out["anchor_tau"] = {t: {k: x for k, x in run_fwd(ev, W, S, VARIANTS["point_end"], tau=t, anchor=True).items()
                             if k != "conf_err"} for t in TAUS}
    out["anchor_sugg"] = run_fwd(ev, W, S, VARIANTS["suggest"], anchor=True)
    out["anchor_sugg_tau"] = {t: {k: x for k, x in run_fwd(ev, W, S, VARIANTS["suggest"], tau=t, anchor=True).items()
                                  if k != "conf_err"} for t in TAUS}
    return name, out


def tap_for(R, key, N, target=0.999):
    best = None
    for t in TAUS:
        taps = sum(r[key][t]["taps"] for r in R) / N; acc = 1 - sum(r[key][t]["err"] for r in R) / N
        if acc >= target and (best is None or taps < best[0]):
            best = (taps, acc, t)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--json")
    ap.add_argument("--loop", action="store_true", help="also run the global confirm_loop at tau 0.999 (slow)")
    ap.add_argument("--only")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    A = ap.parse_args(argv)
    scen = [(n, sc) for n, sc in SCENARIOS if not A.only or A.only in n]
    res = {n: [] for n, _ in scen}
    with ProcessPoolExecutor(A.workers) as ex:
        for name, out in ex.map(run_seed, [(n, sc, s, A.loop) for n, sc in scen for s in range(A.seeds)], chunksize=2):
            res[name].append(out)
    rows = []
    for name, _ in scen:
        R = res[name]; N = sum(r["n"] for r in R)
        row = dict(scenario=name, points=N, g_acc=1 - sum(r["g_err"] for r in R) / N, g_state=sum(r["g_state"] for r in R) / N,
                   g_tap999_oracle=needed_ask_rate([p for r in R for p in r["g_conf_err"]]))
        if A.loop:
            row["g_loop_ask"] = sum(r["g_loop_ask"] for r in R) / N; row["g_loop_acc"] = 1 - sum(r["g_loop_err"] for r in R) / N
        for v in list(VARIANTS) + ["anchor", "anchor_sugg"]:
            row[v] = dict(acc=1 - sum(r[v]["err"] for r in R) / N, state=sum(r[v]["state"] for r in R) / N,
                          ece=ece([p for r in R for p in r[v]["conf_err"]]), voice_edits=sum(r[v]["voice"] for r in R) / N,
                          taps0=sum(r[v]["taps"] for r in R) / N,
                          tap999=tap_for(R, v + "_tau", N), tap99=tap_for(R, v + "_tau", N, 0.99))
        row["anchor"].update(flagged=sum(r["anchor"]["flagged"] for r in R) / len(R),
                             span=sum(r["anchor"]["span"] for r in R) / max(1, sum(r["anchor"]["flagged"] for r in R)),
                             review_taps=sum(r["anchor"]["review"] for r in R) / len(R))
        rows.append(row)
    print(f"SIMULATION, dev_match match, {A.seeds} seeds x {rows[0]['points'] // A.seeds} points (true winners real, evidence simulated)\n")
    print("| scenario | global acc / score right / taps@99.9% | point_end acc / score right / ECE / taps@99.9% | +score checks (Q3) acc / score / taps | +voice (Q2) acc / score / taps | next_serve (Q1) acc / score / taps |")
    print("|---|---|---|---|---|---|")
    f = lambda b: "n/a" if b is None else f"{b[0]:.1%}"  # noqa: E731
    for r in rows:
        p, v, n, c = r["point_end"], r["voice"], r["next_serve"], r["checks"]
        g = f"{r['g_acc']:.3f} / {r['g_state']:.2f} / {r['g_tap999_oracle']:.1%}" + (f" (loop {r['g_loop_ask']:.1%})" if "g_loop_ask" in r else "")
        print(f"| {r['scenario']} | {g} | {p['acc']:.3f} / {p['state']:.2f} / {p['ece']:.3f} / {f(p['tap999'])} | "
              f"{c['acc']:.3f} ({c['taps0']:.1%} taps) / {c['state']:.2f} / {f(c['tap999'])} | {v['acc']:.3f} / {v['state']:.2f} / {f(v['tap999'])} | {n['acc']:.3f} / {n['state']:.2f} / {f(n['tap999'])} |")
    print(f"\nOne anchor before point {ANCHOR_AT} (point_end):")
    for r in rows:
        a = r["anchor"]
        print(f"- {r['scenario']}: flagged {a['flagged']:.0%} of matches, span {a['span']:.0f} points, review taps {a['review_taps']:.1f}/match; "
              f"score right {a['state']:.2f} (vs {r['point_end']['state']:.2f}); taps@99.9% with anchor {f(a['tap999'])}")
    print("\nSuggested backward fixes (user-approved; one approve = 1 tap), taps per 100 points to reach 99.9%:")
    for r in rows:
        p, g, a, sa = r["point_end"], r["suggest"], r["anchor"], r["anchor_sugg"]
        print(f"- {r['scenario']}: forward-only {f(p['tap999'])} | + suggestions: no-tau acc {g['acc']:.3f} ({g['taps0']:.1%} taps), "
              f"score right {g['state']:.2f}, 99.9% at {f(g['tap999'])} | anchor+suggestions 99.9% at {f(sa['tap999'])} (anchor alone {f(a['tap999'])})")
    print("\nSHIPPED (Q2 + Q3 + suggestions; Legal 05:41), taps per 100 points:")
    for r in rows:
        g = r["shipped"]; h = r["shipped_strict_q2"]
        print(f"  strict Q2 gate (p_src>=0.8): no-tau acc {h['acc']:.3f} ({h['taps0']:.1%} taps), score right {h['state']:.2f}; "
              f"99.9% at {f(h['tap999'])}; 99% at {f(h['tap99'])}")
        print(f"- {r['scenario']}: no-tau acc {g['acc']:.3f} ({g['taps0']:.1%} taps), score right {g['state']:.2f}; 99.9% at {f(g['tap999'])}; "
              f"99% at {f(g['tap99'])}")
    if A.json:
        json.dump(rows, open(A.json, "w"), indent=1)


if __name__ == "__main__":
    main()
