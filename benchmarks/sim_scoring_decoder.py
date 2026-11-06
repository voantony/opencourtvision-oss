"""SIMULATION: what global score decoding and selective prediction buy on a synthetic doubles match.

The true point winners come from tests/synthetic_match.py (a seeded rules match, not a recording). Replaying them
through the decoder's state machine reproduces every score column. Everything else is SIMULATED evidence, not model output:
  - per-point winner evidence with accuracy `a`, calibrated by construction: confidence c = 0.5 + 0.5 u, u ~ Beta(alpha, 1),
    and the evidence points to the true winner with probability c (miscal > 1 sharpens the reported logit: overconfident);
  - server-team / serve-side (deuce or ad court) / ends cues per point with accuracies q_srv, q_side, q_ends;
  - spoken score calls on a fraction r_call of points with accuracy q_call (a wrong call is a random other call);
  - optional final score (sets) as an end constraint.
Metrics, pooled over seeds: point-winner accuracy of the per-point argmax ("independent") vs the global decoder (MAP),
score-state accuracy (the score shown before each point equals the truth), and selective prediction: ask the least
confident points (re-decoding after each batch) until every point's posterior >= tau, then the final accuracy and ask rate.
The independent baseline asks its own least confident points (by c) until its pooled accuracy reaches 99.9%.

  python benchmarks/sim_scoring_decoder.py --seeds 40 --json /tmp/sim.json
"""
import argparse
import json
import math
import os
import random
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencourtvision.scoring.callout_gate import CalloutCandidate, gate_callout  # noqa: E402
from opencourtvision.scoring.decoder import OTHER, Obs, PointEvidence, confirm_loop, decode, replay, score_call, serve_side  # noqa: E402
from opencourtvision.scoring.formats import get_format  # noqa: E402

FMT = get_format("tennis", "advantage")
CALLS = ["0-0", "15-0", "0-15", "15-15", "30-0", "0-30", "30-15", "15-30", "40-0", "0-40", "30-30", "40-15", "15-40",
         "40-30", "30-40", "deuce", "ad in", "ad out"]

PARITY = dict(q_srv=0.98, q_side=0.95, q_ends=0.95)
CALLS60 = dict(r_call=0.6, q_call=0.95)
WEAK = dict(q_srv=0.95, q_side=0.80, q_ends=0.90)   # low / partial / behind-glass geometry: side boxes and ball unreliable
# Adjacent courts: Poisson(0.6) foreign calls per between-point window, each a legal-looking score for THEIR match
# (wrong for ours); 25% land during our live rally. Source cues are calibrated LLRs (ours ~ N(+2, 2), theirs ~ N(-2, 2) for the
# in-match speaker clusters; distance/DRR N(+-0.5, 1)); speaker clusters are tied to a team 90% of the time.
COURTS = dict(lam_foreign=0.6)
SCENARIOS = [
    ("a=0.80, ball only", dict(a=0.80)),
    ("a=0.80 + parity cues", dict(a=0.80, **PARITY)),
    ("a=0.80 + parity + calls 60%", dict(a=0.80, **PARITY, **CALLS60)),
    ("a=0.90, ball only", dict(a=0.90)),
    ("a=0.90 + parity cues", dict(a=0.90, **PARITY)),
    ("a=0.90 + parity + calls 60%", dict(a=0.90, **PARITY, **CALLS60)),
    ("a=0.90 + parity + final score", dict(a=0.90, **PARITY, final=True)),
    ("a=0.95 + parity cues", dict(a=0.95, **PARITY)),
    ("a=0.95 + parity + calls 60%", dict(a=0.95, **PARITY, **CALLS60)),
    ("WEAK GEOMETRY a=0.75 + weak parity", dict(a=0.75, **WEAK)),
    ("WEAK GEOMETRY a=0.75 + weak parity + calls 60%", dict(a=0.75, **WEAK, **CALLS60)),
    ("a=0.90 + parity + calls 60% + OTHER COURTS, no gate", dict(a=0.90, **PARITY, **CALLS60, **COURTS, gate=False)),
    ("a=0.90 + parity + calls 60% + OTHER COURTS, gated", dict(a=0.90, **PARITY, **CALLS60, **COURTS, gate=True)),
    ("WEAK GEOMETRY a=0.75 + calls 60% + OTHER COURTS, gated", dict(a=0.75, **WEAK, **CALLS60, **COURTS, gate=True)),
    ("LISTENER ONLY a=0.60 (audio-only winner) + calls 60%", dict(a=0.60, **CALLS60)),
    ("LISTENER ONLY a=0.60 + calls 80%", dict(a=0.60, r_call=0.8, q_call=0.95)),
    ("MISCALIBRATED a=0.90 + parity + calls (logit x2)", dict(a=0.90, **PARITY, **CALLS60, miscal=2.0)),
]
TAUS = (0.99, 0.999)


def truth():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from synthetic_match import MATCH
    P = MATCH["points"]
    W = ["A" if p["winner"] == "host" else "B" for p in P]
    first = "A" if P[0]["server"] == "host" else "B"
    S = replay(W, FMT, first_server=first)
    return W, S, first


def _ob(rng, v, acc, alts, n):
    if rng.random() < acc:
        return Obs(v, acc, n)
    return Obs(rng.choice([x for x in alts if x != v]), acc, n)


def make_evidence(W, S, sc, rng, stats):
    a = sc["a"]; alpha = (2 * a - 1) / (2 - 2 * a)
    ev, conf = [], []
    for w, s in zip(W, S):
        c = 0.5 + 0.5 * rng.betavariate(alpha, 1.0)
        right = rng.random() < c
        pa = c if (w == "A") == right else 1 - c
        if sc.get("miscal"):
            z = math.log(pa / (1 - pa)) * sc["miscal"]; pa = 1 / (1 + math.exp(-z))
        kw = {}
        if "q_srv" in sc:
            kw["server"] = _ob(rng, s.server, sc["q_srv"], "AB", 2)
            side = serve_side(s, FMT)
            if side:
                kw["side"] = _ob(rng, side, sc["q_side"], ("deuce", "ad"), 2)
            kw["ends"] = _ob(rng, s.ends, sc["q_ends"], (0, 1), 2)
        heard = []                          # (candidate, ours?)
        if sc.get("r_call") and rng.random() < sc["r_call"]:
            o = _ob(rng, score_call(s, FMT), sc["q_call"], CALLS, len(CALLS))
            heard.append((CalloutCandidate(t=5.0, call=o.value, asr_conf=sc["q_call"], speaker_llr=rng.gauss(2, 2 ** 0.5),
                                           distance_llr=rng.gauss(0.5, 1), speaker_team=s.server if rng.random() < 0.9 else None), True))
        for _ in range(_poisson(rng, sc.get("lam_foreign", 0))):
            t = 20.0 if rng.random() < 0.25 else 5.0                      # 25% during our live rally
            wrong = rng.choice([c for c in CALLS if c != score_call(s, FMT)])
            heard.append((CalloutCandidate(t=t, call=wrong, asr_conf=sc["q_call"], speaker_llr=rng.gauss(-2, 2 ** 0.5),
                                           distance_llr=rng.gauss(-0.5, 1), speaker_team=rng.choice("AB") if rng.random() < 0.5 else None), False))
        calls = []
        for c, ours in heard:
            if sc.get("gate"):
                live = [(15.0, 30.0)] if rng.random() < 0.98 else []      # our rally detector misses 2% of rallies
                r = gate_callout(c, (0.0, 15.0), live, server_team=s.server if rng.random() < sc["q_srv"] else OTHER[s.server])
                stats["ours" if ours else "foreign"] += 1
                stats[("ours_acc" if ours else "foreign_acc")] += r.accepted
                stats["foreign_weight"] += 0 if ours or not r.accepted else r.p_src
                stats["foreign_strong"] += (not ours) and r.accepted and r.p_src >= 0.5
                if r.accepted:
                    calls.append(r.obs)
            else:
                stats["ours" if ours else "foreign"] += 1
                stats["ours_acc" if ours else "foreign_acc"] += 1
                stats["foreign_weight"] += 0 if ours else 1.0
                stats["foreign_strong"] += not ours
                calls.append(Obs(c.call, c.asr_conf, len(CALLS)))
        ev.append(PointEvidence(p_a=pa, calls=tuple(calls), **kw)); conf.append(max(pa, 1 - pa))
    return ev, conf


def _poisson(rng, lam):
    k, p, L = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def run_seed(args):
    name, sc, seed = args
    W, S, first = truth()
    rng = random.Random(zlib.crc32(f"{name}|{seed}".encode()))
    stats = dict(ours=0, ours_acc=0, foreign=0, foreign_acc=0, foreign_weight=0.0, foreign_strong=0)
    ev, conf = make_evidence(W, S, sc, rng, stats)
    kw = dict(final=S[-1].sets) if sc.get("final") else {}
    ind = ["A" if e.p_a >= 0.5 else "B" for e in ev]
    d = decode(ev, FMT, **kw)
    T = len(W)

    def state_ok(winners):
        try:
            st = replay(winners, FMT, first_server=first)
        except ValueError:
            return 0
        return sum((a.sets, a.games, a.pts) == (b.sets, b.games, b.pts) for a, b in zip(st[:T], S[:T]))

    out = dict(n=T, ind_err=sum(x != y for x, y in zip(ind, W)), dec_err=sum(x != y for x, y in zip(d.winners, W)),
               stats=stats, ind_state=state_ok(ind), dec_state=state_ok(d.winners),
               ind_conf_err=[(c, x != y) for c, x, y in zip(conf, ind, W)],
               dec_conf_err=[(c, x != y) for c, x, y in zip(d.confidence, d.winners, W)])
    for tau in TAUS:
        dd, asked = confirm_loop(ev, FMT, W, tau, batch=3, **kw)
        out[f"ask_{tau}"] = len(asked)
        out[f"err_{tau}"] = sum(x != y for x, y in zip(dd.winners, W))
        out[f"match_ok_{tau}"] = int(out[f"err_{tau}"] == 0)
    return name, out


def needed_ask_rate(pairs, target=0.999):
    """Smallest fraction of points to ask (least confident first) so pooled accuracy >= target."""
    pairs = sorted(pairs, key=lambda x: x[0]); n = len(pairs); errs = sum(e for _, e in pairs)
    for k in range(n + 1):
        if errs <= (1 - target) * n:
            return k / n
        errs -= pairs[k][1] if k < n else 0
    return 1.0


def ece(pairs, bins=10):
    tot, n = 0.0, len(pairs)
    for b in range(bins):
        lo, hi = 0.5 + 0.05 * b, 0.5 + 0.05 * (b + 1)
        sel = [(c, e) for c, e in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            tot += abs(sum(c for c, _ in sel) / len(sel) - sum(1 - e for _, e in sel) / len(sel)) * len(sel) / n
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--json")
    ap.add_argument("--only", help="run only scenarios whose name contains this text")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    A = ap.parse_args(argv)
    scen = [(n, sc) for n, sc in SCENARIOS if not A.only or A.only in n]
    jobs = [(n, sc, s) for n, sc in scen for s in range(A.seeds)]
    res = {n: [] for n, _ in scen}
    with ProcessPoolExecutor(A.workers) as ex:
        for name, out in ex.map(run_seed, jobs, chunksize=4):
            res[name].append(out)
    rows = []
    for name, _ in scen:
        R = res[name]; N = sum(r["n"] for r in R)
        row = dict(scenario=name, points=N, ind_acc=1 - sum(r["ind_err"] for r in R) / N, dec_acc=1 - sum(r["dec_err"] for r in R) / N,
                   ind_state=sum(r["ind_state"] for r in R) / N, dec_state=sum(r["dec_state"] for r in R) / N,
                   ind_ask_999=needed_ask_rate([p for r in R for p in r["ind_conf_err"]]),
                   dec_ask_999_oracle=needed_ask_rate([p for r in R for p in r["dec_conf_err"]]),
                   dec_ece=ece([p for r in R for p in r["dec_conf_err"]]))
        st = {k: sum(r["stats"][k] for r in R) for k in R[0]["stats"]}
        if st["foreign"]:
            row["far"] = st["foreign_acc"] / st["foreign"]; row["frr"] = 1 - st["ours_acc"] / max(st["ours"], 1)
            row["foreign_weight"] = st["foreign_weight"] / max(st["foreign_acc"], 1); row["n_foreign"] = st["foreign"]
            row["far_strong"] = st["foreign_strong"] / st["foreign"]
        for tau in TAUS:
            row[f"ask_{tau}"] = sum(r[f"ask_{tau}"] for r in R) / N
            row[f"acc_{tau}"] = 1 - sum(r[f"err_{tau}"] for r in R) / N
            row[f"errs_{tau}"] = sum(r[f"err_{tau}"] for r in R)
            row[f"match_ok_{tau}"] = sum(r[f"match_ok_{tau}"] for r in R) / len(R)
        rows.append(row)
    print(f"SIMULATION on the dev_match match, {A.seeds} seeds x 101 points per scenario (true winners real, evidence simulated)\n")
    print("| scenario | indep acc | decoded acc | score shown right: indep / decoded | decoder ECE | ask @tau 0.99 -> acc (errs) | ask @tau 0.999 -> acc (errs), matches all-correct | indep ask for 99.9% |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scenario']} | {r['ind_acc']:.3f} | {r['dec_acc']:.3f} | {r['ind_state']:.2f} / {r['dec_state']:.2f} | {r['dec_ece']:.3f} | "
              f"{r['ask_0.99']:.1%} -> {r['acc_0.99']:.4f} ({r['errs_0.99']}) | {r['ask_0.999']:.1%} -> {r['acc_0.999']:.4f} ({r['errs_0.999']}), {r['match_ok_0.999']:.0%} | "
              f"{r['ind_ask_999']:.1%} |")
    print("\nOther-court callouts (adjacent-court injection):")
    for r in rows:
        if "far" in r:
            print(f"- {r['scenario']}: {r['n_foreign']} foreign calls; false acceptance {r['far']:.1%} (mean source weight of accepted "
                  f"{r['foreign_weight']:.2f}); accepted as strong evidence (p_src >= 0.5) {r['far_strong']:.1%}; our calls rejected {r['frr']:.1%}")
    if A.json:
        json.dump(rows, open(A.json, "w"), indent=1)


if __name__ == "__main__":
    main()
