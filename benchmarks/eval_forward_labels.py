"""Forward-only scoring (D1) on REAL signals vs the human-labelled dev_match match, next to the full-app global decoder.

Same inputs as benchmarks/eval_point_labels.py (SwingVision per-point winner + server, gated score callouts from the
between-point STT windows; segments = the labelled segments). Truth answers every prompt (point prompts, score checks,
anchor review). Reports per config: first-call point accuracy, score shown right, final right, taps; the smallest tap
count with zero point errors (99.9% on ~100 points) over a tau sweep; one manual anchor mid-match (flagged span, review
taps).

  python benchmarks/eval_forward_labels.py LABELS.json --sv points.json --stt STT.json [--global]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.eval_point_labels import build, callouts, match_sv, metrics  # noqa: E402
from opencourtvision.labels.points import load_point_labels  # noqa: E402
from opencourtvision.scoring.forward import ForwardScorer  # noqa: E402

TAUS = (None, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01)
VARIANTS = {"point_end": dict(policy="point_end"), "point_end + suggested fixes": dict(policy="point_end", suggestions=True),
            "point_end + score checks (Q3)": dict(policy="point_end", score_checks=0.1),
            "+ voice edits (Q2)": dict(policy="point_end", score_checks=0.1, voice_edits=True),
            "SHIPPED: Q2 + Q3 + suggestions": dict(policy="point_end", score_checks=0.1, voice_edits=True, suggestions=True),
            "next_serve (Q1, disabled)": dict(policy="next_serve")}


def run_fwd(ev, segs, L, truth, TS, kw, tau=None, anchor_at=None):
    def ask(i, k):
        if k == "suggestion":                       # the user approves only a fix that is right on every point
            return all(truth[j] == w for j, w in i.fix)
        if k == "score_check":                      # Q3: fix if the one fix is right, else free edit (true score)
            if i.fix is not None and all(truth[j] == w for j, w in i.fix.fix):
                return ("fix",)
            return ("edit", TS[i.at])
        return TS[i] if k == "score" else truth[i]
    s = ForwardScorer(L.fmt, tau=tau, ask=ask, **kw)
    extra = flagged = span = review = 0
    for i, e in enumerate(ev):
        if anchor_at is not None and i == anchor_at:
            n0 = len(s.prompts)
            f = s.anchor_and_review(TS[i]); extra = 1; review = len(s.prompts) - n0
            if f:
                flagged, span = 1, f.last - f.first + 1
        s.point_ended(e)
    s.finish()
    m = metrics(segs, L, TS, s)
    pts = [i for i, sg in enumerate(segs) if sg.kind == "point"]
    m.update(taps=len(s.prompts) + extra, pt_errs=sum(s.winners[i] != segs[i].choice for i in pts),
             seg_errs=sum(s.winners[i] != truth[i] for i in range(len(segs)) if segs[i].choice is not None),
             flagged=flagged, span=span, review=review, voice_edits=sum(x.source == "voice" for x in s.edits),
             sugg_shown=sum(1 for _ in s.suggestion_log), sugg_approved=sum(v == "approved" for _, v in s.suggestion_log),
             sugg_points_fixed=sum(x.source == "suggestion" for x in s.edits))
    return m


def calibrate_sv(segs, svmap, prior=0.82, k=4.0, prior_pt=0.7):
    """Leave-one-out calibration on the labels (in-sample, one match): P(SV winner right | SV ending category) and
    P(segment is a point | SV matched or not). Each segment's value excludes that segment itself."""
    import collections
    cat = lambda q: (q.get("ending") or "none")  # noqa: E731
    right, n = collections.Counter(), collections.Counter()
    pt, npt = collections.Counter(), collections.Counter()
    for i, sg in enumerate(segs):
        if sg.choice is None:
            continue
        m = i in svmap
        npt[m] += 1; pt[m] += sg.kind == "point"
        if m and sg.kind == "point":
            q = svmap[i]; w = "A" if q["winner"] == "host" else "B"
            n[cat(q)] += 1; right[cat(q)] += w == sg.choice
    out = {}
    for i, sg in enumerate(segs):
        m = i in svmap
        own_pt = 1 if (sg.choice is not None) else 0
        is_pt = 1 if sg.kind == "point" and sg.choice is not None else 0
        p_point = (pt[m] - is_pt + k * prior_pt) / (npt[m] - own_pt + k)
        c = None
        if m:
            q = svmap[i]; w = "A" if q["winner"] == "host" else "B"
            own = 1 if (sg.kind == "point" and sg.choice is not None) else 0
            ok = 1 if own and w == sg.choice else 0
            c = (right[cat(q)] - ok + k * prior) / (n[cat(q)] - own + k)
        out[i] = (c, p_point)
    return out


def build_calibrated(segs, svmap, calls, cal, sv_srv_acc=0.95):
    from opencourtvision.scoring.rules import Obs, PointEvidence
    ev = []
    for i, sg in enumerate(segs):
        c, pp = cal[i]
        kw = dict(p_point=min(max(pp, 0.02), 0.995))
        if c is not None:
            q = svmap[i]; w = "A" if q["winner"] == "host" else "B"
            kw.update(p_a=c if w == "A" else 1 - c, server=Obs("A" if q["server"] == "host" else "B", sv_srv_acc))
        if i in calls:
            obs = [o for o, _, _ in calls[i]]
            kw["call"] = obs[0]
            if len(obs) > 1:
                kw["calls"] = tuple(obs[1:])
        ev.append(PointEvidence(**kw))
    return ev


TAUS_FINE = (None, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 0.999, 1.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("--sv"); ap.add_argument("--stt"); ap.add_argument("--json")
    ap.add_argument("--sv-acc", type=float, default=0.90)
    ap.add_argument("--global", dest="glob", action="store_true", help="also run the full-app global decoder (slow)")
    a = ap.parse_args()
    L = load_point_labels(a.labels)
    sv = json.load(open(a.sv))["points"]; stt = json.load(open(a.stt))
    segs = L.segments; TS = L.states()
    truth = [sg.choice if sg.choice is not None else "skip" for sg in segs]
    svmap = match_sv(segs, sv); calls = callouts(segs, stt)
    n_pts = sum(sg.kind == "point" for sg in segs)
    mid = next(i for i, sg in enumerate(segs) if sg.kind == "point" and sum(x.kind == "point" for x in segs[:i]) >= n_pts // 2)
    out = dict(n_segments=len(segs), n_points=n_pts, anchor_before_segment=mid, configs={})
    for cname, us, uc in (("SV + calls", True, True), ("calls only (no ball curve)", False, True), ("SV only", True, False)):
        ev = build(segs, svmap, calls, us, uc, a.sv_acc, 0.95)
        rows = {}
        for v, kw in VARIANTS.items():
            sweep = {str(t): run_fwd(ev, segs, L, truth, TS, kw, tau=t) for t in TAUS}
            zero = [(m["taps"], t) for t, m in sweep.items() if m["seg_errs"] == 0]
            rows[v] = dict(first=sweep["None"], taps_to_zero_errors=min(zero) if zero else None,
                           anchor=run_fwd(ev, segs, L, truth, TS, kw, anchor_at=mid))
        if a.glob and cname == "SV + calls":
            from opencourtvision.variants.full.scoring_global import confirm_loop, decode
            d0 = decode(ev, L.fmt); d1, asked = confirm_loop(ev, L.fmt, truth, 0.999)
            rows["GLOBAL decoder (full app)"] = dict(first=metrics(segs, L, TS, d0), asks_0999=len(asked),
                                                     after=metrics(segs, L, TS, d1, asked))
        out["configs"][cname] = rows
    cal = calibrate_sv(segs, svmap)
    evc = build_calibrated(segs, svmap, calls, cal)
    n_lab = sum(sg.choice is not None for sg in segs)
    shipped = {}
    for v in ("point_end", "SHIPPED: Q2 + Q3 + suggestions", "point_end + score checks (Q3)", "+ voice edits (Q2)"):
        sweep = {str(t): run_fwd(evc, segs, L, truth, TS, VARIANTS[v], tau=t) for t in TAUS_FINE}
        res = {}
        for tgt in (0.99, 0.999):
            ok = [(m["taps"], t) for t, m in sweep.items() if m["seg_errs"] <= (1 - tgt) * n_lab]
            res[str(tgt)] = (min(ok)[0] * 100 / n_pts, min(ok)[1]) if ok else None
        shipped[v] = dict(first=sweep["None"], taps_per_100=res)
    out["calibrated"] = shipped
    print(f"{len(segs)} segments, {n_pts} points; anchor before segment {mid}")
    print("\n== CALIBRATED SV (leave-one-out by ending category) + calls")
    for v, r in shipped.items():
        f = r["first"]; t = r["taps_per_100"]
        fmt_ = lambda x: "n/a" if x is None else f"{x[0]:.0f} (tau {x[1]})"  # noqa: E731
        print(f"  {v:32s} no-tau: pt acc {f['pt_acc']:.1%} score right {f['score_acc']:.1%} taps {f['taps']} "
              f"(sugg {f['sugg_shown']}/{f['sugg_approved']}) | taps/100 pts @99%: {fmt_(t['0.99'])} @99.9%: {fmt_(t['0.999'])}")
    for cname, rows in out["configs"].items():
        print(f"\n== {cname}")
        for v, r in rows.items():
            f = r["first"]
            if "asks_0999" in r:
                print(f"  {v:32s} first-call pt acc {f['pt_acc']:.1%} score right {f['score_acc']:.1%} final {f['final_ok']} | "
                      f"tau .999 asks {r['asks_0999']} -> score right {r['after']['score_acc']:.1%} unasked acc {r['after']['unasked_acc']:.1%}")
                continue
            z = r["taps_to_zero_errors"]; an = r["anchor"]
            print(f"  {v:32s} first-call pt acc {f['pt_acc']:.1%} score right {f['score_acc']:.1%} final {f['final_ok']} taps {f['taps']} "
                  f"(sugg {f['sugg_shown']}/{f['sugg_approved']} appr, {f['sugg_points_fixed']} pts) | "
                  f"zero errors with {('%d taps (tau %s)' % z) if z else 'n/a'} | anchor: flagged {an['flagged']} span {an['span']} "
                  f"review taps {an['review']} -> pt acc {an['pt_acc']:.1%} score right {an['score_acc']:.1%}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
