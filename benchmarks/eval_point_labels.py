"""Score engine on REAL signals vs a human-labelled match (tools/labeler output, ocv.point_labels.v1).

  python benchmarks/eval_point_labels.py LABELS.json --sv points.json \
      [--stt work/stt_win_cache.json] [--final 2-0] [--tau 0.999] [--json out.json]

Segments = the human-labelled segments (so segmentation errors are out of scope here); truth per segment = A / B / skip
(unlabelled segments are left out of every metric). Signals, each mapped to segments by time:
  sv      SwingVision import: per-point winner (p_a = --sv-acc prior, NOT fitted to the labels), its server (Obs, --sv-srv-acc),
          and p_point (segment has a SwingVision point -> 0.95, else 0.30).
  calls   score callouts from the between-point Whisper windows (stt_win_cache.json), parsed by scoring/callouts.py,
          through the callout source gate (no speaker / distance cues yet -> prior only).
  final   the typed final result as a set-count end constraint (e.g. 2-0).
  vision  engine-v2 per-point winner: NOT AVAILABLE yet (reported as missing).
Metrics per configuration: segment accuracy of the MAP path before any ask, points-only accuracy, score-state accuracy
(state shown before each true point == truth), final score right, then selective prediction at tau (confirm_loop with the
labels as answers): asks, asks per 100 points, accuracy on segments not asked, score-state accuracy after asks.
Also: SwingVision per-point winner agreement with the labels, with every disagreement listed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencourtvision.labels.points import load_point_labels, match_sv_points, ranker_vs_favorites  # noqa: E402
from opencourtvision.scoring.callout_gate import CalloutCandidate, gate_callout  # noqa: E402
from opencourtvision.scoring.callouts import parse_callout  # noqa: E402
from opencourtvision.scoring.decoder import Obs, PointEvidence, canonical_call, confirm_loop, decode  # noqa: E402


def hms(t):
    return f"{int(t // 3600)}:{int(t // 60) % 60:02d}:{t % 60:04.1f}"


def match_sv(segs, sv, offset=2.0, tol=6.0):
    return match_sv_points(segs, sv, offset, tol)


def callouts(segs, stt):
    """Parsed calls from the windowed STT cache -> {segment index: [Obs]} (window ends just before the first hit)."""
    by = {}
    seen = set()
    for k, v in stt.items():
        p = parse_callout(v.get("text", ""))
        cc = canonical_call(p)
        if not cc:
            continue
        key = (round(v["a"], 1), cc)
        if key in seen:
            continue
        seen.add(key)
        fh = float(v.get("first_hit", v["b"]))
        i = next((i for i, s in enumerate(segs) if s.start - 1.0 <= fh <= s.end), None)
        if i is None:
            continue
        g = gate_callout(CalloutCandidate(t=float(v["b"]), call=cc, asr_conf=0.9), window=(v["a"], v["b"]))
        if g.accepted and g.obs is not None:
            by.setdefault(i, []).append((g.obs, v["text"][:60], k.rsplit(":", 1)[0]))
    return by


def build(segs, svmap, calls, use_sv, use_calls, sv_acc, sv_srv_acc):
    ev = []
    for i, s in enumerate(segs):
        kw = {}
        q = svmap.get(i) if use_sv else None
        if use_sv:
            if q:
                w = "A" if q["winner"] == "host" else "B"
                kw.update(p_a=sv_acc if w == "A" else 1 - sv_acc, p_point=0.95,
                          server=Obs("A" if q["server"] == "host" else "B", sv_srv_acc))
            else:
                kw.update(p_point=0.30)
        else:
            kw.update(p_point=0.70)
        if use_calls and i in calls:
            obs = [o for o, _, _ in calls[i]]
            kw["call"] = obs[0]
            if len(obs) > 1:
                kw["calls"] = tuple(obs[1:])
        ev.append(PointEvidence(**kw))
    return ev


def metrics(segs, L, truth_states, d, asked=()):
    lab = [i for i, s in enumerate(segs) if s.choice is not None]
    pts = [i for i in lab if segs[i].kind == "point"]
    acc = sum(d.winners[i] == segs[i].choice for i in lab) / len(lab)
    pacc = sum(d.winners[i] == segs[i].choice for i in pts) / len(pts)

    def same(a, b):
        return (a.sets, a.games, a.pts, a.server) == (b.sets, b.games, b.pts, b.server)
    sacc = sum(same(d.states[i], truth_states[i]) for i in pts) / len(pts)
    fin = d.states[-1].sets == truth_states[-1].sets and d.states[-1].done
    un = [i for i in lab if i not in set(asked)]
    uacc = sum(d.winners[i] == segs[i].choice for i in un) / len(un) if un else 1.0
    return dict(seg_acc=acc, pt_acc=pacc, score_acc=sacc, final_ok=fin, unasked_acc=uacc, n_unasked=len(un))


def run(L, sv, stt, final, tau, sv_acc=0.90, sv_srv_acc=0.95, do_decode=True):
    segs = L.segments
    truth_states = L.states()
    svmap = match_sv(segs, sv) if sv else {}
    calls = callouts(segs, stt) if stt else {}
    truth = [s.choice if s.choice is not None else "skip" for s in segs]
    n_pts = sum(s.kind == "point" for s in segs)
    configs = [("calls alone", False, True, None), ("SwingVision alone", True, False, None),
               ("SwingVision + final", True, False, final), ("fused: SV + calls + final", True, True, final)]
    rows = []
    for name, us, uc, fin in configs:
        if (us and not sv) or (uc and not stt) or not do_decode:
            continue
        ev = build(segs, svmap, calls, us, uc, sv_acc, sv_srv_acc)
        d0 = decode(ev, L.fmt, final=fin)
        m0 = metrics(segs, L, truth_states, d0)
        d1, asked = confirm_loop(ev, L.fmt, truth, tau, final=fin)
        m1 = metrics(segs, L, truth_states, d1, asked)
        rows.append(dict(config=name, before=m0, asks=len(asked), asks_per_100_pts=100 * len(asked) / n_pts, after=m1))
    lab_pts = [i for i, s in enumerate(segs) if s.kind == "point"]
    agree = [(i, svmap[i]) for i in lab_pts if i in svmap]
    dis = [dict(seg=segs[i].idx, t=hms(segs[i].start), sv_point=q["point"], sv_detail=q.get("detail") or "",
                sv="A" if q["winner"] == "host" else "B", label=segs[i].winner, label_server=truth_states[i].server,
                sv_server="A" if q["server"] == "host" else "B")
           for i, q in agree if ("A" if q["winner"] == "host" else "B") != segs[i].winner]
    unmatched = [q["point"] for q in sv if not any(q is v for v in svmap.values())] if sv else []
    on_x = [dict(seg=segs[i].idx, t=hms(segs[i].start), sv_point=q["point"], kind=segs[i].kind) for i, q in svmap.items()
            if segs[i].kind != "point"]
    sv_stats = dict(matched_to_points=len(agree), agree=len(agree) - len(dis), disagreements=dis,
                    sv_points_unmatched=unmatched, sv_points_on_non_points=on_x)
    call_list = [dict(seg=segs[i].idx, t=hms(segs[i].start), call=o.value, truth=None, text=txt, model=m)
                 for i, lst in calls.items() for o, txt, m in lst]
    from opencourtvision.scoring.decoder import score_call
    for c in call_list:
        i = next(j for j, s in enumerate(segs) if s.idx == c["seg"])
        c["truth"] = score_call(truth_states[i], L.fmt) if segs[i].kind == "point" else "(not a point)"
    fav = ranker_vs_favorites(L.highlight_rows(sv)) if sv else {}
    return dict(n_segments=len(segs), n_points=n_pts, rows=rows, favorites=fav, sv=sv_stats, calls=call_list,
                missing=["engine-v2 vision winner (not built yet)"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--sv")
    ap.add_argument("--stt")
    ap.add_argument("--final", default=None, help="sets, e.g. 2-0")
    ap.add_argument("--tau", type=float, default=0.999)
    ap.add_argument("--sv-acc", type=float, default=0.90)
    ap.add_argument("--json")
    ap.add_argument("--no-decode", action="store_true", help="skip the (slow, ~6 min) decoder runs")
    a = ap.parse_args()
    L = load_point_labels(a.labels)
    sv = json.load(open(a.sv))["points"] if a.sv else None
    stt = json.load(open(a.stt)) if a.stt else None
    final = tuple(map(int, a.final.split("-"))) if a.final else None
    R = run(L, sv, stt, final, a.tau, sv_acc=a.sv_acc, do_decode=not a.no_decode)
    print(f"{R['n_segments']} segments, {R['n_points']} points; tau={a.tau}")
    print(f"{'config':28s} {'seg acc':>7s} {'pt acc':>7s} {'score':>6s} {'final':>5s} | {'asks':>4s} {'/100pt':>6s} {'unasked':>7s} {'score':>6s}")
    for r in R["rows"]:
        b, f = r["before"], r["after"]
        print(f"{r['config']:28s} {b['seg_acc']:7.1%} {b['pt_acc']:7.1%} {b['score_acc']:6.1%} {str(b['final_ok']):>5s} | "
              f"{r['asks']:4d} {r['asks_per_100_pts']:6.1f} {f['unasked_acc']:7.1%} {f['score_acc']:6.1%}")
    s = R["sv"]
    print(f"\nSwingVision winner vs labels: {s['agree']}/{s['matched_to_points']} agree; unmatched SV points {s['sv_points_unmatched']};"
          f" SV points on X/unlabelled segments {s['sv_points_on_non_points']}")
    for d in s["disagreements"]:
        print("  ", d)
    print("callouts used:", R["calls"])
    print("missing:", R["missing"])
    print("favourites vs ranker top-N:", R["favorites"])
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
