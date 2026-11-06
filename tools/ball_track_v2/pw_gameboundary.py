"""Ball track at game boundaries: does a ball-based serve-end cue fix the point winner's main failure?

PR #24 (engineering notes 2.11, docs/post_point_cues.md): the score decoder's confident errors come from misplaced game
boundaries -- the serve-end observation (learned from serve-window pose, 63/101 right) is wrong around a game change, so
the decoder flips most of a game. Here the ball tracker runs on each point's SERVE window (first audio hit -0.5 .. +2 s;
extend_clips.py serve_plan/decode + cands_multi heatnet and WASB, noisy-OR fused) and gives the serve direction:
after a near-end serve the ball moves up the image (away from the camera), after a far-end serve it moves down.

Ball serve features (image only, no court model): slope of a robust (RANSAC) line through the fused candidates' image y
in [hit + 0.07 s, hit + 0.8 s], its inlier count, and the fitted y at the hit. They join the pose serve-formation
features in PR #24's per-fold serve-end LR (labels of the fit fold only).

Variants (all PR #24 protocols: halves, tapped flow / automatic, offline / live):
  base         PR #24 as is
  ball_all     ball serve features on every point
  ball_gb      ball serve features only on game-boundary candidates, chosen WITHOUT labels: points the base decoder
               (automatic, offline) puts within +-1 point of a game start, plus points whose pose serve-end call
               disagrees with the base decode's server end (the cascade: ball only where the decoder is unsure)
Research tool (box, CPU); needs a PR #24 checkout (--ppc_repo). Inputs and outputs are private except the summary.

  pw_gameboundary.py --ppc_repo DIR --cands heatnet=DIR --cands wasb=DIR --plan SERVE_PLAN.json --out OUT.json
                     [--tracks V3_DIR] [--pp DIR] <PR #24 data paths>
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# fuse.py is numpy-only: load it by path so the PR #24 checkout's `opencourtvision` package is the one imported below
_spec = importlib.util.spec_from_file_location("ocv_ball_fuse", os.path.join(HERE, "..", "..", "opencourtvision", "ball", "fuse.py"))
_fz = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fz); fuse = _fz.fuse

BALL_FEATS = ("sb_slope", "sb_inl", "sb_y0", "sb_has")
BALL_FEATS_START = ("sb_first", "sb_fast", "sb_y0", "sb_nseg", "sb_has")


def ransac_line(t, y, iters=200, tol=6.0, seed=0):
    """Robust y = a t + b through (t, y); returns (a, b, inliers)."""
    best = (0.0, float(np.median(y)) if len(y) else 0.0, 0)
    if len(t) < 2:
        return best
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        i, j = rng.choice(len(t), 2, replace=False)
        if t[i] == t[j]:
            continue
        a = (y[j] - y[i]) / (t[j] - t[i]); b = y[i] - a * t[i]
        inl = np.abs(y - (a * t + b)) <= tol
        if inl.sum() > best[2]:
            A = np.c_[t[inl], np.ones(inl.sum())]; a2, b2 = np.linalg.lstsq(A, y[inl], rcond=None)[0]
            best = (float(a2), float(b2), int(inl.sum()))
    return best


def serve_ball_features(cand_list, k_hit, fps=30.0, t_lo=0.07, t_hi=0.8, min_peak=0.25):
    """cand_list: per-detector (T, K, 3) candidates of one serve window. Image-only serve-direction features."""
    fused, _ = fuse([c for c in cand_list if c is not None])
    T = len(fused); lo, hi = k_hit + int(round(t_lo * fps)), min(T, k_hit + int(round(t_hi * fps)) + 1)
    ts, ys = [], []
    for k in range(lo, hi):
        for u, v, p in fused[k]:
            if np.isfinite(u) and p >= min_peak:
                ts.append((k - k_hit) / fps); ys.append(v)
    ts, ys = np.array(ts), np.array(ys)
    a, b, n = ransac_line(ts, ys)
    if n < 5:
        return dict(sb_slope=0.0, sb_inl=0.0, sb_y0=0.0, sb_has=0.0)
    return dict(sb_slope=float(np.clip(a / 400.0, -3, 3)), sb_inl=n / 20.0, sb_y0=(b - 360.0) / 360.0, sb_has=1.0)


def fast_segments(fused, k_from=0, seg=12, min_inl=7, min_v=150.0, min_span=50.0, fps=30.0, min_peak=0.3):
    """Greedy, time-ordered straight fast flights in fused candidates: (k_start, vy px/s, inliers, y_start)."""
    out = []; k = k_from; T = len(fused)
    while k + seg <= T:
        ts, ys = [], []
        for j in range(k, k + seg):
            for u, v, p in fused[j]:
                if np.isfinite(u) and p >= min_peak:
                    ts.append((j - k) / fps); ys.append(v)
        a, b, n = ransac_line(np.array(ts), np.array(ys))
        if n >= min_inl and abs(a) >= min_v and abs(a) * seg / fps >= min_span:
            out.append((k, a, n, b)); k += seg
        else:
            k += 2
    return out


def serve_ball_features_start(cand_list, k_anchor, fps=30.0, net_y=360.0):
    """Windows anchored at the point's start (the serve is the first fast flight after it): the first flight's
    direction (near serve -> up the image -> negative vy), the fastest flight in the next 2 s, the y where it starts."""
    fused, _ = fuse([c for c in cand_list if c is not None])
    S = fast_segments(fused, max(0, k_anchor - int(0.5 * fps)))
    if not S:
        return dict(sb_first=0.0, sb_fast=0.0, sb_y0=0.0, sb_nseg=0.0, sb_has=0.0)
    k1, a1, _, b1 = S[0]
    near = [x for x in S if x[0] <= k1 + 2 * fps]
    af = max(near, key=lambda x: abs(x[1]))[1]
    return dict(sb_first=float(np.clip(a1 / 400.0, -3, 3)), sb_fast=float(np.clip(af / 400.0, -3, 3)), sb_y0=(b1 - net_y) / 360.0,
                sb_nseg=min(len(S), 10) / 10.0, sb_has=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppc_repo", required=True); ap.add_argument("--cands", action="append", required=True, help="name=DIR")
    ap.add_argument("--plan", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--refit", help="court_refit.py JSON: per-point net line for sb_y0 (camera drift); default the view's fit")
    for k, d in (("labels", None), ("prelabels", None), ("tracks", None), ("pp", None), ("audio", None), ("court_override", None)):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.ppc_repo, "tools", "point_winner")); sys.path.insert(0, a.ppc_repo)
    import pw_postpoint as PW

    from opencourtvision.scoring import decoder as D
    from opencourtvision.scoring import rules as R
    rows = PW.load(argparse.Namespace(labels=a.labels, prelabels=a.prelabels, tracks=a.tracks, pp=a.pp, audio=a.audio,
                                      court_override=a.court_override))
    plan = {c["idx"]: c for c in json.load(open(a.plan))["clips"]}
    dirs = dict(s.split("=", 1) for s in a.cands)
    bf = {}
    refit = {r["idx"]: r["H_refit"] for r in json.load(open(a.refit))["points"] if r.get("drift_px") is not None} if a.refit else {}
    for i, r in enumerate(rows):
        c = plan.get(r["p"]["idx"])
        if not c:
            continue
        cl = []
        for d in dirs.values():
            f = os.path.join(d, c["name"] + ".npz")
            cl.append(np.load(f)["cands"].astype(float) if os.path.exists(f) else None)
        if any(x is not None for x in cl):
            if c.get("anchor") == "start":
                H = np.asarray(refit[c["idx"]] if c["idx"] in refit else r["view"].H, float)
                q = np.linalg.inv(H) @ [5.485, 11.885, 1.0]
                net_y = float(q[1] / q[2]) / 3.0                       # source px -> 1280x720 candidate px
                bf[i] = serve_ball_features_start(cl, int(round((c["t_anchor"] - c["t0"]) * c["fps"])), net_y=net_y)
            else:
                bf[i] = serve_ball_features(cl, int(round((c["t_first_hit"] - c["t0"]) * c["fps"])))
    start_mode = any(c.get("anchor") == "start" for c in plan.values())
    feats = BALL_FEATS_START if start_mode else BALL_FEATS
    key = "sb_first" if start_mode else "sb_slope"
    truth_end = np.array([r["srv_end_true"] for r in rows])
    # ball cue alone: sign of the slope (near serve -> ball moves up the image -> negative slope)
    sb = [(i, 0 if f[key] < 0 else 1) for i, f in bf.items() if f["sb_has"]]
    first = np.array([r["st"].pts == (0, 0) for r in rows])          # evaluation only (labels)
    res = dict(n_points=len(rows), serve_windows=len(bf), ball_serve_seen=len(sb),
               ball_slope_rule_acc=float(np.mean([e == truth_end[i] for i, e in sb])) if sb else None,
               ball_slope_rule_acc_game_starts=float(np.mean([e == truth_end[i] for i, e in sb if first[i]])) if sb else None,
               n_game_starts=int(first.sum()))
    folds = PW.folds_of(rows, "halves"); allc = list(PW.CUES) + ["ball_rules"] + list(PW.DEC_CUES); dec = list(PW.DEC_CUES)
    sv0 = [dict(r["sv"]) for r in rows]
    base_feats = tuple(PW.SV_FEATS)

    def set_ball(which):
        PW.SV_FEATS = base_feats + (feats if which else ())
        for i, r in enumerate(rows):
            r["sv"] = dict(sv0[i])
            if which and i in which and i in bf:
                if not r["sv"]:                                  # no serve pose: ball features alone (pose feats 0)
                    r["sv"] = {k: 0.0 for k in base_feats}
                r["sv"].update(bf[i])

    def serve_acc():
        PW.serve_classifier(rows, folds)
        ok = [(r["srv_obs"] == r["srv_end_true"]) for r in rows if r["srv_obs"] is not None]
        okg = [(r["srv_obs"] == r["srv_end_true"]) for r, f in zip(rows, first) if f and r["srv_obs"] is not None]
        return dict(serve_obs=len(ok), serve_end_acc=float(np.mean(ok)), serve_end_acc_game_starts=float(np.mean(okg)) if okg else None)

    def boundary_errors(cues):
        p_near = PW.fuse(rows, folds, cues); om_by, pa_by = {}, {}
        sacc = PW.serve_classifier(rows, folds)
        for fit, te in folds:
            om = PW.obs_models(rows, set(fit.tolist()), sacc[int(te[0])]); pa = (sum(rows[i]["p"]["winner"] == "A" for i in fit) + 1) / (len(fit) + 2)
            for i in te:
                om_by[int(i)] = om; pa_by[int(i)] = float(pa)
        ev = PW.evidence(rows, p_near, om_by, cues, pa_by); dd = D.decode(ev, PW.FMT, ends0=PW.ENDS0)
        dec_first = np.array([s.pts == (0, 0) for s in dd.states[:len(rows)]])
        dec_end = np.array([R.server_end(s) for s in dd.states[:len(rows)]])
        return dd, dict(game_starts_missed=int((first & ~dec_first).sum()), game_starts_spurious=int((dec_first & ~first).sum()),
                        server_end_wrong=int((dec_end != truth_end).sum())), dec_first, dec_end

    truth = np.array([r["p"]["winner"] == "A" for r in rows]); h2 = np.array([r["p"]["point_no"] > 50 for r in rows])
    variants = {"EU full": allc, "rules + decoder": ["ball_rules"] + dec, "break + serve + decoder": dec}
    # game-boundary candidates from the BASE decode (no labels)
    set_ball(None); _, _, dfirst, dend = boundary_errors(allc)
    gb = set()
    for i in range(len(rows)):
        if dfirst[max(0, i - 1):i + 2].any() or (rows[i]["srv_obs"] is not None and rows[i]["srv_obs"] != dend[i]):
            gb.add(i)
    res["gb_candidates"] = len(gb); res["gb_candidates_cover_game_starts"] = int(sum(first[i] for i in gb))
    for vname, which in (("base", None), ("ball_all", set(range(len(rows)))), ("ball_gb", gb)):
        set_ball(which); out = dict(serve=serve_acc())
        for k, cues in variants.items():
            v = PW.run_variant(rows, folds, cues, taps=True); pa = v.pop("_pA")
            _, be, _, _ = boundary_errors(cues)
            out[k] = dict({m: round(v[m]["acc"], 3) for m in ("tapped_flow_live", "tapped_flow_offline", "auto_live", "auto_offline")},
                          acc_51_101={m: round(float(((np.array(pa[m]) >= .5) == truth)[h2].mean()), 3) for m in pa},
                          taps99_auto_off=v["auto_offline"].get("taps99"), taps99_auto_live=v["auto_live"].get("taps99"),
                          boundary=be)
        res[vname] = out
        print(vname, json.dumps(out), flush=True)
    set_ball(None)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
