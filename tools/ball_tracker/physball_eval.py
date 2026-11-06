"""Held-out A/B of the physics ball tracker (opencourtvision.ball.physics3d) against the current tracker (own heatmap
net best blob >= 0.5 + the 2D arc filter) on the dev_match clip windows (last 7.5 s of each point), with the same metric
code as eval_heldout.py (SwingVision shot recall, precision proxy, bounce after the last contact, hitter cues), plus
hybrid point-winner tracks: the v3 whole-point npz with the ball (and bounces) inside the clip window replaced.
Mac research tool, CPU only; outputs are private (summary numbers only go to git).

  --split train|heldout|all      which dev_match clips (train = points 1-50, used only for tuning; heldout = 51-101)
  --arms a,b,c                   base_raw, base, p3, p3_top1, p3_nospin, p3_nojoint (see ARMS)
  --grid JSON                    list of Config overrides to tune on (train split); best by sv_recall x precision
  --write_v3 DIR --v3 SEGDIR     hybrid npz for pw_stages (arm given by --write_arm)
  --bounce_gt JSON               hand-marked last bounces (the bounce-marker tool's labels: per point idx, kind bounce|none,
                                 clip frame k, u/v in analysis px, t_last); adds gt_last_* metrics and, when tuning,
                                 rewards configs that hit them (private labels, never committed)
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

from opencourtvision.ball import physics3d as P3  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def camera_for(ctx, manual):
    """manual view-0 court (taps -> DLT) when the clip is inside its time range, else the clip's stored court."""
    if manual and manual["t0"] <= ctx.t0 <= manual["t1"]:
        Hc2i = C.h_court2img(manual["H_img2court"], 3840)
    else:
        Hc2i = ctx.Hc2i
    if Hc2i is None:
        return None
    try:
        return P3.Camera.from_homography(Hc2i, (C.AW, C.AH))
    except (ValueError, np.linalg.LinAlgError):
        return None


def fix_court(ctx, manual, EH):
    """replace the clip's stored court (Hc2i, court gate, net row, ROI) with the manual court when the clip is inside its
    time range. The stored dev_match view-0 court (points 1-50) is oversized; views 1-3 are right and are left alone."""
    if not (manual and manual["t0"] <= ctx.t0 <= manual["t1"]):
        return False
    ctx.Hc2i = C.h_court2img(manual["H_img2court"], 3840)
    ctx.gate = EH.CourtGate(ctx.Hc2i, image_size=(C.AW, C.AH))
    ctx.y_net = C.project(ctx.Hc2i, (C.COURT_W / 2, C.NET_Y))[1]
    ctx.roi = EH.court_roi(np.diag([C.K_NET, C.K_NET, 1.0]) @ ctx.Hc2i, C.NET)
    return True


def ray_dist(cam, uv, X):
    """distance (m) from 3D point X to the camera ray through each pixel uv (N, 2)."""
    Kinv = np.linalg.inv(cam.K)
    d = (np.c_[uv, np.ones(len(uv))] @ Kinv.T) @ cam.R          # world-frame ray directions
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    w = np.asarray(X, float) - cam.centre
    along = d @ w
    return np.where(along > 0, np.linalg.norm(w[None] - along[:, None] * d, axis=1), np.inf)


def sv_geo(G, ctx, cam, pos, evs, sv_shots, win_f=3, tol_m=1.5):
    """SwingVision-anchored truth (independent of our court gate): SV gives each shot's time, 3D hit point and bounce
    court xy. recall: shot found when a reported frame within +-win_f frames passes within tol_m of the SV hit point;
    precision: share of reported frames in those windows that do; bounces: a reported bounce is correct when it lies
    within tol_m of the SV bounce of the shot in flight. Uses the same camera rays for every arm."""
    t0, n = ctx.t0, ctx.n
    ok = np.isfinite(pos[:, 0])
    shots = [s for s in sv_shots if t0 <= s["t"] <= t0 + n / C.AFPS]
    for s in shots:
        k = int(round((s["t"] - t0) * C.AFPS)); lo, hi = max(0, k - win_f), min(n - 1, k + win_f)
        idx = [i for i in range(lo, hi + 1) if ok[i]]
        G["shots"] += 1
        if cam is None or not idx:
            continue
        dd = ray_dist(cam, pos[idx], np.r_[s["hit_xy"], s["hit_z"]])
        G["rep"] += len(idx); G["rep_ok"] += int((dd <= tol_m).sum()); G["found"] += bool((dd <= tol_m).any())
    # last contact: the point's final SV shot (what decides the winner) -- ball found at it (+-win_f, within tol_m)
    if shots and cam is not None and shots[-1] is sv_shots[-1]:
        s = shots[-1]; k = int(round((s["t"] - t0) * C.AFPS)); lo, hi = max(0, k - win_f), min(n - 1, k + win_f)
        idx = [i for i in range(lo, hi + 1) if ok[i]]
        G["last_n"] += 1
        if idx:
            G["last_found"] += bool((ray_dist(cam, pos[idx], np.r_[s["hit_xy"], s["hit_z"]]) <= tol_m).any())
        G["last_hit_ev"] += any(e["kind"] == "hit" and abs(e["i"] - k) <= 4 for e in evs if "i" in e)
    # bounces: SV bounce of shot i happens between shot i and shot i+1 (or the end)
    allsh = sorted(sv_shots, key=lambda s: s["t"])
    bo = [e for e in evs if e["kind"] == "bounce" and np.isfinite(e.get("x", np.nan))]
    G["pts"] += 1; G["bounces"] += len(bo)
    want = [i for i, s in enumerate(allsh) if t0 <= s["t"] <= t0 + n / C.AFPS - 0.3 and s.get("result") in ("In", "Out") and s.get("bounce_xy")]
    G["sv_bounces"] += len(want)
    matched = set()
    for e in bo:
        tb = t0 + e["t"]
        i = max([j for j, s in enumerate(allsh) if s["t"] < tb], default=None)
        if i is None or cam is None or not allsh[i].get("bounce_xy"):
            continue
        g = np.linalg.solve(cam.P[:, [0, 1, 3]], np.r_[e["x"], e["y"], 1.0]); g = g[:2] / g[2]   # pixel -> ground (same camera for every arm)
        good = math.hypot(g[0] - allsh[i]["bounce_xy"][0], g[1] - allsh[i]["bounce_xy"][1]) <= tol_m
        G["bounce_ok"] += good
        if good:
            matched.add(i)
    G["sv_bounce_found"] += len(matched & set(want))
    # the bounce the in/out rules need: first reported bounce 0.1-2.5 s after SV's last shot, vs SV's bounce of it
    last = allsh[-1] if allsh else None
    if last and t0 <= last["t"] <= t0 + n / C.AFPS and last.get("result") in ("In", "Out") and last.get("bounce_xy") and cam is not None:
        G["fin_n"] += 1
        fb = [e for e in bo if last["t"] + 0.1 < t0 + e["t"] <= last["t"] + 2.5]
        if fb:
            e = fb[0]
            g = np.linalg.solve(cam.P[:, [0, 1, 3]], np.r_[e["x"], e["y"], 1.0]); g = g[:2] / g[2]
            G["fin_rep"] += 1
            G["fin_ok"] += math.hypot(g[0] - last["bounce_xy"][0], g[1] - last["bounce_xy"][1]) <= tol_m


def sv_geo_summary(G):
    f = lambda a, b: round(a / b, 3) if b else None
    return dict(sv3d_recall=f(G["found"], G["shots"]), sv3d_precision=f(G["rep_ok"], G["rep"]), sv3d_shots=G["shots"],
                bounces_per_point=f(G["bounces"], G["pts"]), bounce_precision=f(G["bounce_ok"], G["bounces"]),
                bounce_recall=f(G["sv_bounce_found"], G["sv_bounces"]), sv_bounces=G["sv_bounces"],
                final_bounce_found=f(G["fin_rep"], G["fin_n"]), final_bounce_precision=f(G["fin_ok"], G["fin_rep"]),
                final_bounce_ok=G["fin_ok"], final_bounce_n=G["fin_n"],
                last_contact_found=f(G["last_found"], G["last_n"]), last_contact_hit_event=f(G["last_hit_ev"], G["last_n"]), last_contact_n=G["last_n"])


def new_geo():
    return dict(shots=0, found=0, rep=0, rep_ok=0, pts=0, bounces=0, bounce_ok=0, sv_bounces=0, sv_bounce_found=0, fin_n=0, fin_rep=0, fin_ok=0, last_n=0, last_found=0, last_hit_ev=0)


def load_bounce_gt(path):
    """hand-marked last bounces -> {clip idx: answer}; only finished answers count."""
    items = json.load(open(path)).get("items", {})
    return {int(v.get("idx", k)): v for k, v in items.items() if v.get("done") and v.get("kind") in ("bounce", "none")}


def score_last_bounce(gt, events, t0, fps=C.AFPS, cam=None, dk_tol=3, px_tol=20.0):
    """one point's tracker events vs a hand-marked last bounce. The tracker's last bounce is its first reported bounce
    0.1-2.5 s after the last contact (the same rule as final_bounce_*). -> dict(kind, found, ok, dk, px, m)."""
    t_last = float(gt["t_last"])
    fb = [e for e in events if e["kind"] == "bounce" and np.isfinite(e.get("x", np.nan)) and t_last + 0.1 < t0 + e["t"] <= t_last + 2.5]
    r = dict(kind=gt["kind"], found=bool(fb), ok=False, dk=None, px=None, m=None)
    if gt["kind"] == "none" or not fb:
        return r
    e = fb[0]
    r["dk"] = int(round(e["t"] * fps)) - int(gt["k"])
    r["px"] = float(math.hypot(e["x"] - gt["u"], e["y"] - gt["v"]))
    if cam is not None:
        g = [np.linalg.solve(cam.P[:, [0, 1, 3]], np.r_[u, v, 1.0]) for u, v in ((e["x"], e["y"]), (gt["u"], gt["v"]))]
        g = [q[:2] / q[2] for q in g]
        r["m"] = float(math.hypot(*(g[0] - g[1])))
    r["ok"] = abs(r["dk"]) <= dk_tol and r["px"] <= px_tol
    return r


def gt_summary(rows):
    """aggregate score_last_bounce rows: recall/precision of the last bounce vs the hand marks, false alarms on 'none'."""
    b = [r for r in rows if r["kind"] == "bounce"]; nb = [r for r in rows if r["kind"] == "none"]
    f = lambda a, c: round(a / c, 3) if c else None
    med = lambda xs: round(float(np.median(xs)), 2) if xs else None
    return dict(gt_last_n=len(b), gt_last_found=f(sum(r["found"] for r in b), len(b)), gt_last_ok=sum(r["ok"] for r in b),
                gt_last_recall=f(sum(r["ok"] for r in b), len(b)), gt_last_precision=f(sum(r["ok"] for r in b), sum(r["found"] for r in b)),
                gt_last_dk_med=med([abs(r["dk"]) for r in b if r["dk"] is not None]), gt_last_px_med=med([r["px"] for r in b if r["px"] is not None]),
                gt_last_m_med=med([r["m"] for r in b if r["m"] is not None]), gt_none_n=len(nb), gt_none_false=sum(r["found"] for r in nb))


def top1(c, th=0.5):
    pos = np.full((len(c), 2), np.nan)
    ok = np.isfinite(c[:, 0, 0]) & (c[:, 0, 2] >= th)
    pos[ok] = c[ok, 0, :2]
    return pos


def run_p3(cam, cands, cfg, boxes_at):
    out = P3.track(cam, cands, cfg, boxes_at=boxes_at)
    ev = []
    for e in out["events"]:
        if e["kind"] == "bounce" and "uv" in e:
            ev.append(dict(kind="bounce", i=int(round(e["k"])), t=e["k"] / C.AFPS, x=e["uv"][0], y=e["uv"][1], p_in=e.get("p_in"),
                           sd=e.get("sd_xy"), xy=e["xy"]))
        elif e["kind"] == "hit":
            ev.append(dict(kind="hit", i=int(round(e["k"])), t=e["k"] / C.AFPS, x=np.nan, y=np.nan))
    return dict(pos=out["pos"], conf=out["conf"], events=ev, stats=out["stats"], flights=out["flights"])


ARMS = ("base_raw", "base", "base_audio", "base_hardneg", "base_audio_hardneg", "p3", "p3_top1", "p3_nospin", "p3_nojoint", "p3_noterm")


def arm_output(arm, ctx, cam, cands, cfg):
    raw = top1(cands)
    if arm == "base_raw":
        return raw, None
    if arm == "base":
        ph = ctx.phys(raw)
        return ph["pos"], ph
    if arm in ("base_audio", "base_hardneg", "base_audio_hardneg"):
        from opencourtvision.ball.own.physics import filter_track_opt
        # map absolute audio contact times onto clip frames
        breaks = []
        if arm in ("base_audio", "base_audio_hardneg"):
            for ht in getattr(ctx, "hits", []) or []:
                k = int(round((ht - ctx.t0) * C.AFPS))
                if 0 <= k < ctx.n:
                    breaks.append(k)
        ph = filter_track_opt(
            raw, C.AFPS, boxes_at=ctx.bx, y_net=getattr(ctx, "y_net", None), frame_w=C.AW,
            audio_breaks=breaks or None, hardneg_static=arm in ("base_hardneg", "base_audio_hardneg"),
        )
        return ph["pos"], ph
    if cam is None:
        return np.full((len(cands), 2), np.nan), dict(conf=np.zeros(len(cands)), events=[])
    c = cfg
    if arm == "p3_top1":
        cc = np.full_like(cands[:, :1], np.nan); cc[:, 0, :2] = raw; cc[:, 0, 2] = np.where(np.isfinite(raw[:, 0]), cands[:, 0, 2], np.nan)
        cands = cc
    elif arm == "p3_nospin":
        c = P3.Config(**{**cfg.__dict__, "spin": False})
    elif arm == "p3_noterm":
        c = P3.Config(**{**cfg.__dict__, "terminal": False})       # no short-tail (terminal) bounces
    elif arm == "p3_nojoint":
        c = P3.Config(**{**cfg.__dict__, "bounce_bonus": -1e9, "terminal": False})   # no bounces: hits / breaks only
    r = run_p3(cam, cands, c, ctx.bx)
    return r["pos"], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--cands", required=True)
    ap.add_argument("--tracks17", required=True); ap.add_argument("--labels17", required=True)
    ap.add_argument("--prelabels17", required=True); ap.add_argument("--sv17", required=True)
    ap.add_argument("--audio"); ap.add_argument("--pw_stages", required=True); ap.add_argument("--court_override")
    ap.add_argument("--fix_court", action="store_true", help="also use --court_override as the clip's stored court (gate, net row, "
                    "ROI, views) for the clips in its time range, not only for the physics camera")
    ap.add_argument("--split", default="heldout"); ap.add_argument("--arms", default="base_raw,base,p3,p3_top1,p3_nospin,p3_nojoint")
    ap.add_argument("--cfg", default="{}"); ap.add_argument("--grid")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write_v3"); ap.add_argument("--v3"); ap.add_argument("--write_arm", default="p3")
    ap.add_argument("--bounce_gt", help="hand-marked last bounces (bounce-marker labels JSON)")
    ap.add_argument("--dump_pos", help="DIR: save each clip's tracker output (pos, conf, xyz-free events) for the first arm "
                    "(teacher tracks for the learned candidate scorer / event net; private)")
    a = ap.parse_args()
    GT = load_bounce_gt(a.bounce_gt) if a.bounce_gt else {}
    warnings.filterwarnings("ignore")
    EH = load(os.path.join(HERE, "eval_heldout.py"), "eval_heldout")
    pw = EH.load_pw(a.pw_stages)
    index = json.load(open(os.path.join(a.data, "index.json")))["clips"]
    PL = json.load(open(a.prelabels17)); S_SRC = int(PL.get("width", 3840)) / C.AW
    cbv = PL["courts_by_view"]
    if a.fix_court and a.court_override:
        cbv = pw.apply_court_override(cbv, a.court_override)
    views = [pw.View(c) for c in cbv]
    hits_all = sorted(c["t"] for r in PL["rallies"] for c in r["contacts"])
    env = np.load(a.audio)[0] if a.audio else None
    SV = json.load(open(a.sv17))["points"]
    labels = {p["idx"]: p for p in json.load(open(a.labels17))["points"]}
    manual = json.load(open(a.court_override)) if a.court_override else None
    clips = sorted((m["idx"], n) for n, m in index.items() if m["source"] == "dev_match" and (a.split == "all" or m["split"] == a.split))
    base_cfg = json.loads(a.cfg)
    grid = json.load(open(a.grid)) if a.grid else [base_cfg]
    arms = [x for x in a.arms.split(",") if x]
    ctxs, n_fixed = [], 0
    for idx, name in clips:
        if not os.path.exists(os.path.join(a.cands, name + ".npz")) or (a.limit and len(ctxs) >= a.limit):
            continue
        ctx = EH.Ctx(argparse.Namespace(data=a.data, tracks17=a.tracks17), name, index[name], labels[idx], pw, views, hits_all, SV, S_SRC)
        if a.fix_court:
            n_fixed += fix_court(ctx, manual, EH)
        cz = np.load(os.path.join(a.cands, name + ".npz"))["cands"].astype(float)
        cz = cz[:ctx.n] if len(cz) >= ctx.n else np.concatenate([cz, np.full((ctx.n - len(cz),) + cz.shape[1:], np.nan)])
        ctxs.append((idx, ctx, camera_for(ctx, manual), cz))
    print(f"clips {len(ctxs)}, stored court replaced by --court_override: {n_fixed}", flush=True)
    results = []
    for gi, over in enumerate(grid):
        cfg = P3.Config(**{**base_cfg, **over})
        acc, ms, per, geo, gtr = {}, {}, [], {}, {}
        for idx, ctx, cam, cz in ctxs:
            row = dict(idx=idx)
            for arm in arms if gi == 0 else [x for x in arms if x.startswith("p3")]:
                t = time.perf_counter()
                pos, phr = arm_output(arm, ctx, cam, cz, cfg)
                ms[arm] = ms.get(arm, 0.0) + time.perf_counter() - t
                row[arm] = ctx.add(acc.setdefault(arm, EH.new_acc()), pos, phr, pw, hits_all, env)
                evs = phr["events"] if phr is not None else ctx.phys(pos)["events"]
                sv_geo(geo.setdefault(arm, new_geo()), ctx, cam, pos, evs, (SV.get(str(idx)) or {}).get("shots", []))
                if idx in GT:
                    row[arm]["gt_last"] = score_last_bounce(GT[idx], evs, ctx.t0, cam=cam)
                    gtr.setdefault(arm, []).append(row[arm]["gt_last"])
                if phr is not None and "stats" in phr:
                    row[arm]["stats"] = phr["stats"]
                if a.dump_pos and gi == 0 and arm == arms[0]:
                    os.makedirs(a.dump_pos, exist_ok=True)
                    np.savez_compressed(os.path.join(a.dump_pos, f"{ctx.name}.npz"), pos=pos,
                                        conf=(phr["conf"] if phr is not None else np.isfinite(pos[:, 0]).astype(float)),
                                        ev_i=np.array([e["i"] for e in evs if "i" in e], int),
                                        ev_kind=np.array([e["kind"] for e in evs if "i" in e]), t0=ctx.t0)
            per.append(row)
        nfr = sum(c[1].n for c in ctxs)
        table = {k: dict(EH.summarise(v), **sv_geo_summary(geo[k]), **(gt_summary(gtr.get(k, [])) if GT else {}),
                         ms_per_frame=round(1000 * ms[k] / max(1, nfr), 2)) for k, v in acc.items()}
        gt_pts = [dict(idx=r["idx"], **{k: v["gt_last"] for k, v in r.items() if isinstance(v, dict) and "gt_last" in v}) for r in per] if GT else None
        results.append(dict(cfg=over, table=table, gt_points=gt_pts))
        print(json.dumps(dict(cfg=over, p3={k: table[k] for k in table if k.startswith("p3")} if gi else table), default=float), flush=True)
    sel = "p3" if "p3" in results[0]["table"] else next(iter(results[0]["table"]))
    def crit(t):
        c = (t["sv_recall"] or 0) * (t["precision_proxy"] or 0) + 0.002 * t["bounce_found"]
        if GT:          # hand-marked last bounces outrank the proxies: +0.02 per hit, -0.02 per false alarm on a 'none'
            c += 0.02 * t["gt_last_ok"] - 0.02 * t["gt_none_false"]
        return c
    best = max(range(len(results)), key=lambda i: crit(results[i]["table"][sel]))
    C.save_json(dict(split=a.split, n_clips=len(ctxs), results=results, best=best, best_cfg=grid[best],
                     bounce_gt=dict(path=os.path.basename(a.bounce_gt), points=len(GT)) if GT else None), a.out)
    if a.write_v3:
        os.makedirs(a.write_v3, exist_ok=True)
        cfg = P3.Config(**{**base_cfg, **grid[best]})
        for idx, ctx, cam, cz in ctxs:
            src = os.path.join(a.v3, f"seg{idx:03d}.npz")
            if not os.path.exists(src):
                continue
            d = np.load(src); out = {k: d[k] for k in d.files}
            t0v, nv = float(d["t0"]), len(d["ball"])
            pos, phr = arm_output(a.write_arm, ctx, cam, cz, cfg)
            conf = phr["conf"] if phr is not None else np.isfinite(pos[:, 0]).astype(float)
            off = int(round((ctx.t0 - t0v) * C.AFPS))
            lo, hi = max(0, off), min(nv, off + ctx.n)
            ball = out["ball"].astype(np.float32).copy(); pk = out["ball_peak"].astype(np.float32).copy()
            ball[lo:hi] = (pos[lo - off:hi - off] / 2.0).astype(np.float32); pk[lo:hi] = conf[lo - off:hi - off]
            out["ball"], out["ball_peak"] = ball, pk
            tw0, tw1 = t0v + lo / C.AFPS, t0v + hi / C.AFPS
            old = np.atleast_2d(out.get("bounces", np.zeros((0, 3)))).reshape(-1, 3)
            keep = old[(old[:, 0] < tw0) | (old[:, 0] >= tw1)] if len(old) else old
            evb = [e for e in (phr or {}).get("events", []) if e["kind"] == "bounce"]
            new = np.array([[ctx.t0 + e["t"], e["x"] * S_SRC, e["y"] * S_SRC] for e in evb], float).reshape(-1, 3)
            out["bounces"] = np.r_[keep, new].reshape(-1, 3)
            # P(in) of each bounce (NaN where unknown), aligned with out["bounces"]; read by physball_pw.py
            out["bounce_pin"] = np.r_[np.full(len(keep), np.nan), [e.get("p_in") if e.get("p_in") is not None else np.nan for e in evb]].astype(float)
            np.savez_compressed(os.path.join(a.write_v3, f"seg{idx:03d}.npz"), **out)


if __name__ == "__main__":
    main()
