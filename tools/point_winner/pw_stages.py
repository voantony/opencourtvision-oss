"""Point winner v1, stages S1-S5 on the dev_match labels (engineering_notes section 2.9). Research measurement.

Inputs (none of them are committed): the private labels JSON, the per-point end-window tracks from pw_tracks.py
(TrackNet tennis ball + YOLO11-pose players, 640x360 at 30 fps), the CC-footage pre-labels (audio contacts, ball
bounces, one court homography per camera view) and the audio envelope. Truth = the labelled winner of each point.

Protocol: fit on points 1-50, test on 51-101, then swap; the two test halves are pooled into 101 out-of-fold calls.
Each stage is a ridge logistic regression over the features of the layers it includes (pure numpy), with Platt
scaling done inside the fit half. Forward-only (D1): every point is called once from its own evidence plus the
score state before it; nothing is smoothed across points. The score state is the confirmed one (the tapped flow
fixes earlier points), so the L4 prior and the ends mapping are as in a tapped match.

US build: no pose-keypoint features (N3; pose is excluded from the US artefact). EU build: everything.

  python tools/point_winner/pw_stages.py --labels L.json --tracks SEGDIR --prelabels prelabels.json \
      --audio audio_env.npy --out results.json [--per-point private_out.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pw_rally as RL  # noqa: E402
import pw_rules as RU  # noqa: E402

NET_Y, COURT_W, COURT_L = 11.885, 10.97, 23.77
SIDE_M, BACK_M = 2.8, 5.5                       # court polygon for players: bench spectators sit ~3.5 m outside the sideline
SCALE = 6.0                                    # tracks are 640x360, homographies are in 3840x2160 source px


# ---------------------------------------------------------------- geometry
class View:
    def __init__(self, c):
        self.t0, self.t1 = c["t0"], c["t1"]
        self.H = np.array(c["H_img2court"], float)
        self.Hi = np.linalg.inv(self.H)
        self.ok = bool(c.get("detected")) or c.get("line_support", 0) >= 0.5
        nb = self.c2i(COURT_W / 2, 0.0)
        # a view whose near baseline maps far outside the frame is a bad fit (the 3-sample re-aim at ~74 min)
        if not (-2000 < nb[0] < 6000):
            self.ok = False

    def i2c(self, xy):
        xy = np.atleast_2d(xy)
        v = np.c_[xy, np.ones(len(xy))] @ self.H.T
        return v[:, :2] / v[:, 2:3]

    def c2i(self, x, y):
        v = self.Hi @ [x, y, 1.0]
        return v[:2] / v[2]


def apply_court_override(courts_by_view, path):
    """Replace per-view homographies with manual court-setup fits ({view, H_img2court} or a list of them).

    Used by physball_eval (--court_override / --fix_court) and by pw_stages itself. The fixture is the
    N1-style court-setup tap result (tests/fixtures/court_dev_match_view0_manual.json), not private labels.
    """
    cbv = [dict(c) for c in courts_by_view]
    if path:
        ov = json.load(open(path))
        ov = ov if isinstance(ov, list) else [ov]
        for o in ov:
            cbv[o["view"]].update(H_img2court=o["H_img2court"], detected=True, manual=True)
    return cbv


def view_at(views, t):
    for v in views:
        if v.t0 <= t < v.t1:
            return v
    return views[-1]


# ---------------------------------------------------------------- per-point features
def ends_flips(score):
    """Number of end changes before this point (change after game 1, then every two games; set games count on)."""
    sets_done = score.get("set_games_done", [])
    g = sum(score["games"])
    return sum((G + 1) // 2 for G in sets_done) + (g + 1) // 2


def in_court_zone(cxy):
    return -SIDE_M <= cxy[0] <= COURT_W + SIDE_M and -BACK_M <= cxy[1] <= COURT_L + BACK_M


def team_boxes(B, view):
    """boxes (8,5) at one pose frame -> list of (side, cx, cy_top, h, foot_court_y). side +1 near, -1 far."""
    out = []
    for b in B:
        if not np.isfinite(b[0]) or b[4] < 0.4:
            continue
        x0, y0, x1, y1 = b[:4] * SCALE
        cxy = view.i2c([(x0 + x1) / 2, y1])[0]
        if not in_court_zone(cxy):
            continue
        out.append((1 if cxy[1] < NET_Y else -1, (x0 + x1) / 2, y0, y1 - y0, cxy[1]))
    return out


def kp_arm_up(kp):
    """kp (17,3): 1 if either wrist is above the nose/shoulders with confidence."""
    if not np.isfinite(float(kp[0, 2])):
        return 0.0
    sh = np.nanmin([kp[5, 1], kp[6, 1]])
    for w in (9, 10):
        if kp[w, 2] > 0.4 and kp[w, 1] < sh - 2:
            return 1.0
    return 0.0


def slope(t, y):
    if len(t) < 3:
        return np.nan
    A = np.c_[t - t.mean(), np.ones(len(t))]
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def point_features(p, d, view, hits, bounces, env, tc_given=None):
    t0, fps = float(d["t0"]), float(d["fps"])
    ball = d["ball"].astype(float) * SCALE
    pk = d["ball_peak"].astype(float)
    tb = t0 + np.arange(len(ball)) / fps
    good = np.isfinite(ball[:, 0]) & (pk >= 0.5)
    k, B, KP = d["pose_k"], d["boxes"], d["kp"].astype(float)
    tk = t0 + k / fps

    hp = [h for h in hits if p["start"] <= h <= p["end"] - 0.3]
    f = dict(n_hits=len(hp), geom=float(view.ok))
    if not hp:
        return f, None
    tc = hp[-1] if tc_given is None else tc_given
    f["tc_in_window"] = float(tb[0] <= tc <= tb[-1])
    ynet = 0.5 * (view.c2i(0, NET_Y)[1] + view.c2i(COURT_W, NET_Y)[1])
    ynear = view.c2i(COURT_W / 2, 0)[1]
    hpx = max(ynear - ynet, 50.0)

    # ---- L2: who hit last (+ = near team)
    near_t = np.abs(tb - tc) <= 0.17
    m = good & near_t
    bxy = np.nanmedian(ball[m], 0) if m.any() else None
    j = int(np.argmin(np.abs(tk - tc)))
    tb_ = team_boxes(B[j], view) if abs(tk[j] - tc) <= 0.25 and view.ok else []
    if bxy is not None and tb_:
        dist = {1: [], -1: []}
        for s, cx, ytop, h, _ in tb_:
            dist[s].append(math.hypot(bxy[0] - cx, bxy[1] - (ytop + 0.35 * h)) / max(h, 20))
        if dist[1] and dist[-1]:                             # both teams seen; a missing far team is not evidence
            dn, df = min(dist[1]), min(dist[-1])
            f["l2_box"] = float(np.clip(math.log((df + 0.3) / (dn + 0.3)), -3, 3))
    if bxy is not None:
        f["l2_ball_side"] = float(np.clip((bxy[1] - ynet) / hpx, -2, 2))
    a = good & (tb > tc + 0.05) & (tb <= tc + 0.6)
    b = good & (tb >= tc - 0.6) & (tb < tc - 0.05)
    sa, sb = slope(tb[a], ball[a, 1] / hpx), slope(tb[b], ball[b, 1] / hpx)
    if np.isfinite(sa):
        f["l2_dir_after"] = float(np.clip(-sa, -4, 4))      # near hitter -> ball moves up (away)
    if np.isfinite(sb):
        f["l2_dir_before"] = float(np.clip(sb, -4, 4))      # ball came towards the near hitter
    # ---- L1: how it ended, in hitter-relative terms (filled once the hitter sign is known)
    after = good & (tb > tc + 0.25) & (tb <= tc + 2.0)
    f["l1_seen_after"] = float(after.sum() / max(1, ((tb > tc + 0.25) & (tb <= tc + 2.0)).sum()))
    ys = ball[after, 1]
    f["_post_rel"] = ((ys - ynet) / hpx).tolist()
    bo = [x for x in bounces if tc + 0.1 < x["t"] <= tc + 2.5]
    if bo and view.ok:
        cxy = view.i2c([bo[0]["x"], bo[0]["y"]])[0]
        f["_bounce"] = cxy.tolist()
    # ---- L3: reactions 0.5-3 s after the last contact
    w = (tk >= tc + 0.5) & (tk <= tc + 3.0)
    mot, arm = {1: [], -1: []}, {1: [], -1: []}
    prev = None
    for jj in np.nonzero(w)[0]:
        tbx = team_boxes(B[jj], view) if view.ok else []
        for s in (1, -1):
            cur = [(cx, ytop, h) for (ss, cx, ytop, h, _) in tbx if ss == s]
            if prev is not None and cur and prev.get(s):
                mot[s].append(np.mean([min(abs(c[0] - q[0]) / c[2] for q in prev[s]) for c in cur]))
        prev = {s: [(cx, ytop, h) for (ss, cx, ytop, h, _) in tbx if ss == s] for s in (1, -1)}
        for i_, bb in enumerate(B[jj]):
            if not np.isfinite(bb[0]) or bb[4] < 0.4 or not view.ok:
                continue
            x0, y0, x1, y1 = bb[:4] * SCALE
            cxy = view.i2c([(x0 + x1) / 2, y1])[0]; cy = cxy[1]
            if in_court_zone(cxy):
                arm[1 if cy < NET_Y else -1].append(kp_arm_up(KP[jj, i_]))
    if mot[1] or mot[-1]:
        f["l3_motion"] = float(np.clip(np.mean(mot[1] or [0]) - np.mean(mot[-1] or [0]), -1, 1))
    if arm[1] or arm[-1]:
        f["l3_arms_pose"] = float(np.mean(arm[1] or [0]) - np.mean(arm[-1] or [0]))
    if env is not None:
        i0, i1 = int((tc + 0.4) * 100), int((tc + 3.0) * 100)
        base = np.median(env[max(0, int((tc - 4) * 100)):int(tc * 100)])
        f["l3_react_db"] = float(np.clip(np.max(env[i0:i1]) - base, -20, 30)) if i1 > i0 else 0.0
    return f, tc


def ending(f, h):
    """L1 ending relative to the hitter side h (+1 near, -1 far): +1 = shot in and unreturned, -1 = error, 0 = unknown.
    Returns (rule, e_cross, e_out)."""
    rel = np.array(f.get("_post_rel", []))
    cross = float(np.mean(-h * rel > 0.05)) if len(rel) else np.nan  # share of post-contact ball samples past the net
    e_out = np.nan
    if "_bounce" in f:
        x, y = f["_bounce"]
        opp = (y > NET_Y) if h > 0 else (y < NET_Y)
        inside = (-0.3 <= x <= COURT_W + 0.3) and (-0.3 <= y <= COURT_L + 0.3)
        e_out = 1.0 if (opp and not inside) else (-1.0 if (opp and inside) else 0.5)
    if np.isfinite(e_out) and e_out == -1.0:
        rule = 1
    elif np.isfinite(e_out) and e_out > 0:
        rule = -1
    elif np.isfinite(cross) and cross < 0.1 and f.get("l1_seen_after", 0) > 0.3:
        rule = -1                                              # ball seen but never got past the net
    else:
        rule = 0
    return rule, cross, e_out


# ---------------------------------------------------------------- model
def ridge_lr(X, y, lam=1.0, iters=50):
    w = np.zeros(X.shape[1])
    R = lam * np.eye(X.shape[1]); R[0, 0] = 1e-6
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + R @ w
        Hs = X.T @ (X * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit_predict(Xf, yf, Xt, lam):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Z = lambda X: np.c_[np.ones(len(X)), (X - mu) / sd]
    # inner 5-fold for Platt scaling on out-of-fold logits of the fit half
    n = len(yf); idx = np.arange(n); oof = np.zeros(n)
    for k in range(5):
        te = idx % 5 == k
        w = ridge_lr(Z(Xf[~te]), yf[~te], lam)
        oof[te] = Z(Xf[te]) @ w
    w = ridge_lr(Z(Xf), yf, lam)
    a = ridge_lr(np.c_[np.ones(n), oof], yf, 1e-3)
    lt = Z(Xt) @ w
    return 1 / (1 + np.exp(-(a[0] + a[1] * lt)))


def taps_to(conf, correct, target):
    n = len(conf); allowed = math.floor((1 - target) * n + 1e-9)
    order = np.argsort(conf)
    errs = int((~correct).sum())
    for k in range(n + 1):
        if errs <= allowed:
            return k
        if k < n and not correct[order[k]]:
            errs -= 1
    return n


def ece(p, y, bins=10):
    conf = np.maximum(p, 1 - p); pred = p >= 0.5; acc = (pred == (y == 1))
    e = 0.0
    for b in range(bins):
        m = (conf > 0.5 + b * 0.05) & (conf <= 0.5 + (b + 1) * 0.05 + (1e-9 if b == bins - 1 else 0))
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ---------------------------------------------------------------- S6 hard deductions
# rules allowed to override: precision >= 0.95 with >= 10 fires on the dev_match labels (others are measured and reported but
# abstain): net / bounce / serve-phase / pose need the whole-point ball track (S0 v3) to be judged fairly.
ACTIVE = {"next_serve_gap"}
RULE_CONF = {"next_serve_gap": 0.995, "next_serve_side": 0.98, "double_fault": 0.95, "ace_unreturned": 0.95, "bounce_out": 0.9, "bounce_in": 0.9,
             "net": 0.9, "arms_up": 0.85}


def hard_layer(rows, bounces, gap_s):
    """Fire every rule on every point; calls[i] = list of (winner, rule). The ends mapping is the rules one (near team at
    the start = A, checked on dev_match serve frames); the score state is the confirmed one before the point."""
    calls = []; table = {}
    for i, r in enumerate(rows):
        p, rl, d, view = r["p"], r["rl"], r["d"], r["view"]
        nearA = (-1.0) ** r["flips"]; c = []
        w, nm = RU.rule_serve(rl, p["server"], nearA)
        if w: c.append((w, nm))
        if d is not None:
            k, nm = RU.rule_net(rl, d, view, RL.norm_rows)
            if k: c.append((RU.team_of(k, nearA), nm))
            k, nm = RU.rule_bounce(rl, bounces, view)
            if k: c.append((RU.team_of(k, nearA), nm))
        k, nm = RU.rule_pose(r["f"])
        if k: c.append((RU.team_of(k, nearA), nm))
        if i + 1 < len(rows) and rows[i + 1]["p"]["point_no"] == p["point_no"] + 1:
            nx = rows[i + 1]
            side, box = RU.serve_obs(nx["rl"], nx["d"], nx["view"]) if nx["d"] is not None else (None, None)
            gap = nx["p"]["start"] - p["end"]
            ch = True if gap > gap_s else (False if gap < gap_s - 10 else None)
            w, nm = RU.rule_next_serve(p["score_before"], p["server"], r["flips"], side, box, obs_change=ch)
            if w:
                w2, _ = RU.rule_next_serve(p["score_before"], p["server"], r["flips"], None, None, obs_change=ch)
                c.append((w, "next_serve_gap" if w2 == w else "next_serve_side"))   # break length alone vs + serve side/box
        calls.append(c)
        for w, nm in c:
            t = table.setdefault(nm, dict(fires=0, correct=0))
            t["fires"] += 1; t["correct"] += int(w == p["winner"])
    for t in table.values():
        t["precision"] = round(t["correct"] / t["fires"], 3) if t["fires"] else None
    return dict(calls=calls, table=table)


def apply_hard(pS5, calls, build):
    """Override S5 where the firing rules agree. US build: pose is not available, and the next-serve rule is a
    consistency flag only (forward-only, D1): it never changes the call; if it disagrees the point is asked first,
    if it agrees the confidence rises. Disagreeing rules -> no override, asked first."""
    p6 = pS5.copy(); conf = np.maximum(pS5, 1 - pS5)
    for i, c in enumerate(calls):
        c = [(w, nm) for w, nm in c if nm in ACTIVE and not (build == "US" and nm == "arms_up")]
        flag = [w for w, nm in c if nm.startswith("next_serve")] if build == "US" else []
        over = [(w, nm) for w, nm in c if not (build == "US" and nm.startswith("next_serve"))]
        ws = {w for w, _ in over}
        if len(ws) == 1:
            w = ws.pop(); cf = max(RULE_CONF[nm] for _, nm in over)
            p6[i] = cf if w == "A" else 1 - cf; conf[i] = cf
        elif len(ws) > 1:
            conf[i] = 0.5
        if flag:
            call = "A" if p6[i] >= 0.5 else "B"
            conf[i] = max(conf[i], RULE_CONF["next_serve_side"]) if flag[0] == call else 0.0
    return p6, conf


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--tracks", required=True)
    ap.add_argument("--prelabels", required=True); ap.add_argument("--audio")
    ap.add_argument("--out", required=True); ap.add_argument("--per-point")
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--bounces-from-tracks", action="store_true",
                    help="use each track npz's own 'bounces' array (t, x, y in source px; e.g. the own ball tracker's physics "
                    "bounces) instead of the pre-label bounces, for points whose npz has one")
    ap.add_argument("--anchor", choices=("audio", "rally"), default="audio",
                    help="last contact = our last audio hit (v1/v2) or the rally end from the whole-point ball track (pw_rally, S0 v3)")
    ap.add_argument("--hard_rules", action="store_true", help="S6: hard deductions (pw_rules) override S5 where they fire")
    ap.add_argument("--next_gap_s", type=float, default=60.0, help="segment gap above which the break is a changeover")
    ap.add_argument("--court_override", help="manual court-setup homography JSON (e.g. tests/fixtures/court_dev_match_view0_manual.json)")
    ap.add_argument("--tc-from", help="DIAGNOSTIC ONLY: point_features JSON whose last_contact (SwingVision shot times "
                    "snapped to our audio hits) replaces our own last audio contact. Not a product number.")
    a = ap.parse_args()
    L = json.load(open(a.labels))
    P = json.load(open(a.prelabels))
    views = [View(c) for c in apply_court_override(P["courts_by_view"], getattr(a, "court_override", None))]
    hits = sorted(c["t"] for r in P["rallies"] for c in r["contacts"])
    bounces = sorted((b for r in P["rallies"] for b in r.get("bounces", [])), key=lambda b: b["t"])
    env = np.load(a.audio)[0] if a.audio else None
    tc_map = {q["point"]: q.get("last_contact") for q in json.load(open(a.tc_from))["points"]} if a.tc_from else {}

    pts = [p for p in L["points"] if p.get("kind") == "point" and p.get("winner") in ("A", "B")]
    pts.sort(key=lambda p: p["point_no"])
    # games of completed sets, for the ends mapping
    set_games, last_sets, prev_games = [], [0, 0], [0, 0]
    rows = []
    for p in pts:
        sb = p["score_before"]
        if sb["sets"] != last_sets:
            set_games.append(sum(prev_games) + 1)   # the set-winning game is not in games_before of the next set
            last_sets = sb["sets"]
        prev_games = sb["games"]
        sc = dict(games=sb["games"], set_games_done=list(set_games))
        npz = os.path.join(a.tracks, f"seg{p['idx']:03d}.npz")
        d = np.load(npz) if os.path.exists(npz) else None
        view = view_at(views, p["end"])
        bl = bounces
        if a.bounces_from_tracks and d is not None and "bounces" in d.files:
            bl = [dict(t=float(b[0]), x=float(b[1]), y=float(b[2])) for b in np.atleast_2d(d["bounces"]) if len(b) == 3]
        rl = RL.rally(d, view, hits, p["start"], p["end"]) if d is not None else None   # rally structure (hard rules)
        tcg = tc_map.get(p["point_no"]) if a.tc_from else (rl["t_last"] if (rl and a.anchor == "rally") else None)
        f, tc = point_features(p, d, view, hits, bl, env, tcg) if d is not None else ({}, None)
        rows.append(dict(p=p, f=f, tc=tc, rl=rl, d=d, view=view, flips=ends_flips(sc), srvA=1.0 if p["server"] == "A" else -1.0,
                         y=1.0 if p["winner"] == "A" else 0.0))
    n = len(rows)
    half = np.array([0 if r["p"]["point_no"] <= 50 else 1 for r in rows])
    y = np.array([r["y"] for r in rows])

    def design(stage, build, near_A0, flips_rule):
        X, names = [], None
        for r in rows:
            f = r["f"]
            nearA = (1.0 if near_A0 else -1.0) * ((-1.0) ** (r["flips"] if flips_rule else 0))
            # hitter evidence, + = team A hit last
            l2 = [f.get(k, 0.0) for k in ("l2_box", "l2_ball_side", "l2_dir_after", "l2_dir_before")]
            hs = float(np.tanh(0.8 * l2[0] + 0.6 * l2[1] + 0.5 * l2[2] + 0.3 * l2[3]))
            hA = nearA * hs
            v = {"hitA_box": nearA * l2[0], "hitA_ball": nearA * l2[1], "hitA_after": nearA * l2[2],
                 "hitA_before": nearA * l2[3]}
            if stage >= 2:
                rule, cross, e_out = ending(f, 1 if hs >= 0 else -1)
                v["rule_core"] = hA * (rule if rule else -0.4)            # unknown -> errors dominate in rec doubles
                v["hitA_x_cross"] = hA * (0.0 if not np.isfinite(cross) else cross - 0.5)
                v["hitA_x_seen"] = hA * (f.get("l1_seen_after", 0.0) - 0.5)
            if stage >= 3:
                nh = r["f"].get("n_hits", 0)
                v["srvA"] = r["srvA"]
                v["parity_lastA"] = r["srvA"] * (1.0 if nh % 2 == 1 else -1.0) * (1.0 if nh >= 2 else 0.0)
                v["srvA_x_short"] = r["srvA"] * (1.0 if nh <= 2 else 0.0)  # 1-2 contacts: ace, return error or DF
            if stage >= 4:
                v["reactA_motion"] = nearA * f.get("l3_motion", 0.0)
                v["react_db_x_hitA"] = hA * f.get("l3_react_db", 0.0) / 10
                if build == "EU":
                    v["reactA_arms_pose"] = nearA * f.get("l3_arms_pose", 0.0)
            if stage >= 5:
                v["rule_x_parity"] = v["rule_core"] * v["parity_lastA"]
                v["hitA_x_parity"] = hA * v["parity_lastA"]
            names = list(v)
            X.append([v[k] for k in names])
        return np.array(X, float), names

    def run(stage, build):
        lam = a.lam if stage < 5 else a.lam * 2
        p_oof = np.zeros(n); chosen = {}
        for h in (0, 1):
            fit, te = half != h, half == h
            best = None
            for near_A0 in (True, False):            # ends mapping, calibrated from the fit half's confirmed points
                for flips_rule in (True, False):
                    X, _ = design(stage, build, near_A0, flips_rule)
                    pf = fit_predict(X[fit], y[fit], X[fit], lam)
                    ll = float(np.sum(y[fit] * np.log(pf + 1e-9) + (1 - y[fit]) * np.log(1 - pf + 1e-9)))
                    if best is None or ll > best[0]:
                        best = (ll, near_A0, flips_rule, X)
            _, near_A0, flips_rule, X = best
            chosen[f"test_half_{h + 1}"] = dict(near_A_at_start=near_A0, ends_change_by_rules=flips_rule)
            p_oof[te] = fit_predict(X[fit], y[fit], X[te], lam)
        pred = p_oof >= 0.5; correct = pred == (y == 1); conf = np.maximum(p_oof, 1 - p_oof)
        res = dict(first_call_acc=float(correct.mean()), correct=int(correct.sum()), n=n,
                   acc_test_51_101=float(correct[half == 1].mean()), acc_test_1_50=float(correct[half == 0].mean()),
                   ece=ece(p_oof, y), taps_per_100_at_99=100 * taps_to(conf, correct, 0.99) / n,
                   taps_per_100_at_999=100 * taps_to(conf, correct, 0.999) / n, ends=chosen)
        return res, correct, p_oof

    out = dict(schema="ocv.point_winner_stages.v1", match="2026-10-17 doubles", n_points=n,
               last_contact="DIAGNOSTIC: SwingVision-snapped" if a.tc_from else "ours: last audio contact in the segment",
               protocol="fit 1-50 / test 51-101, then swap; pooled out-of-fold; ridge LR + Platt (numpy)",
               baseline_swingvision="83/100 = 83% (section 2.8.4b)",
               s0=dict(points_with_tracks=sum(1 for r in rows if r["f"]), points_with_last_contact=sum(1 for r in rows if r["tc"] is not None),
                       last_contact_in_window=sum(1 for r in rows if r["f"].get("tc_in_window") == 1.0),
                       geometry_ok=sum(1 for r in rows if r["f"].get("geom") == 1.0),
                       ball_at_last_contact=sum(1 for r in rows if "l2_ball_side" in r["f"]),
                       boxes_at_last_contact=sum(1 for r in rows if "l2_box" in r["f"]),
                       bounce_after_last_contact=sum(1 for r in rows if "_bounce" in r["f"])),
               stages={})
    prior_majority = float(max(y.mean(), 1 - y.mean()))
    out["majority_baseline"] = prior_majority
    per_point = {}
    for build in ("US", "EU"):
        prev = None
        for stage, nm in ((1, "S1 L2 alone"), (2, "S2 L1+L2 rule core"), (3, "S3 + L4 score/serve prior"),
                          (4, "S4 + L3 reactions"), (5, "S5 fusion + calibration")):
            res, corr, p = run(stage, build)
            if prev is not None:
                res["fixed_vs_prev"] = int((corr & ~prev).sum()); res["broken_vs_prev"] = int((~corr & prev).sum())
            out["stages"][f"{build} {nm}"] = res
            per_point[f"{build} S{stage}"] = [float(x) for x in p]
            prev = corr
    if a.hard_rules:
        hard = hard_layer(rows, bounces, a.next_gap_s)
        out["hard_rules"] = hard["table"]
        for build in ("US", "EU"):
            pS5 = np.array(per_point[f"{build} S5"])
            p6, conf6 = apply_hard(pS5, hard["calls"], build)
            pred = p6 >= 0.5; correct = pred == (y == 1)
            out["stages"][f"{build} S6 hard rules over S5"] = dict(
                first_call_acc=float(correct.mean()), correct=int(correct.sum()), n=n,
                taps_per_100_at_99=100 * taps_to(conf6, correct, 0.99) / n, taps_per_100_at_999=100 * taps_to(conf6, correct, 0.999) / n)
            per_point[f"{build} S6"] = [float(x) for x in p6]
    json.dump(out, open(a.out, "w"), indent=1)
    if a.per_point:
        json.dump(dict(point_no=[r["p"]["point_no"] for r in rows], truth=[r["p"]["winner"] for r in rows],
                       p_A=per_point, features=[{k: v for k, v in r["f"].items() if not k.startswith("_")} for r in rows]),
                  open(a.per_point, "w"))
    print(json.dumps(out["s0"]), "majority", round(prior_majority, 3))
    if "hard_rules" in out:
        print("hard rules:", json.dumps(out["hard_rules"]))
    for k, r in out["stages"].items():
        if "S6" in k:
            print(f"{k:32s} acc {r['first_call_acc']:.1%} ({r['correct']}/{r['n']}) taps/100 @99% {r['taps_per_100_at_99']:.1f} "
                  f"@99.9% {r['taps_per_100_at_999']:.1f}")
            continue
        print(f"{k:32s} acc {r['first_call_acc']:.1%} ({r['correct']}/{r['n']}; 51-101 {r['acc_test_51_101']:.1%}, "
              f"1-50 {r['acc_test_1_50']:.1%}) ECE {r['ece']:.3f} taps/100 @99% {r['taps_per_100_at_99']:.1f} "
              f"@99.9% {r['taps_per_100_at_999']:.1f} fixed/broken {r.get('fixed_vs_prev', '-')}/{r.get('broken_vs_prev', '-')}")


if __name__ == "__main__":
    main()
