"""Identity truth for evaluation when nobody has hand-labelled who is who.

Serve-frame truth ("rules truth"). At each serve contact the four people are told apart by the rules plus the user's
point labels (server team, deuce/ad, score): the server is the serving-side person further from the net, the receiver
is on the diagonal half; within a set each team's serve order and receiving sides are fixed. So the people at every
serve frame fall into per-(team, set) role classes, and only a few binary links are left (which team is which pair
per changeover block; which serve-order class is which receiving class per team and set; set-to-set and A/B naming).
Those links are decided by a majority over many points of appearance (OSNet + colour), so one noisy frame cannot
flip them, and each link's margin is reported for a visual check. Not hand truth: see docs/player_id.md.

Rally truth ("motion chain"): from the serve frame, each side's two people are followed frame to frame by court
position only, and the chain stops for that side at the first ambiguous step (crossing, merge, missed detection).
Frames past that are left without truth. Motion is the truth source here, so the motion cue cannot be ablated on it.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .geometry import outside_dist
from .rules import end_flips, receive_side, serve_parity, set_index, side_roles


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 0 else v


def appearance(d):
    parts = [unit(d[k]) for k in ("osnet", "colour_mask" if d.get("colour_mask") is not None else "colour_box") if d.get(k) is not None]
    return np.concatenate(parts) if parts else None


def is_edge(d, sw=3840, sh=2160, px=4):
    b = d["box"]
    return b[0] <= px or b[2] >= sw - px or b[3] >= sh - px


def serve_people(frame, keep):
    """kept people per side at a serve frame (1 or 2 each, clear: conf >= 0.4, < 2.5 m outside, not cut by the frame
    edge), or None if a side has none."""
    dets = frame["dets"]; out = {}
    for s in (0, 1):
        v = [i for i in keep if dets[i]["side"] == s and dets[i]["conf"] >= 0.4 and outside_dist([dets[i]["xy"]])[0] < 2.5 and not is_edge(dets[i])]
        if not 1 <= len(v) <= 2:
            return None
        out[s] = v
    return out


def serving_side_evidence(frames, keep, fi, half=1):
    """max arm-up (serve pose) per side over the frames around the serve contact; nan where not measurable."""
    ev = {0: np.nan, 1: np.nan}; pt = frames[fi]["point"]
    for j in range(max(0, fi - half), min(len(frames), fi + half + 1)):
        if frames[j]["point"] != pt:
            continue
        for i in keep[j]:
            d = frames[j]["dets"][i]; v = d.get("serve_pose", np.nan)
            if np.isfinite(v) and not (np.isfinite(ev[d["side"]]) and ev[d["side"]] >= v):
                ev[d["side"]] = v
    return ev


def build_serve_truth(frames, keep, points, serve_idx, a_near_first_block=None):
    """frames: all frames; keep[f] = in-play det indices (hand rule); points: label dicts (match order) with idx;
    serve_idx: point idx -> frame index of its serve contact.
    Returns (truth: {frame_index: {det_index: 'A1'..}}, report dict)."""
    flips = dict(zip([p["idx"] for p in points], end_flips(points)))
    P = {}
    for p in points:
        fi = serve_idx.get(p["idx"])
        if fi is None:
            continue
        pr = serve_people(frames[fi], keep[fi])
        if pr is None:
            continue
        dets = frames[fi]["dets"]; A = {i: appearance(dets[i]) for i in pr[0] + pr[1]}
        if any(v is None for v in A.values()):
            continue
        P[p["idx"]] = dict(p=p, fi=fi, pairs=pr, app=A, flip=flips[p["idx"]], ev=serving_side_evidence(frames, keep, fi))
    rep = dict(points_with_serve_people=len(P), people_per_serve=dict(Counter(len(v["pairs"][0]) + len(v["pairs"][1]) for v in P.values())))
    if len(P) < 4:
        return {}, rep
    # 1) which pair is which team, per changeover block (schedule = start hypothesis; appearance majority decides)
    pair = {k: {s: np.mean([v["app"][i] for i in v["pairs"][s]], 0) for s in (0, 1)} for k, v in P.items()}
    blocks = defaultdict(list)
    for k, v in P.items():
        blocks[v["flip"]].append(k)
    x_near = {b: b % 2 == 0 for b in blocks}             # team X on the near side at the start (schedule)
    for _ in range(4):
        cx = np.mean([pair[k][0 if x_near[blocks_of] else 1] for blocks_of, ks in blocks.items() for k in ks], 0)
        cy = np.mean([pair[k][1 if x_near[blocks_of] else 0] for blocks_of, ks in blocks.items() for k in ks], 0)
        new = {}
        for b, ks in blocks.items():
            keep_cost = sum(np.linalg.norm(pair[k][0] - cx) + np.linalg.norm(pair[k][1] - cy) for k in ks)
            swap_cost = sum(np.linalg.norm(pair[k][1] - cx) + np.linalg.norm(pair[k][0] - cy) for k in ks)
            new[b] = keep_cost <= swap_cost
        if new == x_near:
            break
        x_near = new
    sched_agree = float(np.mean([x_near[b] == (b % 2 == 0) for b in blocks]))
    block_margin = {}
    for b, ks in blocks.items():
        kc = sum(np.linalg.norm(pair[k][0] - cx) + np.linalg.norm(pair[k][1] - cy) for k in ks)
        sc = sum(np.linalg.norm(pair[k][1] - cx) + np.linalg.norm(pair[k][0] - cy) for k in ks)
        block_margin[int(b)] = round(float(abs(kc - sc) / len(ks)), 3)
    rep.update(changeover_blocks=len(blocks), schedule_agreement=sched_agree, block_margin=block_margin)
    # 2) which of X / Y is team A: the serving side holds the arm-up (toss / trophy) person around the contact
    ev = []
    for k, v in P.items():
        xn = x_near[v["flip"]]; e = v["ev"]
        if not (np.isfinite(e[0]) and np.isfinite(e[1])):
            continue
        srv_side_if_x_is_a = (0 if xn else 1) if v["p"]["server"] == "A" else (1 if xn else 0)
        ev.append(e[srv_side_if_x_is_a] - e[1 - srv_side_if_x_is_a])
    x_is_a = (np.mean(ev) >= 0) if ev else True
    x_is_a_override = None
    if a_near_first_block is not None:                  # checked by eye on a frame (or given by the user's tap)
        b0 = min(blocks); x_is_a_override = bool(x_near[b0]) == bool(a_near_first_block); x_is_a = x_is_a_override
    rep.update(team_naming_evidence=dict(n=len(ev), mean=round(float(np.mean(ev)), 3) if ev else None,
                                         frac_positive=round(float(np.mean(np.array(ev) > 0)), 3) if ev else None, x_is_A=bool(x_is_a),
                                         override=x_is_a_override, heuristic_x_is_A=bool(np.mean(ev) >= 0) if ev else None))
    def side_of_team(v, team):
        xn = x_near[v["flip"]]; x_side = 0 if xn else 1
        return x_side if (team == "A") == x_is_a else 1 - x_side
    # 3) roles -> per (team, set) classes
    cls = {}                                             # point -> {det: (team, set, 'P'|'Q', k)}
    for k, v in P.items():
        p = v["p"]; f = frames[v["fi"]]; srv_team = p["server"]; rcv_team = "B" if srv_team == "A" else "A"
        ss, rs = side_of_team(v, srv_team), side_of_team(v, rcv_team)
        S = v["pairs"][ss]; R = v["pairs"][rs]
        sr, rr = side_roles([f["dets"][i]["xy"] for i in S], [f["dets"][i]["xy"] for i in R])
        st = set_index(p["score_before"]); par = serve_parity(p); rsd = receive_side(p); c = {}
        for i, (role, cf) in zip(S, sr):
            if role is not None and cf >= 0.2:
                c[i] = (srv_team, st, "P", par if role == "main" else 1 - par)
        for i, (role, cf) in zip(R, rr):
            if role is not None and cf >= 0.2:
                c[i] = (rcv_team, st, "Q", rsd if role == "main" else 1 - rsd)
        cls[k] = (c, {d: (srv_team if d in S else rcv_team) for d in S + R})
    groups = defaultdict(list)
    for k, (c, _) in cls.items():
        for d, key in c.items():
            groups[key].append(P[k]["app"][d])
    G = {key: np.mean(v, 0) for key, v in groups.items()}
    # 4) link classes to persons: per (team, set) P0 ~ Q0 or P0 ~ Q1; person 0 := P0
    person = {}; links = {}
    for team in ("A", "B"):
        for st in sorted({key[1] for key in G}):
            g = {(kind, kk): G.get((team, st, kind, kk)) for kind in "PQ" for kk in (0, 1)}
            hasP = g[("P", 0)] is not None and g[("P", 1)] is not None; hasQ = g[("Q", 0)] is not None and g[("Q", 1)] is not None
            base, other = ("P", "Q") if hasP else ("Q", "P") if hasQ else (None, None)
            if base is None:
                continue
            person[(team, st, base, 0)] = 0; person[(team, st, base, 1)] = 1
            if base == "P" and hasQ or base == "Q" and hasP:
                straight = np.linalg.norm(g[(base, 0)] - g[(other, 0)]) + np.linalg.norm(g[(base, 1)] - g[(other, 1)])
                cross = np.linalg.norm(g[(base, 0)] - g[(other, 1)]) + np.linalg.norm(g[(base, 1)] - g[(other, 0)])
                person[(team, st, other, 0)] = 0 if straight <= cross else 1; person[(team, st, other, 1)] = 1 - person[(team, st, other, 0)]
                links[f"{team}/set{st + 1}/serve-order~receive-side"] = round(float(abs(straight - cross)), 3)
            else:                                        # one class present with a single member: link it by appearance
                for kk in (0, 1):
                    if g[(other, kk)] is not None:
                        dd = [np.linalg.norm(g[(other, kk)] - g[(base, q)]) for q in (0, 1)]
                        person[(team, st, other, kk)] = int(np.argmin(dd))
    # 5) set-to-set: person ids of later sets follow set 1 by appearance
    def pmean(team, st, pid):
        v = [G[key] for key, q in person.items() if key[0] == team and key[1] == st and q == pid and key in G]
        return np.mean(v, 0) if v else None
    flipset = {}
    for team in ("A", "B"):
        sets = sorted({key[1] for key in person if key[0] == team})
        for st in sets[1:]:
            a0, a1, b0, b1 = pmean(team, sets[0], 0), pmean(team, sets[0], 1), pmean(team, st, 0), pmean(team, st, 1)
            if any(x is None for x in (a0, a1, b0, b1)):
                continue
            straight = np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1); cross = np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)
            flipset[(team, st)] = cross < straight; links[f"{team}/set{st + 1}~set{sets[0] + 1}"] = round(float(abs(straight - cross)), 3)
    rep["link_margins"] = links
    truth = {}; npts = 0
    for k, (c, _) in cls.items():
        fi = P[k]["fi"]; t = {}
        for d, key in c.items():
            if key not in person:
                continue
            pid = person[key] ^ int(flipset.get((key[0], key[1]), False))
            t[d] = f"{key[0]}{pid + 1}"
        if t:
            truth[fi] = t; npts += 1
    rep["points_with_id_truth"] = npts
    # 6) outliers: truth people whose appearance is nearer another identity's centroid (for the visual check)
    cen = defaultdict(list)
    for fi, t in truth.items():
        for d, s in t.items():
            cen[s].append(appearance(frames[fi]["dets"][d]))
    C = {s: np.mean(v, 0) for s, v in cen.items()}
    out = []
    for fi, t in truth.items():
        for d, s in t.items():
            a = appearance(frames[fi]["dets"][d]); same = [q for q in C if q[0] == s[0]]
            if min(same, key=lambda q: np.linalg.norm(a - C[q])) != s:
                out.append((fi, d, s))
    rep["appearance_outliers"] = len(out); rep["outlier_list"] = out[:60]
    return truth, rep


def motion_chain(frames, keep, start_fi, start_truth, point, direction=1, vmax=7.0, margin_m=0.8):
    """extend serve truth along one point by court position only (per side, 1 or 2 people); a side's chain stops at
    the first frame where its people count changes, a step is too fast, or two people are within margin_m of a swap.
    Returns {frame_index: {det: id}}."""
    out = {}; f0 = frames[start_fi]
    pos = {s: {sid: np.asarray(f0["dets"][d]["xy"], float) for d, sid in start_truth.items() if f0["dets"][d]["side"] == s} for s in (0, 1)}
    alive = {s: len(pos[s]) > 0 for s in (0, 1)}; last_t = {s: f0["t"] for s in (0, 1)}
    fi = start_fi + direction
    while 0 <= fi < len(frames) and frames[fi]["point"] == point and any(alive.values()):
        f = frames[fi]; t = {}
        for s in (0, 1):
            if not alive[s]:
                continue
            dis = [i for i in keep[fi] if f["dets"][i]["side"] == s]; ids = list(pos[s]); dt = abs(f["t"] - last_t[s])
            if len(dis) != len(ids) or dt > 0.45:
                alive[s] = False; continue
            X = [np.asarray(f["dets"][i]["xy"], float) for i in dis]
            if len(ids) == 1:
                a = (dis[0],); ok = np.linalg.norm(X[0] - pos[s][ids[0]]) <= vmax * dt + 0.5
            else:
                keepc = np.linalg.norm(X[0] - pos[s][ids[0]]) + np.linalg.norm(X[1] - pos[s][ids[1]])
                swapc = np.linalg.norm(X[1] - pos[s][ids[0]]) + np.linalg.norm(X[0] - pos[s][ids[1]])
                a = (dis[0], dis[1]) if keepc <= swapc else (dis[1], dis[0])
                step = [np.linalg.norm(np.asarray(f["dets"][a[j]]["xy"]) - pos[s][ids[j]]) for j in (0, 1)]
                ok = abs(keepc - swapc) >= margin_m and max(step) <= vmax * dt + 0.5
            if not ok:
                alive[s] = False; continue
            for j, sid in enumerate(ids):
                t[a[j]] = sid; pos[s][sid] = np.asarray(f["dets"][a[j]]["xy"], float)
            last_t[s] = f["t"]
        if t:
            out[fi] = t
        fi += direction
    return out
