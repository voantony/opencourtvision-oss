"""Eval stage of pid_run.py (see its docstring). Reads det/*.npz + crops/*.vision.jsonl under OUT."""
from __future__ import annotations

import glob
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from opencourtvision.players import geometry as G
from opencourtvision.players.cues import colour_hist, pose_norm, serve_pose_score
from opencourtvision.players.deduce import deduce_ids
from opencourtvision.players.inplay import det_features, hand_score, select
from opencourtvision.players.metrics import IDS, evaluate
from opencourtvision.players.rules import end_flips, receive_side, serve_parity, set_index, side_roles
from opencourtvision.players.tracker import IDTracker
from opencourtvision.players.truth import build_serve_truth, motion_chain

FPS, SERVE_I = 2.5, 3
HAND = ("court", "motion", "role_pos", "colour_box", "colour_mask", "osnet", "role_pose")
LEARNED = ("head", "role_posenet")


def pct(v, q):
    v = [x for x in v if np.isfinite(x)]
    return float(np.percentile(v, q)) if v else float("nan")


# ------------------------------------------------------------------------------------------------------------ loading
def load(out, P, log):
    import cv2
    frames, ms = [], defaultdict(list)
    for p in P:
        fp = os.path.join(out, "det", f"p{p['idx']:03d}.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=False); meta = json.loads(str(z["meta"])); rows = meta["rows"]
        for k in ("ms_det", "ms_osnet", "ms_colour_box"):
            ms[k] += z[k].tolist()
        V = {}
        vf = os.path.join(out, "crops", f"p{p['idx']:03d}.vision.jsonl")
        if os.path.exists(vf):
            for ln in open(vf):
                r = json.loads(ln); V[os.path.relpath(r["path"], out)] = r; ms["vision"].append(r["ms_mask"] + r["ms_pose"])
        sb = p["score_before"]; base = len(frames)
        for f in range(meta["n"]):
            frames.append(dict(t=meta["t0"] + f / FPS, point=p["idx"], fi_in_point=f, serve=(f == SERVE_I), set=set_index(sb), server=p["server"],
                               parity=serve_parity(p), rside=receive_side(p), dets=[]))
        for j, r in enumerate(rows):
            d = dict(box=r["box"], conf=r["conf"], xy=np.array(r["xy"]), side=int(r["xy"][1] >= G.NET_Y), h_ratio=r["h_ratio"], crop=r["crop"],
                     osnet=z["osnet"][j].astype(np.float32), colour_box=z["colour_box"][j].astype(np.float32))
            v = V.get(r["crop"])
            if v is not None:
                kp = np.array(v["kp"], float) if "kp" in v else None
                t1 = time.perf_counter(); img = cv2.imread(os.path.join(out, r["crop"])); mask = None
                if v.get("mask"):
                    mm = np.fromfile(os.path.join(out, r["crop"]) + ".mask.u8", np.uint8).reshape(v["mh"], v["mw"])
                    mask = cv2.resize(mm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                    if mask.sum() < 30:
                        mask = None
                d["colour_mask"] = colour_hist(img, mask, kp) if mask is not None else None
                ms["colour_mask"].append(1000 * (time.perf_counter() - t1))
                d["kp"] = kp; d["serve_pose"] = serve_pose_score(kp) if kp is not None else float("nan"); d["pose_vec"] = pose_norm(kp) if kp is not None else None
            frames[base + r["f"]]["dets"].append(d)
    log(f"loaded {len(frames)} frames, {sum(len(f['dets']) for f in frames)} in-zone detections")
    return frames, ms


def hand_keep(frames, score_fn=None):
    keep, prev = [], None
    for i, f in enumerate(frames):
        if i == 0 or frames[i - 1]["point"] != f["point"]:
            prev = None
        ns = Counter(d["side"] for d in f["dets"])
        F = [det_features(d, prev, (f["fi_in_point"] - SERVE_I) / FPS, ns[d["side"]]) for d in f["dets"]]
        sc = [hand_score(x) for x in F] if score_fn is None else score_fn(F)
        for d, x in zip(f["dets"], F):
            d["feat_inplay"] = x
        k = select(f["dets"], sc, thr=-1.0 if score_fn is None else 0.3)
        keep.append(k); prev = np.array([f["dets"][j]["xy"] for j in k]) if k else prev
    return keep


# ------------------------------------------------------------------------------------------------------------ prep
def prep_roles(frames, keep, posenet=None):
    """per serve frame: position roles (role_pos), hand pose roles (role_pose), learned pose roles (role_posenet)."""
    for f, k in zip(frames, keep):
        if not f["serve"]:
            continue
        srv_side = 0 if f["near_team"] == f["server"] else 1
        S = [i for i in k if f["dets"][i]["side"] == srv_side]; R = [i for i in k if f["dets"][i]["side"] != srv_side]
        if S and R:
            sr, rr = side_roles([f["dets"][i]["xy"] for i in S], [f["dets"][i]["xy"] for i in R])
            for i, (role, cf) in zip(S, sr):
                if role == "main":
                    f["dets"][i].update(role_pos="srv", p_role_pos=0.5 + 0.5 * cf)
            for i, (role, cf) in zip(R, rr):
                if role == "main":
                    f["dets"][i].update(role_pos="rcv", p_role_pos=0.5 + 0.5 * cf)
        if len(S) == 2:
            sp = [f["dets"][i].get("serve_pose", np.nan) for i in S]
            if np.isfinite(sp).all():
                j = int(np.argmax(sp)); f["dets"][S[j]].update(role_pose="srv", p_role_pose=0.5 + 0.5 * float(np.tanh(abs(sp[0] - sp[1]))))
            if posenet is not None:
                pv = [f["dets"][i].get("pose_vec") for i in S]
                if all(v is not None for v in pv):
                    pr = posenet.predict_proba(np.array(pv))[:, 1]; j = int(np.argmax(pr))
                    f["dets"][S[j]].update(role_posenet="srv", p_role_posenet=float(0.5 + 0.5 * abs(pr[0] - pr[1])))


def set_near(frames, P, a_near_at_tap, tap_point):
    fl = dict(zip([p["idx"] for p in P], end_flips(P))); ft = fl[tap_point]
    for f in frames:
        same = (fl[f["point"]] - ft) % 2 == 0
        f["near_team"] = ("A" if same else "B") if a_near_at_tap else ("B" if same else "A")
        for d in f["dets"]:
            for k in ("role_pos", "p_role_pos", "role_pose", "p_role_pose", "role_posenet", "p_role_posenet"):
                d.pop(k, None)


# ------------------------------------------------------------------------------------------------------------ tracking
def run_tracker(frames, keep, start, mapping, cues, unit_ms, recorder=None, **kw):
    tr = IDTracker(cues=cues, unit_ms=unit_ms, **kw); tr.recorder = recorder
    sub = lambda x: dict(frames[x], dets=[frames[x]["dets"][i] for i in keep[x]])
    inv = {i: j for j, i in enumerate(keep[start])}
    tr.init(sub(start), {inv[i]: s for i, s in mapping.items() if i in inv})
    preds = [{} for _ in frames]; i = start
    while i < len(frames):
        j = i
        while j < len(frames) and frames[j]["point"] == frames[i]["point"]:
            j += 1
        pr, _ = tr.point([sub(x) for x in range(i, j)])
        for x, q in zip(range(i, j), pr):
            preds[x] = {keep[x][d]: s for d, s in q.items()}
        i = j
    return preds, tr


def relabel_best(preds, truth, groups):
    """best naming inside each group of interchangeable ids (anonymous taps)."""
    import itertools
    best, bv = preds, -1
    for perm in itertools.product(*[list(itertools.permutations(g)) for g in groups]):
        mp = {}
        for g, pg in zip(groups, perm):
            mp.update(dict(zip(g, pg)))
        P2 = [{d: mp.get(s, s) for d, s in p.items()} for p in preds]
        v = sum(int(P2[fi].get(d) == s) for fi, t in truth.items() for d, s in t.items())
        if v > bv:
            best, bv = P2, v
    return best


# ------------------------------------------------------------------------------------------------------------ main
def run_eval(out, lab, points_fn, log, loader=None):
    T0 = time.time(); P, L = points_fn(); P = sorted(P, key=lambda p: p["start"])
    frames, ms = (loader or load)(out, P, log)
    P = [p for p in P if any(f["point"] == p["idx"] for f in frames)]
    keep = hand_keep(frames)
    serve_idx = {f["point"]: i for i, f in enumerate(frames) if f["serve"]}
    an = os.environ.get("PID_A_NEAR_FIRST_BLOCK")
    truth_s, trep = build_serve_truth(frames, keep, P, serve_idx, None if an is None else an == "1")
    log("serve truth", json.dumps({k: v for k, v in trep.items() if k != "outlier_list"}))
    truth_r = {}
    for fi, t in truth_s.items():
        for dr in (1, -1):
            truth_r.update(motion_chain(frames, keep, fi, t, frames[fi]["point"], direction=dr))
    truth = dict(truth_r); truth.update(truth_s)
    # tap frame: the app asks for the taps on the first frame where all four are visible (2 per side) with known ids
    full4 = [fi for fi, t in sorted(truth.items()) if len(t) == 4 and Counter(frames[fi]["dets"][d]["side"] for d in keep[fi]) == Counter({0: 2, 1: 2})]
    first = full4[0] if full4 else max(truth, key=lambda fi: (len(truth[fi]), -fi))
    tap_point = frames[first]["point"]; tdet = truth[first]
    trep["tap_frame"] = dict(point=tap_point, t=round(frames[first]["t"], 2), ids=sorted(tdet.values()), frames_with_all4_visible=len(full4))
    a_near = any(frames[first]["dets"][d]["side"] == 0 and s.startswith("A") for d, s in tdet.items())
    set_near(frames, P, a_near, tap_point)
    # schedule (court cue) vs truth: is each truth person on the side the schedule says?
    side_ok = [int((frames[fi]["dets"][d]["side"] == 0) == (frames[fi]["near_team"] == s[0])) for fi, t in truth_s.items() for d, s in t.items()]
    trep["schedule_vs_truth_side"] = float(np.mean(side_ok)) if side_ok else None
    # unit costs (ms): p10 ~ unthrottled (the job may run at background QoS), p50 as measured
    unit = {k: pct(ms[k], 10) for k in ("ms_osnet", "vision", "ms_colour_box", "colour_mask")}
    unit_ms = dict(osnet=unit["ms_osnet"], vision=unit["vision"], colour_box=unit["ms_colour_box"], colour_mask=unit["colour_mask"])
    det_ms = dict(p10=pct(ms["ms_det"], 10), p50=pct(ms["ms_det"], 50))
    s2 = lambda f: f["set"] >= 1
    # ---------------- learned models, trained on set 1 (index 0) only
    tl = time.time(); learned_rep = {}
    from opencourtvision.players import learned as LN
    app = lambda d: np.r_[d["osnet"], d["colour_mask"] if d.get("colour_mask") is not None else d["colour_box"]]
    Xh, yh, gh = [], [], []; gid = {}
    for fi, t in truth.items():
        if frames[fi]["set"] != 0:
            continue
        for d, s in t.items():
            key = (frames[fi]["point"], s); gid.setdefault(key, len(gid))
            Xh.append(app(frames[fi]["dets"][d])); yh.append(gid[key]); gh.append(fi)
    head = LN.train_metric_head(np.array(Xh), np.array(yh), np.array(gh)) if len(gid) >= 4 else None
    if head is not None:
        allX = [(f, d) for f in frames for d in f["dets"]]
        t1 = time.perf_counter(); H = head(np.array([app(d) for _, d in allX])); unit_ms["head"] = 1000 * (time.perf_counter() - t1) / max(1, len(allX))
        for (f, d), h in zip(allX, H):
            d["head"] = h
    learned_rep["head_train"] = dict(tracklets=len(gid), samples=len(yh))
    # pose-role MLP (labels: position roles of set 1)
    Xp, yp = [], []
    prep_roles(frames, keep)
    for fi in truth_s:
        f = frames[fi]
        if f["set"] != 0:
            continue
        srv_side = 0 if f["near_team"] == f["server"] else 1
        for i in keep[fi]:
            d = f["dets"][i]
            if d["side"] == srv_side and d.get("pose_vec") is not None and any(f["dets"][j].get("role_pos") == "srv" for j in keep[fi]):
                Xp.append(d["pose_vec"]); yp.append(int(d.get("role_pos") == "srv" and d.get("p_role_pos", 0) >= 0.6))
    posenet = LN.train_mlp(np.array(Xp), np.array(yp)) if len(set(yp)) == 2 and len(yp) >= 20 else None
    if posenet is not None:
        t1 = time.perf_counter(); posenet.predict_proba(np.array(Xp[:64])); unit_ms["posenet"] = 1000 * (time.perf_counter() - t1) / min(64, len(Xp))
    set_near(frames, P, a_near, tap_point); prep_roles(frames, keep, posenet)
    # server pick accuracy on later sets (position roles with conf >= 0.6 as reference)
    sp = defaultdict(list)
    for fi in truth_s:
        f = frames[fi]
        if f["set"] == 0:
            continue
        ref = [i for i in keep[fi] if f["dets"][i].get("role_pos") == "srv" and f["dets"][i]["p_role_pos"] >= 0.8]
        if not ref:
            continue
        for c in ("role_pose", "role_posenet"):
            got = [i for i in keep[fi] if f["dets"][i].get(c) == "srv"]
            if got:
                sp[c].append(int(got[0] == ref[0]))
    learned_rep["server_pick_vs_position_later_sets"] = {c: dict(acc=float(np.mean(v)), n=len(v)) for c, v in sp.items()}
    # learned in-play MLP: truth people vs other in-zone people (frames where the side has 2 truth people)
    Xi, yi = [], []
    for fi, t in truth.items():
        f = frames[fi]
        if f["set"] != 0:
            continue
        for s in (0, 1):
            ts = [d for d in t if f["dets"][d]["side"] == s]
            if len(ts) != 2:
                continue
            for j, d in enumerate(f["dets"]):
                if d["side"] == s:
                    Xi.append(d["feat_inplay"]); yi.append(int(j in ts))
    inplay = LN.train_mlp(np.array(Xi), np.array(yi)) if len(set(yi)) == 2 else None
    keep_l = hand_keep(frames, (lambda F: inplay.predict_proba(np.array(F))[:, 1]) if inplay is not None else None) if inplay is not None else keep

    def inplay_quality(K, flt):
        tp = fp = fn = 0
        for fi, t in truth.items():
            f = frames[fi]
            if not flt(f):
                continue
            for s in (0, 1):
                ts = {d for d in t if f["dets"][d]["side"] == s}
                if len(ts) != 2:
                    continue
                ks = {i for i in K[fi] if f["dets"][i]["side"] == s}
                tp += len(ks & ts); fp += len(ks - ts); fn += len(ts - ks)
        return dict(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), false_per_100_sides=100 * fp / max(1, (tp + fn) / 2))
    learned_rep["inplay_later_sets"] = dict(hand=inplay_quality(keep, s2), mlp=inplay_quality(keep_l, s2))
    # association LR (learned fusion weights) on set 1 decisions of the full learned tracker
    cues_full = HAND + tuple(c for c in LEARNED if (c != "head" or head is not None) and (c != "role_posenet" or posenet is not None))
    recs = []
    def recorder(F, tks, comps, side_team, rows):
        if F[0]["set"] != 0:
            return
        for comp, row in zip(comps, rows):
            side, col = comp; team = side_team[side]; votes = [0, 0]
            for ti, c in col.items():
                for p, di in tks[ti].items:
                    t = truth.get(F[p]["_fi"], {}).get(keep[F[p]["_fi"]][di])
                    if t is None or t[0] != team:
                        continue
                    for lab in (0, 1):
                        votes[lab] += int(t == f"{team}{1 + (c ^ lab)}")
            if max(votes) == 0 or votes[0] == votes[1]:
                continue
            right = int(votes[1] > votes[0])
            recs.append([row[c][1 - right] - row[c][right] for c in sorted(row)])
    learned_rep["train_s"] = round(time.time() - tl, 1)
    # ---------------- variants
    for i, f in enumerate(frames):
        f["_fi"] = i
    def go(name, cues, K=None, mapping=None, start=None, **kw):
        K = K or keep; start = first if start is None else start; mapping = mapping or truth[start]
        for i, f in enumerate(frames):
            f["_map"] = {j: K[i][j] for j in range(len(K[i]))}
        tw = time.time(); preds, tr = run_tracker(frames, K, start, mapping, cues, unit_ms, **kw)
        return preds, tr, time.time() - tw
    R = {}
    def score(name, preds, tr, K=None, extra=None):
        fl = lambda f: f["_fi"] > first
        r = dict(serve=evaluate(frames, preds, truth_s, lambda f: f["serve"] and fl(f)),
                 rally=evaluate(frames, preds, truth_r, lambda f: not f["serve"] and fl(f), n_points=len(P)),
                 all=evaluate(frames, preds, truth, fl),
                 later_sets=evaluate(frames, preds, truth, lambda f: fl(f) and s2(f)),
                 ms_features=tr.ms_feat / max(1, tr.n_frames), ms_track=tr.ms_track / max(1, tr.n_frames), cues=list(tr.cues))
        r["ms_frame"] = det_ms["p10"] + r["ms_features"] + r["ms_track"]
        if extra:
            r.update(extra)
        R[name] = r; log(name, f"serve={r['serve']['id_acc']:.3f} rally={r['rally']['id_acc']:.3f} sw/100={r['all']['switches_per_100pts']:.1f} ms={r['ms_frame']:.0f}")
        return r
    base = ("court", "motion")
    variants = {"court_only": ("court",), "court+motion": base}
    for c in ("role_pos", "colour_box", "colour_mask", "osnet", "role_pose") + tuple(c for c in LEARNED if c in cues_full):
        variants[f"court+motion+{c}"] = base + (c,)
    variants["full_hand"] = HAND
    for c in HAND:
        variants[f"full_hand-{c}"] = tuple(x for x in HAND if x != c)
    variants["full_learned"] = cues_full
    if head is not None:
        variants["full_hand-osnet+head"] = tuple(x for x in HAND if x != "osnet") + ("head",)
    for n, cues in variants.items():
        preds, tr, _ = go(n, cues); score(n, preds, tr)
    # learned fusion weights
    for i, f in enumerate(frames):
        f["_fi"] = i
    run_tracker(frames, keep, first, truth[first], cues_full, unit_ms, recorder=recorder)
    if len(recs) >= 20:
        dC = np.array(recs); w = LN.train_assoc(np.r_[dC, -dC], np.r_[np.ones(len(dC)), np.zeros(len(dC))])
        names = sorted([c for c in cues_full if c not in ("court", "motion")]); wd = dict(zip(names, (w / max(w.max(), 1e-9) * 3).tolist()))
        learned_rep["assoc_weights"] = {k: round(v, 3) for k, v in wd.items()}; learned_rep["assoc_samples"] = len(recs)
        preds, tr, _ = go("full_learned+assoc", cues_full, weights=wd); score("full_learned+assoc", preds, tr)
    if inplay is not None:
        preds, tr, _ = go("full_hand+inplay_mlp", HAND, K=keep_l, mapping={d: s for d, s in truth[first].items() if d in keep_l[first]})
        score("full_hand+inplay_mlp", preds, tr)
    # ---------------- per-cue ablation table: gain alone (over court+motion) and loss when removed (from full_hand)
    abl = {}
    for c in ("court", "motion", "role_pos", "colour_box", "colour_mask", "osnet", "role_pose", "head", "role_posenet"):
        e = {}
        if f"court+motion+{c}" in R:
            for k in ("serve", "rally"):
                e[f"gain_alone_{k}"] = R[f"court+motion+{c}"][k]["id_acc"] - R["court+motion"][k]["id_acc"]
        if f"full_hand-{c}" in R:
            for k in ("serve", "rally"):
                e[f"loss_removed_{k}"] = R["full_hand"][k]["id_acc"] - R[f"full_hand-{c}"][k]["id_acc"]
        if c == "court":
            e["gain_alone_serve"] = R["court_only"]["serve"]["id_acc"] - 0.25
        if c == "motion":
            e["gain_alone_serve"] = R["court+motion"]["serve"]["id_acc"] - R["court_only"]["serve"]["id_acc"]
            e["gain_alone_rally"] = R["court+motion"]["rally"]["id_acc"] - R["court_only"]["rally"]["id_acc"]
        need = {"colour_box": ["colour_box"], "colour_mask": ["vision", "colour_mask"], "osnet": ["osnet"], "head": ["osnet", "head"],
                "role_pose": ["vision"], "role_posenet": ["vision", "posenet"]}.get(c, [])
        e["ms_per_player"] = sum(unit_ms.get(x, 0.0) for x in need); e["ms_per_frame_4p"] = 4 * e["ms_per_player"]
        abl[c] = e
    # ---------------- cascade: cheap-first order by gain per ms, sweep tau
    gain = {c: max(abl[c].get("gain_alone_serve", 0), 0) + max(abl[c].get("gain_alone_rally", 0), 0) for c in abl if c in cues_full}
    rest = [c for c in cues_full if c not in ("court", "motion", "role_pos")]
    order = ["court", "motion", "role_pos"] + sorted(rest, key=lambda c: -(gain.get(c, 0) / max(0.05, abl[c]["ms_per_player"])))
    casc = {}
    for tau in (0.5, 1.0, 2.0, 3.0, 5.0):
        preds, tr, _ = go(f"cascade_tau{tau}", tuple(order), order=tuple(order), tau=tau)
        r = score(f"cascade_tau{tau}", preds, tr, extra=dict(order=order, cue_use={c: tr.cue_used[c] / max(1, tr.decisions) for c in order}))
        casc[tau] = r
    # ---------------- labelling: 4 taps vs 2 taps vs fewer
    labr = {}
    t4 = truth[first]; inv_t = {s: d for d, s in t4.items()}
    fk = dict(frames[first], dets=[frames[first]["dets"][i] for i in keep[first]]); kinv = {i: j for j, i in enumerate(keep[first])}
    def taps(names):
        return [(kinv[inv_t[n]], n) for n in names]
    for name, tp, groups in (("4_taps", ["A1", "A2", "B1", "B2"], []), ("2_taps_one_per_team", ["A1", "B1"], []),
                             ("2_taps_same_team", ["A1", "A2"], [("B1", "B2")]), ("1_tap", ["A1"], [("B1", "B2")]),
                             ("0_taps", [], [("A1", "A2"), ("B1", "B2")])):
        if len(tp) == 4:
            m = {kinv[d]: s for d, s in t4.items()}; st = "ok"
        else:
            m, st = deduce_ids(fk["dets"], taps(tp), near_default="A" if a_near else "B")
        if not m:
            labr[name] = dict(status=st); continue
        mapping = {keep[first][j]: s for j, s in m.items()}
        preds, tr, _ = go(name, HAND, mapping=mapping)
        if groups:
            preds = relabel_best(preds, truth, groups)
        r = score(f"label_{name}", preds, tr); labr[name] = dict(status=st, serve=r["serve"]["id_acc"], rally=r["rally"]["id_acc"], all=r["all"]["id_acc"],
                                                                 switches=r["all"]["switches_per_100pts"])
    # deduction failure analysis: tap A1 + B1 on every frame that has truth for all four (and on every frame: can we deduce?)
    stat = Counter(); wrong = 0; okn = 0; any_frame = Counter()
    for fi, f in enumerate(frames):
        fk2 = [f["dets"][i] for i in keep[fi]]; kv = {i: j for j, i in enumerate(keep[fi])}
        t = truth.get(fi, {})
        ns = Counter(d["side"] for d in fk2)
        any_frame["deducible" if ns[0] == 2 and ns[1] == 2 else f"defer:side_counts_{ns[0]}_{ns[1]}"] += 1
        if len(t) != 4:
            continue
        it = {s: d for d, s in t.items()}
        if any(it[n] not in kv for n in ("A1", "B1")):
            stat["tapped_person_not_in_play"] += 1; continue
        m, st = deduce_ids(fk2, [(kv[it["A1"]], "A1"), (kv[it["B1"]], "B1")]); stat[st] += 1
        if st == "ok":
            okn += 1; wrong += int(any(m.get(kv.get(d)) != s for d, s in t.items()))
    labr["deduction_on_truth_frames"] = dict(status=dict(stat), wrong_given_ok=wrong / max(1, okn), n_ok=okn)
    labr["deduction_any_frame"] = {k: v / len(frames) for k, v in any_frame.most_common(6)}
    # ---------------- check set + contact sheets
    sheets = check_sheets(out, frames, truth_s, R, keep, P)
    rep = dict(schema="vamos.player_id.eval.v1", video=os.path.basename(lab), fps=FPS, points=len(P), frames=len(frames),
               truth=trep, truth_rally_frames=len(truth_r), tap_point=tap_point, det_ms=det_ms, unit_ms=unit_ms,
               ms_samples=dict(det=len(ms["ms_det"]), vision=len(ms["vision"])), variants=R, ablation=abl, cascade_order=order,
               learned=learned_rep, labelling=labr, sheets=sheets, runtime_s=round(time.time() - T0, 1),
               note="Truth is a proxy (rules truth at serve frames, motion-chain truth in rallies), not hand labels.")
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_md(rep, os.path.join(out, "report.md")); log("eval done", round(time.time() - T0), "s")
    return rep


def check_sheets(out, frames, truth_s, R, keep, P):
    """contact sheets: (1) identity gallery, (2) 24 serve frames with truth ids, (3) frames where full_hand disagrees."""
    import cv2
    os.makedirs(os.path.join(out, "check"), exist_ok=True)
    def crop(d, h=160):
        im = cv2.imread(os.path.join(out, d["crop"]))
        return cv2.resize(im, (max(8, int(im.shape[1] * h / im.shape[0])), h)) if im is not None else np.zeros((h, 60, 3), np.uint8)
    def tile(cells, w=120, h=160):
        im = np.full((h + 22, w, 3), 255, np.uint8); c, txt = cells; c = c[:, :w]
        im[22:22 + c.shape[0], : c.shape[1]] = c; cv2.putText(im, txt, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1); return im
    gal = defaultdict(list); keys = sorted(truth_s)
    for fi in keys[:: max(1, len(keys) // 8)]:
        for d, s in truth_s[fi].items():
            gal[s].append((crop(frames[fi]["dets"][d]), f"{s} p{frames[fi]['point']}"))
    rows = [np.hstack([tile(c) for c in gal[s][:8]] + [np.full((182, 120, 3), 255, np.uint8)] * (8 - len(gal[s][:8]))) for s in IDS if gal[s]]
    if rows:
        cv2.imwrite(os.path.join(out, "check", "gallery.jpg"), np.vstack(rows))
    sel = keys[:: max(1, len(keys) // 24)][:24]; rows = []; items = []
    for fi in sel:
        f = frames[fi]; cells = []
        for d, s in sorted(truth_s[fi].items(), key=lambda q: q[1]):
            cells.append(tile((crop(f["dets"][d]), f"{s} p{f['point']}"))); items.append(dict(point=f["point"], t=round(f["t"], 2), box=[round(x) for x in f["dets"][d]["box"]], id=s))
        while len(cells) < 4:
            cells.append(np.full((182, 120, 3), 255, np.uint8))
        rows.append(np.hstack(cells))
    for k in range(0, len(rows), 12):
        cv2.imwrite(os.path.join(out, "check", f"serve_sheet_{k // 12}.jpg"), np.vstack(rows[k:k + 12]))
    json.dump(dict(note="Check set: for each serve frame, the 4 people and the id the rules truth gave them. "
                        "A1/A2 = your team (Host), B1/B2 = Guest. Mark any wrong id.", items=items), open(os.path.join(out, "check", "check_set.json"), "w"), indent=1)
    return dict(gallery="check/gallery.jpg", serve_sheets=sorted(os.path.basename(x) for x in glob.glob(os.path.join(out, "check", "serve_sheet_*.jpg"))),
                check_set="check/check_set.json", n_items=len(items))


def write_md(rep, path):
    V = rep["variants"]; f = lambda x: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.1f}%"
    L = ["# 4-player ID tracker (proxy truth; see notes)", "",
         f"points {rep['points']}, frames {rep['frames']} at {rep['fps']} fps; serve-frame truth on {rep['truth'].get('points_with_id_truth')} points, "
         f"motion-chain truth on {rep['truth_rally_frames']} rally frames; tap frame {rep['truth'].get('tap_frame')}", "",
         "| variant | ID acc serve | ID acc rally | ID acc later sets | switches /100 pts | recall near | recall far | false /100 fr | ms/frame |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in V.items():
        L.append(f"| {n} | {f(r['serve']['id_acc'])} | {f(r['rally']['id_acc'])} | {f(r['later_sets']['id_acc'])} | {r['all']['switches_per_100pts']:.1f} | "
                 f"{f(r['all']['recall_near'])} | {f(r['all']['recall_far'])} | {r['all']['false_per_100_frames']:.1f} | {r['ms_frame']:.0f} |")
    L += ["", "| cue | gain alone serve | gain alone rally | loss removed serve | loss removed rally | ms/player |", "|---|---|---|---|---|---|"]
    for c, e in rep["ablation"].items():
        L.append(f"| {c} | {f(e.get('gain_alone_serve'))} | {f(e.get('gain_alone_rally'))} | {f(e.get('loss_removed_serve'))} | {f(e.get('loss_removed_rally'))} | {e['ms_per_player']:.1f} |")
    full = V.get("full_hand", {}).get("all", {}).get("per_player", {})
    L += ["", "per player (full_hand, all truth frames): " + ", ".join(f"{k} {f(v)}" for k, v in full.items()), "",
          "labelling: " + json.dumps(rep["labelling"], default=str), "", "learned: " + json.dumps(rep["learned"], default=str), "",
          "truth: " + json.dumps({k: v for k, v in rep["truth"].items() if k != "outlier_list"}, default=str), "",
          f"cascade order: {rep['cascade_order']}; unit ms: {json.dumps(rep['unit_ms'])}; detector ms: {json.dumps(rep['det_ms'])}"]
    open(path, "w").write("\n".join(L) + "\n")
