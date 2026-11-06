"""Learned hit / bounce event detector (opencourtvision.ball.learned EventNet) vs the physics tracker's own events.

Input: tracker outputs dumped by physball_eval --dump_pos (pos, conf, physics events) for one arm, on the TRAIN
points (to fit) and the HELD-OUT points (to test). Labels from SwingVision on the train points: hit = SV shot time,
bounce = the frame between shot i and i+1 where the track passes closest (<= 25 px) to SV's bounce spot of shot i.
Features: track kinematics, physics flags, audio envelope, nearest player's wrist speed and distance (attention.py).
Held-out metrics (vs SV, same rules for both): hit recall/precision (+-3 frames), last-contact hit found, final bounce
found/precision (first bounce 0.1-2.5 s after the last SV shot within 1.5 m of SV's bounce). Optionally writes v3
hybrid tracks with the learned bounces for the point-winner stages. CPU (box); private outputs.

  eventnet.py --train_dump DIR --test_dump DIR --model OUT.pt --out RESULT.json [--write_v3 SRC_V3 DST] [--no_audio]
              [--write_tc TC.json]   # learned last hit per test clip, for the point winner (pw_learned_tc.py)
Arms scored on the test clips: physics (physics hits + bounces), learned (learned hits + bounces) and hits_phys
(learned hits + physics bounces: the hybrid).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ball_tracker"))
import common as C  # noqa: E402

from opencourtvision.ball import attention as AT  # noqa: E402
from opencourtvision.ball import learned as L  # noqa: E402

V = os.environ.get("OCV_VF", "/workspace/vf"); DATA = os.environ.get("OCV_DATA", f"{V}/ballbench/run1/data"); TR = f"{V}/dev_match_tracks_v2/seg_r"
SV = json.load(open(f"{V}/ballbench/sv_events_dev_match.json"))["points"]
MAN = json.load(open(f"{V}/ballbench/ocv_phys/tests/fixtures/court_dev_match_view0_manual.json"))
IDX = json.load(open(f"{DATA}/index.json"))["clips"]


def court2img(name):
    z = np.load(f"{DATA}/clips/{name}.npz"); t0 = float(z["t0"])
    if MAN["t0"] <= t0 <= MAN["t1"]:
        return C.h_court2img(MAN["H_img2court"], 3840)
    H = z["Hc2i"]
    return H if np.isfinite(H).all() else None


def swing_track(name, pos):
    m = IDX[name]; n = len(pos)
    z = np.load(f"{DATA}/clips/{name}.npz"); d = np.load(f"{TR}/seg{m['idx']:03d}.npz")
    ps = AT.players_from_npz(d, n, int(round((float(z["t0"]) - float(d["t0"])) * 30)))
    sw = np.zeros(n); pdist = np.full(n, 10.0)
    p = pos.copy(); ok = np.isfinite(p[:, 0])
    if ok.any():
        for c in (0, 1):
            p[:, c] = np.interp(np.arange(n), np.arange(n)[ok], pos[ok, c])
    for i in range(n):
        w = AT.wrist_speed(ps, i)
        if w and ok.any():
            j = int(np.argmin([np.hypot(*(p[i] - c)) / h for _, _, c, h in w]))
            sw[i] = w[j][0]; pdist[i] = np.hypot(*(p[i] - w[j][2])) / w[j][3]
    return sw, pdist


def clip_xy(name, dump, env, audio=True):
    t = np.load(os.path.join(dump, name + ".npz")); pos, conf = t["pos"], t["conf"]; t0 = float(t["t0"]); n = len(pos)
    evs = [dict(kind=str(k), i=int(i)) for i, k in zip(t["ev_i"], t["ev_kind"])]
    adb = env[0][np.clip(((t0 + np.arange(n) / 30.0) * 100).astype(int), 0, env.shape[1] - 1)] if audio else None
    sw, pdist = swing_track(name, pos)
    return L.event_features(pos, conf, evs, adb, sw, pdist), pos, evs, t0


def labels(name, pos, t0):
    n = len(pos); Y = np.zeros((2, n), np.float32)
    shots = sorted((SV.get(str(IDX[name]["idx"])) or {}).get("shots", []), key=lambda s: s["t"])
    H = court2img(name)
    for j, s in enumerate(shots):
        k = int(round((s["t"] - t0) * 30))
        for dk, w in ((0, 1.0), (-1, 0.5), (1, 0.5)):
            if 0 <= k + dk < n:
                Y[0, k + dk] = max(Y[0, k + dk], w)
        if H is not None and s.get("bounce_xy") and s.get("result") in ("In", "Out"):
            b = C.project(H, s["bounce_xy"]); k1 = int(round((shots[j + 1]["t"] - t0) * 30)) if j + 1 < len(shots) else n
            lo, hi = max(0, k + 3), min(n, k1 - 1)
            if hi > lo:
                d = np.hypot(pos[lo:hi, 0] - b[0], pos[lo:hi, 1] - b[1])
                if np.isfinite(d).any() and np.nanmin(d) <= 25:
                    kb = lo + int(np.nanargmin(d))
                    for dk, w in ((0, 1.0), (-1, 0.5), (1, 0.5)):
                        if 0 <= kb + dk < n:
                            Y[1, kb + dk] = max(Y[1, kb + dk], w)
    return Y, shots


def score(name, t0, n, pos, hits, bounces, shots, G):
    """hits/bounces: frame lists. Updates G (same rules for physics and learned events)."""
    sk = [int(round((s["t"] - t0) * 30)) for s in shots if 0 <= (s["t"] - t0) * 30 < n]
    G["sv_hits"] += len(sk); G["hit_found"] += sum(any(abs(h - k) <= 3 for h in hits) for k in sk)
    G["hits_rep"] += len(hits); G["hits_ok"] += sum(any(abs(h - k) <= 3 for k in sk) for h in hits)
    if shots and 0 <= (shots[-1]["t"] - t0) * 30 < n:
        kl = int(round((shots[-1]["t"] - t0) * 30)); G["last_n"] += 1; G["last_hit"] += any(abs(h - kl) <= 4 for h in hits)
        last = shots[-1]; H = court2img(name)
        if last.get("result") in ("In", "Out") and last.get("bounce_xy") and H is not None:
            G["fin_n"] += 1
            fb = [b for b in bounces if kl + 3 < b <= kl + 75 and np.isfinite(pos[b, 0])]
            if fb:
                G["fin_rep"] += 1
                g = np.linalg.solve(H, np.r_[pos[fb[0]], 1.0]); g = g[:2] / g[2]
                G["fin_ok"] += math.hypot(g[0] - last["bounce_xy"][0], g[1] - last["bounce_xy"][1]) <= 1.5
    G["pts"] += 1; G["bounces"] += len(bounces)


def summ(G):
    f = lambda a, b: round(a / b, 3) if b else None
    return dict(hit_recall=f(G["hit_found"], G["sv_hits"]), hit_precision=f(G["hits_ok"], G["hits_rep"]), last_contact_hit=f(G["last_hit"], G["last_n"]),
                last_n=G["last_n"], final_bounce_found=f(G["fin_rep"], G["fin_n"]), final_bounce_precision=f(G["fin_ok"], G["fin_rep"]),
                final_bounce_ok=G["fin_ok"], final_bounce_n=G["fin_n"], bounces_per_point=f(G["bounces"], G["pts"]))


def newG():
    return dict(sv_hits=0, hit_found=0, hits_rep=0, hits_ok=0, last_n=0, last_hit=0, fin_n=0, fin_rep=0, fin_ok=0, pts=0, bounces=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dump", required=True); ap.add_argument("--test_dump", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_audio", action="store_true"); ap.add_argument("--no_swing", action="store_true")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--write_v3", nargs=2); ap.add_argument("--write_tc")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4); torch.manual_seed(0)
    env = np.load(f"{V}/ballbench/audio_env_dev_match.npy")
    tr = sorted(f[:-4] for f in os.listdir(a.train_dump) if f.endswith(".npz"))
    te = sorted(f[:-4] for f in os.listdir(a.test_dump) if f.endswith(".npz"))
    Xs, Ys = [], []
    for nm in tr:
        X, pos, _, t0 = clip_xy(nm, a.train_dump, env, not a.no_audio)
        if a.no_swing:
            X[:, 12:14] = 0
        Y, _ = labels(nm, pos, t0); Xs.append(X); Ys.append(Y)
    print(f"train clips {len(Xs)} hit labels {int(sum((y[0] == 1).sum() for y in Ys))} bounce labels {int(sum((y[1] == 1).sum() for y in Ys))}", flush=True)
    net = L.build_eventnet(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for ep in range(a.epochs):
        for j in np.random.default_rng(ep).permutation(len(Xs)):
            x = torch.from_numpy(Xs[j].T[None]); y = torch.from_numpy(Ys[j][None])
            lo = net(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, y, pos_weight=torch.tensor([[[15.0], [25.0]]]))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); torch.save(dict(model=net.state_dict(), audio=not a.no_audio, swing=not a.no_swing), a.model)
    # threshold per head picked on TRAIN (max F1), then frozen for held-out
    def probs(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(X.T[None])))[0].numpy()
    best = {}
    for h in (0, 1):
        sc = []
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            tp = fp = fn = 0
            for X, Y in zip(Xs, Ys):
                ev = L.pick_events(probs(X)[h], th); gt = [k for k in range(len(Y[h])) if Y[h][k] == 1]
                tp += sum(any(abs(e - g) <= 3 for g in gt) for e in ev); fp += sum(not any(abs(e - g) <= 3 for g in gt) for e in ev)
                fn += sum(not any(abs(e - g) <= 3 for e in ev) for g in gt)
            sc.append((2 * tp / max(1, 2 * tp + fp + fn), th))
        best[h] = max(sc)[1]
    Gp, Gl, Gh = newG(), newG(), newG(); ms = 0.0; nfr = 0; learned_bounces = {}; learned_tc = {}
    for nm in te:
        t = time.perf_counter()
        X, pos, evs, t0 = clip_xy(nm, a.test_dump, env, not a.no_audio)
        if a.no_swing:
            X[:, 12:14] = 0
        P = probs(X); ms += time.perf_counter() - t; nfr += len(pos)
        _, shots = labels(nm, pos, t0)
        score(nm, t0, len(pos), pos, [e["i"] for e in evs if e["kind"] == "hit"], [e["i"] for e in evs if e["kind"] == "bounce"], shots, Gp)
        lh = L.pick_events(P[0], best[0]); lb = [k for k in L.pick_events(P[1], best[1]) if np.isfinite(pos[k, 0])]
        score(nm, t0, len(pos), pos, lh, lb, shots, Gl)
        score(nm, t0, len(pos), pos, lh, [e["i"] for e in evs if e["kind"] == "bounce"], shots, Gh)
        learned_tc[nm] = dict(idx=IDX[nm]["idx"], t_last=round(t0 + lh[-1] / 30.0, 3) if lh else None, hits=[round(t0 + k / 30.0, 3) for k in lh])
        learned_bounces[nm] = (t0, [(t0 + k / 30.0, float(pos[k, 0]), float(pos[k, 1])) for k in lb])
    res = dict(thresholds=best, physics=summ(Gp), learned=summ(Gl), hits_phys=summ(Gh), ms_per_frame=round(1000 * ms / max(1, nfr), 2),
               train_clips=len(Xs), test_clips=len(te), audio=not a.no_audio, swing=not a.no_swing)
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))
    if a.write_tc:
        json.dump(dict(note="PRIVATE. Learned last hit per clip (video time).", clips=learned_tc), open(a.write_tc, "w"))
    if a.write_v3:
        src, dst = a.write_v3; os.makedirs(dst, exist_ok=True); S = 3840 / 1280.0
        for f in os.listdir(src):
            if not f.endswith(".npz"):
                continue
            d = np.load(os.path.join(src, f)); out = {k: d[k] for k in d.files}
            nm = f"dev_match_{f[:-4]}"
            if nm in learned_bounces:
                t0, lb = learned_bounces[nm]; tw0, tw1 = t0, t0 + IDX[nm]["n"] / 30.0
                old = np.atleast_2d(out.get("bounces", np.zeros((0, 3)))).reshape(-1, 3)
                keep = old[(old[:, 0] < tw0) | (old[:, 0] >= tw1)] if len(old) else old
                new = np.array([[tb, x * S, y * S] for tb, x, y in lb], float).reshape(-1, 3)
                out["bounces"] = np.r_[keep, new].reshape(-1, 3); out["bounce_pin"] = np.full(len(out["bounces"]), np.nan)
            np.savez_compressed(os.path.join(dst, f), **out)


if __name__ == "__main__":
    main()
