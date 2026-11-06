"""Synthetic doubles match for tests: 4 players with outfit colours + re-ID vectors, rule-following serve positions,
end changes, rally motion, spectators and detection noise. Frames come in the tracker/eval format (no images)."""
from __future__ import annotations

import numpy as np

from .cues import COLOUR_DIM
from .geometry import CW, NET_Y


def match_points(sets=((6, 1), (6, 2)), first_server="B", seed=0):
    rng = np.random.default_rng(seed); P = []; idx = 0; t = 100.0; sets_won = [0, 0]
    for sa, sb in sets:
        seq = ["A"] * sa + ["B"] * sb; rng.shuffle(seq)
        games = [0, 0]; nset = sum(sets_won); srv = first_server if nset % 2 == 0 else ("A" if first_server == "B" else "B")
        for gi, gw in enumerate(seq):
            npts = int(rng.integers(4, 7)); pts = [0, 0]
            for k in range(npts):
                idx += 1; dur = float(rng.uniform(4, 12))
                P.append(dict(idx=idx, start=t, end=t + dur, kind="point", winner=gw, server=srv, side="deuce" if k % 2 == 0 else "ad",
                              score_before=dict(sets=list(sets_won), games=list(games), pts=list(pts), tiebreak=False), t_serve=t + 1.0))
                t += dur + 20; pts[0 if gw == "A" else 1] += 1
            games[0 if gw == "A" else 1] += 1; srv = "A" if srv == "B" else "B"
        sets_won[0 if sa > sb else 1] += 1
    return P


def make_frames(P, fps=2.5, serve_i=3, seed=0, miss=0.03, spectators=1, id_noise=0.35, colour_noise=0.25):
    """returns (frames, people) where people maps 'A1'.. to its appearance prototypes; each det has `_true` id."""
    from .rules import end_flips, receive_side, serve_parity, set_index
    rng = np.random.default_rng(seed)
    proto = {s: dict(osnet=rng.normal(size=512), colour=np.abs(rng.normal(size=COLOUR_DIM))) for s in ("A1", "A2", "B1", "B2")}
    for s in proto:
        proto[s]["osnet"] /= np.linalg.norm(proto[s]["osnet"])
    proto["A2"]["colour"] = 0.7 * proto["A1"]["colour"] + 0.3 * proto["A2"]["colour"]      # partners dress alike
    flips = end_flips(P); frames = []
    for p, fl in zip(P, flips):
        a_near = fl % 2 == 0; st = set_index(p["score_before"]); par = serve_parity(p); rs = receive_side(p)
        srv_team = p["server"]; rcv_team = "B" if srv_team == "A" else "A"
        # who serves / receives: per set the order is fixed
        server = f"{srv_team}{1 + (par ^ (st % 2))}"; srv_partner = server[0] + ("2" if server[1] == "1" else "1")
        receiver = f"{rcv_team}{1 + (rs ^ (st % 2))}"; rcv_partner = receiver[0] + ("2" if receiver[1] == "1" else "1")
        srv_near = (srv_team == "A") == a_near
        sx = CW / 2 + (1.5 if rs == 0 else -1.5) * (1 if srv_near else -1)
        def place(near, x, depth):
            return np.array([x, depth if near else 23.77 - depth])
        pos = {server: place(srv_near, sx, -0.5), srv_partner: place(srv_near, CW - sx, 9.0),
               receiver: place(not srv_near, CW - sx, -0.3), rcv_partner: place(not srv_near, sx, 9.0)}
        n = int((p["end"] - p["t_serve"] + 1.2) * fps)
        vel = {s: np.zeros(2) for s in pos}; home = {s: bool(q[1] < NET_Y) for s, q in pos.items()}
        for f in range(n):
            dets = []
            for s, q in pos.items():
                if f > serve_i:
                    vel[s] = 0.6 * vel[s] + rng.normal(scale=1.2, size=2); q += vel[s] / fps
                    near = home[s]; q[1] = np.clip(q[1], -3, NET_Y - 0.8) if near else np.clip(q[1], NET_Y + 0.8, 26.7); q[0] = np.clip(q[0], -2, CW + 2)
                if rng.random() < miss:
                    continue
                o = proto[s]["osnet"] + rng.normal(scale=id_noise / np.sqrt(512), size=512) * np.sqrt(512) / 8
                c = np.abs(proto[s]["colour"] + rng.normal(scale=colour_noise, size=COLOUR_DIM)); c = np.sqrt(c / c.sum())
                kp = np.zeros((17, 3)); kp[:, 2] = 1.0; kp[5:7, 1] = 0.25; kp[11:13, 1] = 0.55; kp[[9, 10], 1] = 0.45
                if f == serve_i and s == server:
                    kp[9, 1] = 0.02
                dets.append(dict(box=[100, 100, 140, 220], conf=float(rng.uniform(0.6, 0.95)), xy=q.copy() + rng.normal(scale=0.1, size=2), h_ratio=1.0,
                                 crop="", osnet=(o / np.linalg.norm(o)).astype(np.float32), colour_box=c.astype(np.float32),
                                 colour_mask=c.astype(np.float32), kp=kp, _true=s))
            for k in range(spectators):
                xy = np.array([CW + 3.2 + k, 5 + 3 * k]) + rng.normal(scale=0.2, size=2)
                v = rng.normal(size=512)
                dets.append(dict(box=[100, 100, 140, 220], conf=0.5, xy=xy, h_ratio=1.0, crop="", osnet=(v / np.linalg.norm(v)).astype(np.float32),
                                 colour_box=np.sqrt(np.full(COLOUR_DIM, 1.0 / COLOUR_DIM)).astype(np.float32), colour_mask=None, kp=None, _true=None))
            for d in dets:
                d["side"] = int(d["xy"][1] >= NET_Y)
            frames.append(dict(t=p["t_serve"] - 1.2 + f / fps, point=p["idx"], fi_in_point=f, serve=(f == serve_i), set=st, server=srv_team,
                               parity=par, rside=rs, dets=dets))
    return frames, proto
