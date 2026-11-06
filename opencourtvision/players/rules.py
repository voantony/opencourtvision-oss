"""Doubles rules used as identity constraints: which team is on the near side (changeovers), which partner serves
(serve rotation) and which partner receives on the deuce side. All derived from the point's score_before, which comes
from the point labels (ground truth) or from the scorer.
"""
from __future__ import annotations

from collections import defaultdict


def set_index(sb) -> int:
    return int(sum(sb["sets"]))


def end_flips(points) -> list[int]:
    """number of end changes before each point (players change ends after odd games of a set, and every 6 points of
    a tiebreak). `points` in match order, each with score_before {sets, games, pts, tiebreak}."""
    last_games = defaultdict(int)
    for p in points:
        sb = p["score_before"]; last_games[set_index(sb)] = max(last_games[set_index(sb)], sum(sb["games"]))
    out = []
    for p in points:
        sb = p["score_before"]; s = set_index(sb)
        f = sum((last_games[k] + 1 + 1) // 2 for k in range(s))    # finished set k had last_games[k]+1 games
        g = sum(sb["games"]); f += (g + 1) // 2
        if sb.get("tiebreak"):
            f += sum(sb["pts"]) // 6
        out.append(f)
    return out


def near_team(points, a_near_at_start: bool = True) -> list[str]:
    """'A' or 'B' on the near (camera) side for each point."""
    return [("A" if (f % 2 == 0) == a_near_at_start else "B") for f in end_flips(points)]


def serve_parity(p) -> int:
    """which partner of the serving team serves this game: 0 for the team's 1st, 3rd, .. service game in the set, 1
    for its 2nd, 4th, ..  (teams alternate games, so the team's k-th service game is game 2k or 2k+1 of the set)."""
    g = sum(p["score_before"]["games"])
    return (g // 2) % 2


def receive_side(p) -> int:
    """0 = deuce court, 1 = ad court (from the label's side, else from the points score)."""
    s = p.get("side")
    if s in ("deuce", "ad"):
        return 0 if s == "deuce" else 1
    return sum(p["score_before"]["pts"]) % 2


def serve_roles(serving_xy, receiving_xy, mid_x: float = 5.485, net_y: float = 11.885):
    """Position-only roles at the serve contact.
    serving_xy / receiving_xy: (2, 2) court metres of the two people on each side (order arbitrary).
    Returns (server_i, receiver_i, conf_server, conf_receiver): server = the serving-side person further from the net; receiver = the
    receiving-side person on the diagonal half (the opposite x-half to the server); confidences in [0, 1]."""
    import numpy as np
    s = np.asarray(serving_xy, float); r = np.asarray(receiving_xy, float)
    depth = np.abs(s[:, 1] - net_y); si = int(np.argmax(depth)); c_s = min(1.0, abs(depth[0] - depth[1]) / 3.0)
    sx = np.sign(s[si, 0] - mid_x) or 1.0
    opp = -(r[:, 0] - mid_x) * sx                     # > 0: on the diagonal half
    if (opp > 0).sum() == 1:
        ri = int(np.argmax(opp)); c_r = min(1.0, float(abs(r[ri, 0] - mid_x)) / 1.5)
    else:                                             # both on one half: take the one further back
        ri = int(np.argmax(np.abs(r[:, 1] - net_y))); c_r = 0.3
    return si, ri, float(c_s), float(c_r)


def side_roles(serving_xy, receiving_xy, mid_x: float = 5.485, net_y: float = 11.885, deep: float = 9.0, at_net: float = 7.0):
    """Roles at the serve contact when a side may show only one person (the camera often misses a near player).
    Returns (srv, rcv): per side a list (one entry per person) of (role, conf) with role 'main' (server / receiver),
    'partner' or None. Two people on a side: relative rule (`serve_roles`). One person: absolute depth from the net
    (>= `deep` m: baseline -> main; <= `at_net` m -> partner; in between -> unknown)."""
    import numpy as np
    s = [np.asarray(q, float) for q in serving_xy]; r = [np.asarray(q, float) for q in receiving_xy]
    srv = [(None, 0.0)] * len(s); rcv = [(None, 0.0)] * len(r); sx = None
    if len(s) == 2 and len(r) == 2:
        si, ri, cs, cr = serve_roles(s, r, mid_x, net_y)
        srv = [("main", cs) if i == si else ("partner", cs) for i in range(2)]
        rcv = [("main", cr) if i == ri else ("partner", cr) for i in range(2)]
        return srv, rcv
    if len(s) == 2:
        depth = [abs(q[1] - net_y) for q in s]; si = int(np.argmax(depth)); cs = min(1.0, abs(depth[0] - depth[1]) / 3.0)
        srv = [("main", cs) if i == si else ("partner", cs) for i in range(2)]; sx = s[si][0]
    elif len(s) == 1:
        d = abs(s[0][1] - net_y)
        srv = [("main", 1.0)] if d >= deep else [("partner", 1.0)] if d <= at_net else [(None, 0.0)]
        sx = s[0][0] if d >= deep else None
    if len(r) == 2:
        depth = [abs(q[1] - net_y) for q in r]
        if sx is not None:
            opp = [-(q[0] - mid_x) * (np.sign(sx - mid_x) or 1.0) for q in r]
            if (np.array(opp) > 0).sum() == 1:
                ri = int(np.argmax(opp)); cr = min(1.0, abs(r[ri][0] - mid_x) / 1.5)
                return srv, [("main", cr) if i == ri else ("partner", cr) for i in range(2)]
        ri = int(np.argmax(depth)); cr = min(1.0, abs(depth[0] - depth[1]) / 3.0)
        rcv = [("main", cr) if i == ri else ("partner", cr) for i in range(2)]
    elif len(r) == 1:
        d = abs(r[0][1] - net_y)
        rcv = [("main", 1.0)] if d >= deep else [("partner", 1.0)] if d <= at_net else [(None, 0.0)]
    return srv, rcv
