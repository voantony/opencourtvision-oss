"""Hard deductions for the point winner (S6, engineering_notes section 2.9). Each rule abstains unless its evidence is
clear; when one fires it overrides the soft S1-S5 fusion. Rules work in near/far terms from the rally structure
(pw_rally) and map to teams with the ends mapping; the game-state rule uses the confirmed score before the point.

Kinds: +1 = near team, -1 = far team. Teams 'A'/'B'.
"""
import numpy as np

NET_Y, COURT_W, COURT_L = 11.885, 10.97, 23.77


def other(t):
    return "B" if t == "A" else "A"


def team_of(kind, nearA):
    return "A" if kind * nearA > 0 else "B"


def side_of(team, nearA):
    return 1 if (team == "A") == (nearA > 0) else -1


# ------------------------------------------------------------------ serve phase
def rule_serve(r, server, nearA):
    """r = pw_rally.rally(...) dict. Ace / unreturned serve: exactly one shot, on the server's side -> server wins.
    Double fault: two shots >= 3 s apart, both on the server's side, nothing from the receivers -> receivers win."""
    if not r:
        return None, None
    sk = side_of(server, nearA)
    sh = [q for q in r["shots"] if q[2] != 0]
    if not sh or any(q[2] != sk for q in sh):
        return None, None
    if len(sh) == 1 and r["n_shots"] == 1:
        return server, "ace_unreturned"
    if len(sh) == 2 and sh[1][0] - sh[0][0] >= 3.0:
        return other(server), "double_fault"
    return None, None


# ------------------------------------------------------------------ ending
def rule_net(r, d, view, rows_fn):
    """Ball never crosses after the last shot (seen >= 0.4 s in the next 1.5 s, always on the hitter's side) -> the
    hitter lost (net). Returns (+1/-1 winner side, name) in near/far terms."""
    if not r or r["last_kind"] == 0:
        return None, None
    t, y, _, _ = rows_fn(d, view)
    k = r["last_kind"]; t0 = r["t_last"]
    w = (t > t0 + 0.15) & (t <= t0 + 1.5)
    if w.sum() < 12:
        return None, None
    past = (-k * y[w]) > 0.12                                 # far side for a near hitter (y < -0.12) and vice versa
    if past.mean() < 0.05:
        return -k, "net"
    return None, None


def rule_bounce(r, bounces, view, ok_views=True, out_m=0.4, in_m=0.6):
    """First ball-track bounce 0.2-2.5 s after the last shot on the receiving half: clearly out -> hitter lost; clearly
    in and no further shot -> hitter won. Needs a trustworthy court fit."""
    if not r or r["last_kind"] == 0 or not view.ok or not ok_views:
        return None, None
    k = r["last_kind"]; t0 = r["t_last"]
    bo = [b for b in bounces if t0 + 0.2 < b["t"] <= t0 + 2.5]
    for b in bo:
        x, yc = view.i2c([b["x"], b["y"]])[0]
        recv = (yc > NET_Y) if k > 0 else (yc < NET_Y)
        if not recv:
            continue
        dx = min(x, COURT_W - x); dy = (COURT_L - yc) if k > 0 else yc  # distance inside the lines (negative = out)
        m = min(dx, dy)
        if m < -out_m:
            return -k, "bounce_out"
        if m > in_m:
            return k, "bounce_in"
        return None, None
    return None, None


# ------------------------------------------------------------------ game state from the next serve
def next_state(score, winner, server):
    """score: dict(pts=[A,B], games=[A,B], sets=[A,B]) before the point (advantage scoring, no tiebreak handling beyond
    6-6). Returns (game_over, next_server, points_in_game_after, total_games_after_in_set, set_over)."""
    pa, pb = score["pts"]; ga, gb = score["games"]
    pa, pb = (pa + 1, pb) if winner == "A" else (pa, pb + 1)
    game_over = (pa >= 4 or pb >= 4) and abs(pa - pb) >= 2
    if not game_over:
        return False, server, pa + pb, ga + gb, False
    ga, gb = (ga + 1, gb) if pa > pb else (ga, gb + 1)
    set_over = (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2
    return True, other(server), 0, ga + gb, set_over


def rule_next_serve(score, server, flips_before, obs_side, obs_box, nearA0=1.0, obs_change=None):
    """Which outcome explains the next point's serve? obs_side: +1 near / -1 far server (None if unseen); obs_box:
    'deuce'/'ad'/None. obs_change: True if the break before the next serve is a changeover (> ~75 s; points <= 65 s on
    17 Oct), False if short, None unknown. flips_before = end changes before this point. Fires only if exactly one
    outcome matches."""
    if obs_side is None and obs_change is None:
        return None, None
    ok = []
    for w in ("A", "B"):
        go, nsrv, pig, gtot, _ = next_state(score, w, server)
        fl = flips_before + (1 if go and gtot % 2 == 1 else 0)
        nA = nearA0 * (-1) ** fl
        side = side_of(nsrv, nA)
        box = "deuce" if pig % 2 == 0 else "ad"
        change = go and gtot % 2 == 1
        if (obs_side is None or side == obs_side) and (obs_box is None or box == obs_box) and (obs_change is None or change == obs_change):
            ok.append(w)
    if len(ok) == 1:
        return ok[0], "next_serve"
    return None, None


def serve_obs(r, d, view, scale=6.0):
    """(side, box) of the serve that opens rally r: side = kind of its first shot; box from the server's foot x."""
    if not r or r["first_kind"] == 0:
        return None, None
    side = r["first_kind"]; t = r["t_first"]
    tk = float(d["t0"]) + d["pose_k"] / float(d["fps"]); j = int(np.argmin(np.abs(tk - t)))
    if abs(tk[j] - t) > 0.3:
        return side, None
    best = None
    for b in d["boxes"][j]:
        if not np.isfinite(b[0]) or b[4] < 0.3:
            continue
        x0, y0, x1, y1 = b[:4] * scale
        cx, cy = view.i2c([((x0 + x1) / 2, y1)])[0]
        behind = cy < 1.0 if side > 0 else cy > COURT_L - 1.0     # the server stands at/behind the baseline
        if behind and -1.5 <= cx <= COURT_W + 1.5:
            dist = abs(cy) if side > 0 else abs(cy - COURT_L)
            if best is None or dist < best[0]:
                best = (dist, cx)
    if best is None:
        return side, None
    right = best[1] > COURT_W / 2
    deuce = right if side > 0 else not right                  # near server faces away from the camera, far server faces it
    return side, ("deuce" if deuce else "ad")


# ------------------------------------------------------------------ reactions (EU build: pose)
def rule_pose(f, thr=0.5):
    v = f.get("l3_arms_pose")
    if v is None or abs(v) < thr:
        return None, None
    return (1 if v > 0 else -1), "arms_up"


# ------------------------------------------------------------------ rules-first p(A)
# Hard rules (Antony's order on issue 1): ace/unreturned serve, double fault, net, bounce clearly in or out,
# and what the next serve implies. Each of those functions already abstains unless the evidence is clear.
# Pose and other behaviour are soft. Audio contributes hit timing only: a team side, never samples, and nothing
# is written. The decoder then applies score constraints, callouts and an optional final score.
# This module does not read a label file and does not set p(A) from the winner it is being scored against.

RULE_P = 0.92          # a fired rule, not 1: leaves room for the decoder and for calibration
SOFT_SHRINK = 0.5
ABSTAIN_P = 0.5
ABSTAIN_BAND = 0.08
# Rules the ask queue must not send back for re-labelling. They already fired.
RESOLVED_RULES = frozenset({"ace_unreturned", "double_fault", "net", "bounce_in", "bounce_out", "next_serve"})


def _team_from_kind(kind, near_is_A):
    """kind +1 is the near side. near_is_A maps that side to team A."""
    if kind in ("A", "B"):
        return kind
    if kind in (1, -1):
        near = kind > 0
        return "A" if near == bool(near_is_A) else "B"
    return None


def behaviour_cue(pose_out, near_is_A=True):
    """Turn a pose rule (near/far) into a soft cue. None when the pose rule abstained."""
    if not pose_out or pose_out[0] is None:
        return None
    team = _team_from_kind(pose_out[0], near_is_A)
    if team not in ("A", "B"):
        return None
    return (0.7 if team == "A" else 0.3, 0.5)


def _audio_cue(audio_side):
    """Hit-timing side only. Not a path, not samples, and this function does not write."""
    if audio_side is None:
        return None
    if audio_side not in ("A", "B"):
        raise TypeError("audio_side must be 'A' or 'B' from in-memory hit timing, not audio samples or a file")
    return (0.65 if audio_side == "A" else 0.35, 0.5)


def p_a_from_cues(serve=None, net=None, bounce=None, next_serve=None, near_is_A=True,
                  ball=None, behaviour=None, audio_side=None):
    """Fuse rule outputs into one p(A).

    Hard rules, each (side, name) or None when that rule abstained:
      serve — ace_unreturned or double_fault, already a team
      net, bounce — near/far (+1/-1), bounce only when clearly in or clearly out
      next_serve — the team implied by the next serve
    Soft, used only when every hard rule abstained or they contradict:
      ball, behaviour — (p_a, weight)
      audio_side — 'A' or 'B' from hit timing. Samples are not accepted and nothing is saved.
    Contradicting hard rules abstain. Returns dict(p_a, abstain, rules, tier).
    """
    fired = []
    for item in (serve, net, bounce, next_serve):
        if not item or item[0] is None:
            continue
        team = _team_from_kind(item[0], near_is_A)
        if team in ("A", "B"):
            fired.append((team, item[1]))
    teams = {t for t, _ in fired}
    names = [n for _, n in fired]
    if len(teams) == 1:
        team = next(iter(teams))
        return {"p_a": RULE_P if team == "A" else 1.0 - RULE_P, "abstain": False, "rules": names, "tier": "rule"}
    cues = []
    for item in (ball, behaviour, _audio_cue(audio_side)):
        if not item:
            continue
        p, w = float(item[0]), float(item[1])
        if w > 0 and 0.0 <= p <= 1.0:
            cues.append((p, w))
    if len(teams) > 1 or not cues:
        return {"p_a": ABSTAIN_P, "abstain": True, "rules": names, "tier": "conflict" if teams else "abstain"}
    raw = sum(p * w for p, w in cues) / sum(w for _, w in cues)
    p = ABSTAIN_P + (raw - ABSTAIN_P) * SOFT_SHRINK
    return {"p_a": float(p), "abstain": abs(p - ABSTAIN_P) < ABSTAIN_BAND, "rules": names, "tier": "soft"}


def decode_points(p_a, sport="tennis", preset="advantage", first_server="A", calls=None, final=None):
    """Score decoder over p(A). Abstentions leave the path to the rules of tennis.

    `calls` aligns with p_a. Each entry is an Obs (a heard score call) or None.
    `final` is an optional sets tuple, e.g. (2, 0), passed through as a soft-or-hard constraint.
    """
    from opencourtvision.scoring.decoder import PointEvidence, decode
    from opencourtvision.scoring.formats import get_format
    calls = list(calls or [])
    ev = []
    for i, p in enumerate(p_a):
        ev.append(PointEvidence(p_a=float(p), p_point=1.0, call=calls[i] if i < len(calls) else None))
    return decode(ev, get_format(sport, preset), first_server=first_server, final=final)
