"""Highlight ranking: point_quality, personal reel, coaching reel (highlight_algorithm.md v3.2, Coach O1-O16).

Ported from the spec's reference script (rallycut/work/hl_sanity.py: score4 / personal4 / interleave). Three outputs per match:
  match reel     top points by PQ (point_quality), won or lost;
  personal reel  the user's reel: won points x1.2, great lost points x1.0 and always included, ordinary lost points
                 x loss_weight (UserConfig, default 0.5) and at most 1; missed set/match points go to the coaching reel;
  coaching reel  top points by coaching_value (missed set/match points, unforced errors, double faults, serve +1).

Formulas (§3.3, §3.4; weights in `Weights`):
  M       = pressure_multiplier(): min(1.75, max(BP 1.3, SP 1.45, MP 1.6, TB game point 1.2) x 1.15 if 30-30/deuce)  [O1, O7]
  M_eff   = 1 + kappa_score*(M - 1)
  G_rally = max(0.75*v_rally^2, 0.60*v_net_exchange) + 0.05*v_rally_duration            [O3; v = within-match percentile]
  G_out   = winner 0.45 if <= 3 shots (return winners incl.), 0.70 from 4 shots or high difficulty; ace/service winner 0.40  [O2, O11]
  G_react = 0.30*p(react_db) if the reaction is present (react_db >= 8 dB), else 0         [MD C1c]
  content = sum of the largest 3 of {G_rally, G_out*kappa_score, G_diff, G_react};  PQ = M_eff*content
  great lost point = lost by the user's team and in the top 5% of the match by max(G_rally, G_diff) (pressure-free, ties at
                     the cut included); at least 1 and at most 3 per match                  [O8, O15]
O5 (MD): an audio-only reaction raises PQ and lets the point clip extend (to onset + 2.5 s), but it is a standalone reel beat
only once pose confirms it (MatchSettings.pose_confirmed_reactions).
The user's team is a per-match setting (MatchSettings.user_team, default SwingVision "host"; O10 the app confirms it).
All lists are PROVISIONAL until tuned on the user's own picks (§7).
"""
from dataclasses import asdict, dataclass

from ..scoring.formats import PRESETS, game_point_state, game_pressure, get_format, pressure_multiplier
from .config import MatchSettings, UserConfig

_LEVEL = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


@dataclass(frozen=True)
class Weights:
    kappa_score: float = 0.5          # SV/T3 outcome and score
    kappa_rally: float = 1.0          # rally count stands in for our T2 audio count
    kappa_audio: float = 1.0
    rally: float = 0.75               # 0.75*v^2 (O3)
    net_exchange: float = 0.60
    rally_duration: float = 0.05
    winner_short: float = 0.45        # <= 3 shots (O11)
    winner_long: float = 0.70         # >= long_from shots or high difficulty
    long_from: int = 4
    ace: float = 0.40
    react: float = 0.30
    react_present_db: float = 8.0
    react_present_dur_s: float = 0.5  # only checked when the point carries react_dur
    team_bias: float = 1.2            # won points in the personal reel
    great_top: float = 0.05           # O8
    great_min: int = 1
    great_max: int = 3
    max_ordinary_lost: int = 1        # O4 / O13
    clip: tuple = (("shot_quality", 0.26), ("rally_intensity", 0.20), ("reaction", 0.12))
    shot_quality_max: float = 0.75
    coach_missed_big: float = 0.80
    coach_serve_plus_one: float = 0.40
    coach_unforced_error: float = 0.60
    coach_double_fault: float = 0.60


DEFAULT_WEIGHTS = Weights()


@dataclass
class PointInput:
    """One point, sport-neutral. Teams are 'host' / 'guest' (the two sides of the score)."""
    point: int
    server: str
    winner: str
    host_pts: int = 0                 # points won in the game before this point (0, 1, 2, 3, 4 = AD ...)
    guest_pts: int = 0
    games_before: tuple = (0, 0)      # (host, guest) games in the current set before this game
    sets_before: tuple = (0, 0)       # (host, guest) sets won before this set
    rally_shots: int = 0              # in-play contacts from the last serve (0 for a double fault)
    rally_span_s: float = 0.0
    fast_exchanges: int = 0           # contact gaps < 0.8 s
    net_exchange: int = 0             # volleys in the longest alternating net exchange
    ending: str = None                # winner | ace | service_winner | unforced_error | double_fault | None
    detail: str = ""
    react_db: float = None
    react_dur: float = None
    high_difficulty: bool = False     # passing shot on the run / T1 reflex volley (O11 override); T0 on SV data
    game: int = None
    set: int = None
    t_last: float = None
    pressure_flags: dict = None       # score-free input (audio pipeline): {match_point, set_point, break_point, deuce, tb_game_point,
                                      # user_set_point, user_match_point} instead of the game score above

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("games_before", "sets_before"):
            if k in d: d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)


def score_level(s):
    """'0'/'15'/'30'/'40'/'AD' -> points won (0..4)."""
    return _LEVEL[str(s).strip().upper()]


def ending_from_detail(detail):
    d = (detail or "").strip()
    if d == "Ace": return "ace"
    if d == "Service Winner": return "service_winner"
    if d == "Double Fault": return "double_fault"
    if "Unforced" in d: return "unforced_error"
    if d.endswith("Winner"): return "winner"
    return None


def percentiles(vals, ndigits=3):
    """Mid-rank within-match percentile p(x) = (#{y < x} + 0.5*#{y = x}) / n (§3.1), quantised to `ndigits` so ties are stable."""
    srt = sorted(vals); n = len(vals)
    out = []
    for v in vals:
        lo = sum(1 for x in srt if x < v); eq = sum(1 for x in srt if x == v)
        out.append(round((lo + 0.5 * eq) / n, ndigits) if ndigits is not None else (lo + 0.5 * eq) / n)
    return out


def _deuce_rule(settings):
    if settings.deuce_rule:
        for f in PRESETS[settings.sport.lower()].values():
            if f.deuce_rule == settings.deuce_rule:
                return f, settings.deuce_rule
        raise ValueError(f"no {settings.sport} format with deuce rule {settings.deuce_rule!r}")
    f = get_format(settings.sport)
    return f, f.deuce_rule


def pressure(p, fmt, deuce_rule, user_team="host"):
    """Pressure state before point p (§3.3). -> dict(M, game_state, set_point, match_point, user_set_point, user_match_point,
    game_point_for) using scoring.formats game_pressure() / pressure_multiplier(). A point carrying pressure_flags (no game
    score) uses those flags instead."""
    if p.pressure_flags is not None:
        f = p.pressure_flags
        gstate = "break_point" if f.get("break_point") else "deuce" if (f.get("deuce") or f.get("thirty_all")) else None
        nested = {k: bool(f.get(k)) for k in ("match_point", "set_point", "tb_game_point", "pickleball_game_point")}
        return dict(M=pressure_multiplier(nested, gstate), game_state=gstate, set_point=nested["set_point"], match_point=nested["match_point"],
                    break_point=gstate == "break_point", user_set_point=bool(f.get("user_set_point")),
                    user_match_point=bool(f.get("user_match_point")), game_point_for=None)
    if p.pressure_flags is None and p.server not in ("host", "guest"):
        raise ValueError(f"point {p.point}: server must be 'host' or 'guest' (or pass pressure_flags)")
    s, r = (p.host_pts, p.guest_pts) if p.server == "host" else (p.guest_pts, p.host_pts)
    receiver = "guest" if p.server == "host" else "host"
    gstate = game_pressure(s, r, deuce_rule, winner=None if p.winner is None else ("S" if p.winner == p.server else "R"))
    gpf = game_point_state(s, r, deuce_rule)["game_point_for"]
    at_gp = {"server": {p.server}, "receiver": {receiver}, "both": {"host", "guest"}}.get(gpf, set())
    games = dict(zip(("host", "guest"), p.games_before)); sets = dict(zip(("host", "guest"), p.sets_before))

    def sp(t):
        me, op = games[t], games["guest" if t == "host" else "host"]
        return t in at_gp and ((me >= fmt.games_per_set - 1 and me > op) or me == op == fmt.tiebreak_at)

    def mp(t):
        return sp(t) and sets[t] == fmt.sets_to_win - 1

    flags = dict(set_point=sp("host") or sp("guest"), match_point=mp("host") or mp("guest"))
    return dict(M=pressure_multiplier(flags, gstate), game_state=gstate, set_point=flags["set_point"], match_point=flags["match_point"],
                break_point=gstate == "break_point", user_set_point=sp(user_team), user_match_point=mp(user_team), game_point_for=gpf)


def score_points(points, settings=None, weights=DEFAULT_WEIGHTS):
    """PointInput list (chronological) -> one dict per point with every term (explainable; §10 `explain`)."""
    st = settings or MatchSettings(); w = weights; ut = st.user_team
    fmt, deuce_rule = _deuce_rule(st)
    pts = [p if isinstance(p, PointInput) else PointInput.from_dict(p) for p in points]
    n = len(pts)
    if n == 0: return []
    v_rally = percentiles([p.rally_shots for p in pts]); v_dur = percentiles([p.rally_span_s for p in pts])
    v_fast = percentiles([p.fast_exchanges for p in pts])
    nref = max(4, sorted(p.net_exchange for p in pts)[min(n - 1, int(0.95 * n))])
    rv = sorted(p.react_db for p in pts if p.react_db is not None)
    rint = percentiles([a + 0.5 * b for a, b in zip(v_rally, v_fast)], None)
    out = []
    for i, p in enumerate(pts):
        pr = pressure(p, fmt, deuce_rule, ut)
        M = pr["M"]; M_eff = 1 + w.kappa_score * (M - 1)
        v_netx = round(min(1.0, p.net_exchange / nref), 3)
        g_rally_raw = max(w.rally * v_rally[i] ** 2, w.net_exchange * v_netx) + w.rally_duration * v_dur[i]
        w_win = w.winner_long if (p.rally_shots >= w.long_from or p.high_difficulty) else w.winner_short
        g_out_raw = w_win if p.ending == "winner" else w.ace if p.ending in ("ace", "service_winner") else 0.0
        present = (p.react_db is not None and p.react_db >= w.react_present_db
                   and (p.react_dur is None or p.react_dur >= w.react_present_dur_s))
        v_react = 0.0
        if p.react_db is not None:
            v_react = round((sum(x < p.react_db for x in rv) + 0.5 * sum(x == p.react_db for x in rv)) / len(rv), 3)
        g_react = w.react * v_react * w.kappa_audio if present else 0.0
        g_diff_raw = 0.0                                     # difficulty group: T0 until track/players + pose (§9 #1)
        g_rally, g_out, g_diff = g_rally_raw * w.kappa_rally, g_out_raw * w.kappa_score, g_diff_raw * w.kappa_score
        content = sum(sorted([g_rally, g_out, g_diff, g_react], reverse=True)[:3])
        pq = M_eff * content
        lost = p.winner is not None and p.winner != ut
        cw = dict(w.clip)
        cs = M_eff * (cw["shot_quality"] * g_out / w.shot_quality_max + cw["rally_intensity"] * rint[i] * w.kappa_rally
                      + cw["reaction"] * (v_react if present else 0.0) * w.kappa_audio) / sum(cw.values())
        missed_big = lost and (pr["user_set_point"] or pr["user_match_point"])
        s1 = p.ending == "winner" and p.rally_shots == 3 and p.winner is not None and p.winner == p.server == ut
        tags = dict(missed_set_or_match_point=missed_big, serve_plus_one_winner=s1, unforced_error_end=p.ending == "unforced_error" and lost,
                    double_fault_end=p.ending == "double_fault" and p.server == ut)
        cv = M_eff * w.kappa_score * (w.coach_missed_big * tags["missed_set_or_match_point"] + w.coach_serve_plus_one * tags["serve_plus_one_winner"]
                                      + w.coach_unforced_error * tags["unforced_error_end"] + w.coach_double_fault * tags["double_fault_end"])
        out.append(dict(point=p.point, game=p.game, set=p.set, server=p.server, winner=p.winner, ending=p.ending, detail=p.detail,
                        rally_shots=p.rally_shots, lost=lost, v_rally=v_rally[i], v_dur=v_dur[i], v_fast=v_fast[i], v_netx=v_netx,
                        v_react=v_react if p.react_db is not None else None, react_db=p.react_db,
                        M=M, M_eff=M_eff, pressure=pr, g_rally=g_rally, g_out=g_out, g_diff=g_diff, g_react=g_react, w_win=w_win if p.ending == "winner" else 0.0,
                        content=content, pq=pq, rally_difficulty=max(g_rally_raw, g_diff_raw), clip_score=cs, coaching_value=cv,
                        coaching_tags=[k for k, v in tags.items() if v], missed_big=missed_big, serve_plus_one=s1,
                        reaction_present=present, reaction_extends_clip=present,               # O5: may extend the clip and raise rank
                        reaction_beat_ok=present and p.point in st.pose_confirmed_reactions))  # O5: standalone beat only with pose
    return out


def great_lost_points(scored, weights=DEFAULT_WEIGHTS):
    """O8 + O15: lost points in the top `great_top` of the match by max(G_rally, G_diff); ties at the cut included;
    at least great_min (best lost point by that measure), at most great_max."""
    if not scored: return []
    rd = sorted(scored, key=lambda s: (-s["rally_difficulty"], s["point"]))
    cut_n = max(1, round(weights.great_top * len(scored)))
    cut = rd[cut_n - 1]["rally_difficulty"]
    great = [s["point"] for s in rd if s["rally_difficulty"] >= cut and s["lost"]]
    if len(great) < weights.great_min:
        great = [s["point"] for s in rd if s["lost"]][:weights.great_min]
    return great[:weights.great_max]


def match_top(scored, n=10):
    return [s["point"] for s in sorted(scored, key=lambda s: (-s["pq"], s["point"]))[:n]]


def clip_score_top(scored, n=10):
    return [s["point"] for s in sorted(scored, key=lambda s: (-s["clip_score"], -s["pq"], s["point"]))[:n]]


def coaching_top(scored, n=10):
    return [s["point"] for s in sorted(scored, key=lambda s: (-s["coaching_value"], s["point"]))[:n]]


def interleave(seq, is_lost, chrono=None):
    """O8 + O14: keep chronology but defer a lost point that would follow another lost point until after the next won beat;
    the climax (last item) stays last. Where that is impossible, each leftover goes where it adds the fewest adjacent lost
    pairs, ties to the slot nearest its chronological place. chrono: sort key for chronological order (default identity)."""
    if not seq: return []
    key = chrono or (lambda x: x)
    build, climax = list(seq[:-1]), seq[-1]
    out, pend = [], []
    for x in build:
        if is_lost(x) and out and is_lost(out[-1]):
            pend.append(x); continue
        out.append(x)
        if not is_lost(x) and pend:
            out.append(pend.pop(0))

    def pairs(s):
        return sum(is_lost(a) and is_lost(b) for a, b in zip(s, s[1:]))

    for x in pend:
        place = sorted(out + [x], key=key).index(x)
        best = min(range(len(out) + 1), key=lambda k: (pairs(out[:k] + [x] + out[k:] + [climax]), abs(k - place)))
        out.insert(best, x)
    return out + [climax]


def personal_scores(scored, great, loss_weight, weights=DEFAULT_WEIGHTS):
    """§5.6 personal score per point: won x team_bias, great lost x1.0, ordinary lost x loss_weight;
    None = excluded (a missed set/match point that isn't great goes to the coaching reel only, O4)."""
    sc = {}
    for s in scored:
        if s["lost"] and s["missed_big"] and s["point"] not in great:
            sc[s["point"]] = None; continue
        f = weights.team_bias if not s["lost"] else (1.0 if s["point"] in great else loss_weight)
        sc[s["point"]] = s["pq"] * f
    return sc


def personal_reel(scored, loss_weight=None, n=10, weights=DEFAULT_WEIGHTS, great=None, user=None):
    """The user's personal reel. loss_weight: 0.0 (only wins) .. 1.0 (best points, win or lose); default from UserConfig.
    -> dict(selected [by personal score], order [reel order, climax last], climax, great, scores, loss_weight)."""
    if loss_weight is None:
        loss_weight = (user or UserConfig()).loss_weight
    loss_weight = UserConfig(loss_weight).loss_weight
    great = list(great_lost_points(scored, weights) if great is None else great)
    by = {s["point"]: s for s in scored}
    sc = personal_scores(scored, great, loss_weight, weights)
    order = sorted((s["point"] for s in scored if sc[s["point"]] is not None), key=lambda x: (-sc[x], x))
    sel = sorted(great, key=lambda x: (-sc[x], x))[:n]       # O8 + Antony: great lost points always in, never scaled
    ordinary = 0
    for x in order:
        if len(sel) >= n: break
        if x in sel: continue
        if by[x]["lost"]:                                     # ordinary lost point
            if loss_weight == 0.0 or ordinary >= weights.max_ordinary_lost: continue
            ordinary += 1
        sel.append(x)
    sel = sorted(sel, key=lambda x: (-sc[x], x))
    if not sel:
        return dict(selected=[], order=[], climax=None, great=great, scores=sc, loss_weight=loss_weight)
    cands = sel if loss_weight == 1.0 else [x for x in sel if not by[x]["lost"]]      # O9
    climax = max(cands or sel, key=lambda x: (sc[x], -x))
    seq = sorted(x for x in sel if x != climax) + [climax]
    seq = interleave(seq, lambda x: by[x]["lost"])
    return dict(selected=sel, order=seq, climax=climax, great=great, scores=sc, loss_weight=loss_weight)


def highlights(points, settings=None, user=None, n=10, weights=DEFAULT_WEIGHTS, loss_weight=None):
    """Everything for one match: scored points, match / personal / coaching top-n, great lost points, O5 reaction flags."""
    st = settings or MatchSettings(); user = user or UserConfig()
    scored = score_points(points, st, weights)
    great = great_lost_points(scored, weights)
    pr = personal_reel(scored, user.loss_weight if loss_weight is None else loss_weight, n, weights, great)
    return dict(user_team=st.user_team, user_team_confirmed=st.user_team_confirmed, provisional=True,
                points=scored, match_top=match_top(scored, n), personal=pr, coaching_top=coaching_top(scored, n),
                clip_score_top=clip_score_top(scored, n), great_lost=great,
                reaction_extends=[s["point"] for s in scored if s["reaction_extends_clip"]],
                reaction_beats=[s["point"] for s in scored if s["reaction_beat_ok"]])


# ---------------------------------------------------------------- SwingVision adapter

_INPLAY = {"first_serve", "second_serve", "first_return", "second_return", "serve_plus_one", "return_plus_one", "in_play"}


def points_from_swingvision(m, segments=None, lag_s=2.0):
    """SVMatch (importers.swingvision.load) -> [PointInput]. segments: rows of our segments.csv (read_segments_csv(full=True))
    for react_db, joined on the segment containing the point's last shot Video Time + lag_s (raw file clock)."""
    from ..importers.swingvision import NET_Y, SERVICE_FROM_NET
    byp = m.shots_by_point()
    team = m.team
    # games / sets before each game, from the Games sheet
    gctx, h, g, cur, sets = {}, 0, 0, None, {"host": 0, "guest": 0}
    for r in m.games:
        if cur is not None and r.get("set") != cur:
            sets["host" if h > g else "guest"] += 1; h = g = 0
        cur = r.get("set")
        gctx[r.get("game")] = ((h, g), (sets["host"], sets["guest"]))
        if r.get("game_winner") == "host": h += 1
        else: g += 1
    out = []
    for p in m.points:
        sh = sorted(byp.get(p["point"], []), key=lambda s: s["video_time"])
        serves = [i for i, s in enumerate(sh) if s.get("type") in ("first_serve", "second_serve")]
        det = p.get("detail") or ""
        rally = [s for s in sh[serves[-1]:] if s.get("type") in _INPLAY] if serves and det != "Double Fault" else []
        t = [s["video_time"] for s in rally]
        gaps = [b - a for a, b in zip(t, t[1:])]

        def netvol(s):
            return "Volley" in (s.get("stroke") or "") and s.get("hit_y") is not None and abs(s["hit_y"] - NET_Y) <= SERVICE_FROM_NET
        run = best = 0; prev = None
        for s in rally:                                      # volleys in the longest alternating net exchange
            run = run + 1 if netvol(s) and prev is not None and netvol(prev) and team(prev["player"]) != team(s["player"]) else 0
            if run: best = max(best, run + 1)
            prev = s
        react = None
        if segments and sh:
            tl = sh[-1]["video_time"] + lag_s
            seg = next((r for r in segments if float(r["start_s"]) <= tl <= float(r["end_s"])), None)
            if seg is not None and seg.get("react_db") not in (None, ""):
                react = float(seg["react_db"])
        (gb, sb) = gctx.get(p.get("game"), ((0, 0), (0, 0)))
        out.append(PointInput(point=p["point"], server=p.get("match_server"), winner=p.get("point_winner"),
                              host_pts=score_level(p.get("host_game_score")), guest_pts=score_level(p.get("guest_game_score")),
                              games_before=gb, sets_before=sb, rally_shots=len(rally), rally_span_s=round(t[-1] - t[0], 2) if len(t) > 1 else 0.0,
                              fast_exchanges=sum(1 for x in gaps if x < 0.8), net_exchange=best, ending=ending_from_detail(det), detail=det,
                              react_db=react, game=p.get("game"), set=p.get("set"),
                              t_last=round(sh[-1]["video_time"] + lag_s, 3) if sh else None))
    return out


def main(argv=None):
    """ocv highlights EXPORT.xlsx [--segments segments.csv] [--loss-weight 0.5] [--user-team host|guest] [--json out.json]"""
    import argparse
    import json

    from ..importers import swingvision as sv
    ap = argparse.ArgumentParser(prog="ocv highlights", description="Rank a match's points (highlight_algorithm.md v3.2). PROVISIONAL.")
    ap.add_argument("export", help="SwingVision export .xlsx")
    ap.add_argument("--segments", help="our segments.csv (react_db)")
    ap.add_argument("--loss-weight", type=float, help="0.0 only wins .. 1.0 win or lose (default: user config, else 0.5)")
    ap.add_argument("--user-config", help="user config JSON (default $OCV_USER_CONFIG or ~/.config/opencourtvision/user.json)")
    ap.add_argument("--user-team", choices=("host", "guest"), default="host", help="which score side is the user's team (default host)")
    ap.add_argument("--sport", default="tennis")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--json", help="write the full result here")
    A = ap.parse_args(argv)
    user = UserConfig.load(A.user_config)
    if A.loss_weight is not None: user = UserConfig(A.loss_weight)
    segs = sv.read_segments_csv(A.segments, full=True) if A.segments else None
    res = highlights(points_from_swingvision(sv.load(A.export), segs), MatchSettings(user_team=A.user_team, sport=A.sport), user, A.n)
    pr = res["personal"]
    print(json.dumps(dict(user_team=res["user_team"], loss_weight=pr["loss_weight"], match_top=res["match_top"], personal_top=pr["selected"],
                          personal_order=pr["order"], climax=pr["climax"], great_lost=res["great_lost"], coaching_top=res["coaching_top"],
                          provisional=True), indent=1))
    if A.json:
        with open(A.json, "w") as fh: json.dump(res, fh, indent=1, default=list)


if __name__ == "__main__":
    main()
