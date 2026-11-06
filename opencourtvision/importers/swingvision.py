"""SwingVision per-shot export (.xlsx) -> OpenCourtVision Events, plus a plausibility audit of the export.

Sheets used: Settings, Shots, Points, Games, Sets, Stats (the "Export Shots" workbook). Needs openpyxl
(`pip install opencourtvision[swingvision]`).

Coordinates (stated in the Settings sheet and confirmed on a real export): metres, X positive to the camera's right with
0 on the centre line, Y from the near baseline (0) towards the far baseline (23.77), Z up. `to_court()` converts to the
OpenCourtVision court frame used by `labels.schema.court_position` (x from the near-left doubles sideline).

Clocks: "Video Time" counts seconds from the SwingVision session start (Settings "Start Time"). On the one export
checked so far (one 86-min doubles match), a shot row's Video Time sits SV_SHOT_LAG_S = 2.0 s BEFORE the contact heard
in the raw recording, and every Points row's Video Time equals one of its serve rows + exactly 2.0 s, so
contact = Video Time + lag (+ any clock offset between the session and the file, 0 when the camera file starts with
the session). Treat the lag as measured, not guaranteed; `to_events(lag_s=...)` overrides it.

Every Event is marked source="swingvision", verified=False: SwingVision output is a reference to compare against,
not ground truth (see `audit()` for why). Player names are replaced by roles H1/H2 (host team) and G1/G2 (guest team)
from the Settings team strings; `players` in the returned meta maps roles back to names for local use only.
"""
import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..events import Event
from ..labels.schema import ShotLabel, court_position, validate

SOURCE = "swingvision"
SV_SHOT_LAG_S = 2.0

# court geometry (m), SwingVision frame: x centred, y from the near baseline
COURT_L, NET_Y = 23.77, 23.77 / 2
HALF_W_DOUBLES, HALF_W_SINGLES = 10.97 / 2, 8.23 / 2
SERVICE_FROM_NET = 6.40
RUNOFF_BACK, RUNOFF_SIDE = 6.40, 3.66        # ITF recommended minimum run-off; beyond this a hit/bounce is implausible

_DIRECTION = {"down the T": "T", "out wide": "wide", "body": "body", "cross court": "cross_court",
              "down the line": "down_the_line", "inside out": "inside_out", "inside in": "inside_in"}
SERVE_TYPES = ("first_serve", "second_serve")
_DETAIL_WIN = ("Winner", "Ace", "Service Winner")

_COLS = {"Speed (MPH)": "speed_mph", "Bounce (x)": "bounce_x", "Bounce (y)": "bounce_y", "Hit (x)": "hit_x", "Hit (y)": "hit_y",
         "Hit (z)": "hit_z"}


def _key(h):
    return _COLS.get(h) or str(h).strip().lower().replace(" ", "_")


def _rows(ws):
    it = ws.iter_rows(values_only=True)
    try: head = [_key(h) if h is not None else None for h in next(it)]
    except StopIteration: return []
    out = []
    for r in it:
        if r is None or all(v is None for v in r): continue
        out.append({h: v for h, v in zip(head, r) if h is not None})
    return out


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def _hms(t):
    if t is None or not math.isfinite(t): return "?"
    t = max(0.0, t); return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


@dataclass
class SVMatch:
    settings: dict
    shots: list
    points: list
    games: list = field(default_factory=list)
    sets: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)            # stat name -> {"Host Set 1": v, ...}
    roles: dict = field(default_factory=dict)            # player name -> H1/H2/G1/G2

    def team(self, player):
        r = self.roles.get(player, ""); return "host" if r.startswith("H") else "guest" if r.startswith("G") else None

    def shots_by_point(self):
        d = defaultdict(list)
        for s in self.shots: d[s["point"]].append(s)
        return d


def load(path):
    """Read a SwingVision export workbook -> SVMatch (rows as dicts with snake_case keys)."""
    try:
        import openpyxl
    except ImportError as e:   # pragma: no cover
        raise SystemExit("reading SwingVision exports needs openpyxl: pip install 'opencourtvision[swingvision]'") from e
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = lambda n: _rows(wb[n]) if n in wb.sheetnames else []
    settings = (sheet("Settings") or [{}])[0]
    shots = sheet("Shots")
    for s in shots:
        for k in ("speed_mph", "bounce_x", "bounce_y", "hit_x", "hit_y", "hit_z", "video_time"): s[k] = _f(s.get(k))
        for k in ("shot", "point", "game", "set"): s[k] = int(s[k]) if _f(s.get(k)) is not None else None
    points = sheet("Points")
    for p in points:
        for k in ("video_time", "duration"): p[k] = _f(p.get(k))
        for k in ("point", "game", "set"): p[k] = int(p[k]) if _f(p.get(k)) is not None else None
    stats = {str(r.get("stat_name")): {k: v for k, v in r.items() if k != "stat_name"} for r in sheet("Stats")}
    roles = {}
    for team, tag in (("host_team", "H"), ("guest_team", "G")):
        names = [n.strip() for n in str(settings.get(team) or "").split("/") if n.strip()]
        for i, n in enumerate(names): roles[n] = f"{tag}{i + 1}"
    for s in shots:                                     # players not named in Settings get U1, U2, ...
        if s.get("player") not in roles: roles[s.get("player")] = f"U{sum(r.startswith('U') for r in roles.values()) + 1}"
    return SVMatch(settings, shots, points, sheet("Games"), sheet("Sets"), stats, roles)


def to_court(x, y):
    """SwingVision (x centred, y from near baseline) -> OpenCourtVision court frame (x from the near-left doubles sideline)."""
    return x + HALF_W_DOUBLES, y


# ---------------------------------------------------------------------------------------------------------------- events

def shot_label(s, last=False, won=None):
    """One Shots row -> ShotLabel (fields SwingVision does not provide stay None)."""
    stroke, typ = str(s.get("stroke") or ""), str(s.get("type") or "")
    side = "forehand" if stroke.startswith("Forehand") else "backhand" if stroke.startswith("Backhand") else None
    if stroke == "Serve" or typ in SERVE_TYPES: shot_type = "second_serve" if typ == "second_serve" else "serve"
    elif stroke.endswith("Volley"): shot_type = "volley"
    elif stroke == "Overhead": shot_type = "smash"
    elif typ in ("first_return", "second_return"): shot_type = "return"
    elif stroke in ("Forehand", "Backhand"): shot_type = "swing"
    else: shot_type = None                                   # "Feed" and unknown strokes
    direction = _DIRECTION.get(str(s.get("direction") or "").strip())
    res = s.get("result")
    outcome = None
    if res in ("Net", "Out"): outcome = "error" if last or shot_type not in ("serve", "second_serve") else None
    elif res == "In": outcome = "win" if last and won else "in"
    position = None
    if s.get("hit_x") is not None and s.get("hit_y") is not None:
        position = court_position(*to_court(s["hit_x"], s["hit_y"]))
    return ShotLabel(position=position, side=side, shot_type=shot_type, direction=direction, outcome=outcome)


def to_events(m, lag_s=SV_SHOT_LAG_S, offset_s=0.0, keep_names=False):
    """SVMatch -> (events, meta). One "contact" Event per Shots row (t = Video Time + lag + offset), plus "point_start"
    (serve contact) and "point_end" (Video Time + Duration) per Points row with the outcome in label/attrs."""
    who = (lambda n: n) if keep_names else (lambda n: m.roles.get(n, n))
    pts = {p["point"]: p for p in m.points}
    by_pt = m.shots_by_point(); ev = []
    for pno, shots in by_pt.items():
        p = pts.get(pno, {})
        for i, s in enumerate(shots):
            last = i == len(shots) - 1
            won = m.team(s.get("player")) == p.get("point_winner") if p.get("point_winner") else None
            attrs = {k: s.get(k) for k in ("point", "game", "set", "shot", "type", "stroke", "spin", "speed_mph", "result",
                                           "bounce_x", "bounce_y", "bounce_side", "hit_x", "hit_y", "hit_z", "hit_side") if s.get(k) is not None}
            attrs["video_time"] = s.get("video_time")
            ev.append(Event("contact", s["video_time"] + lag_s + offset_s, who(s.get("player")), shot_label(s, last, won),
                            confidence=0.0, evidence=[f"{SOURCE}: {s.get('type')} {s.get('stroke')} {s.get('result')}"],
                            verified=False, source=SOURCE, attrs=attrs))
    for p in m.points:
        if p.get("video_time") is None: continue
        t0 = p["video_time"] - SV_SHOT_LAG_S + lag_s + offset_s
        detail = p.get("detail") or None
        outcome = None if not detail else "win" if any(detail.endswith(w) for w in _DETAIL_WIN) else "error"
        attrs = {k: p.get(k) for k in ("point", "game", "set", "serve_state", "match_server", "host_game_score", "guest_game_score",
                                       "point_winner", "detail", "break_point", "set_point", "duration")}
        shots = by_pt.get(p["point"], [])
        last_p = who(shots[-1].get("player")) if shots else None
        ev.append(Event("point_start", t0, None, None, verified=False, source=SOURCE, attrs=attrs, evidence=[f"{SOURCE}: point {p['point']}"]))
        ev.append(Event("point_end", t0 + (p.get("duration") or 0.0), last_p, ShotLabel(outcome=outcome) if outcome else None,
                        verified=False, source=SOURCE, attrs=attrs, evidence=[f"{SOURCE}: {detail or 'no detail'}"]))
    ev.sort(key=lambda e: (e.t, e.type))
    meta = dict(source=SOURCE, lag_s=lag_s, offset_s=offset_s, session_start=str(m.settings.get("start_time")),
                players={v: k for k, v in m.roles.items()} if keep_names else sorted(m.roles.values()))
    return ev, meta


# ----------------------------------------------------------------------------------------------------------------- audit

def _ex(s, m, extra=""):
    t = s.get("video_time")
    return (f"P{s.get('point')} shot {s.get('shot')} {m.roles.get(s.get('player'), '?')} {s.get('type')}/{s.get('stroke')} "
            f"{(s.get('speed_mph') or 0):.1f} mph {s.get('result')} @VT {t:.2f} ({_hms(t)}){(' ' + extra) if extra else ''}")


def _pex(p, extra=""):
    t = p.get("video_time")
    return f"P{p.get('point')} (G{p.get('game')} S{p.get('set')}) {p.get('detail') or 'no detail'} @VT {t:.2f} ({_hms(t)}){(' ' + extra) if extra else ''}"


def _inside(x, y, tol=0.0):
    return abs(x) <= HALF_W_DOUBLES + tol and -tol <= y <= COURT_L + tol


def _service_box_ok(s, tol):
    """Serve bounce inside the diagonal service box (singles width). Server at hit_x>0 on the near end serves to x<0, etc."""
    x, y, hx, hy = s["bounce_x"], s["bounce_y"], s["hit_x"], s["hit_y"]
    far_box = hy < NET_Y
    in_y = (NET_Y - tol <= y <= NET_Y + SERVICE_FROM_NET + tol) if far_box else (NET_Y - SERVICE_FROM_NET - tol <= y <= NET_Y + tol)
    in_x = abs(x) <= HALF_W_SINGLES + tol and (x * hx <= tol * abs(hx) or abs(x) <= tol)   # opposite half of the server
    return in_y and in_x


def audit(m, serve_max=90.0, rally_max=80.0, tol=0.10, short_postroll=1.5, lag_s=SV_SHOT_LAG_S, segments=None, examples=6):
    """Plausibility audit of an SVMatch. -> report dict {"summary": {...}, "checks": [{"id", "title", "count", "of", "examples"}]}.
    segments: optional list of (start_s, end_s) of our own point segments on the raw clock, for the point-ending comparison."""
    checks = []

    def add(cid, title, items, of=None, note=None, **extra):
        checks.append(dict(id=cid, title=title, count=len(items), of=of, note=note, examples=items[:examples], **extra))

    S, P = m.shots, m.points
    by_pt = m.shots_by_point(); pts = {p["point"]: p for p in P}
    serves = [s for s in S if s.get("type") in SERVE_TYPES]

    # --- coordinate frame check (where do serves get hit from?)
    near = [s["hit_y"] for s in serves if s["hit_y"] is not None and s["hit_y"] < NET_Y]
    far = [s["hit_y"] for s in serves if s["hit_y"] is not None and s["hit_y"] >= NET_Y]
    frame = dict(serve_hit_y_median_near=round(statistics.median(near), 2) if near else None,
                 serve_hit_y_median_far=round(statistics.median(far), 2) if far else None,
                 bounce_x_median=round(statistics.median([s["bounce_x"] for s in S if s["bounce_x"] is not None]), 2) if S else None,
                 hit_x_range=[round(min(s["hit_x"] for s in S), 2), round(max(s["hit_x"] for s in S), 2)] if S else None)

    # --- speeds
    add("serve_speed", f"Serve speed > {serve_max:.0f} mph", [_ex(s, m) for s in sorted(serves, key=lambda s: -(s["speed_mph"] or 0))
                                                             if (s["speed_mph"] or 0) > serve_max], of=len(serves))
    rally = [s for s in S if s.get("type") not in SERVE_TYPES]
    add("rally_speed", f"Non-serve speed > {rally_max:.0f} mph", [_ex(s, m) for s in rally if (s["speed_mph"] or 0) > rally_max], of=len(rally))
    cnt = Counter(round(s["speed_mph"], 6) for s in S if s["speed_mph"] is not None)
    rep = sorted(((v, c) for v, c in cnt.items() if c >= 3), key=lambda x: (-x[1], -x[0]))
    vmax = max(cnt) if cnt else None
    add("speed_repeats", "Exactly repeated speed values (>= 3 identical)", [f"{v:.6f} mph x{c}" for v, c in rep], of=len(cnt),
        note=(f"max speed {vmax:.6f} appears {cnt[vmax]}x: looks like a cap/default" if vmax and cnt[vmax] > 1 else None))
    step = math.sqrt(3) / 2
    lat = [s for s in S if s["speed_mph"] and abs(s["speed_mph"] / step - round(s["speed_mph"] / step)) < 1e-4]
    checks.append(dict(id="speed_grid", title="Speeds that are exact multiples of sqrt(3)/2 mph (quantised fallback estimate?)",
                       count=len(lat), of=len(S), note=None,
                       examples=[f"{k} x{c}" for k, c in Counter(s.get("stroke") for s in lat).most_common()]))
    sec_fast = []
    for pno, shots in by_pt.items():
        firsts = [s for s in shots if s.get("type") == "first_serve" and s["speed_mph"]]
        for s in shots:
            if s.get("type") == "second_serve" and s["speed_mph"] and firsts:
                f = [x for x in firsts if x["video_time"] < s["video_time"]]
                if f and s["speed_mph"] > f[-1]["speed_mph"]:
                    sec_fast.append(_ex(s, m, f"> first serve {f[-1]['speed_mph']:.1f} mph"))
    add("second_faster", "Second serve faster than the first serve of the same point", sec_fast,
        of=sum(1 for s in S if s.get("type") == "second_serve"))

    # --- coordinates
    b_in_out = [_ex(s, m, f"bounce ({s['bounce_x']:.2f}, {s['bounce_y']:.2f})") for s in S
                if s.get("result") == "In" and not _inside(s["bounce_x"], s["bounce_y"], tol)]
    add("in_but_outside", f"Result In but bounce outside the doubles court (> {tol:.2f} m)", b_in_out, of=sum(s.get("result") == "In" for s in S))
    o_inside = [_ex(s, m, f"bounce ({s['bounce_x']:.2f}, {s['bounce_y']:.2f})") for s in rally if s.get("result") == "Out"
                and _inside(s["bounce_x"], s["bounce_y"], -tol) and (s["bounce_side"] != s["hit_side"])]
    add("out_but_inside", f"Non-serve Result Out but bounce inside the doubles court (> {tol:.2f} m inside)", o_inside,
        of=sum(s.get("result") == "Out" for s in rally))
    s_out_in = [_ex(s, m, f"bounce ({s['bounce_x']:.2f}, {s['bounce_y']:.2f}) from ({s['hit_x']:.2f}, {s['hit_y']:.2f})") for s in serves
                if s.get("result") == "Out" and _service_box_ok(s, -tol)]
    add("serve_out_but_in_box", f"Serve Result Out but bounce inside the diagonal service box (> {tol:.2f} m inside)", s_out_in,
        of=sum(s.get("result") == "Out" for s in serves))
    srv_box = [_ex(s, m, f"bounce ({s['bounce_x']:.2f}, {s['bounce_y']:.2f}) from ({s['hit_x']:.2f}, {s['hit_y']:.2f})")
               for s in serves if s.get("result") == "In" and not _service_box_ok(s, tol)]
    add("serve_box", "Serve marked In but bounce outside the diagonal service box", srv_box, of=sum(s.get("result") == "In" for s in serves))
    own = [_ex(s, m) for s in S if s.get("result") == "In" and s.get("bounce_side") == s.get("hit_side")]
    add("in_own_side", "Result In but the ball bounced on the hitter's own side", own, of=sum(s.get("result") == "In" for s in S))
    far_b = [_ex(s, m, f"bounce ({s['bounce_x']:.2f}, {s['bounce_y']:.2f})") for s in S
             if abs(s["bounce_x"]) > HALF_W_DOUBLES + RUNOFF_SIDE or not (-RUNOFF_BACK <= s["bounce_y"] <= COURT_L + RUNOFF_BACK)]
    add("bounce_beyond_runoff", "Bounce beyond the court run-off (6.40 m back / 3.66 m side)", far_b, of=len(S))
    far_h = [_ex(s, m, f"hit ({s['hit_x']:.2f}, {s['hit_y']:.2f}, z {s['hit_z']:.2f})") for s in S
             if abs(s["hit_x"]) > HALF_W_DOUBLES + RUNOFF_SIDE or not (-RUNOFF_BACK <= s["hit_y"] <= COURT_L + RUNOFF_BACK)]
    add("hit_beyond_runoff", "Hit position beyond the court run-off", far_h, of=len(S))
    hz = [_ex(s, m, f"z {s['hit_z']:.2f} m") for s in S if s["hit_z"] is not None and (s["hit_z"] < 0 or
          (s["hit_z"] > 3.5 and s.get("stroke") not in ("Serve", "Overhead")))]
    add("hit_height", "Hit height below the ground or > 3.5 m on a groundstroke/volley", hz, of=len(S))
    side_y = [_ex(s, m, f"hit_y {s['hit_y']:.2f} but Hit Side {s['hit_side']}") for s in S
              if (s["hit_y"] < NET_Y) != (s.get("hit_side") == "near")]
    add("hit_side_mismatch", "Hit Side disagrees with Hit (y)", side_y, of=len(S))

    # --- schema / sequence rules
    lab_err = []
    for s in S:
        for msg in validate(shot_label(s)): lab_err.append(_ex(s, m, msg))
    add("schema_rules", "Label fails OpenCourtVision schema rules (serve direction etc.)", lab_err, of=len(S))
    pre = [_ex(s, m) for s in S if s.get("type") == "none"]
    add("type_none", "Shots with Type 'none' (feeds/returned faults/between-point hits, assigned to a point)", pre, of=len(S))
    feed = [_ex(s, m) for s in S if s.get("stroke") == "Feed"]
    add("feed_stroke", "Stroke 'Feed' in a match", feed, of=len(S))
    first_not_serve, no_serve, multi, sec_wo_fault, first_in_then_second, ret_order = [], [], [], [], [], []
    team_both_ends, srv_team = [], []
    for pno, shots in sorted(by_pt.items()):
        p = pts.get(pno, {})
        play = [s for s in shots if s.get("type") != "none"]
        sv = [s for s in shots if s.get("type") in SERVE_TYPES]
        if not sv: no_serve.append(_pex(p or {"point": pno, "video_time": shots[0]["video_time"]}, f"{len(shots)} shot(s), no serve row"))
        elif play and play[0].get("type") not in SERVE_TYPES: first_not_serve.append(_ex(play[0], m, "first in-play shot is not a serve"))
        c = Counter(s.get("type") for s in sv)
        if c["first_serve"] > 1 or c["second_serve"] > 1: multi.append(_pex(p, f"{c['first_serve']} first / {c['second_serve']} second serve rows"))
        for i, s in enumerate(shots):
            if s.get("type") == "second_serve":
                f = [x for x in shots[:i] if x.get("type") == "first_serve"]
                if not f: sec_wo_fault.append(_ex(s, m, "no first serve before it"))
                elif f[-1].get("result") == "In": first_in_then_second.append(_ex(s, m, "first serve was In"))
            if s.get("type") in ("first_return", "second_return"):
                prev = [x for x in shots[:i] if x.get("type") != "none"]
                if not prev or prev[-1].get("type") not in SERVE_TYPES: ret_order.append(_ex(s, m, "return not right after a serve"))
        sides = defaultdict(set)
        for s in play: sides[m.team(s.get("player"))].add(s.get("hit_side"))
        if any(len(v) > 1 for v in sides.values()): team_both_ends.append(_pex(p, "a team hits from both ends"))
        if sv and p.get("match_server") and any(m.team(s.get("player")) != p["match_server"] for s in sv):
            srv_team.append(_pex(p, f"server row by {', '.join(sorted({m.roles.get(s.get('player'), '?') for s in sv}))}, "
                                    f"Points says {p['match_server']} serves"))
    add("no_serve", "Points with no serve row", no_serve, of=len(by_pt))
    add("first_not_serve", "Points whose first in-play shot is not a serve", first_not_serve, of=len(by_pt))
    add("multi_serve", "Points with more than one first or second serve row", multi, of=len(by_pt))
    add("second_without_fault", "Second serve without a first serve before it", sec_wo_fault)
    add("second_after_first_in", "Second serve although the first serve was In", first_in_then_second)
    add("return_order", "Return not directly after a serve", ret_order)
    add("team_both_ends", "A team hits from both ends of the court within one point", team_both_ends, of=len(by_pt))
    add("server_team", "Serve row player's team differs from Points 'Match Server'", srv_team, of=len(by_pt))
    game_srv = defaultdict(Counter)
    for s in serves: game_srv[s.get("game")][m.roles.get(s.get("player"), "?")] += 1
    multi_srv = [f"G{g}: {dict(c)}" for g, c in sorted(game_srv.items()) if len(c) > 1]
    add("server_changes_in_game", "Games with more than one serving player (one player serves a whole game)", multi_srv, of=len(game_srv))

    # --- points vs detail
    nodet, df_bad, ue_bad, ace_bad, winner_bad, hand_bad, state_bad = [], [], [], [], [], [], []
    for p in P:
        shots = by_pt.get(p["point"], []); d = p.get("detail") or ""
        if not d: nodet.append(_pex(p, f"winner {p.get('point_winner')}")); continue
        play = [s for s in shots if s.get("type") != "none"]
        last = play[-1] if play else None
        fs = [s for s in shots if s.get("type") == "first_serve"]; ss = [s for s in shots if s.get("type") == "second_serve"]
        if d == "Double Fault":
            ok = fs and ss and fs[-1].get("result") != "In" and ss[-1].get("result") != "In"
            if not ok: df_bad.append(_pex(p, f"first {[s.get('result') for s in fs]} second {[s.get('result') for s in ss]}"))
            if p.get("point_winner") == p.get("match_server"): df_bad.append(_pex(p, "double fault won by the server"))
        elif d in ("Ace", "Service Winner"):
            srv = (ss or fs)[-1] if (ss or fs) else None
            after = [s for s in play if srv and s["video_time"] > srv["video_time"]]
            bad = srv is None or srv.get("result") != "In" or (d == "Ace" and after) or any(s.get("result") == "In" for s in after[:1])
            if bad or p.get("point_winner") != p.get("match_server"):
                ace_bad.append(_pex(p, f"serve {srv.get('result') if srv else None}, then {[s.get('result') for s in after][:3]}"))
        elif d.endswith("Winner") and last is not None:
            if last.get("result") != "In" or m.team(last.get("player")) != p.get("point_winner"):
                winner_bad.append(_pex(p, f"last shot {_ex(last, m)} by {m.team(last.get('player'))}, winner {p.get('point_winner')}"))
        elif d.endswith("Unforced Error") and last is not None:
            if last.get("result") == "In" or m.team(last.get("player")) == p.get("point_winner"):
                ue_bad.append(_pex(p, f"last shot {_ex(last, m)} by {m.team(last.get('player'))}, winner {p.get('point_winner')}"))
        if last is not None and d.split()[0] in ("Forehand", "Backhand") and not str(last.get("stroke")).startswith(d.split()[0]):
            hand_bad.append(_pex(p, f"last shot stroke {last.get('stroke')}"))
        if p.get("serve_state") == "second" and not ss and d != "Double Fault": state_bad.append(_pex(p, "Serve State second, no second serve row"))
        if p.get("serve_state") == "first" and ss: state_bad.append(_pex(p, "Serve State first, but a second serve row"))
    add("no_detail", "Points without a Detail (how the point ended)", nodet, of=len(P))
    add("double_fault", "'Double Fault' not backed by two serve faults", df_bad)
    add("ace_sw", "'Ace'/'Service Winner' inconsistent with the shot rows", ace_bad)
    add("winner", "'Winner' whose last shot is not In, or not by the point winner", winner_bad)
    add("unforced_error", "'Unforced Error' whose last shot is In, or by the point winner", ue_bad)
    add("detail_stroke", "Detail says Forehand/Backhand but the last shot's stroke differs", hand_bad)
    add("serve_state", "Serve State disagrees with the serve rows", state_bad, of=len(P))
    lw = []
    for p in P:
        play = [s for s in by_pt.get(p["point"], []) if s.get("type") != "none"]
        if not play or not p.get("point_winner") or p.get("detail") in ("Double Fault", "Ace", "Service Winner"): continue
        last = play[-1]; t = m.team(last.get("player"))
        exp = t if last.get("result") == "In" else ({"host": "guest", "guest": "host"}.get(t))
        if exp and exp != p["point_winner"]: lw.append(_pex(p, f"last shot {last.get('result')} by {t}, but {p['point_winner']} won"))
    add("winner_vs_last_shot", "Point winner disagrees with the last shot (In -> hitter wins, Net/Out -> hitter loses; "
        "double faults, aces and service winners excluded)", lw, of=sum(p.get("detail") not in ("Double Fault", "Ace", "Service Winner") for p in P))

    # --- score replay vs the Points sheet, totals vs Stats
    checks.append(_score_replay(m, examples))
    checks.append(_stats_check(m, by_pt, examples))

    # --- point endings (post-roll after the last shot)
    pe = point_endings(m, lag_s=lag_s, short=short_postroll, segments=segments)
    checks.append(dict(id="short_postroll", title=f"Points ending < {short_postroll} s after the last contact", count=pe["n_short"],
                       of=pe["n"], note=pe["note"], examples=pe["worst"][:max(examples, 10)], stats=pe))
    summary = dict(shots=len(S), points=len(P), games=len(m.games), sets=len(m.sets), frame=frame,
                   settings_points=m.settings.get("points"), roles=sorted(m.roles.values()))
    return dict(summary=summary, checks=checks)


_ADV = ["0", "15", "30", "40"]


def _score_replay(m, examples):
    """Replay game scores from Point Winner (ad scoring) and compare with Host/Guest Game Score before each point."""
    bad = []; h = g = 0; game = None; games_won = Counter(); gw_bad = []
    last_game_winner = {}
    for p in m.points:
        if p.get("game") != game:
            if game is not None and (h or g): bad.append(f"G{game} ended unfinished at {h}-{g} points")
            game, h, g = p.get("game"), 0, 0
        def lab(a, b):
            if a >= 3 and b >= 3: return ("40", "40") if a == b else (("AD", "40") if a > b else ("40", "AD"))
            return _ADV[min(a, 3)], _ADV[min(b, 3)]
        exp = lab(h, g); got = (str(p.get("host_game_score")), str(p.get("guest_game_score")))
        if exp != got: bad.append(_pex(p, f"sheet says {got[0]}-{got[1]}, replay gives {exp[0]}-{exp[1]}"))
        if p.get("point_winner") == "host": h += 1
        elif p.get("point_winner") == "guest": g += 1
        if (h >= 4 or g >= 4) and abs(h - g) >= 2:
            w = "host" if h > g else "guest"; games_won[w] += 1; last_game_winner[game] = w; h = g = 0
    for gm in m.games:
        if last_game_winner.get(gm.get("game")) not in (None, gm.get("game_winner")):
            gw_bad.append(f"G{gm.get('game')}: Games sheet {gm.get('game_winner')}, replay {last_game_winner.get(gm.get('game'))}")
    items = bad + gw_bad
    return dict(id="score_replay", title="Game scores / game winners that do not follow from the point winners", count=len(items),
                of=len(m.points), examples=items[:examples], note=f"replayed games won: host {games_won['host']}, guest {games_won['guest']}")


def _stats_check(m, by_pt, examples):
    """Recompute Stats rows per set and team from the Points and Shots sheets and list disagreements."""
    calc = defaultdict(Counter)                  # (set, team) -> stat -> value
    for p in m.points:
        st, srv = p.get("set"), p.get("match_server"); rcv = {"host": "guest", "guest": "host"}.get(srv)
        if srv is None: continue
        d = p.get("detail") or ""; win = p.get("point_winner")
        c = calc[(st, srv)]
        c["1st Serves"] += 1
        if p.get("serve_state") == "first": c["1st Serves In"] += 1; c["1st Serves Won"] += win == srv
        else:
            c["2nd Serves"] += 1
            if d != "Double Fault": c["2nd Serves In"] += 1
            c["2nd Serves Won"] += win == srv
        r = calc[(st, rcv)]
        if p.get("serve_state") == "first": r["1st Returns"] += 1; r["1st Returns Won"] += win == rcv
        else: r["2nd Returns"] += 1; r["2nd Returns Won"] += win == rcv
        for t in ("host", "guest"): calc[(st, t)]["Total Points"] += 1
        calc[(st, win)]["Total Points Won"] += 1
        if d == "Ace": calc[(st, win)]["Aces"] += 1
        if d == "Service Winner": calc[(st, win)]["Service Winners"] += 1
        if d in ("Forehand Winner", "Backhand Winner"): calc[(st, win)][f"{d}s"] += 1
        if d.endswith("Unforced Error"): calc[(st, {"host": "guest", "guest": "host"}[win])][f"{d}s"] += 1
    # shots-derived first serves in, for cross-checking the Points-derived numbers
    shot_calc = defaultdict(Counter)
    for p in m.points:
        sv = [s for s in by_pt.get(p["point"], []) if s.get("type") == "first_serve"]
        if sv and p.get("match_server"):
            shot_calc[(p.get("set"), p["match_server"])]["1st Serves In"] += sv[-1].get("result") == "In"
    diffs = []
    for name, row in m.stats.items():
        for col, v in row.items():
            if v is None or not str(col).startswith(("host_set_", "guest_set_")): continue
            team, st = col.split("_set_")[0], int(col.split("_set_")[1])
            if (st, team) not in calc or name not in calc[(st, team)] and name not in ("Aces", "Service Winners", "Forehand Winners",
                                                                                      "Backhand Winners", "Forehand Unforced Errors",
                                                                                      "Backhand Unforced Errors"): continue
            mine = calc[(st, team)][name]
            if int(v) != mine: diffs.append(f"Set {st} {team} '{name}': Stats {int(v)}, from Points {mine}")
    sdiff = []
    for (st, team), c in sorted(shot_calc.items()):
        a = calc[(st, team)]["1st Serves In"]
        if a != c["1st Serves In"]: sdiff.append(f"Set {st} {team} '1st Serves In': from Points {a}, from Shots (last first-serve row In) {c['1st Serves In']}")
    items = diffs + sdiff
    return dict(id="stats_consistency", title="Stats sheet vs numbers recomputed from Points (and Shots)", count=len(items),
                examples=items[:max(examples, 12)], of=None, n_stats_vs_points=len(diffs), n_points_vs_shots=len(sdiff))


def point_endings(m, lag_s=SV_SHOT_LAG_S, short=1.5, segments=None, worst=10):
    """Post-roll after each point's last contact: (Video Time + Duration) - (last shot Video Time + lag).
    With segments [(start_s, end_s)] on the raw clock, also how far our own segment end is from SwingVision's."""
    by_pt = m.shots_by_point(); rows = []
    for p in m.points:
        shots = by_pt.get(p["point"], [])
        if not shots or p.get("video_time") is None or p.get("duration") is None: continue
        end_raw = p["video_time"] - SV_SHOT_LAG_S + lag_s + p["duration"]        # Points Video Time is already contact-aligned
        last_c = max(s["video_time"] for s in shots) + lag_s
        r = dict(point=p["point"], end=end_raw, last_contact=last_c, post=end_raw - last_c, naive=p["video_time"] + p["duration"] - (last_c - lag_s))
        if segments:
            seg = min(segments, key=lambda ab: (0 if ab[0] - 1 <= last_c <= ab[1] + 1 else 1, abs((ab[0] + ab[1]) / 2 - last_c)))
            if seg[0] - 1 <= last_c <= seg[1] + 1: r["ours_end"] = seg[1]; r["ours_post"] = seg[1] - last_c
        rows.append(r)
    post = sorted(r["post"] for r in rows)
    q = lambda f: round(post[min(len(post) - 1, int(f * (len(post) - 1) + 0.5))], 2) if post else None
    hist = Counter()
    for v in post:
        hist["<0.5" if v < 0.5 else "0.5-1.0" if v < 1 else "1.0-1.5" if v < 1.5 else "1.5-2.0" if v < 2 else "2.0-3.0" if v < 3 else ">=3.0"] += 1
    bad = sorted([r for r in rows if r["post"] < short], key=lambda r: r["post"])
    ex = [f"P{r['point']}: last contact {r['last_contact']:.2f} s ({_hms(r['last_contact'])}), SV end {r['end']:.2f} s, post-roll {r['post']:.2f} s"
          + (f", ours ends {r['ours_end']:.2f} s (+{r['ours_post']:.2f} s)" if "ours_end" in r else "") for r in bad[:worst]]
    ours = [r["ours_post"] for r in rows if "ours_post" in r]
    out = dict(n=len(rows), n_short=len(bad), short_s=short, lag_s=lag_s, min=q(0), p10=q(0.1), median=q(0.5), p90=q(0.9), max=q(1),
               mean=round(sum(post) / len(post), 2) if post else None, histogram={k: hist[k] for k in
               ("<0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">=3.0")},
               naive_min=round(min(r["naive"] for r in rows), 2) if rows else None,
               naive_median=round(statistics.median(r["naive"] for r in rows), 2) if rows else None, worst=ex,
               ours_matched=len(ours), ours_post_median=round(statistics.median(ours), 2) if ours else None,
               ours_post_min=round(min(ours), 2) if ours else None,
               ours_later=sum(1 for r in rows if "ours_end" in r and r["ours_end"] > r["end"]))
    out["note"] = (f"post-roll = SV point end - (last shot Video Time + {lag_s:g} s lag); without the lag correction the median would read "
                   f"{out['naive_median']} s. Distribution (s): {out['histogram']}")
    return out


def format_markdown(rep, title="SwingVision export audit"):
    s = rep["summary"]
    L = [f"# {title}", "", f"{s['shots']} shots, {s['points']} points, {s['games']} games, {s['sets']} sets "
         f"(Settings says {s['settings_points']} points). Players are shown as roles {', '.join(s['roles'])}.", "",
         f"Coordinate frame check: serve hit y median near {s['frame']['serve_hit_y_median_near']} m / far "
         f"{s['frame']['serve_hit_y_median_far']} m (baselines at 0 and 23.77), bounce x median {s['frame']['bounce_x_median']} m "
         f"(centre line at 0), hit x range {s['frame']['hit_x_range']}.", "",
         "Every row here is a reference, not ground truth. Times: VT = SwingVision Video Time (s from session start).", "",
         "| check | count | of |", "|---|---:|---:|"]
    for c in rep["checks"]:
        L.append(f"| {c['title']} | {c['count']} | {c['of'] if c['of'] is not None else ''} |")
    for c in rep["checks"]:
        L += ["", f"## {c['title']}: {c['count']}" + (f" / {c['of']}" if c["of"] is not None else "")]
        if c.get("note"): L.append(f"_{c['note']}_")
        L += [f"- {e}" for e in c["examples"]]
    pe = next((c["stats"] for c in rep["checks"] if c["id"] == "short_postroll"), None)
    if pe:
        L += ["", "### Post-roll distribution", f"min {pe['min']} / p10 {pe['p10']} / median {pe['median']} / p90 {pe['p90']} / max {pe['max']} s "
              f"(mean {pe['mean']}); n = {pe['n']}"]
        if pe["ours_matched"]:
            L.append(f"Our segments matched {pe['ours_matched']} of these points; our post-roll median {pe['ours_post_median']} s, min "
                     f"{pe['ours_post_min']} s; ours ends later than SwingVision on {pe['ours_later']}.")
    return "\n".join(L) + "\n"


def read_segments_csv(path, full=False):
    """Our `<name> - segments.csv` -> [(start_s, end_s)], or the full rows as dicts (numbers converted) with full=True."""
    import csv
    rows = list(csv.DictReader(open(path)))
    if not full: return [(float(r["start_s"]), float(r["end_s"])) for r in rows]
    for r in rows:
        for k, v in r.items():
            f = _f(v)
            if f is not None: r[k] = f
    return rows


def _nearest(sorted_vals, t):
    import bisect
    i = bisect.bisect_left(sorted_vals, t)
    c = [sorted_vals[j] for j in (i - 1, i) if 0 <= j < len(sorted_vals)]
    return min(c, key=lambda v: abs(v - t)) if c else None


def estimate_offset(m, hits, lag_s=SV_SHOT_LAG_S, search=(-60.0, 60.0), step=0.01, tol=0.15):
    """Clock offset (s) that best lines up SwingVision contacts (Video Time + lag) with our audio hit times.
    -> dict(offset, matched, of, baseline) where baseline is the mean match count at offsets 10-60 s away (chance level)."""
    hs = sorted(hits); sv = [s["video_time"] + lag_s for s in m.shots if s.get("video_time") is not None]
    def resid(o):
        return [abs(n - t - o) for t in sv if (n := _nearest(hs, t + o)) is not None and abs(n - t - o) <= tol]
    count = lambda o: len(resid(o))
    n = int(round((search[1] - search[0]) / step)); grid = [search[0] + i * step for i in range(n + 1)]
    coarse = max(grid[::10], key=count)
    best = max((g for g in grid if abs(g - coarse) <= 0.15), key=lambda o: (len(r := resid(o)), -sum(r)))   # centre of the plateau
    far = [g for g in grid[::50] if 10 <= abs(g - best) <= 60]
    return dict(offset=round(best, 2), matched=count(best), of=len(sv), tol=tol,
                baseline=round(sum(map(count, far)) / len(far), 1) if far else None)


def compare_points(m, segments, hits=None, score_rows=None, lag_s=SV_SHOT_LAG_S, offset_s=0.0, pad=1.0):
    """Our point segments (rows of segments.csv on the raw clock) vs SwingVision points.
    A SwingVision point [first shot contact - pad, last contact + pad] is matched to every segment overlapping it.
    -> dict with matched / split / missed points, our extra segments, timing deltas, contact agreement and score agreement."""
    by_pt = m.shots_by_point(); sv = []
    for p in m.points:
        shots = by_pt.get(p["point"], [])
        if not shots or p.get("video_time") is None: continue
        serve_t = p["video_time"] - SV_SHOT_LAG_S + lag_s + offset_s          # the serve that started the played point
        play = [s["video_time"] + lag_s + offset_s for s in shots if s.get("type") != "none"] or [serve_t]
        sv.append(dict(point=p["point"], game=p.get("game"), serve=serve_t, first=min(play), last=max(play),
                       end=serve_t + (p.get("duration") or 0.0)))
    segs = sorted(segments, key=lambda r: r["start_s"])
    hit_of = lambda r: r.get("first_hit_s", r["start_s"])
    used, matched, split, missed, d_start, d_end, d_first = set(), 0, [], [], [], [], []
    for q in sv:
        ov = [i for i, r in enumerate(segs) if r["start_s"] < q["last"] + pad and r["end_s"] > q["first"] - pad]
        if not ov: missed.append(q); continue
        matched += 1; used.update(ov)
        if len(ov) > 1: split.append((q["point"], len(ov)))
        home = [i for i in ov if segs[i]["start_s"] <= q["serve"] <= segs[i]["end_s"]] or ov
        r = segs[home[0]]
        d_first.append(hit_of(r) - q["serve"])
        d_start.append(r["start_s"] - q["serve"])
        d_end.append(segs[ov[-1]]["end_s"] - q["end"])
    extra = [r for i, r in enumerate(segs) if i not in used]
    fh = sorted(hit_of(r) for r in segs)
    srv_rows = [s["video_time"] + lag_s + offset_s for s in m.shots if s.get("type") in SERVE_TYPES]
    d_srv = [abs(_nearest(fh, t) - t) for t in srv_rows] if fh else []
    st = lambda v: dict(n=len(v), median=round(statistics.median(v), 2), mean=round(sum(v) / len(v), 2),
                        p10=round(sorted(v)[int(0.1 * (len(v) - 1))], 2), p90=round(sorted(v)[int(0.9 * (len(v) - 1))], 2),
                        within_0_5s=sum(abs(x) <= 0.5 for x in v), within_1s=sum(abs(x) <= 1.0 for x in v)) if v else None
    out = dict(sv_points=len(sv), our_segments=len(segs), matched=matched, split=len(split), split_examples=split[:10],
               missed=[dict(point=q["point"], serve=round(q["serve"], 2), at=_hms(q["serve"])) for q in missed],
               extra=len(extra), extra_fault_like=sum(1 for r in extra if r.get("fault_like") in (1, 1.0, "1", True)),
               extra_examples=[dict(idx=int(r.get("idx", 0)), start=round(r["start_s"], 2), at=_hms(r["start_s"]), hits=r.get("hits"))
                               for r in extra[:10]],
               first_hit_minus_sv_serve=st(d_first),
               serve_rows_vs_nearest_first_hit=dict(n=len(d_srv), within_150ms=sum(d <= 0.15 for d in d_srv), within_500ms=sum(d <= 0.5 for d in d_srv),
                                                    median_abs_s=round(statistics.median(d_srv), 2) if d_srv else None), start_minus_sv_serve=st(d_start), end_minus_sv_end=st(d_end))
    # changeovers: our changeover_after flag vs a SwingVision game change right after that segment
    co = [r for r in segs if r.get("changeover_after") in (1, 1.0, "1")]
    games_after = []
    for r in co:
        nxt = next((q for q in sv if q["serve"] > r["end_s"]), None); prev = [q for q in sv if q["serve"] <= r["end_s"]]
        games_after.append(bool(nxt and prev and nxt["game"] != prev[-1]["game"]))
    out["changeovers"] = dict(ours=len(co), at_sv_game_change=sum(games_after))
    if hits is not None:
        hs = sorted(hits); sc = [s["video_time"] + lag_s + offset_s for s in m.shots]
        dt = [(_nearest(hs, t) - t) for t in sc]
        out["contacts"] = dict(sv_shots=len(sc), within_80ms=sum(abs(d) <= 0.08 for d in dt), within_150ms=sum(abs(d) <= 0.15 for d in dt),
                               within_300ms=sum(abs(d) <= 0.3 for d in dt),
                               median_abs_ms_within_150=round(1000 * statistics.median([abs(d) for d in dt if abs(d) <= 0.15]), 1)
                               if any(abs(d) <= 0.15 for d in dt) else None)
        spans = [(q["first"] - 0.5, q["last"] + 0.5) for q in sv]
        in_pts = [h for h in hs if any(a <= h <= b for a, b in spans)]
        out["contacts"]["our_hits_inside_sv_points"] = len(in_pts)
        out["contacts"]["our_hits_inside_with_sv_shot_150ms"] = sum(1 for h in in_pts if abs(_nearest(sorted(sc), h) - h) <= 0.15)
    if score_rows is not None:
        known = [r for r in score_rows if str(r.get("score_before", "?")).strip() not in ("", "?")]
        out["score"] = dict(our_rows=len(score_rows), our_rows_with_score=len(known),
                            note="no comparable scores: our timeline has no recognised score callouts" if not known else None)
    return out


def format_comparison(c):
    L = ["", "# Our point detection vs SwingVision", "",
         f"SwingVision points {c['sv_points']}, our segments {c['our_segments']}. Matched {c['matched']} SwingVision points to at least one of "
         f"our segments ({c['split']} split over several), missed {len(c['missed'])}; {c['extra']} of our segments match no SwingVision point "
         f"({c['extra_fault_like']} of them flagged fault_like)."]
    for k, t in (("first_hit_minus_sv_serve", "our first hit - SV serve contact"), ("start_minus_sv_serve", "our segment start - SV serve contact"),
                 ("end_minus_sv_end", "our segment end - SV point end")):
        v = c.get(k)
        if v: L.append(f"- {t}: median {v['median']} s, p10 {v['p10']}, p90 {v['p90']} (|d| <= 0.5 s: {v['within_0_5s']}/{v['n']}, <= 1 s: {v['within_1s']}/{v['n']})")
    v = c["serve_rows_vs_nearest_first_hit"]
    L.append(f"- every SwingVision serve row (incl. faults) vs the nearest first hit of any of our segments: {v['within_150ms']}/{v['n']} within 150 ms, "
             f"{v['within_500ms']}/{v['n']} within 0.5 s (median |d| {v['median_abs_s']} s)")
    if c["missed"]: L.append("- missed: " + ", ".join(f"P{q['point']} @{q['at']}" for q in c["missed"][:15]))
    if c["extra_examples"]: L.append("- extra (first 10): " + ", ".join(f"#{e['idx']} @{e['at']} ({e['hits']:.0f} hits)" for e in c["extra_examples"]))
    L.append(f"- changeovers: we flagged {c['changeovers']['ours']}, {c['changeovers']['at_sv_game_change']} of them fall at a SwingVision game change")
    if "contacts" in c:
        k = c["contacts"]
        L.append(f"- contacts: of {k['sv_shots']} SwingVision shots, {k['within_80ms']} have one of our audio hits within 80 ms, {k['within_150ms']} within "
                 f"150 ms, {k['within_300ms']} within 300 ms (median |d| {k['median_abs_ms_within_150']} ms); of our {k['our_hits_inside_sv_points']} hits inside "
                 f"SwingVision points, {k['our_hits_inside_with_sv_shot_150ms']} have a SwingVision shot within 150 ms")
    if "score" in c:
        L.append(f"- score: {c['score']['our_rows_with_score']} of our {c['score']['our_rows']} timeline rows carry a score"
                 + (f" ({c['score']['note']})" if c["score"]["note"] else ""))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-sv-audit", description="Audit a SwingVision shot export (.xlsx) for implausible data.")
    ap.add_argument("xlsx")
    ap.add_argument("--out", help="write the Markdown report here")
    ap.add_argument("--json", help="write the report JSON here")
    ap.add_argument("--segments", help="our segments.csv (raw-clock point segments) for the point-ending comparison")
    ap.add_argument("--hits", help="our rallies.json (audio hit times) for clock-offset and contact agreement")
    ap.add_argument("--score", help="our score timeline.csv for score agreement")
    ap.add_argument("--serve-max", type=float, default=90.0, help="flag serves faster than this (mph, default 90)")
    ap.add_argument("--rally-max", type=float, default=80.0, help="flag non-serves faster than this (mph, default 80)")
    ap.add_argument("--tol", type=float, default=0.10, help="line tolerance (m)")
    ap.add_argument("--short", type=float, default=1.5, help="flag points ending less than this after the last contact (s)")
    ap.add_argument("--lag", type=float, default=SV_SHOT_LAG_S, help="shot Video Time -> contact lag (s)")
    ap.add_argument("--examples", type=int, default=6)
    A = ap.parse_args(argv)
    m = load(A.xlsx)
    rep = audit(m, A.serve_max, A.rally_max, A.tol, A.short, A.lag, read_segments_csv(A.segments) if A.segments else None, A.examples)
    md = format_markdown(rep)
    if A.segments:
        import csv
        import json
        hits = [t for r in json.load(open(A.hits)) for t in r["times"]] if A.hits else None
        if hits:
            rep["offset"] = estimate_offset(m, hits, A.lag)
            o = rep["offset"]
            md += (f"\n# Clock offset\n\nBest offset {o['offset']:+.2f} s on top of the {A.lag:g} s lag: {o['matched']}/{o['of']} SwingVision contacts "
                   f"within {o['tol'] * 1000:.0f} ms of our audio hits (chance level about {o['baseline']}).\n")
        rows = read_segments_csv(A.segments, full=True)
        score = list(csv.DictReader(open(A.score))) if A.score else None
        rep["comparison"] = compare_points(m, rows, hits, score, A.lag, rep.get("offset", {}).get("offset", 0.0))
        md += format_comparison(rep["comparison"])
    print(md)
    if A.out: open(A.out, "w").write(md)
    if A.json:
        import json
        json.dump(rep, open(A.json, "w"), indent=1, default=str)
    return rep


if __name__ == "__main__":
    main()
