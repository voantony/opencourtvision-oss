"""Caption trigger rules: the only way a reel or highlight caption gets on screen.

Rule R3 / editorial spec §6: captions come only from verified Events, and "hype" phrases only when their trigger is
verified. A moment asks for a caption by naming a trigger (`"caption_trigger": "rally_of_match"`); the trigger is
evaluated on the verified events inside the moment and returns the caption text and the time it may appear (at or
after the event, never before), or None. A fixed caption string in a config is never shown on its own: it can only
override the wording of a trigger that passed. Events from reference sources (SwingVision) never count as verified.

Triggers:
  shot            caption() of the best verified contact in the moment ("Forehand down the line", "Serve T")
  net_exchange    "Quick hands": >= 2 verified volleys within 1.2 s, by players of both teams, each at the net
                  (attrs net_dist_m <= 3 m). Without player-to-net distances (no player tracks yet) it never fires.
  rally_of_match  "Rally of the match": the moment's rally has the most verified contacts of the match (>= 4).
  ace             "Ace": verified serve, verified in-bounce (bounce event, outcome in) after it, and no verified return
                  contact before the point's end.
  running_get     "Running get": verified contact with attrs displacement_m >= 4 (tennis; 3 for padel/pickleball).
  break_point / set_point / match_point
                  Stakes card from a verified "score_state" event whose attrs carry the flag (break_point: true, ...).
"""
from .events import REFERENCE_SOURCES, caption

NET_DIST_M, QUICK_WINDOW_S, MIN_QUICK_VOLLEYS = 3.0, 1.2, 2
RALLY_MIN_HITS, RALLY_GAP_S = 4, 4.0
RUN_M = {"tennis": 4.0, "padel": 3.0, "pickleball": 3.0}
MAX_WORDS = {"9:16": 4, "16:9": 6}


def ok(e):
    """Verified and not from a reference import."""
    return bool(e.verified) and e.source not in REFERENCE_SOURCES


def team_of(e):
    """Team key for an event: attrs['team'], else from the player id (near*/far*, H*/G*). None if unknown."""
    if e.attrs.get("team"): return str(e.attrs["team"])
    p = str(e.player or "")
    for pre in ("near", "far"):
        if p.startswith(pre): return pre
    if p[:1] in ("H", "G") and p[1:].isdigit(): return p[:1]
    return None


def _st(e):
    return str(e.label.shot_type) if e.label is not None and e.label.shot_type is not None else None


def rallies(events, gap=RALLY_GAP_S):
    """Verified contacts grouped into rallies: split at verified point_end events, else at gaps > `gap` s."""
    ends = sorted(e.t for e in events if e.type == "point_end" and ok(e))
    cs = sorted((e for e in events if e.type == "contact" and ok(e)), key=lambda e: e.t)
    out, cur = [], []
    for e in cs:
        if cur and (e.t - cur[-1].t > gap or any(cur[-1].t < t <= e.t for t in ends)): out.append(cur); cur = []
        cur.append(e)
    if cur: out.append(cur)
    return out


def _in(events, a, b, typ=None):
    return [e for e in events if a <= e.t <= b and ok(e) and (typ is None or e.type == typ)]


def trig_shot(events, a, b, **_):
    best = None
    for e in sorted(_in(events, a, b, "contact"), key=lambda e: (-e.confidence, e.t)):
        c = caption(e)
        if c: best = (c, e.t); break
    return best, ("no verified contact with a caption" if best is None else "")


def trig_net_exchange(events, a, b, window=QUICK_WINDOW_S, min_volleys=MIN_QUICK_VOLLEYS, net_dist=NET_DIST_M, **_):
    vol = [e for e in _in(events, a, b, "contact") if _st(e) == "volley"]
    if len(vol) < min_volleys: return None, f"{len(vol)} verified volley(s) < {min_volleys}"
    at_net = [e for e in vol if e.attrs.get("net_dist_m") is not None and float(e.attrs["net_dist_m"]) <= net_dist]
    if len(at_net) < min_volleys: return None, "players' distance to the net unknown or > 3 m (needs player tracks)"
    at_net.sort(key=lambda e: e.t)
    for i in range(len(at_net)):
        grp = [e for e in at_net[i:] if e.t - at_net[i].t <= window]
        teams = {team_of(e) for e in grp}
        if len(grp) >= min_volleys and None not in teams and len(teams) >= 2:
            return ("Quick hands", grp[-1].t), ""
    return None, f"no {min_volleys} net volleys by both teams within {window} s"


def trig_rally_of_match(events, a, b, min_hits=RALLY_MIN_HITS, **_):
    R = rallies(events)
    if not R: return None, "no verified rallies"
    longest = max(len(r) for r in R)
    mine = [r for r in R if any(a <= e.t <= b for e in r)]
    if not mine: return None, "no verified rally in the moment"
    r = max(mine, key=len)
    if len(r) < min_hits: return None, f"rally has {len(r)} verified contacts < {min_hits}"
    if len(r) < longest: return None, f"rally has {len(r)} verified contacts, the longest has {longest}"
    return ("Rally of the match", r[-1].t), ""


def trig_ace(events, a, b, **_):
    for s in _in(events, a, b, "contact"):
        if _st(s) not in ("serve", "second_serve"): continue
        end = min((e.t for e in events if e.type == "point_end" and e.t > s.t), default=b)
        bounce = [e for e in _in(events, s.t, end, "bounce") if e.label is not None and str(e.label.outcome) == "in"]
        returns = [e for e in _in(events, s.t + 1e-6, end, "contact")]
        if bounce and not returns: return ("Ace", bounce[0].t), ""
    return None, "no verified serve with a verified in-bounce and no return"


def trig_running_get(events, a, b, sport="tennis", **_):
    need = RUN_M.get(sport, 4.0)
    for e in _in(events, a, b, "contact"):
        if e.attrs.get("displacement_m") is not None and float(e.attrs["displacement_m"]) >= need: return ("Running get", e.t), ""
    return None, f"no verified contact with displacement >= {need} m (needs player tracks)"


def _stakes(flag, text):
    def f(events, a, b, **_):
        for e in _in(events, a, b, "score_state"):
            if e.attrs.get(flag): return (text, e.t), ""
        return None, f"no verified score state with {flag}"
    return f


TRIGGERS = {"shot": trig_shot, "net_exchange": trig_net_exchange, "rally_of_match": trig_rally_of_match, "ace": trig_ace,
            "running_get": trig_running_get, "break_point": _stakes("break_point", "Break point"),
            "set_point": _stakes("set_point", "Set point"), "match_point": _stakes("match_point", "Match point")}


def moment_caption(trigger, events, a, b, text=None, aspect="9:16", **params):
    """-> (caption or None, t_event or None, reason). `text` may only reword a trigger that passed (max words per aspect)."""
    if not trigger: return None, None, "no caption_trigger: captions only come from a verified trigger"
    if trigger not in TRIGGERS: raise ValueError(f"unknown caption_trigger {trigger!r}; one of {', '.join(TRIGGERS)}")
    res, why = TRIGGERS[trigger](events or [], a, b, **params)
    if res is None: return None, None, why
    cap, t = res
    if text: cap = text
    if len(cap.split()) > MAX_WORDS.get(aspect, 4): return None, None, f"caption longer than {MAX_WORDS.get(aspect, 4)} words"
    return cap, t, ""
