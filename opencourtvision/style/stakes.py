"""Stakes captions from the REAL score state (rules engine), never from a mock-up or a guess.

A caption needs (1) a verified score state before the point (every earlier winner confirmed by the user: labels, fix
screen or confirmed watch taps), (2) a label on the preset's stakes allowlist, and (3) a clip that shows the whole point
(the planner only ever has whole points). MATCH POINT exists only when one more point won by a team ends the match."""
from ..scoring.formats import game_point_state
from ..scoring.rules import advance

PRIORITY = ("MATCH POINT", "SET POINT", "BREAK POINT")


def _who_wins(st, fmt):
    """{team: 'match'|'set'|'game'|None} = what winning the next point ends, for each team."""
    out = {}
    for t in ("A", "B"):
        nx = advance(st, t, fmt)
        if nx.done: out[t] = "match"
        elif nx.sets != st.sets: out[t] = "set"
        elif nx.games != st.games or (st.tb and not nx.tb): out[t] = "game"
        else: out[t] = None
    return out


def stakes(st, fmt, verified):
    """-> (label or None, team or None). Highest stake wins: MATCH > SET > BREAK; deciding point (golden / Star) via the
    format's deuce rule. Unverified state -> (None, None)."""
    if not verified or st.done: return None, None
    w = _who_wins(st, fmt)
    for lvl, lab in (("match", "MATCH POINT"), ("set", "SET POINT")):
        ts = [t for t in ("A", "B") if w[t] == lvl]
        if ts: return lab, ts[0]
    recv = "B" if st.server == "A" else "A"
    if w[recv] == "game" and not st.tb:
        return "BREAK POINT", recv
    if not st.tb and fmt.system == "games_sets":
        si = st.idx(st.server)
        g = game_point_state(st.pts[si], st.pts[1 - si], fmt.deuce_rule)
        if g.get("deciding"):
            return ("STAR POINT" if fmt.deuce_rule == "star_point" else "PUNTO DE ORO"), None
    return None, None


def saved(label, team, winner):
    """MATCH POINT SAVED (post-resolution only): the team that had match point lost the point."""
    return "MATCH POINT SAVED" if label == "MATCH POINT" and team and winner and winner != team else None


def caption_for(st, fmt, verified, allowlist):
    lab, team = stakes(st, fmt, verified)
    return (lab, team) if lab in set(allowlist) else (None, None)
