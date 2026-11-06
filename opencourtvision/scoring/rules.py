"""Score rules and evidence types shared by every scorer (common code; ships in both builds).

Split out of decoder.py (v3.9): the rules-exact state machine (`advance`, `replay`), score calls, serve side and the
per-point evidence records. The forward-only scorer (`forward.py`) and the full-app global decoder (`decoder.py`) both
build on this module; the US build packages this and `forward.py`, not `decoder.py`.
"""
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

from .formats import ScoringFormat, _deciding_after, game_point_state

NEG = -1e18
OTHER = {"A": "B", "B": "A"}
EPS = 1e-6


# ---------------------------------------------------------------- state machine

@dataclass(frozen=True)
class ScoreState:
    """Score before a point. games_sets (tennis/padel): sets, games, points (or tiebreak points) per team, the serving
    team, ends (0/1, which end team A is at) and tiebreak bookkeeping. Pickleball: games, points, serving team, server
    number. done = the match is over."""
    sets: tuple = (0, 0)
    games: tuple = (0, 0)
    pts: tuple = (0, 0)
    server: str = "A"
    ends: int = 0
    tb: int = 0                 # 0 = regular game, 7 or 10 = tiebreak target
    tb_first: str = None        # team that served the first tiebreak point
    server_no: int = 0          # pickleball doubles: 1 or 2
    done: bool = False

    def idx(self, team):
        return 0 if team == "A" else 1


def initial_state(fmt: ScoringFormat, first_server="A", ends=0):
    if fmt.system == "games_sets":
        return ScoreState(server=first_server, ends=ends)
    return ScoreState(server=first_server, ends=ends, server_no=2 if fmt.first_server_exception else 1)


def _add(t, i, k=1):
    return (t[0] + k, t[1]) if i == 0 else (t[0], t[1] + k)


@lru_cache(maxsize=200_000)
def advance(st: ScoreState, winner: str, fmt: ScoringFormat) -> ScoreState:
    """Exact rules: the state after `winner` ('A'/'B') wins the point played from `st`."""
    if st.done:
        raise ValueError("match already over")
    if fmt.system == "games_sets":
        return _advance_games_sets(st, winner, fmt)
    return _advance_pickleball(st, winner, fmt)


def _advance_games_sets(st, w, fmt):
    wi = st.idx(w)
    pts = _add(st.pts, wi)
    if st.tb:
        n = pts[0] + pts[1]
        if max(pts) >= st.tb and abs(pts[0] - pts[1]) >= 2:
            return _end_set(st, w, fmt, next_server=OTHER[st.tb_first], games_played_odd=True)
        # serve: first point by tb_first, then each team serves two points
        server = st.tb_first if ((n + 1) // 2) % 2 == 0 else OTHER[st.tb_first]
        ends = st.ends ^ (n % 6 == 0)
        return replace(st, pts=pts, server=server, ends=ends)
    # regular game: whose game point is decided by the deuce rule (server-relative)
    si = st.idx(st.server)
    s, r = pts[si], pts[1 - si]
    dec = _deciding_after(fmt.deuce_rule)
    s0, r0 = st.pts[si], st.pts[1 - si]
    deciding = dec is not None and s0 >= 3 and r0 >= 3 and s0 == r0 and (min(s0, r0) - 2) >= dec
    if not (deciding or (s >= 4 and s - r >= 2) or (r >= 4 and r - s >= 2)):
        if fmt.deuce_rule == "advantage" and min(pts) > 3:
            pts = (pts[0] - 1, pts[1] - 1)       # 4-4 == 3-3: merge equal futures (keeps the lattice small)
        return replace(st, pts=pts)
    games = _add(st.games, wi)
    ga, gb = games
    n_games = ga + gb
    target = fmt.games_per_set
    sets_total = st.sets[0] + st.sets[1]
    final_set = sets_total == 2 * fmt.sets_to_win - 2
    set_won = (max(ga, gb) >= target and abs(ga - gb) >= 2) or (st.games in ((target, target - 1), (target - 1, target)) and max(ga, gb) == target + 1)
    if final_set and fmt.final_set == "advantage_set":
        set_won = max(ga, gb) >= target and abs(ga - gb) >= 2
    if set_won:
        return _end_set(replace(st, games=games), w, fmt, next_server=OTHER[st.server], games_played_odd=n_games % 2 == 1)
    ends = st.ends ^ (n_games % 2 == 1)
    nxt = replace(st, games=games, pts=(0, 0), server=OTHER[st.server], ends=ends)
    if ga == gb == fmt.tiebreak_at and not (final_set and fmt.final_set == "advantage_set"):
        nxt = replace(nxt, tb=fmt.tiebreak_to, tb_first=nxt.server)
    return nxt


def _end_set(st, w, fmt, next_server, games_played_odd):
    sets = _add(st.sets, st.idx(w))
    done = max(sets) >= fmt.sets_to_win
    ends = st.ends ^ bool(games_played_odd)
    nxt = ScoreState(sets=sets, games=(0, 0), pts=(0, 0), server=next_server, ends=ends, done=done)
    if not done and fmt.final_set == "match_tiebreak_10" and sets == (fmt.sets_to_win - 1, fmt.sets_to_win - 1):
        nxt = replace(nxt, tb=10, tb_first=next_server)
    return nxt


def _advance_pickleball(st, w, fmt):
    """Mirrors formats.play_pickleball_game; `sets` holds games won, `pts` the game score."""
    si = st.idx(st.server)
    pts, server, sno = st.pts, st.server, st.server_no
    if w == st.server:
        pts = _add(pts, si)
    elif fmt.system == "side_out":
        if sno == 1 and fmt.servers_per_side_out == 2:
            sno = 2
        else:
            server, sno = OTHER[server], 1
    else:
        wi = st.idx(w)
        would = pts[wi] + 1 >= fmt.points_to_win and pts[wi] + 1 - pts[1 - wi] >= fmt.win_by
        if not (fmt.game_point_only_when_serving and would):
            pts = _add(pts, wi)
        server, sno = w, 1
    for i in (0, 1):
        if pts[i] >= fmt.points_to_win and pts[i] - pts[1 - i] >= fmt.win_by:
            games = _add(st.sets, i)
            done = max(games) >= (fmt.games_to_win or 1)
            first = OTHER[st.tb_first or "A"] if st.tb_first else OTHER["A"]
            return ScoreState(sets=games, pts=(0, 0), server=first, ends=st.ends ^ 1, tb_first=first,
                              server_no=2 if fmt.first_server_exception else 1, done=done)
    return replace(st, pts=pts, server=server, server_no=sno, tb_first=st.tb_first or "A")


def serve_side(st: ScoreState, fmt: ScoringFormat):
    """Court the serve goes from: 'deuce' (right) or 'ad' (left); None when not determined by the score (the receivers
    choose on a golden / Star point; pickleball needs player identity, see design §3.3)."""
    if fmt.system != "games_sets":
        return None
    if st.tb:
        return "deuce" if (st.pts[0] + st.pts[1]) % 2 == 0 else "ad"
    si = st.idx(st.server)
    if game_point_state(st.pts[si], st.pts[1 - si], fmt.deuce_rule)["deciding"]:
        return None
    return "deuce" if (st.pts[0] + st.pts[1]) % 2 == 0 else "ad"


def score_call(st: ScoreState, fmt: ScoringFormat):
    """Canonical spoken call before the point, server first: '30-15', 'deuce', 'deuce 2', 'ad in', tiebreak '3-2',
    pickleball '4-2-1' (side-out doubles) or '4-2'."""
    si = st.idx(st.server)
    s, r = st.pts[si], st.pts[1 - si]
    if fmt.system != "games_sets":
        return f"{s}-{r}-{st.server_no}" if fmt.score_call_numbers == 3 else f"{s}-{r}"
    if st.tb:
        return f"{s}-{r}"
    return game_point_state(s, r, fmt.deuce_rule)["call"]


def canonical_call(parsed):
    """callouts.parse_callout() output -> score_call() vocabulary (None if not comparable)."""
    if not parsed:
        return None
    kind, v = parsed
    names = {0: "0", 15: "15", 30: "30", 40: "40"}
    if kind == "pts":
        if v[0] == v[1] == 40:
            return "deuce"
        return f"{names.get(v[0], v[0])}-{names.get(v[1], v[1])}"
    if kind == "deuce":
        return "deuce"
    if kind == "deuce_n":
        return f"deuce {v}"
    if kind in ("ad", "ad_n"):
        side = v if kind == "ad" else v[1]
        return {"in": "ad in", "server": "ad in", "saque": "ad in", "out": "ad out", "receiver": "ad out", "resto": "ad out"}.get(side)
    return None


# ---------------------------------------------------------------- evidence

@dataclass(frozen=True)
class Obs:
    value: object
    acc: float              # P(observation correct); errors spread uniformly over the other `n_values - 1` values
    n_values: int = 2
    p_src: float = 1.0      # P(the observation is about OUR court/players). Callouts: the source gate's speaker / position /
                            # distance posterior (design §3.6). Mixture: p_src * P(obs | state) + (1 - p_src) / n_values,
                            # so a legal call from an uncertain source is weak evidence and a foreign one is ignored.


@dataclass(frozen=True)
class PointEvidence:
    p_a: float = 0.5                 # calibrated P(A won | per-point evidence) with a 0.5 prior
    p_point: float = 1.0             # P(segment is a real point)
    server: Obs = None               # 'A' / 'B'
    side: Obs = None                 # 'deuce' / 'ad'
    ends: Obs = None                 # 0 / 1
    call: Obs = None                 # score_call() string; n_values ~ 20
    calls: tuple = ()                # more callouts heard in this between-point window (partner repeats, other courts)
    confirmed: str = None            # 'A' / 'B' / 'skip' from the user: hard evidence
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def _obs_ll(o, truth):
    if o is None or truth is None:
        return 0.0
    p = o.acc if o.value == truth else (1 - o.acc) / max(o.n_values - 1, 1)
    if o.p_src < 1.0:
        p = o.p_src * p + (1 - o.p_src) / o.n_values
    return math.log(max(p, EPS))


def state_ll(st, ev, fmt):
    """log P(state observations at this segment | state)."""
    ll = _obs_ll(ev.server, st.server) + _obs_ll(ev.ends, st.ends)
    if ev.side is not None:
        ll += _obs_ll(ev.side, serve_side(st, fmt))
    if ev.call is not None or ev.calls:
        sc = score_call(st, fmt)
        for c in ((ev.call,) if ev.call is not None else ()) + tuple(ev.calls):
            ll += _obs_ll(c, sc)
    return ll


def choice_ll(st, ev):
    """[(choice, log weight)] for this segment from state st."""
    if ev.confirmed:
        if ev.confirmed == "skip":
            return [("skip", 0.0)]
        return [] if st.done else [(ev.confirmed, 0.0)]
    pp = min(max(ev.p_point, EPS), 1 - EPS) if ev.p_point < 1 else 1.0
    pa = min(max(ev.p_a, EPS), 1 - EPS)
    out = []
    if pp < 1 or st.done:
        out.append(("skip", math.log(1 - pp) if pp < 1 else math.log(EPS)))
    if not st.done:
        lp = math.log(pp)
        out += [("A", lp + math.log(pa)), ("B", lp + math.log(1 - pa))]
    return out


def independent_guess(evidence):
    """Per-point baseline with no rules: argmax of the per-point winner likelihood."""
    return ["A" if ev.p_a >= 0.5 else "B" for ev in evidence]


def replay(winners, fmt, first_server="A", ends0=0):
    """States before each point for a known winner sequence (skips allowed); raises if a point is played after match end."""
    st = initial_state(fmt, first_server, ends0); out = [st]
    for w in winners:
        st = st if w == "skip" else advance(st, w, fmt)
        out.append(st)
    return out
