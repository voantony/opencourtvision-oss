"""Global score decoder: a rules-exact score state machine plus Viterbi / forward-backward over noisy per-point evidence.

Design: rallycut/scoring_engine_design.md §4. Each rally segment t carries
  - a calibrated per-point winner likelihood (p_a = P(team A won | ball, behaviour, audio evidence)),
  - optional observations of the score state BEFORE the point: serving team, serve side (deuce/ad court), ends, the
    spoken score call (server first), each with its own accuracy,
  - p_point: the probability the segment is a real point (not a let, a first-serve fault cut as its own rally, or a warm
    up hit), and an optional user confirmation (hard evidence).
The hidden variable per segment is c in {A, B, skip}. The score state is a deterministic function of the choices so far,
so equal states merge and the lattice stays small (tens of states per step); decoding a match is O(T * states).
Outputs: the MAP (Viterbi) winner sequence, per-point posterior marginals (forward-backward), and an ask order.

Teams are "A" and "B" (A = the recording user's side, "host" in the SwingVision import). Pure Python, no dependencies,
so it ports 1:1 to Swift on device.
"""
import itertools
import math
from dataclasses import dataclass, replace

from .rules import (  # noqa: F401  (re-exported: decoder.X keeps working)
    EPS,
    NEG,
    OTHER,
    Obs,
    PointEvidence,
    ScoreState,
    ScoringFormat,
    _add,
    _advance_games_sets,
    _advance_pickleball,
    _deciding_after,
    _end_set,
    _obs_ll,
    advance,
    canonical_call,
    choice_ll,
    game_point_state,
    independent_guess,
    initial_state,
    replay,
    score_call,
    serve_side,
    state_ll,
)


def _lse(xs):
    m = max(xs)
    if m <= NEG / 2:
        return NEG
    return m + math.log(sum(math.exp(x - m) for x in xs))


@dataclass
class Decoded:
    winners: list            # MAP choice per segment: 'A' / 'B' / 'skip'
    states: list             # MAP state before each segment (len T + 1, last = final)
    p_a: list                # posterior P(A won segment t)
    p_skip: list             # posterior P(segment t is not a point)
    confidence: list         # posterior probability of the MAP choice at t
    log_z: float

    def ask_order(self, k=None, exclude=()):
        """Segments to confirm, least confident first."""
        idx = sorted((i for i in range(len(self.confidence)) if i not in set(exclude)), key=lambda i: self.confidence[i])
        return idx[:k] if k else idx


class ScoreConflict(ValueError):
    """No legal score path satisfies the hard constraints (anchors, confirmations, final score). `conflicts` lists the
    clashing anchors when they are known (see anchor_conflicts); the UI shows them and asks which one is right."""

    def __init__(self, msg, conflicts=()):
        super().__init__(msg)
        self.conflicts = list(conflicts)


def decode(evidence, fmt, first_server=None, ends0=None, final=None, final_acc=1.0, anchors=None, set_scores=None,
           beam=25.0, max_states=600, _retries=2):
    """Viterbi + forward-backward over the score lattice.
    first_server / ends0: known values or None (uniform prior over both).
    final: sets tuple of the final state, e.g. (2, 0) (games won in pickleball); final_acc < 1 makes it soft.
    anchors: {segment index: Anchor} hard constraints on the state BEFORE that segment (index len(evidence) = after the
      last one); see attach_anchors(). set_scores: the result per set, e.g. [(6, 1), (6, 2)] (A first): a hard constraint
      on every set-ending transition. Anchors and set scores constrain the path both ways: forward-backward re-infers
      every point before and after them.
    beam: drop states whose forward log mass is more than `beam` nats (25 = e^-25 ~ 1e-11) below the best, and keep at most
    `max_states` per step. If hard constraints leave no state, the decode is retried with a wider beam before a
    ScoreConflict is raised (a real conflict, not a pruning artefact)."""
    try:
        return _decode(evidence, fmt, first_server, ends0, final, final_acc, anchors or {}, set_scores, beam, max_states)
    except ScoreConflict:
        if _retries <= 0 or not (anchors or set_scores or final is not None or any(e.confirmed for e in evidence)):
            raise
        return decode(evidence, fmt, first_server, ends0, final, final_acc, anchors, set_scores, beam * 2, max_states * 4,
                      _retries - 1)


def _set_ok(s, c, s2, set_scores):
    """A set-ending transition must produce the stated set score."""
    if not set_scores or c == "skip" or s2.sets == s.sets:
        return True
    k = s.sets[0] + s.sets[1]
    if k >= len(set_scores):
        return True
    games = _add(s.games, s.idx(c))
    return games == tuple(set_scores[k])


def _decode(evidence, fmt, first_server, ends0, final, final_acc, anchors, set_scores, beam, max_states):
    T = len(evidence)
    starts = [initial_state(fmt, s, e) for s in ([first_server] if first_server else ["A", "B"])
              for e in ([ends0] if ends0 is not None else [0, 1])]
    lp0 = -math.log(len(starts))
    alpha = [{s: lp0 for s in starts}]
    vit = [{s: (lp0, None, None) for s in starts}]
    edges = []                   # edges[t] = list of (s, choice, s', weight)
    a_idx = sorted(anchors)

    def constrain(states, t):
        """Hard constraints on the states before segment t: the anchor at t, monotone bounds from the next anchors."""
        a = anchors.get(t)
        out = [s for s in states if (a is None or a.matches(s, fmt))]
        nxt = [anchors[i] for i in a_idx if i > t]
        if nxt:
            out = [s for s in out if nxt[0].reachable_from(s)]
        if final is not None and final_acc >= 1.0:
            out = [s for s in out if _can_reach(s, final, T - t, fmt)]
        return out

    keep0 = constrain(list(alpha[0]), 0)
    alpha[0] = {s: alpha[0][s] for s in keep0}; vit[0] = {s: vit[0][s] for s in keep0}
    for t, ev in enumerate(evidence):
        if not alpha[t]:
            raise ScoreConflict(f"no legal score path reaches segment {t}")
        a_next, v_next, E = {}, {}, []
        for s, la in alpha[t].items():
            e = state_ll(s, ev, fmt)
            for c, w in choice_ll(s, ev):
                s2 = s if c == "skip" else advance(s, c, fmt)
                if not _set_ok(s, c, s2, set_scores):
                    continue
                wt = e + w
                E.append((s, c, s2, wt))
                a_next.setdefault(s2, []).append(la + wt)
                cand = vit[t][s][0] + wt
                if s2 not in v_next or cand > v_next[s2][0]:
                    v_next[s2] = (cand, s, c)
        a_next = {s: _lse(v) for s, v in a_next.items()}
        keep = constrain(list(a_next), t + 1)
        if keep:
            top = max(a_next[s] for s in keep)
            keep = sorted((s for s in keep if a_next[s] >= top - beam), key=lambda s: -a_next[s])[:max_states]
        a_next = {s: a_next[s] for s in keep}
        v_next = {s: v_next[s] for s in keep}
        alpha.append(a_next)
        vit.append(v_next)
        edges.append([x for x in E if x[2] in a_next])
    if not alpha[T]:
        raise ScoreConflict("no legal score path satisfies the constraints at the end of the match")

    def term(s):
        if final is None:
            return 0.0
        return math.log(max(final_acc, EPS)) if tuple(s.sets) == tuple(final) else math.log(max(1 - final_acc, EPS) / 10)

    log_z = _lse([la + term(s) for s, la in alpha[T].items()])
    beta = [None] * (T + 1)
    beta[T] = {s: term(s) for s in alpha[T]}
    pa, pskip, pc = [0.0] * T, [0.0] * T, []
    for t in range(T - 1, -1, -1):
        b, mass = {}, {"A": [], "B": [], "skip": []}
        for s, c, s2, wt in edges[t]:
            x = wt + beta[t + 1].get(s2, NEG)
            b.setdefault(s, []).append(x)
            mass[c].append(alpha[t][s] + x)
        beta[t] = {s: _lse(v) for s, v in b.items()}
        m = {c: math.exp(_lse(v) - log_z) if v else 0.0 for c, v in mass.items()}
        pa[t], pskip[t] = m["A"], m["skip"]
        pc.append(m)
    pc.reverse()
    end = max(vit[T], key=lambda s: vit[T][s][0] + term(s))
    states, winners, s = [end], [], end
    for t in range(T, 0, -1):
        _, prev, c = vit[t][s]
        winners.append(c); states.append(prev); s = prev
    winners.reverse(); states.reverse()
    conf = [pc[t][winners[t]] for t in range(T)]
    return Decoded(winners=winners, states=states, p_a=pa, p_skip=pskip, confidence=conf, log_z=log_z)


# ---------------------------------------------------------------- score anchors (MD moment_search_spec v0.6)

TENNIS_PTS = {"0": 0, "love": 0, "15": 1, "30": 2, "40": 3}


@dataclass(frozen=True)
class Anchor:
    """A score the user says was true BEFORE the point at match time t (typed, tapped or spoken; partial or full).
    Always in the user's perspective: us = team A. Any field may be None (partial anchor):
      sets / games: (us, them); pts: tennis/padel game score as labels ("30", "15") or "deuce" / "ad us" / "ad them",
      or raw ints in a tiebreak or pickleball; server: "us" / "them". Stored by match + timestamp, not point id, so it
      survives re-analysis (re-attached to segments with attach_anchors)."""
    t: float
    sets: tuple = None
    games: tuple = None
    pts: object = None
    server: str = None
    source: str = "typed"           # typed | tap | voice | final

    def matches(self, st, fmt):
        if self.sets is not None and st.sets != tuple(self.sets):
            return False
        if self.games is not None and fmt.system == "games_sets" and st.games != tuple(self.games):
            return False
        if self.server is not None and st.server != ("A" if self.server == "us" else "B"):
            return False
        if self.pts is None:
            return True
        p = self.pts
        if isinstance(p, str):
            a, b = st.pts
            if st.tb or fmt.system != "games_sets":
                return False
            if p == "deuce":
                return a == b and a >= 3
            if p in ("ad us", "ad them"):
                return min(a, b) >= 3 and (a - b) == (1 if p == "ad us" else -1)
            return False
        p = tuple(p)
        if all(isinstance(x, str) for x in p):
            if st.tb or fmt.system != "games_sets":
                return False
            return (st.pts[0], st.pts[1]) == (TENNIS_PTS[p[0]], TENNIS_PTS[p[1]]) and not (p[0] == p[1] == "40")
        return st.pts == p

    def reachable_from(self, st):
        """Necessary condition: sets and (in the same set) games never decrease."""
        if self.sets is None:
            return True
        if st.sets[0] > self.sets[0] or st.sets[1] > self.sets[1]:
            return False
        if self.games is not None and st.sets == tuple(self.sets):
            return st.games[0] <= self.games[0] and st.games[1] <= self.games[1]
        return True


def parse_anchor(text, t):
    """Typed anchor 'Us X – Them Y' (games in the current set), optional 'sets 1-0', points '30-15' (us first), 'deuce',
    'ad us' / 'ad them', and a serving toggle 'we serve' / 'they serve' (or 'serving: us/them'). Returns an Anchor or None."""
    import re
    s = text.lower().replace("–", "-").replace("—", "-")
    m = re.search(r"us\s*(\d+)\s*-?\s*them\s*(\d+)", s)
    games = (int(m.group(1)), int(m.group(2))) if m else None
    ms = re.search(r"sets?\s*(\d)\s*-\s*(\d)", s)
    sets = (int(ms.group(1)), int(ms.group(2))) if ms else None
    rest = s[m.end():] if m else s
    pts = None
    if re.search(r"\bdeuce\b|\biguales\b", rest):
        pts = "deuce"
    elif re.search(r"\bad(vantage)?\s*(us|them)\b", rest):
        pts = "ad " + re.search(r"\bad(vantage)?\s*(us|them)\b", rest).group(2)
    else:
        mp = re.search(r"\b(0|15|30|40|love)\s*-\s*(0|15|30|40|love)\b", rest)
        if mp:
            pts = (mp.group(1), mp.group(2))
    server = "us" if re.search(r"\bwe serve|serving:?\s*us\b", s) else "them" if re.search(r"\bthey serve|serving:?\s*them\b", s) else None
    if games is None and sets is None and pts is None:
        return None
    return Anchor(t=t, sets=sets, games=games, pts=pts, server=server)


def attach_anchors(anchors, segment_starts, tol=1.0):
    """Anchors (by match time) -> {segment index: Anchor}: an anchor describes the score before the segment whose start
    is nearest to anchor.t (within tol), else the first one starting after it (index len(segments) = after the last point). Two anchors on one segment are kept
    only if identical; otherwise a ScoreConflict is raised for the UI."""
    out = {}
    for a in sorted(anchors, key=lambda a: a.t):
        near = [k for k, t0 in enumerate(segment_starts) if abs(t0 - a.t) <= tol]
        i = min(near, key=lambda k: abs(segment_starts[k] - a.t)) if near else \
            next((k for k, t0 in enumerate(segment_starts) if t0 >= a.t), len(segment_starts))
        if i in out and out[i] != a:
            b = out[i]
            if any(x is not None and y is not None and x != y for x, y in ((a.sets, b.sets), (a.games, b.games), (a.pts, b.pts), (a.server, b.server))):
                raise ScoreConflict(f"two anchors disagree at segment {i}", [("anchor_anchor", b, a)])
            a = Anchor(t=b.t, sets=a.sets or b.sets, games=a.games or b.games, pts=a.pts or b.pts, server=a.server or b.server, source=b.source)
        out[i] = a
    return out


def anchor_conflicts(evidence, fmt, anchors, decoded=None, **kw):
    """Conflicts to show the user, never resolve silently: [("anchor_anchor", a, b)] for anchor pairs that no legal path
    satisfies together (rules only, evidence ignored), and [("anchor_call", anchor, segment, heard, path)] for heard calls
    (p_src >= 0.5) that disagree with the anchored best path within 3 segments of an anchor."""
    out = []
    neutral = [PointEvidence(p_point=e.p_point, confirmed=e.confirmed) for e in evidence]
    items = sorted(anchors.items())
    for (i, a), (j, b) in itertools.combinations(items, 2):
        try:
            _decode(neutral, fmt, kw.get("first_server"), kw.get("ends0"), None, 1.0, {i: a, j: b}, None, 60.0, 20000)
        except ScoreConflict:
            out.append(("anchor_anchor", a, b))
    if out:
        return out
    d = decoded or decode(evidence, fmt, anchors=anchors, **kw)
    for i, a in items:
        for k in range(max(0, i - 3), min(len(evidence), i + 4)):
            ev = evidence[k]
            for c in ((ev.call,) if ev.call else ()) + tuple(ev.calls):
                path = score_call(d.states[k], fmt)
                if c.p_src >= 0.5 and c.value != path:
                    out.append(("anchor_call", a, k, c.value, path))
    return out


def next_best_check(evidence, fmt, decoded, k=5, **kw):
    """Value of information: among the k least confident unconfirmed points, the one whose answer most reduces the
    expected number of wrong points (sum of 1 - confidence after re-decoding, weighted by the current posterior)."""
    cand = [i for i in decoded.ask_order() if evidence[i].confirmed is None][:k]
    best, best_v = None, None
    for i in cand:
        v = 0.0
        for c, p in (("A", decoded.p_a[i]), ("B", 1 - decoded.p_a[i] - decoded.p_skip[i]), ("skip", decoded.p_skip[i])):
            if p < 1e-6:
                continue
            ev = list(evidence); ev[i] = replace(ev[i], confirmed=c)
            try:
                d = decode(ev, fmt, **kw)
            except ScoreConflict:
                continue
            v += p * sum(1 - x for x in d.confidence)
        if best_v is None or v < best_v:
            best, best_v = i, v
    return best


def add_anchor(evidence, fmt, anchors, new, segment_starts, previous=None, **kw):
    """Add an anchor, re-decode, and report what changed for the UI:
    -> dict(decoded, anchors, changed=[segment...], summary="Updated 14 points in set 2", next_check=segment, conflicts=[...]).
    Conflicts are returned (and nothing is applied) instead of being resolved silently."""
    try:
        att = attach_anchors(list(anchors.values()) + [new] if isinstance(anchors, dict) else list(anchors) + [new], segment_starts)
    except ScoreConflict as e:
        return dict(decoded=previous, anchors=anchors, changed=[], summary="conflict", next_check=None, conflicts=e.conflicts)
    conf = anchor_conflicts(evidence, fmt, att, **kw)
    if any(c[0] == "anchor_anchor" for c in conf):
        return dict(decoded=previous, anchors=anchors, changed=[], summary="conflict", next_check=None, conflicts=conf)
    d = decode(evidence, fmt, anchors=att, **kw)
    prev = previous or decode(evidence, fmt, anchors={i: a for i, a in att.items() if a is not new}, **kw)
    changed = [i for i in range(len(evidence)) if d.winners[i] != prev.winners[i]]
    sets = sorted({d.states[i].sets[0] + d.states[i].sets[1] + 1 for i in changed})
    summary = f"Updated {len(changed)} point{'s' * (len(changed) != 1)}" + (f" in set {', '.join(map(str, sets))}" if changed else "")
    return dict(decoded=d, anchors=att, changed=changed, summary=summary, next_check=next_best_check(evidence, fmt, d, **kw),
                conflicts=conf)


def _can_reach(st, final, remaining, fmt):
    """Cheap necessary condition for reaching final sets (games won in pickleball) within `remaining` segments."""
    need = [final[i] - st.sets[i] for i in (0, 1)]
    if min(need) < 0 or (st.done and need != [0, 0]):
        return False
    per_set = 4 * fmt.games_per_set if fmt.system == "games_sets" else (fmt.points_to_win or 11)
    if fmt.final_set == "match_tiebreak_10":
        per_set = min(per_set, 10)
    progress = 4 * max(st.games) + max(st.pts) if fmt.system == "games_sets" else max(st.pts)
    return sum(need) * per_set - progress <= remaining


def confirm_loop(evidence, fmt, truth, threshold, max_asks=None, batch=1, **kw):
    """Selective prediction with re-decoding: while the least confident unconfirmed segment is below `threshold`, ask about
    it (up to `batch` per round, least confident first), treat each answer as hard evidence and re-decode. batch=1 is the
    product UX (the review queue re-sorts after every tap); batch > 1 is a cheaper approximation that can over-ask slightly.
    `truth` stands in for the user's answers. Returns (decoded, asked indices)."""
    ev = list(evidence); asked = []
    while True:
        d = decode(ev, fmt, **kw)
        order = [i for i in d.ask_order() if ev[i].confirmed is None and d.confidence[i] < threshold]
        if max_asks is not None:
            order = order[:max(0, max_asks - len(asked))]
        if not order:
            return d, asked
        for i in order[:batch]:
            ev[i] = replace(ev[i], confirmed=truth[i]); asked.append(i)
