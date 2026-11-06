"""Forward-only scoring (design D1 as tightened by Legal on 18 Oct 2026, v3.9; BOTH builds).

Rules this module enforces (engineering_notes §2.8.9):
  1. Each point is decided ONCE, when it ends, from that point's live cues (winner likelihood fused from ball track, hit
     audio and in-point voice; p_point). Uncertain cues inside the point are weighed before the commit; after the commit
     the point is frozen.
  2. One committed score state only: no tree, pruned lattice or set of alternative score histories, no backward pass, no hindsight
     smoothing, no re-choosing of any past winner by the engine.
  3. Past points change only through a USER edit: a tap (`user_edit`) or a voice correction. After an edit, the score from
     that point on is recomputed deterministically by replaying the committed winners through the rules.
  4. Low-confidence points prompt the user before the commit (selective prediction at `tau`).

Legal rulings of 18 Oct 05:41 (both builds):
  - Q3 score-check prompts (`score_checks=<p>`): SHIPPED. Prompt when this point's pre-serve cues are unlikely under the
    ONE shown score (likelihood ratio vs uninformative < p) or a point is played after the shown match end. The
    decision never enumerates or weights alternative histories; then at most one obvious fix is offered with
    fix / keep / free edit; no answer leaves the score unchanged; every answer is logged.
  - Q2 players' spoken score calls (`voice_edits=True`): SHIPPED as user input. A clear, gated, score-shaped call that
    differs only by the point just committed is applied as a user edit with feedback ("Heard 15-30, updated") and
    one-tap undo; any older conflict goes to the Q3 tap prompt with no re-anchoring (Legal 05:55).
  - Q1 hold-until-next-serve (`policy="next_serve"`): kept in code, DISABLED (worse on real data, 73% vs 82%).

Manual score anchors (user-set truth, `set_anchor`): the user types the score before any point. From there the engine
keeps scoring forward. Points before the anchor are NEVER re-derived: if they do not add up to the typed score the
scorer raises a review flag ("points 12-15 don't add up to your entered score, review?") and lists suggested points to
check. The suggestion list is ONLY a sort of the committed points in that span by each point's own stored first-call
confidence, lowest first; no alternative score history is built or weighed. The user fixes it with taps (`user_edit`).

Suggested backward fixes (Antony via Legal, 18 Oct 05:28; both builds; `suggestions=True`): when later evidence (a
manual anchor, a gated spoken score call, or an observed server / serve side) is inconsistent with the replay of the
committed points, the scorer creates ONE `Suggestion` ("points 12-15 look wrong, fix them?") with ONE proposed fix and
changes nothing. Only `approve(suggestion)` applies it, as user edits with source "suggestion"; `decline` drops it.
Fix logic (rule / consistency checks only): the smallest set (<= 3) of committed points, taken from the lowest stored
first-call confidence first (ties: nearest the conflict), whose A<->B flip makes the replay pass a boolean consistency
check. No tree or ranked set of probability-weighted score histories is kept, nothing is renormalised over candidate
scores, and nothing is rewritten without the user's approval.

Pure Python; builds on rules.py only (never on decoder.py, which is full-app only and off by default there).
"""
import math
from dataclasses import dataclass, field, replace

from .rules import OTHER, advance, choice_ll, initial_state, score_call, state_ll

POLICIES = ("point_end", "next_serve")


@dataclass(frozen=True)
class Commit:
    idx: int
    choice: str              # 'A' / 'B' / 'skip'
    confidence: float        # normalised weight of the choice among this point's options (1.0 when the user answered)
    asked: bool              # the user was prompted before the commit


@dataclass
class Edit:
    idx: int
    choice: str
    source: str              # 'tap' / 'voice'


def _norm(scored):
    m = max(w for _, w in scored)
    z = sum(math.exp(w - m) for _, w in scored)
    return [(c, math.exp(w - m) / z) for c, w in scored]


def start_state(fmt, ev0=None, first_server=None, ends0=None):
    """Initial state from setup; if first server / ends are unknown, the best fit to the first point's pre-serve cues,
    chosen once before the first point."""
    cands = [initial_state(fmt, s, e) for s in ([first_server] if first_server else ["A", "B"])
             for e in ([ends0] if ends0 is not None else [0, 1])]
    if ev0 is None or len(cands) == 1:
        return cands[0]
    return max(cands, key=lambda s: state_ll(s, ev0, fmt))


def point_options(st, ev, fmt, nxt=None):
    """[(choice, probability)] for ONE point from the committed state. nxt (next_serve flag only): the next point's
    pre-serve cues, scored against the single state each option leads to."""
    out = []
    for c, w in choice_ll(st, ev):
        if nxt is not None:
            s2 = st if c == "skip" else advance(st, c, fmt)
            w += 0.0 if s2.done else state_ll(s2, nxt, fmt)
        out.append((c, w))
    return _norm(out)


def _ll1(o, truth):
    """log [P(obs | truth) / (1 / n_values)] for one observation (source mixture included)."""
    p = o.acc if o.value == truth else (1 - o.acc) / max(o.n_values - 1, 1)
    if o.p_src < 1.0:
        p = o.p_src * p + (1 - o.p_src) / o.n_values
    return math.log(max(p, 1e-9)) - math.log(1.0 / o.n_values)


def _heard(ev, min_src):
    return [c for c in ((ev.call,) if ev.call is not None else ()) + tuple(ev.calls) if c.p_src >= min_src]


def same_score(a, b):
    return (a.sets, a.games, a.pts, a.tb) == (b.sets, b.games, b.pts, b.tb)


@dataclass
class ReviewFlag:
    anchor_idx: int          # the anchor sits before this point
    first: int               # span of committed points that don't add up: first..last (inclusive)
    last: int
    suggestions: list        # point indices in the span, lowest stored first-call confidence first
    resolved: bool = False
    proposed: object = None  # Suggestion (ONE best fix) when ForwardScorer.suggestions is on

    def message(self):
        return (f"points {self.first + 1}-{self.last + 1} don't add up to your entered score, review?"
                if self.last >= self.first else "your entered score differs from the start of the match, review?")


@dataclass(frozen=True)
class Suggestion:
    kind: str                # 'anchor' / 'call' / 'server_side'
    at: int                  # the evidence sits before this point
    first: int               # span searched (inclusive)
    last: int
    fix: tuple               # ((point index, new winner), ...): the ONE proposed fix
    message: str


@dataclass(frozen=True)
class ScorePrompt:
    at: int
    fix: object              # Suggestion or None
    options: tuple           # ('fix', 'keep', 'edit') or ('keep', 'edit')
    message: str


def state_from_call(shown, call, fmt):
    """Q2(a): the state a spoken point call sets going forward: the called point score with the shown sets / games /
    server kept. Parsing the call text into points (not a search over histories). None if not a regular-game call."""
    if shown.tb or shown.done:
        return None
    best = None
    for a in range(0, 8):
        for b in range(0, 8):
            st = replace(shown, pts=(a, b))
            try:
                ok = score_call(st, fmt) == call
            except Exception:
                ok = False
            if ok:
                d = abs((a + b) - sum(shown.pts))
                if best is None or d < best[0]:
                    best = (d, st)
    return best[1] if best else None


def _consistent(st, target, fmt):
    """Boolean consistency of a replayed state with the conflicting evidence."""
    kind, val = target
    if kind == "anchor":
        return same_score(st, val)
    if kind == "call":
        return score_call(st, fmt) in val
    srv, side = val
    from .rules import serve_side
    return (srv is None or st.server == srv) and (side is None or serve_side(st, fmt) == side)


@dataclass
class ForwardScorer:
    """Incremental forward-only scorer. Feed points with `point_ended(ev)` (and, for the next_serve flag, the next
    point's cues with `next_cues(ev)`). `ask(i, kind) -> answer` stands in for the user prompt."""
    fmt: object
    policy: str = "point_end"
    tau: float = None
    ask: object = None
    voice_edits: bool = False
    score_checks: float = None
    suggestions: bool = False
    max_fix: int = 3
    pool: int = 12
    span_cap: int = 40
    min_src: float = 0.5
    first_server: str = None
    ends0: int = None
    commits: list = field(default_factory=list)
    states: list = field(default_factory=list)     # committed state before each point (+ after the last)
    edits: list = field(default_factory=list)
    prompts: list = field(default_factory=list)    # (index, kind) of every user prompt
    anchors: dict = field(default_factory=dict)    # point index -> user-set state before that point
    flags: list = field(default_factory=list)      # ReviewFlag per inconsistent anchor
    min_call_acc: float = 0.8
    feedback: list = field(default_factory=list)   # Q2(b) visible messages ("Heard 15-30, updated") with undo info
    log: list = field(default_factory=list)        # Q3(d) / Q2 audit log of prompts and answers
    pending_suggestions: list = field(default_factory=list)
    suggestion_log: list = field(default_factory=list)   # (Suggestion, 'approved' | 'declined')
    _pending: object = None
    _ss_conflict: object = None

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(self.policy)

    @property
    def state(self):
        return self.states[-1]

    def _start(self, ev):
        if not self.states:
            self.states.append(start_state(self.fmt, ev, self.first_server, self.ends0))

    def _clear_calls(self, ev):
        """Q2(d): only clear, score-shaped utterances that passed the callout gate (our players, legal call text)."""
        return [c for c in _heard(ev, self.min_src) if c.acc >= self.min_call_acc and isinstance(c.value, str)]

    def _voice(self, ev):
        """Q2 (Legal 05:41, amended 05:55): a players' spoken score call is USER INPUT.
        (b) if it differs from the shown score only by the point just committed: user edit (source 'voice'), visible
            feedback "Heard 15-30, updated" and one-tap undo (`undo_voice`);
        (a/c) any older conflict goes to the Q3 tap prompt (fix / keep / edit); no re-anchoring, nothing rewritten."""
        if not self.voice_edits or not self.commits:
            return
        heard = self._clear_calls(ev)
        shown = score_call(self.state, self.fmt)
        if not heard or any(c.value == shown for c in heard):
            return
        call = heard[0].value
        last = self.commits[-1]
        if last.choice in ("A", "B"):
            prev = self.states[-2]
            if not prev.done:
                flip = advance(prev, OTHER[last.choice], self.fmt)
                if score_call(flip, self.fmt) == call:
                    self.user_edit(last.idx, OTHER[last.choice], source="voice")
                    self.feedback.append(dict(at=len(self.commits), msg=f"Heard {call}, updated", undo=(last.idx, last.choice)))
                    self.log.append(dict(at=len(self.commits), kind="voice", response="applied", call=call, idx=last.idx))
                    return
        # Legal 05:55: an older conflict is NOT re-anchored; it goes to the Q3 tap prompt (fix / keep / edit), which
        # `_check_score` raises for a clear call that differs from the shown score.
        self.log.append(dict(at=len(self.commits), kind="voice", response="sent_to_q3", call=call))

    def undo_voice(self):
        """One-tap undo of the last voice edit (a user action)."""
        for fb in reversed(self.feedback):
            if fb.get("undo"):
                idx, old = fb["undo"]
                fb["undo"] = None
                self.user_edit(idx, old, source="undo")
                self.log.append(dict(at=len(self.commits), kind="voice", response="undone", idx=idx))
                return True
        return False

    def _shown_fit(self, ev):
        """log likelihood ratio of this point's pre-serve cues under the shown state vs uninformative cues."""
        lr = 0.0
        st = self.state
        for o, truth in ((ev.server, st.server), (ev.ends, st.ends)):
            if o is not None:
                lr += _ll1(o, truth)
        if ev.side is not None:
            from .rules import serve_side
            lr += _ll1(ev.side, serve_side(st, self.fmt))
        if ev.call is not None or ev.calls:
            sc = score_call(st, self.fmt)
            for c in ((ev.call,) if ev.call is not None else ()) + tuple(ev.calls):
                lr += _ll1(c, sc)
        return lr

    def _cue_target(self, ev):
        heard = self._clear_calls(ev)
        if heard:
            return "call", ("call", frozenset(c.value for c in heard))
        srv = ev.server.value if ev.server is not None and ev.server.acc >= 0.9 else None
        side = ev.side.value if ev.side is not None and ev.side.acc >= 0.9 else None
        if srv is not None or side is not None:
            return "server_side", ("ss", (srv, side))
        return None, None

    def _check_score(self, ev):
        """Q3 (Legal 05:41, shipped). (a) The decision to prompt evaluates ONLY the one shown state against this point's
        pre-serve cues (likelihood ratio vs uninformative), never alternative histories. (b) After deciding, at most one
        obvious fix is computed (`suggest`); nothing changes without the user's answer. (c) No answer = no change.
        (d) Every answer is logged. (e) The prompt offers fix / keep / free edit."""
        if self.score_checks is None or not self.commits:
            return
        shown = score_call(self.state, self.fmt)
        clear_mismatch = any(c.value != shown for c in self._clear_calls(ev)) and \
            not any(c.value == shown for c in self._clear_calls(ev))
        if not (self.state.done or clear_mismatch or self._shown_fit(ev) < math.log(self.score_checks)):
            return
        i = len(self.commits)
        kind, target = self._cue_target(ev)
        fix = None
        if target is not None:
            first = self._game_start(i) if kind == "call" else None
            fix = self.suggest(kind, target, first=first)
        pr = ScorePrompt(i, fix, ("fix", "keep", "edit") if fix else ("keep", "edit"),
                         fix.message if fix else "The score may be off. Keep it or edit?")
        self.prompts.append((i, "score_check"))
        ans = self.ask(pr, "score_check") if self.ask is not None else None
        resp = ans[0] if isinstance(ans, tuple) and ans else None
        if resp not in pr.options:
            resp = None                                             # (c) timeout / no answer: unchanged
        if resp == "fix":
            self.approve(fix)
        elif resp == "edit":
            self.anchor_and_review(ans[1])
        elif fix is not None:
            self.decline(fix)
        self.log.append(dict(at=i, kind="score_check", response=resp or "no_answer",
                             fix=fix.fix if fix else None))

    def anchor_and_review(self, state):
        """User enters the score; if earlier points don't add up, the user checks the suggestions (each check is a prompt)
        and edits the wrong ones until the span adds up."""
        f = self.set_anchor(state)
        if f is None or self.ask is None:
            return f
        if f.proposed is not None:
            self._offer(f.proposed)
            if f.resolved:
                return f
        for j in f.suggestions:
            self.prompts.append((j, "review"))
            ans = self.ask(j, "point")
            if ans is not None and ans != self.commits[j].choice:
                self.user_edit(j, ans)
            if f.resolved:
                break
        return f

    def _replay_from(self, first, upto, flips):
        """State before point `upto`, replaying committed winners from the committed state before `first` with flips."""
        st = self.states[first]
        for j in range(first, upto):
            w = flips.get(j, self.commits[j].choice)
            if j in self.anchors and j != first:
                st = self.anchors[j]
            st = st if (w == "skip" or st.done) else advance(st, w, self.fmt)
        return st

    def _game_start(self, at):
        """Index of the first point of the current game (committed states)."""
        j = at
        while j > 0 and self.states[j].pts != (0, 0):
            j -= 1
        return j

    def suggest(self, kind, target, at=None, first=None):
        """ONE suggested fix for a conflict before point `at` (default: now), or None. Changes nothing."""
        import itertools
        at = len(self.commits) if at is None else at
        prev = [k for k in self.anchors if k < at]
        lo = max(prev) if prev else 0
        first = max(lo, at - self.span_cap) if first is None else max(lo, first)
        if at <= first or _consistent(self._replay_from(first, at, {}), target, self.fmt):
            return None
        cand = [j for j in range(first, at) if self.commits[j].choice in ("A", "B")
                and not any(e.idx == j for e in self.edits)]
        cand = sorted(cand, key=lambda j: (self.commits[j].confidence, at - j))[:self.pool]
        for k in range(1, self.max_fix + 1):
            for combo in itertools.combinations(cand, k):          # generated lowest-confidence first
                flips = {j: OTHER[self.commits[j].choice] for j in combo}
                if _consistent(self._replay_from(first, at, flips), target, self.fmt):
                    fix = tuple(sorted(flips.items()))
                    lo, hi = fix[0][0], fix[-1][0]
                    msg = (f"point {lo + 1} looks wrong, fix it?" if lo == hi else f"points {lo + 1}-{hi + 1} look wrong, fix them?")
                    sg = Suggestion(kind, at, first, at - 1, fix, msg)
                    self.pending_suggestions.append(sg)
                    return sg
        return None

    def approve(self, sg):
        """User approval: apply the proposed fix as user edits (source 'suggestion')."""
        for j, w in sg.fix:
            self.user_edit(j, w, source="suggestion")
        if sg in self.pending_suggestions:
            self.pending_suggestions.remove(sg)
        self.suggestion_log.append((sg, "approved"))

    def decline(self, sg):
        if sg in self.pending_suggestions:
            self.pending_suggestions.remove(sg)
        self.suggestion_log.append((sg, "declined"))

    def _offer(self, sg):
        """Show a suggestion; with an `ask` stand-in, ask(sg, 'suggestion') -> True approves."""
        if sg is None or self.ask is None:
            return
        self.prompts.append((sg.at, "suggestion"))
        if self.ask(sg, "suggestion"):
            self.approve(sg)
        else:
            self.decline(sg)

    def _suggest_from_cues(self, ev):
        """Pre-serve cues before this point vs the shown score: a gated call, or a confident server / side."""
        if not self.suggestions or not self.commits:
            return
        at = len(self.commits)
        heard = [] if self.voice_edits else _heard(ev, self.min_src)
        if heard:
            vals = frozenset(c.value for c in heard)
            if score_call(self.state, self.fmt) not in vals:
                # a point call constrains the current game: search only its points
                self._offer(self.suggest("call", ("call", vals), first=self._game_start(at)))
                return
        srv = ev.server.value if ev.server is not None and ev.server.acc >= 0.9 else None
        side = ev.side.value if ev.side is not None and ev.side.acc >= 0.9 else None
        bad = (self.score_checks is None and (srv is not None or side is not None)
               and not _consistent(self.state, ("ss", (srv, side)), self.fmt))
        # rule check: server / side must disagree with the shown score in two consecutive pre-serve windows
        if bad and self._ss_conflict == at - 1:
            self._offer(self.suggest("server_side", ("ss", (srv, side))))
        self._ss_conflict = at if bad else None

    def _commit(self, ev, nxt=None):
        i = len(self.commits)
        st = self.state
        if ev.confirmed:
            c, p, asked = ev.confirmed, 1.0, False
        else:
            c, p = max(point_options(st, ev, self.fmt, nxt), key=lambda x: x[1])
            asked = False
            if self.tau is not None and self.ask is not None and p < self.tau:
                self.prompts.append((i, "point"))
                c, p, asked = self.ask(i, "point"), 1.0, True
        if c != "skip" and st.done:
            c = "skip"
        self.commits.append(Commit(i, c, p, asked))
        self.states.append(st if c == "skip" else advance(st, c, self.fmt))
        return self.commits[-1]

    def point_ended(self, ev):
        """A point has ended with its evidence. point_end policy: commit now. next_serve flag: hold it as pending."""
        self._start(ev)
        if self.policy == "point_end":
            self._voice(ev)
            self._suggest_from_cues(ev)
            self._check_score(ev)
            return self._commit(ev)
        out = None
        if self._pending is not None:               # a new point started without separate cues: use its pre-serve cues
            out = self._commit(self._pending, nxt=ev)
            self._pending = None
        self._voice(ev)
        self._suggest_from_cues(ev)
        self._check_score(ev)
        self._pending = ev
        return out

    def finish(self):
        if self._pending is not None:
            self._commit(self._pending)
            self._pending = None

    def _derived(self, j):
        """State after point j from the state before it and its committed winner (rules only)."""
        st, w = self.states[j], self.commits[j].choice
        return st if (w == "skip" or st.done) else advance(st, w, self.fmt)

    def user_edit(self, idx, choice, source="tap"):
        """User tap / voice correction of a past point: set its winner, then recompute the score from that point on by
        replaying the committed winners through the rules, up to the next user anchor (user truth stays). No other
        winner is re-chosen. Review flags covering the point are re-checked."""
        self.edits.append(Edit(idx, choice, source))
        c = self.commits[idx]
        self.commits[idx] = Commit(idx, choice, 1.0, c.asked)
        for j in range(idx, len(self.commits)):
            if j + 1 in self.anchors:
                break
            self.states[j + 1] = self._derived(j)
        self._recheck()

    def _span_start(self, a):
        prev = [k for k in self.anchors if k < a]
        return max(prev) if prev else 0

    def _check(self, a):
        """Does the replay from the previous anchor (or the start) reach the anchor at a?"""
        if a == 0:
            return True
        return same_score(self._derived(a - 1), self.anchors[a])

    def _recheck(self):
        for f in self.flags:
            if not f.resolved and self._check(f.anchor_idx):
                f.resolved = True

    def set_anchor(self, state, ev0=None):
        """User-typed score before the next point (user-set truth from here). Earlier points are left as committed; an
        inconsistency raises a ReviewFlag (returned) whose suggestions are a sort by stored first-call confidence."""
        self._start(ev0)
        if self._pending is not None:               # next_serve flag: the held point is committed first, as usual
            self._commit(self._pending)
            self._pending = None
        a = len(self.commits)
        self.anchors[a] = state
        self.states[-1] = state
        if self._check(a):
            return None
        first = self._span_start(a)
        span = [i for i in range(first, a) if self.commits[i].choice in ("A", "B") or self.commits[i].confidence < 1]
        sugg = sorted(span, key=lambda i: (self.commits[i].confidence, i))
        f = ReviewFlag(a, first, a - 1, sugg)
        if self.suggestions:
            f.proposed = self.suggest("anchor", ("anchor", state), at=a)
        self.flags.append(f)
        return f

    def open_flags(self):
        return [f for f in self.flags if not f.resolved]

    @property
    def winners(self):
        return [c.choice for c in self.commits]

    @property
    def confidence(self):
        return [c.confidence for c in self.commits]


def score_forward(evidence, fmt, policy="point_end", **kw):
    """Batch helper: feed every point in order (a stream, never a look-back) and return the scorer."""
    s = ForwardScorer(fmt, policy=policy, **kw)
    for ev in evidence:
        s.point_ended(ev)
    s.finish()
    return s
