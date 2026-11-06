"""D1 forward-only scoring (Legal, 18 Oct 2026): committed points and the running score never change without a user edit.

Property-style (stdlib random, many seeds; no extra dependency): stream random evidence point by point and assert that
everything committed so far is byte-identical (pickle) after any amount of later evidence. Plus taps, voice edits,
manual anchors and an import guard (forward.py never touches the global decoder).
"""
import ast
import os
import pickle
import random

import pytest

from opencourtvision.scoring.formats import get_format
from opencourtvision.scoring.forward import ForwardScorer, same_score, score_forward
from opencourtvision.scoring.rules import OTHER, Obs, PointEvidence, advance, initial_state, replay, score_call, serve_side

FMT = get_format("tennis", "advantage")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def random_match(rng, n):
    """True winners + noisy evidence (winner likelihood, parity cues, sometimes a wrong or foreign score call)."""
    W = [rng.choice("AB") for _ in range(n)]
    W2, S2, st = [], [], initial_state(FMT, "A", 0)
    for w in W:
        if st.done:
            break
        W2.append(w); S2.append(st); st = advance(st, w, FMT)
    ev = []
    for w, s in zip(W2, S2):
        c = rng.uniform(0.5, 0.99)
        pa = c if (w == "A") == (rng.random() < c) else 1 - c
        kw = dict(p_a=pa, p_point=rng.choice([1.0, 0.9, 0.6]))
        if rng.random() < 0.8:
            kw["server"] = Obs(s.server if rng.random() < 0.95 else OTHER[s.server], 0.95)
            side = serve_side(s, FMT)
            if side:
                kw["side"] = Obs(side, 0.9)
        if rng.random() < 0.5:
            call = score_call(s, FMT) if rng.random() < 0.8 else rng.choice(["15-0", "30-30", "deuce", "40-15"])
            kw["call"] = Obs(call, 0.9, 20, p_src=rng.choice([0.3, 0.7, 0.95]))
        ev.append(PointEvidence(**kw))
    return W2, S2, ev


def snap(s, k):
    return pickle.dumps((s.commits[:k], s.states[:k + 1]))


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("policy", ["point_end", "next_serve"])
def test_committed_points_never_change_without_user_edit(seed, policy):
    rng = random.Random(seed)
    W, S, ev = random_match(rng, rng.randint(5, 140))
    tau = rng.choice([None, 0.8, 0.95])
    s = ForwardScorer(FMT, policy=policy, tau=tau, ask=lambda i, kind: W[i] if kind == "point" else None)
    seen = []                                    # (k, snapshot of the first k commits + states) at each step
    for e in ev:
        s.point_ended(e)
        k = len(s.commits)
        for kk, b in seen:
            assert snap(s, kk) == b, f"earlier output changed at k={kk} after later evidence"
        seen.append((k, snap(s, k)))
    s.finish()
    for kk, b in seen:
        assert snap(s, kk) == b
    assert not s.edits


@pytest.mark.parametrize("seed", range(40))
def test_voice_edits_are_the_only_changes_and_are_logged(seed):
    rng = random.Random(1000 + seed)
    W, S, ev = random_match(rng, rng.randint(10, 120))
    s = ForwardScorer(FMT, voice_edits=True)
    prev = []
    for e in ev:
        n_ed = len(s.edits)
        s.point_ended(e)
        changed = [i for i, c in enumerate(prev) if s.commits[i].choice != c]
        edited = {x.idx for x in s.edits[n_ed:]}
        assert set(changed) <= edited
        assert all(x.source == "voice" for x in s.edits)
        assert all(i == len(prev) - 1 for i in edited)          # only the point just before the call
        prev = [c.choice for c in s.commits]


def test_user_edit_recomputes_deterministically():
    rng = random.Random(7)
    W, S, ev = random_match(rng, 60)
    s = score_forward(ev, FMT, first_server="A", ends0=0)
    before = [c.choice for c in s.commits]
    s.user_edit(10, OTHER[before[10]] if before[10] in "AB" else "A")
    assert [c.choice for i, c in enumerate(s.commits) if i != 10] == [c for i, c in enumerate(before) if i != 10]
    assert s.states == replay(s.winners, FMT, "A", 0)


def test_anchor_sets_running_score_and_scoring_continues_forward():
    rng = random.Random(3)
    W, S, ev = random_match(rng, 80)
    s = ForwardScorer(FMT, first_server="A", ends0=0)
    for e in ev[:40]:
        s.point_ended(e)
    s.set_anchor(S[40])
    assert same_score(s.state, S[40])
    for e in ev[40:41]:
        s.point_ended(e)
    assert s.states[40] == S[40] and len(s.commits) == 41


def test_anchor_never_changes_earlier_points_and_flags_mismatch():
    rng = random.Random(5)
    W, S, ev = random_match(rng, 80)
    ev = [PointEvidence(p_a=0.6 if w == "B" else 0.4) if i == 12 else e for i, (w, e) in enumerate(zip(W, ev))]
    s = ForwardScorer(FMT, first_server="A", ends0=0)
    for e in ev[:30]:
        s.point_ended(e)
    before = pickle.dumps((s.commits[:30], s.states[:30]))
    W_eng = s.winners[:]
    flag = s.set_anchor(S[30])
    assert pickle.dumps((s.commits[:30], s.states[:30])) == before   # the anchor changed no earlier point
    if W_eng == W[:30]:
        assert flag is None
        return
    assert flag is not None and not flag.resolved and "don't add up" in flag.message()
    assert (flag.first, flag.last) == (0, 29)
    conf = [s.commits[i].confidence for i in flag.suggestions]
    assert conf == sorted(conf)                                   # a sort by stored first-call confidence only
    assert s.winners == W_eng                                     # nothing re-derived
    for i in flag.suggestions:                                    # user taps through the suggestions
        if s.winners[i] != W[i]:
            s.user_edit(i, W[i])
        if not s.open_flags():
            break
    assert not s.open_flags()
    assert same_score(s.states[30], S[30])


def test_inconsistent_anchor_leaves_points_and_flags():
    s = ForwardScorer(FMT, first_server="A", ends0=0)
    for _ in range(3):
        s.point_ended(PointEvidence(p_a=0.9))                     # engine: A wins 3 -> 40-0
    before = pickle.dumps((s.commits, s.states[:3]))
    st = initial_state(FMT, "A", 0)
    for w in "ABB":
        st = advance(st, w, FMT)                                  # user types 15-30
    flag = s.set_anchor(st)
    assert flag is not None and flag.message().startswith("points 1-3")
    assert pickle.dumps((s.commits, s.states[:3])) == before
    assert same_score(s.state, st)
    s.point_ended(PointEvidence(p_a=0.9))
    assert same_score(s.states[3], st) and s.states[4] == advance(st, "A", FMT)


def test_forward_module_never_uses_global_decoder_or_histories():
    src = open(os.path.join(ROOT, "opencourtvision", "scoring", "forward.py")).read()
    tree = ast.parse(src)
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("decoder" in (m or "") for m in mods)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"decode", "confirm_loop", "beam", "forward_backward", "viterbi", "alpha", "vit"}


# ---- suggested backward fixes (user-confirmed only; Antony via Legal, 05:28) ----

@pytest.mark.parametrize("seed", range(40))
def test_suggestions_never_change_committed_points_without_user_action(seed):
    rng = random.Random(5000 + seed)
    W, S, ev = random_match(rng, rng.randint(10, 120))
    s = ForwardScorer(FMT, suggestions=True)              # no user present: suggestions are only shown
    seen = []
    for e in ev:
        s.point_ended(e)
        k = len(s.commits)
        for kk, b in seen:
            assert snap(s, kk) == b
        seen.append((k, snap(s, k)))
    assert not s.edits


def _conflicted():
    """Engine commits A,A,A (40-0); the players then call '15-30' -> one suggestion."""
    s = ForwardScorer(FMT, first_server="A", ends0=0, suggestions=True)
    for c in (0.9, 0.55, 0.6):
        s.point_ended(PointEvidence(p_a=c))
    return s


def test_suggestion_leaves_state_untouched_until_approved():
    import copy
    s = _conflicted()
    before = pickle.dumps((s.commits, s.states))
    sg = s.suggest("call", ("call", frozenset({"15-30"})))
    assert sg is not None and len(sg.fix) == 2 and "look wrong" in sg.message
    assert {j for j, _ in sg.fix} == {1, 2}                       # the two lowest-confidence points
    assert pickle.dumps((s.commits, s.states)) == before         # nothing changed
    s2 = copy.deepcopy(s)
    s.approve(sg)
    assert pickle.dumps((s.commits, s.states)) != before
    assert all(e.source == "suggestion" for e in s.edits) and len(s.edits) == 2
    s2.decline(s2.pending_suggestions[0])
    assert pickle.dumps((s2.commits, s2.states)) == before and not s2.edits


def test_approving_a_suggestion_equals_the_same_manual_edits():
    import copy
    s = _conflicted()
    sg = s.suggest("call", ("call", frozenset({"15-30"})))
    m = copy.deepcopy(s)
    s.approve(sg)
    for j, w in sg.fix:
        m.user_edit(j, w)
    assert s.winners == m.winners and s.states == m.states
    assert score_call(s.state, FMT) == "15-30"


def test_anchor_suggestion_is_one_fix_and_waits_for_approval():
    s = ForwardScorer(FMT, first_server="A", ends0=0, suggestions=True)
    for c in (0.95, 0.6, 0.9):
        s.point_ended(PointEvidence(p_a=c))                       # A A A -> 40-0
    st = initial_state(FMT, "A", 0)
    for w in "ABA":
        st = advance(st, w, FMT)                                  # user types 30-15
    before = pickle.dumps(s.commits)
    f = s.set_anchor(st)
    assert f.proposed is not None and f.proposed.fix == ((1, "B"),)
    assert pickle.dumps(s.commits) == before and not f.resolved
    s.approve(f.proposed)
    assert f.resolved and s.winners == ["A", "B", "A"]


@pytest.mark.parametrize("seed", range(30))
def test_with_user_every_change_is_an_explicit_user_action(seed):
    rng = random.Random(9000 + seed)
    W, S, ev = random_match(rng, rng.randint(10, 120))
    s = ForwardScorer(FMT, suggestions=True, tau=0.8,
                      ask=lambda x, k: (all(W[j] == w for j, w in x.fix) if k == "suggestion" else W[x]))
    prev = []
    for e in ev:
        n_ed = len(s.edits)
        s.point_ended(e)
        changed = {i for i, c in enumerate(prev) if s.commits[i].choice != c}
        assert changed <= {x.idx for x in s.edits[n_ed:]}
        assert all(x.source in ("suggestion", "tap") for x in s.edits)
        prev = [c.choice for c in s.commits]


# ---- Legal rulings 05:41: Q3 score-check prompts and Q2 spoken score calls (shipped), Q1 disabled ----

def _call(v, acc=0.95, p_src=0.9):
    return Obs(v, acc, 20, p_src=p_src)


def _two_points(**kw):
    s = ForwardScorer(FMT, first_server="A", ends0=0, **kw)
    s.point_ended(PointEvidence(p_a=0.95))
    s.point_ended(PointEvidence(p_a=0.6))                         # engine: 30-0
    return s


def test_q3a_prompt_decision_uses_only_the_shown_state():
    import math
    calls = []
    s = _two_points(score_checks=0.05, ask=lambda pr, k: ("keep",))
    s.suggest = lambda *a, **k: calls.append(a) or None          # spy: the fix search must not decide the prompt
    ok = PointEvidence(p_a=0.5, call=_call("30-0"))
    assert s._shown_fit(ok) > math.log(0.05)
    s.point_ended(ok)
    assert not calls and not any(k == "score_check" for _, k in s.prompts)
    bad = PointEvidence(p_a=0.5, call=_call("15-30"))
    expect = s._shown_fit(bad) < math.log(0.05) or score_call(s.state, FMT) != "15-30"   # one shown state only
    s.point_ended(bad)
    assert expect and any(k == "score_check" for _, k in s.prompts)


def test_q3b_c_at_most_one_fix_and_no_change_without_answer():
    seen = []
    s = _two_points(score_checks=0.05, ask=lambda pr, k: seen.append(pr) or None)   # never answers (timeout)
    before = pickle.dumps((s.commits[:2], s.states[:3]))
    s.point_ended(PointEvidence(p_a=0.5, call=_call("15-15")))
    assert len(seen) == 1 and seen[0].fix is not None and len(seen[0].fix.fix) == 1   # one obvious fix
    assert pickle.dumps((s.commits[:2], s.states[:3])) == before                     # unanswered: unchanged
    assert [c.choice for c in s.commits[:2]] == ["A", "A"] and not s.edits
    assert s.log[-1]["response"] == "no_answer"


def test_q3d_e_options_and_logged_approval():
    s = _two_points(score_checks=0.05, ask=lambda pr, k: ("fix",) if "fix" in pr.options else ("keep",))
    s.point_ended(PointEvidence(p_a=0.5, call=_call("15-15")))
    assert s.log[-1]["kind"] == "score_check" and s.log[-1]["response"] == "fix" and s.log[-1]["fix"] == ((1, "B"),)
    assert s.winners[:2] == ["A", "B"] and all(e.source == "suggestion" for e in s.edits)
    st = initial_state(FMT, "A", 0)
    for w in "BB":
        st = advance(st, w, FMT)
    s2 = _two_points(score_checks=0.05, ask=lambda pr, k: ("edit", st))         # free edit
    s2.point_ended(PointEvidence(p_a=0.5, call=_call("0-30")))
    assert same_score(s2.states[2], st) and s2.log[-1]["response"] == "edit"
    offered = []
    s3 = _two_points(score_checks=0.05, ask=lambda pr, k: offered.append(pr.options) or ("keep",))
    s3.point_ended(PointEvidence(p_a=0.5, call=_call("15-15")))
    assert s3.winners[:2] == ["A", "A"] and s3.log[-1]["response"] == "keep"
    assert offered == [("fix", "keep", "edit")]                   # (e) fix + keep + free edit


def test_q2b_call_changing_only_last_point_is_a_user_edit_with_feedback_and_undo():
    s = _two_points(voice_edits=True)
    s.point_ended(PointEvidence(p_a=0.5, call=_call("15-15")))    # flip of point 2 only
    assert s.winners[1] == "B" and s.edits[-1].source == "voice"
    assert s.feedback[-1]["msg"] == "Heard 15-15, updated"
    assert s.undo_voice() and s.winners[1] == "A" and s.edits[-1].source == "undo"


def test_q2_older_conflict_goes_to_q3_prompt_without_reanchoring():
    seen = []
    s = _two_points(voice_edits=True, score_checks=0.1, ask=lambda pr, k: seen.append(pr) or None)
    before = pickle.dumps((s.commits, s.states))
    s.point_ended(PointEvidence(p_a=0.9, call=_call("0-30")))     # needs both earlier points flipped
    assert pickle.dumps((s.commits[:2], s.states[:3])) == before  # never silently rewritten
    assert not s.anchors and not s.edits                         # no re-anchoring (Legal 05:55)
    assert len(seen) == 1 and s.log[-1]["kind"] == "score_check"  # sent to the Q3 tap prompt
    assert any(x.get("response") == "sent_to_q3" for x in s.log)
    assert s.states[3] == advance(s.states[2], "A", FMT)          # scoring continues from the unchanged score


def test_q2d_only_clear_gated_score_calls_count():
    for c in (_call("15-15", p_src=0.3), _call("15-15", acc=0.5)):
        s = _two_points(voice_edits=True)
        s.point_ended(PointEvidence(p_a=0.5, call=c))
        assert not s.edits and not s.anchors and s.winners[:2] == ["A", "A"]


def test_q1_hold_until_next_serve_is_disabled_in_every_profile():
    from opencourtvision.editloop.build import flags
    from opencourtvision.scoring.engine import make_scorer
    for r in ("US", "UK", "ES", "ROW"):
        assert make_scorer(flags(r), FMT).policy == "point_end"
        assert make_scorer(flags(r, pending_commit=True), FMT).policy == "point_end"
