"""Score decoder: exact rules state machine, Viterbi / forward-backward, selective prediction, callout gate (design §3-§5)."""
import itertools
import math
import random

import pytest

from opencourtvision.scoring.callout_gate import CalloutCandidate, flag_mismatches, gate_callout
from opencourtvision.scoring.callouts import parse_callout
from opencourtvision.scoring.decoder import (
    Anchor,
    Obs,
    PointEvidence,
    ScoreConflict,
    ScoreState,
    add_anchor,
    advance,
    anchor_conflicts,
    attach_anchors,
    canonical_call,
    choice_ll,
    confirm_loop,
    decode,
    independent_guess,
    initial_state,
    parse_anchor,
    replay,
    score_call,
    serve_side,
    state_ll,
)
from opencourtvision.scoring.formats import get_format, play_game, play_pickleball_game

TENNIS = get_format("tennis", "advantage")


def _match():
    """Seeded synthetic match (tests/synthetic_match.py), not a recording."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from synthetic_match import MATCH
    P = MATCH["points"]
    W = ["A" if p["winner"] == "host" else "B" for p in P]
    first = "A" if P[0]["server"] == "host" else "B"
    return P, W, first, MATCH


def test_replay_matches_a_hand_checked_game():
    """Guest serves; host wins four straight points. Scores are written out, not copied from a recording."""
    P = [
        {"server": "guest", "winner": "host", "host_pts": 0, "guest_pts": 0, "games_before": [0, 0], "sets_before": [0, 0]},
        {"server": "guest", "winner": "host", "host_pts": 1, "guest_pts": 0, "games_before": [0, 0], "sets_before": [0, 0]},
        {"server": "guest", "winner": "host", "host_pts": 2, "guest_pts": 0, "games_before": [0, 0], "sets_before": [0, 0]},
        {"server": "guest", "winner": "host", "host_pts": 3, "guest_pts": 0, "games_before": [0, 0], "sets_before": [0, 0]},
    ]
    W = ["A", "A", "A", "A"]
    S = replay(W, TENNIS, first_server="B")
    for p, s in zip(P, S):
        assert s.server == ("A" if p["server"] == "host" else "B")
        assert s.games == tuple(p["games_before"]) and s.sets == tuple(p["sets_before"])
        assert s.pts == (p["host_pts"], p["guest_pts"])
    assert S[-1].games == (1, 0) and S[-1].pts == (0, 0) and not S[-1].done


@pytest.mark.parametrize("rule", ["advantage", "golden_point", "star_point"])
def test_game_logic_matches_formats_play_game(rule):
    fmt = get_format("padel", {"advantage": "advantage", "golden_point": "golden_point", "star_point": "fip_star_point"}[rule])
    rng = random.Random(7)
    for _ in range(300):
        seq = [rng.choice("SR") for _ in range(30)]
        g = play_game(seq, rule)
        st = initial_state(fmt, "A")
        for w in seq[:g["points_played"]]:
            assert st.games == (0, 0)
            st = advance(st, "A" if w == "S" else "B", fmt)
        assert st.games == ((1, 0) if g["winner"] == "S" else (0, 1))


def test_star_point_and_golden_point_deciding_point_side_is_unobservable():
    fmt = get_format("padel", "fip_star_point")
    st = initial_state(fmt, "A")
    for w in "AAABBB" + "AB" + "AB":           # deuce 1 -> ad 1 -> deuce 2 -> ad 2 -> deuce 3
        st = advance(st, w, fmt)
    assert score_call(st, fmt) == "deuce 3" and serve_side(st, fmt) is None   # the receivers choose the side
    assert advance(st, "B", fmt).games == (0, 1)
    gold = get_format("padel", "golden_point")
    st = initial_state(gold, "A")
    for w in "AAABBB":
        st = advance(st, w, gold)
    assert serve_side(st, gold) is None and advance(st, "A", gold).games == (1, 0)


def test_tiebreak_serve_rotation_and_next_set_server():
    st = ScoreState(games=(6, 5), server="B")
    for _ in range(4):
        st = advance(st, "B", TENNIS)          # B holds: 6-6
    assert st.games == (6, 6) and st.tb == 7 and st.tb_first == "A" and st.server == "A"
    servers = []
    for w in "ABABABABABAA":                   # to 6-6 in the tiebreak, then A wins two
        servers.append(st.server)
        st = advance(st, w, TENNIS)
    assert servers[:7] == ["A", "B", "B", "A", "A", "B", "B"]
    assert st.sets == (1, 0) and st.server == "B"         # the tiebreak's first receiver serves first in the next set


def test_pickleball_side_out_matches_play_pickleball_game():
    fmt = get_format("pickleball", "usap_side_out_doubles")
    rng = random.Random(3)
    for _ in range(50):
        rallies = [rng.choice("AB") for _ in range(80)]
        ref = play_pickleball_game(rallies, fmt, first="A")
        st = initial_state(fmt, "A")
        for i, w in enumerate(rallies[:len(ref["calls"])]):
            assert score_call(st, fmt) == ref["calls"][i]
            st = advance(st, w, fmt)
        assert st.sets == ((1, 0) if ref["winner"] == "A" else (0, 1))


def _brute_force(evidence, fmt, first):
    """Exact posteriors by enumerating every winner sequence (no skips)."""
    tot, pa = 0.0, [0.0] * len(evidence)
    for seq in itertools.product("AB", repeat=len(evidence)):
        st, lp = initial_state(fmt, first), 0.0
        for ev, w in zip(evidence, seq):
            lp += state_ll(st, ev, fmt) + dict(choice_ll(st, ev))[w]
            st = advance(st, w, fmt)
        p = math.exp(lp); tot += p
        for i, w in enumerate(seq):
            pa[i] += p * (w == "A")
    return [x / tot for x in pa]


def test_forward_backward_equals_brute_force():
    rng = random.Random(11)
    ev = [PointEvidence(p_a=rng.uniform(0.2, 0.9), server=Obs(rng.choice("AB"), 0.8), side=Obs(rng.choice(["deuce", "ad"]), 0.7),
                        call=Obs(rng.choice(["15-0", "30-15", "deuce"]), 0.6, 20, p_src=0.7)) for _ in range(9)]
    d = decode(ev, TENNIS, first_server="A", ends0=0)
    for a, b in zip(d.p_a, _brute_force(ev, TENNIS, "A")):
        assert abs(a - b) < 1e-9


def test_perfect_evidence_decodes_the_truth():
    _, W, first, _m = _match()
    d = decode([PointEvidence(p_a=0.99 if w == "A" else 0.01) for w in W], TENNIS, first_server=first)
    assert d.winners == W and min(d.confidence) > 0.98


def test_next_server_fixes_a_wrong_point_guess():
    # A holds to love; the ball evidence wrongly favours B on point 4. B serving next is only possible if the game ended.
    ev = [PointEvidence(p_a=0.9)] * 3 + [PointEvidence(p_a=0.4), PointEvidence(p_a=0.5, server=Obs("B", 0.99))]
    assert independent_guess(ev)[3] == "B"
    d = decode(ev, TENNIS, first_server="A", ends0=0)
    assert d.winners[3] == "A" and d.confidence[3] > 0.9


def test_spoken_call_fixes_a_wrong_point_guess():
    call = Obs(canonical_call(parse_callout("30-0")), 0.95, 20)
    d = decode([PointEvidence(p_a=0.8), PointEvidence(p_a=0.45), PointEvidence(p_a=0.5, call=call)], TENNIS, first_server="A", ends0=0)
    assert d.winners[:2] == ["A", "A"]


def test_foreign_call_with_low_source_weight_cannot_flip_the_score():
    foreign = Obs("0-30", 0.95, 20, p_src=0.05)
    d = decode([PointEvidence(p_a=0.8), PointEvidence(p_a=0.8), PointEvidence(p_a=0.5, call=foreign)], TENNIS, first_server="A", ends0=0)
    assert d.winners[:2] == ["A", "A"]
    trusted = Obs("0-30", 0.95, 20)
    d = decode([PointEvidence(p_a=0.8), PointEvidence(p_a=0.8), PointEvidence(p_a=0.5, call=trusted)], TENNIS, first_server="A", ends0=0)
    assert d.winners[:2] == ["B", "B"]       # same call treated as ours does move it: the weight is what protects us


def test_let_segment_is_skipped():
    ev = [PointEvidence(p_a=0.9), PointEvidence(p_a=0.5, p_point=0.05), PointEvidence(p_a=0.9, side=Obs("ad", 0.95))]
    d = decode(ev, TENNIS, first_server="A", ends0=0)
    assert d.winners[1] == "skip" and d.p_skip[1] > 0.9
    assert all(0 <= a + s <= 1 + 1e-9 for a, s in zip(d.p_a, d.p_skip))


def test_confirm_loop_reaches_threshold_and_contradictions_raise():
    _, W, first, _m = _match()
    rng = random.Random(5)
    ev = [PointEvidence(p_a=(0.75 if (w == "A") == (rng.random() > 0.2) else 0.25)) for w in W]
    d, asked = confirm_loop(ev, TENNIS, W, 0.999, first_server=first)
    assert d.winners == W and min(d.confidence) >= 0.999 and 0 < len(asked) < len(W)
    with pytest.raises(ValueError):             # 48 confirmed points = 6-0 6-0; a 49th confirmed point is impossible
        decode([PointEvidence(confirmed="A")] * 49, TENNIS, first_server="A", ends0=0)


def test_final_score_constraint_and_canonical_calls():
    assert canonical_call(parse_callout("treinta quince")) == "30-15"
    assert canonical_call(parse_callout("40 iguales")) == "deuce"
    assert canonical_call(parse_callout("ventaja resto")) == "ad out"
    assert canonical_call(parse_callout("30-15")) == "30-15"     # Whisper writes digits
    _, W, first, match = _match()
    ev = [PointEvidence(p_a=0.6 if w == "A" else 0.4) for w in W]
    assert decode(ev, TENNIS, first_server=first, final=(2, 0)).states[-1].sets == (2, 0)


def test_callout_gate_rules():
    win, live = (10.0, 25.0), [(0.0, 9.0), (26.0, 40.0)]
    ours = CalloutCandidate(t=15, call="30-15", speaker_llr=2.0, distance_llr=0.5, speaker_team="A")
    r = gate_callout(ours, win, live, server_team="A", legal_calls={"30-15", "15-30"})
    assert r.accepted and r.p_src > 0.9 and r.obs.p_src == r.p_src
    assert gate_callout(CalloutCandidate(t=30, call="30-15"), win, live).reasons == ("during_our_live_rally",)
    assert gate_callout(CalloutCandidate(t=5.0, call="30-15"), win, []).reasons == ("outside_between_point_window",)
    far = CalloutCandidate(t=15, call="30-15", speaker_llr=-3.0, distance_llr=-1.0)
    assert gate_callout(far, win, live).reasons == ("source_not_our_players",)
    rcv = gate_callout(CalloutCandidate(t=15, call="30-15", speaker_llr=1.0, speaker_team="B"), win, live, server_team="A")
    assert rcv.accepted and "called_by_receiving_team" in rcv.reasons
    illegal = gate_callout(ours, win, live, server_team="A", legal_calls={"0-0"})
    assert not illegal.accepted and "not_a_legal_score_for_our_match" in illegal.reasons


def test_flag_mismatches_keeps_the_players_call():
    ev = [PointEvidence(p_a=0.97), PointEvidence(p_a=0.97), PointEvidence(p_a=0.5, call=Obs("15-15", 0.3, 20))]
    d = decode(ev, TENNIS, first_server="A", ends0=0)
    assert d.winners[:2] == ["A", "A"] and flag_mismatches(d, ev, TENNIS) == [(2, "15-15", "30-0")]


# ---------------------------------------------------------------- score anchors (MD v0.6)

def test_parse_anchor_us_them_with_serving_toggle():
    a = parse_anchor("Us 5 – Them 4, 30-15, we serve", 120.0)
    assert (a.games, a.pts, a.server, a.t) == ((5, 4), ("30", "15"), "us", 120.0)
    b = parse_anchor("sets 1-0, Us 2 - Them 1, deuce, serving: them", 5.0)
    assert (b.sets, b.games, b.pts, b.server) == ((1, 0), (2, 1), "deuce", "them")
    assert parse_anchor("ad us", 1.0).pts == "ad us" and parse_anchor("hello", 1.0) is None


def test_anchor_is_a_hard_constraint_both_ways():
    # weak, wrong evidence everywhere; one anchor "Us 1 - Them 0, 0-0, they serve" before point 5 forces A to win game 1
    ev = [PointEvidence(p_a=0.3)] * 8
    free = decode(ev, TENNIS, first_server="A", ends0=0)
    assert free.winners[:4] == ["B"] * 4
    d = decode(ev, TENNIS, first_server="A", ends0=0, anchors={4: Anchor(t=0, sets=(0, 0), games=(1, 0), pts=("0", "0"), server="them")})
    assert d.winners[:4] == ["A"] * 4 and d.states[4].games == (1, 0)    # backwards from the anchor
    assert d.winners[4:] == ["B"] * 4                                  # forwards: evidence still decides after it


def test_partial_anchor_and_final_set_scores():
    _, W, first, match = _match()
    set_scores = [tuple(int(x) for x in part.split("-")) for part in match["final_score"].split()]
    ev = [PointEvidence(p_a=0.55 if w == "A" else 0.45, server=Obs(s.server, 0.98)) for w, s in zip(W, replay(W, TENNIS, first_server=first))]
    d = decode(ev, TENNIS, final=tuple(match["final_sets"]), set_scores=set_scores)
    games_end_set1 = [s.games for s, n in zip(d.states, d.states[1:]) if n.sets == (1, 0) and s.sets == (0, 0)]
    won = 0 if set_scores[0][0] > set_scores[0][1] else 1
    before = list(set_scores[0])
    before[won] -= 1
    assert d.states[-1].sets == tuple(match["final_sets"]) and games_end_set1 and games_end_set1[0] == tuple(before)
    part = decode(ev, TENNIS, anchors={70: Anchor(t=0, sets=(1, 0))})
    assert part.states[70].sets == (1, 0)


def test_conflicting_anchors_are_reported_not_resolved():
    ev = [PointEvidence(p_a=0.5)] * 10
    anchors = {2: Anchor(t=2, games=(0, 0), pts=("30", "0")), 4: Anchor(t=4, games=(0, 0), pts=("0", "15"))}   # can't go 30-0 -> 0-15
    with pytest.raises(ScoreConflict):
        decode(ev, TENNIS, first_server="A", ends0=0, anchors=anchors)
    conf = anchor_conflicts(ev, TENNIS, anchors, first_server="A", ends0=0)
    assert conf and conf[0][0] == "anchor_anchor"
    with pytest.raises(ScoreConflict):
        attach_anchors([Anchor(t=10, games=(1, 0)), Anchor(t=10.2, games=(0, 1))], [0.0, 10.0, 20.0])


def test_anchor_vs_callout_conflict_is_shown():
    heard = Obs("0-30", 0.95, 20)                         # players called 0-30 (server first), the user anchored 30-0 for A
    ev = [PointEvidence(p_a=0.5), PointEvidence(p_a=0.5), PointEvidence(p_a=0.5, call=heard), PointEvidence(p_a=0.5)]
    anchors = {2: Anchor(t=2, games=(0, 0), pts=("30", "0"))}
    conf = anchor_conflicts(ev, TENNIS, anchors, first_server="A", ends0=0)
    assert ("anchor_call", anchors[2], 2, "0-30", "30-0") in conf


def test_add_anchor_reports_changes_and_next_check():
    _, W, first, _m = _match()
    rng = random.Random(9)
    ev = [PointEvidence(p_a=(0.7 if (w == "A") == (rng.random() > 0.25) else 0.3)) for w in W[:24]]
    S = replay(W, TENNIS, first_server=first)
    i = next(i for i, st in enumerate(S) if 8 <= i < 24 and max(st.pts) < 3 and not st.tb)
    st = S[i]
    new = Anchor(t=float(i), sets=st.sets, games=st.games, pts=tuple(["0", "15", "30", "40"][x] for x in st.pts), server="us" if st.server == "A" else "them")
    r = add_anchor(ev, TENNIS, {}, new, [float(k) for k in range(24)], first_server=first)
    assert not r["conflicts"] and r["decoded"].states[i].games == st.games
    assert r["summary"].startswith("Updated") and r["next_check"] is not None and ev[r["next_check"]].confirmed is None
