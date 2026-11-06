import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "point_winner"))
import pw_rally as RL  # noqa: E402
import pw_rules as RU  # noqa: E402


class _View:                                    # image row 1000 = net line, 2000 = near baseline; x passes through
    def c2i(self, x, y):
        return np.array([x * 100.0, 2000.0 - (y / 11.885) * 1000.0])

    def i2c(self, xy):
        xy = np.atleast_2d(xy)
        return np.c_[xy[:, 0] / 100.0, (2000.0 - xy[:, 1]) / 1000.0 * 11.885]


def _track(contacts, t_end, feed=()):
    """sawtooth ball rows: far contact at -0.5 (row 500), near contact at +0.6 (row 1600); then flat feeding."""
    fps = 30.0; t = np.arange(0, t_end, 1 / fps); y = np.full(len(t), np.nan)
    for (ta, ya), (tb, yb) in zip(contacts[:-1], contacts[1:]):
        m = (t >= ta) & (t < tb); y[m] = ya + (yb - ya) * (t[m] - ta) / (tb - ta)
    for ta, tb in feed:                                     # slow, small drift: not a rally
        m = (t >= ta) & (t < tb); y[m] = -0.1 + 0.05 * np.sin(t[m])
    rows = 1000.0 + 1000.0 * y
    ball = np.c_[np.full(len(t), 3000.0), rows] / 6.0
    return dict(t0=0.0, fps=fps, ball=ball, ball_peak=np.where(np.isfinite(rows), 1.0, 0.0),
                pose_k=np.arange(0, len(t), 3), boxes=np.full((len(range(0, len(t), 3)), 8, 5), np.nan))


def test_rally_end_is_last_alternating_contact_not_post_point_feeding():
    shots = [(0.3, -0.15), (1.0, -0.5), (2.2, 0.6), (3.4, -0.5), (4.6, 0.6), (5.8, -0.5), (6.6, 0.0)]
    d = _track(shots, 20.0, feed=[(10.0, 14.0)])
    hits = [1.0, 2.2, 3.4, 4.6, 5.8, 11.0, 13.5]           # the two late hits are ball feeding after the point
    r = RL.rally(d, _View(), hits, 0.0, 20.0, use_players=False)
    assert abs(r["t_last"] - 5.8) < 0.1 and r["last_kind"] == -1
    assert r["first_kind"] == -1 and r["n_shots"] >= 4


def test_next_state_game_and_ends():
    s = dict(pts=[3, 1], games=[2, 2], sets=[0, 0])
    assert RU.next_state(s, "A", "A")[:3] == (True, "B", 0)          # 40-15 won by A: game, server changes
    assert RU.next_state(s, "B", "A")[:3] == (False, "A", 5)


def test_next_serve_rule_uses_changeover_and_box():
    s = dict(pts=[3, 1], games=[2, 2], sets=[0, 0])                   # game 5 if A wins -> odd total -> changeover
    assert RU.rule_next_serve(s, "A", 2, None, None, obs_change=True)[0] == "A"
    assert RU.rule_next_serve(s, "A", 2, None, "ad", obs_change=False)[0] == "B"   # 40-30: same server, ad court
    assert RU.rule_next_serve(dict(pts=[1, 1], games=[0, 0], sets=[0, 0]), "A", 0, 1, "deuce")[0] is None  # 15-15: no info


def test_serve_rules():
    r = dict(shots=[(1.0, -0.5, 1, True)], n_shots=1)                 # one shot from the near server, nothing back
    assert RU.rule_serve(r, "A", 1.0) == ("A", "ace_unreturned")
    r = dict(shots=[(1.0, -0.5, 1, True), (9.0, -0.5, 1, True)], n_shots=2)
    assert RU.rule_serve(r, "A", 1.0) == ("B", "double_fault")
    r = dict(shots=[(1.0, -0.5, 1, True), (2.2, 0.6, -1, True)], n_shots=2)
    assert RU.rule_serve(r, "A", 1.0) == (None, None)


def test_us_build_never_changes_a_call_from_the_next_serve():
    import pw_stages as S
    p5 = np.array([0.8, 0.3])
    calls = [[("B", "next_serve_gap")], [("B", "next_serve_gap")]]
    p_us, c_us = S.apply_hard(p5, calls, "US")
    assert np.allclose(p_us, p5) and c_us[0] == 0.0 and c_us[1] >= 0.98   # disagree -> ask first; agree -> confident
    p_eu, _ = S.apply_hard(p5, calls, "EU")
    assert p_eu[0] < 0.5 and p_eu[1] < 0.5
