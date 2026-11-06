import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "point_winner"))
import pw_stages as S  # noqa: E402


def test_taps_ask_lowest_confidence_first():
    conf = np.array([0.55, 0.9, 0.6, 0.99, 0.7])
    correct = np.array([False, True, False, True, True])
    assert S.taps_to(conf, correct, 0.999) == 2          # the two wrong calls are the two least confident
    assert S.taps_to(conf, correct, 0.5) == 0            # 2 errors allowed of 5


def test_ends_change_after_odd_games_and_carry_across_sets():
    assert S.ends_flips(dict(games=[0, 0], set_games_done=[])) == 0
    assert S.ends_flips(dict(games=[1, 0], set_games_done=[])) == 1
    assert S.ends_flips(dict(games=[1, 1], set_games_done=[])) == 1
    assert S.ends_flips(dict(games=[2, 1], set_games_done=[])) == 2
    assert S.ends_flips(dict(games=[0, 0], set_games_done=[7])) == 4   # 6-1: changes after games 1, 3, 5 and 7


def test_ridge_lr_learns_a_separable_sign():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200); y = (x > 0).astype(float)
    p = S.fit_predict(x[:, None], y, np.array([[2.0], [-2.0]]), 1.0)
    assert p[0] > 0.9 and p[1] < 0.1
