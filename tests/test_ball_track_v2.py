"""Ball tracker v2: rule fusion, player-attention prior, cascade mask, learned-model plumbing (synthetic data only)."""
import importlib.util
import os

import numpy as np
import pytest

from opencourtvision.ball import attention as AT
from opencourtvision.ball import learned as L
from opencourtvision.ball.fuse import fuse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cands(T=20, K=4):
    return np.full((T, K, 3), np.nan)


def test_fuse_agreement_raises_peak_and_counts():
    a = _cands(); b = _cands()
    a[:, 0] = (100, 100, 0.6); b[:, 0] = (103, 101, 0.6)      # same blob, 3 px apart
    b[:, 1] = (500, 300, 0.9)                                   # seen by b only
    f, ag = fuse([a, b])
    j = int(np.nanargmin(np.abs(f[0, :, 0] - 101)))
    assert ag[0, j] == 2 and f[0, j, 2] == pytest.approx(1 - 0.4 * 0.4)
    assert 100 <= f[0, j, 0] <= 103
    assert f[0, 0, 2] == pytest.approx(0.9) and ag[0, 0] == 1


def test_fuse_keeps_single_detector_peak():
    a = _cands(); a[:, 0] = (10, 10, 0.3)
    f, ag = fuse([a, _cands()])
    assert f[0, 0, 2] == pytest.approx(0.3) and ag[0, 0] == 1


def _player(cx, cy, face=(0.0, -1.0), h=60.0, wrist=None):
    kp = np.zeros((17, 3)); kp[:, 2] = 0.9
    head = np.array([cx, cy - h / 2]); kp[AT.LEAR, :2] = head; kp[AT.REAR, :2] = head
    kp[AT.NOSE, :2] = head + 5 * np.asarray(face); kp[AT.LEYE, :2] = kp[AT.NOSE, :2]; kp[AT.REYE, :2] = kp[AT.NOSE, :2]
    kp[AT.LSH, :2] = (cx - 8, cy - h / 3); kp[AT.RSH, :2] = (cx + 8, cy - h / 3)
    kp[AT.LWR, :2] = wrist if wrist is not None else (cx - 10, cy); kp[AT.RWR, :2] = kp[AT.LWR, :2]
    return kp, np.array([cx - 15, cy - h / 2, cx + 15, cy + h / 2, 0.9])


def test_attention_lone_candidate_is_neutral_and_static_is_downweighted():
    T = 20; c = _cands(T, 2)
    c[:, 0] = np.c_[np.linspace(100, 400, T), np.full(T, 200), np.full(T, 0.6)]    # moving ball
    ps = [[_player(300, 500)] for _ in range(T)]
    F = AT.features(c, ps)
    r = AT.reweight(c, F, ps, use=("gaze",))
    assert r[5, 0, 2] == pytest.approx(0.6)                        # one candidate: gaze cannot penalise it
    c[:, 1] = (900, 650, 0.7)                                      # a ball lying still in a corner
    F = AT.features(c, ps); r = AT.reweight(c, F, ps, use=("static",))
    assert F[10, 1, 2] == pytest.approx(1.0) and F[10, 0, 2] < 0.2
    assert r[10, 0, 0] == pytest.approx(c[10, 0, 0])              # the moving ball now ranks first


def test_gaze_prefers_the_watched_candidate():
    T = 3; c = _cands(T, 2)
    c[:, 0] = (300, 100, 0.5); c[:, 1] = (300, 700, 0.55)          # up (watched) vs down
    ps = [[_player(250, 400), _player(350, 400)] for _ in range(T)]
    F = AT.features(c, ps, y_net=300)
    assert F[1, 0, 0] > F[1, 1, 0]
    r = AT.reweight(c, F, ps, use=("gaze",))
    assert r[1, 0, 1] == pytest.approx(100)


def test_gaze_point_meets_rays():
    rays = [(np.array([0.0, 0.0]), np.array([1.0, 1.0]) / np.sqrt(2)), (np.array([10.0, 0.0]), np.array([-1.0, 1.0]) / np.sqrt(2))]
    x = AT.gaze_point(rays)
    assert x == pytest.approx([5.0, 5.0], abs=1e-6)


def test_alternation():
    assert AT.alternation_ok([(1, 1), (10, -1), (20, 1)]) == 1.0
    assert AT.alternation_ok([(1, 1), (10, 1)]) == 0.0


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_cascade_escalates_gaps_and_tail_in_whole_windows():
    cas = _load("cascade", "tools/ball_track_v2/cascade.py")
    conf = np.ones(30); conf[10:14] = 0                            # frames 11, 12 have no inlier within +-1
    m = cas.escalate_mask(30, conf, [20.0], "gap:1", win=3)
    assert m[9:15].all() and not m[:9].any() and not m[15:].any()
    m = cas.escalate_mask(30, np.ones(30), [20.0], "tail:0.1", win=3)
    assert m[18:].all() and not m[:15].any()


def test_cand_features_and_events():
    T = 12; c = _cands(T, 3); c[:, 0] = np.c_[np.linspace(100, 210, T), np.full(T, 300), np.full(T, 0.8)]
    X, v = L.cand_features(c, [c], None, np.full(T, -30.0))
    assert X.shape == (T, 3, L.CAND_F) and v[:, 0].all() and not v[:, 1].any()
    assert X[5, 0, 21] < 0.05                                       # on a constant-velocity line
    assert L.pick_events(np.array([0, 0.9, 0.95, 0, 0, 0, 0, 0, 0.8, 0]), 0.5, 3) == [2, 8]
    E = L.event_features(c[:, 0, :2], np.ones(T), [dict(kind="hit", i=3)], None, np.zeros(T), np.ones(T))
    assert E.shape == (T, L.EV_F) and E[3, 9] == 1.0


def test_learned_nets_forward():
    torch = pytest.importorskip("torch")
    net = L.build_candnet()
    y = net(torch.zeros(1, 10, 4, L.CAND_F), torch.ones(1, 10, 4))
    assert tuple(y.shape) == (1, 10, 4)
    ev = L.build_eventnet()
    assert tuple(ev(torch.zeros(1, L.EV_F, 50)).shape) == (1, 2, 50)
