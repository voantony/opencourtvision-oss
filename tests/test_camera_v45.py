"""Camera v4.5 (Engineer, md-v4 crop_height_check): gate spread fix + hysteresis, graduated widening, no player-framing
widen when gates frame the players, far-partner stats. Real-data before/after: docs/engineering_notes.md §2.8.4b."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from opencourtvision.reel import camera as C  # noqa: E402
from tests.test_camera_v43 import FPS, SW, S, _clip, _x  # noqa: E402


def _roles(boxes, bxy, contacts):
    return C.contact_roles(boxes, bxy, contacts, 0.0, FPS)


def test_gate_spread_never_unions_hitter_and_receiver_across_the_flight():
    boxes, bxy, bx = _clip(1.0, 2.0, xn=(100, 140), xf=(1100, 1140))      # near hitter, far receiver 3000 src px apart
    roles = _roles(boxes, bxy, [1.0, 2.0]); n = len(boxes)
    Lr, Rr, _ = C.relevant_span(n, roles, bxy[:, 0], 0.0, FPS)
    old = np.nan_to_num((Rr - Lr) * S)
    new = C.gate_spread(n, roles, bx, 0.0, FPS, S)
    assert old.max() > 2500                                                # v4.1: across the court -> full height
    assert new.max() < 400                                                 # v4.5: one player (+ ball) at a time
    assert new[int(1.5 * FPS)] == 0                                        # mid-flight: no zoom need, the pan moves


def test_gate_spread_hysteresis_and_ball_reach():
    boxes, bxy, bx = _clip(1.0, 2.0)
    roles = _roles(boxes, bxy, [1.0, 2.0]); n = len(boxes)
    sp = C.gate_spread(n, roles, bx, 0.0, FPS, S)
    win, hold = C.gate_win(FPS), int(round(0.5 * FPS / 2))
    i = roles[0]["i"]
    assert sp[i + win + hold] > 0 and sp[i + win + hold + 2] == 0            # held 0.25 s past the gate window, then released
    far = bx.copy(); far[i - win:i + win + 1] = 3700                       # a misdetected ball across the court is ignored
    assert C.gate_spread(n, roles, far, 0.0, FPS, S).max() <= sp.max() + 1e-6


def test_widen_request_steps_and_counts_soft_gates():
    n, cw = 200, np.full(200, 1216.0); h = np.full(n, 0.75); cx = np.full(n, 1000.0)
    gates = [(100, 2500, 2600, 2600, 100, False)]                           # a soft gate the path misses by far
    r0 = C.widen_request(gates, cx, cw, h, n, FPS, SW, soft=False)
    assert r0[1] == 0
    req, viol, inf, missed = C.widen_request(gates, cx, cw, h, n, FPS, SW, soft=True)
    assert viol == 1 and not inf and missed
    assert np.exp(req[100]) == pytest.approx(0.75 * 1.1)                     # <= 10 % per re-plan, not straight to full
    req2 = C.widen_request(gates, cx, cw, h, n, FPS, SW, cfg=dict(C.CAMERA, widen_step=None), soft=True)[0]
    assert np.exp(req2[100]) == pytest.approx(1.0)


def test_plan_clip_v45_holds_gates_tighter_than_v41():
    boxes, bxy, bx = _clip(1.0, 3.0)
    fb = np.ones(len(boxes), bool)                                          # ball lost everywhere (player framing)
    v41 = dict(C.CAMERA, spread_mode="relevant", widen_soft=False, widen_step=None, fallback_wide_with_gates=True, w_ball=1.0)
    p_old = C.plan_clip(boxes, bxy, bx, [1.0, 3.0], 0.0, FPS, scale=S, fallback=fb, cfg=v41)
    v45 = dict(C.CAMERA, preserve=False, far_keep=False)                   # v4.5 behaviour only (v4.6 features: test_camera_v46)
    p_new = C.plan_clip(boxes, bxy, bx, [1.0, 3.0], 0.0, FPS, scale=S, fallback=fb, cfg=v45)
    assert p_new["feasible"] and all(p_new["reached"])
    assert p_new["h"].mean() < p_old["h"].mean() - 0.05
    v, a, quick = p_new["caps"]
    assert v <= 1 + 1e-6 and a <= 1 + 1e-6 and not quick
    n = len(boxes)
    q = C.role_clip_qa(p_new["roles"], _x(p_new), np.zeros(n, int), p_new["cw"], np.full(n, 2160), S, SW, FPS, 0.0)
    assert q["pass"]


def test_partner_stats():
    boxes = [[(100, 120, 1, 500, 700, 0), (900, 920, 1, 200, 260, 1)]] * 3
    x, cw = np.array([200, 2600, 2720]), np.full(3, 1216)
    st = C.partner_stats(boxes, x, cw, S)
    assert st["far_out"].tolist() == [True, False, True] and st["any_out"].all()
    assert st["sliced"].tolist() == [False, False, True]                    # 2700-2760 cut by the left edge at 2720
    assert not C.partner_stats(boxes, x, cw, S, wide=True)["any_out"].any()
