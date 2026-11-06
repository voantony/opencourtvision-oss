"""Camera v4.6 (Engineer; MD reel v4.5 review of 0c59fa6 + far-partner concern from 37302d7): pre-serve framing (server
kept in, slow, no reversal, opening frame on the server), lead to the returner after a serve, source-edge
infeasibility, far-partner soft keep-in, reel-level framing checks. Real-data before/after: docs/engineering_notes.md §2.8.4b."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from opencourtvision.reel import camera as C  # noqa: E402
from tests.test_camera_v43 import FPS, SW, S, _x  # noqa: E402


def _serve_clip(pre=3.0, rally=(1.3, 2.4), dur=None, server_x=(330, 370), noise=True):
    """Doubles clip opening `pre` s before a near serve (server deep at server_x, partner at the net), far return
    rally[0] s after it, near volley rally[1] s after it. Pre-serve 'ball' detections jump around (bounces + false
    positives) to drag a ball-led camera. Analysis px."""
    ts_c = [pre, pre + rally[0], pre + rally[1]]
    dur = dur or ts_c[-1] + 1.5; n = int(dur * FPS); ts = np.arange(n) / FPS
    boxes = [[(server_x[0], server_x[1], 1, 520, 710, 0), (520, 560, 1, 430, 600, 0),
              (760, 790, 1, 200, 262, 1), (560, 585, 1, 205, 262, 1)] for _ in range(n)]
    xs = [sum(server_x) / 2, 775, 540]; ys = [600, 230, 520]
    bx = np.interp(ts, ts_c, xs); by = np.interp(ts, ts_c, ys)
    if noise:
        pre_f = ts < pre
        bx[pre_f] = np.where((np.arange(pre_f.sum()) // 40) % 2 == 0, 1100.0, 300.0)   # 0.67 s alternating false detections
    return boxes, np.stack([bx, by], 1), bx * S, ts_c


def _y(n):
    return np.zeros(n, int), np.full(n, 2160)


def test_serve_role_needs_a_pre_serve_phase():
    boxes, bxy, bx, ct = _serve_clip(pre=0.5)
    roles = C.contact_roles(boxes, bxy, ct, 0.0, FPS)
    assert C.serve_role(roles, 0.0) is None                                   # starts 0.5 s before: a rally cut
    boxes, bxy, bx, ct = _serve_clip()
    roles = C.contact_roles(boxes, bxy, ct, 0.0, FPS)
    assert C.serve_role(roles, 0.0) is roles[0]
    assert C.serve_role(roles, 0.0, kinds=["forehand", None, None]) is None
    assert C.serve_role(roles, 0.0, dict(C.CAMERA, preserve=False)) is None
    st = C.server_track(boxes, roles[0])
    assert min(st) == 0 and max(st) == roles[0]["i"] and st[0][0] == 330      # back to the first frame, the deep player


def test_preserve_frames_the_server_still_no_reversal():
    boxes, bxy, bx, ct = _serve_clip()
    n = len(boxes); y, ch = _y(n)
    old = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S, cfg=dict(C.CAMERA, preserve=False, far_keep=False))
    new = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S)
    q_old = C.preserve_qa(dict(old, serve_i=new["serve_i"], server=new["server"]), _x(old), y, old["cw"], ch, S, FPS, 0.0)
    q_new = C.preserve_qa(new, _x(new), y, new["cw"], ch, S, FPS, 0.0)
    assert q_old["reversals"] > 0 or q_old["server_out_frames"] > 0           # the ball-led v4.5 camera chases the noise
    assert q_new["reversals"] == 0 and q_new["opening_has_server"] is True and q_new["server_out_frames"] == 0
    i_s = new["serve_i"]; v = np.abs(np.diff(new["cx"][:i_s - C.gate_win(FPS)])) * FPS / new["cw"][1:i_s - C.gate_win(FPS)]
    assert v.max() <= C.CAMERA["preserve_pan_max"] + 1e-6                     # slow
    assert new["feasible"] and all(r for r, g in zip(new["reached"], new["gates"]) if g[5])
    q = C.role_clip_qa(new["roles"], _x(new), y, new["cw"], ch, S, SW, FPS, 0.0)
    assert q["pass"]
    vv, aa, quick = new["caps"]
    assert vv <= 1 + 1e-6 and aa <= 1 + 1e-6 and not quick


def test_serve_lead_reaches_the_returner_earlier():
    boxes, bxy, bx, ct = _serve_clip(noise=False)
    a = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S, cfg=dict(C.CAMERA, serve_lead_s=0.35, far_keep=False))
    b = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S, cfg=dict(C.CAMERA, far_keep=False))
    g = [g_ for g_ in b["gates"] if g_[5]][1]                                 # the return gate
    lo, hi = g[1], g[2]
    def first_in(p): return next(i for i in range(len(p["cx"])) if all(lo - 1 <= p["cx"][j] <= hi + 1 for j in range(i, g[0] + 1)))
    assert first_in(b) <= first_in(a)


def test_source_edge_check_flags_a_runner_leaving_the_frame():
    boxes, bxy, bx, ct = _serve_clip(noise=False)
    roles = C.contact_roles(boxes, bxy, ct, 0.0, FPS)
    assert C.source_edge_check(roles, S, SW, FPS, 0.0) == []
    r = roles[1]                                                               # far returner runs off the right edge
    for i in list(r["hitter"])[:]:
        b = r["hitter"][i]; r["hitter"][i] = (b[0] + 340, 1280, b[2], b[3], b[4], 1)
    e = C.source_edge_check(roles, S, SW, FPS, 0.0)
    assert len(e) == 1 and e[0][0] <= r["t"] <= e[0][1]
    plan = dict(feasible=True, infeasible=[], source_edge=e)
    act, why = C.infeasible_decision(plan, 0, dict(C.CAMERA, infeasible_action="wide", max_wide=1))
    assert act == "reject" and "source frame" in why                           # wide shows the same frame: reject


def test_far_keep_reduces_far_partner_out():
    boxes, bxy, bx, ct = _serve_clip(noise=False)
    off = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S, cfg=dict(C.CAMERA, far_keep=False))
    on = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S)
    f_off = C.partner_stats(boxes, _x(off), off["cw"], S)["far_out"].mean()
    f_on = C.partner_stats(boxes, _x(on), on["cw"], S)["far_out"].mean()
    assert f_on < f_off
    assert on["h"].mean() <= off["h"].mean() + 1e-6                           # no widening for it
    assert all(r for r, g in zip(on["reached"], on["gates"]) if g[5])


def test_partner_stats_infers_side_without_field():
    boxes = [[(100, 120, 1, 500, 700), (900, 920, 1, 200, 260)]] * 3
    st = C.partner_stats(boxes, np.array([200, 2600, 2720]), np.full(3, 1216), S)
    assert st["far_out"].tolist() == [True, False, True]


def test_framing_checks():
    pq = dict(reversals=0, opening_has_server=True, server_out_frames=0, server_out_span=None)
    ok = C.framing_checks({"a": dict(preserve=pq, source_edge=[]), "b": dict(preserve=None, source_edge=[])}, 30.0)
    assert all(v["pass"] for v in ok.values())
    bad = C.framing_checks({"a": dict(preserve=dict(pq, reversals=3, opening_has_server=False, server_out_frames=60,
                                                    server_out_span=[1.0, 2.0]), source_edge=[(3.0, 3.5)])}, 55.0)
    assert not any(v["pass"] for v in bad.values())


def test_gate_plan_slow_and_mono_windows():
    n, cw = 240, np.full(240, 1216.0)
    gates = [(200, 2900, 3000, 2950, 50, True)]
    x, _, _ = C.gate_plan(gates, n, FPS, cw, SW, keep=[(0, 600, 700, 500.0)], slow=[(0, 150, 0.25)], mono=[(0, 200, 1)])
    v = np.diff(x) * FPS / cw[1:]
    assert v[:149].max() <= 0.25 + 1e-6 and v[:199].min() >= -1e-6


def _p101_clip(server=True, bench_blob=True):
    """MD P101 (match4 reel v4.5, 5162.23-5173.31): far server deep right (analysis x ~700-716), near returner far left
    (209-370) 1.14 s after the serve, far volley 2.44 s later. Motion blobs never saw the server (still, small) but did
    see a deeper blob on the left benches; the toss ball sits over the server."""
    pre, ts_c = 3.5, [3.5, 4.64, 7.08]; n = int(8.5 * FPS); ts = np.arange(n) / FPS
    fr = [(209, 370, 1, 250, 475, 0), (520, 580, 1, 300, 470, 0), (610, 627, 1, 168, 218, 1)]
    if server: fr.append((699, 716, 1, 160, 207, 1))
    if bench_blob: fr.append((476, 503, 1, 137, 169, 1))
    boxes = [list(fr) for _ in range(n)]
    xs, ys = [705, 290, 618], [150, 400, 190]
    bx = np.interp(ts, ts_c, xs); by = np.interp(ts, ts_c, ys)
    pre_f = ts < pre - 1.0
    bx[pre_f] = np.nan; by[pre_f] = np.nan                                    # only the toss is seen before the serve
    return boxes, np.stack([bx, by], 1), bx * S, ts_c, ["far", None, None], ["serve", None, None]


def test_p101_server_identity_from_side_and_toss():
    boxes, bxy, bx, ct, sides, kinds = _p101_clip()
    roles = C.contact_roles(boxes, bxy, ct, 0.0, FPS, sides=sides, kinds=kinds)
    chk = roles[0]["server_check"]
    assert chk["ok"] and chk["box"][0] == 699                                   # not the deeper bench blob at 476
    assert C.serve_role(roles, 0.0, kinds=kinds) is roles[0]


def test_p101_unidentified_server_fails_the_check():
    """d09205a regression: the planner followed a wrong (near / bench) blob as 'server' and preserve_qa, using the same
    wrong track, reported 0 frames out. Now: no far box near the toss -> not identified, no pre-serve planning on a
    guess, and the reel check fails."""
    boxes, bxy, bx, ct, sides, kinds = _p101_clip(server=False)
    roles = C.contact_roles(boxes, bxy, ct, 0.0, FPS, sides=sides, kinds=kinds)
    assert roles[0]["server_check"]["ok"] is False and roles[0]["server_check"]["ball_ok"] is False
    assert C.serve_role(roles, 0.0, kinds=kinds) is None
    p = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S, sides=sides, kinds=kinds)
    n = len(boxes); y, ch = _y(n)
    q = C.preserve_qa(p, _x(p), y, p["cw"], ch, S, FPS, 0.0)
    assert q is not None and q["server_identified"] is False and q["opening_has_server"] is None
    fc = C.framing_checks({"w4_P101": dict(preserve=q, source_edge=[])}, 30.0)
    assert not fc["server_in_crop_preserve_check"]["pass"] and not fc["opening_frame_server_check"]["pass"]


def test_p101_far_server_right_near_returner_left_is_held():
    boxes, bxy, bx, ct, sides, kinds = _p101_clip()
    p = C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S, sides=sides, kinds=kinds)
    n = len(boxes); y, ch = _y(n)
    q = C.preserve_qa(p, _x(p), y, p["cw"], ch, S, FPS, 0.0)
    assert q["server_identified"] and q["opening_has_server"] is True and q["server_out_frames"] == 0 and q["reversals"] == 0
    assert C.role_clip_qa(p["roles"], _x(p), y, p["cw"], ch, S, SW, FPS, 0.0)["pass"]
    assert p["feasible"]
