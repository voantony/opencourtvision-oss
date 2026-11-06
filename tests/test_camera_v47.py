"""Camera v4.7 (MD 20-point A/B on a0e0a44): planner budget (never hang), safe_plan fallback (full height, caps,
clamped, no LP), locked wide broadcast shot (16:9), plan_clip_capped selection."""
import time

import numpy as np
import pytest

pytest.importorskip("scipy")
from opencourtvision.reel import camera as C  # noqa: E402
from tests.test_camera_v43 import FPS, SW, S, _x  # noqa: E402
from tests.test_camera_v46 import _serve_clip, _y  # noqa: E402


def test_planner_budget_raises_instead_of_hanging():
    boxes, bxy, bx, ct = _serve_clip()
    with pytest.raises(C.PlanTimeout):
        C.plan_clip(boxes, bxy, bx, ct, 0.0, FPS, scale=S, cfg=dict(C.CAMERA, _deadline=time.monotonic() - 1))


def test_capped_falls_back_to_safe_plan_on_timeout():
    boxes, bxy, bx, ct = _serve_clip()
    t = time.monotonic()
    gp = C.plan_clip_capped(boxes, bxy, bx, ct, 0.0, FPS, cfg=dict(C.CAMERA, plan_budget_s=0.0), scale=S)
    assert gp["planner"] == "safe_plan" and "budget" in gp["fallback"] and time.monotonic() - t < 10
    n = len(boxes); y, ch = _y(n)
    q = C.role_clip_qa(gp["roles"], _x(gp), y, gp["cw"], ch, S, SW, FPS, 0.0)
    assert q["pass"]                                                     # hitter + receiver framed at every contact
    pq = C.preserve_qa(gp, _x(gp), y, gp["cw"], ch, S, FPS, 0.0)
    assert pq["opening_has_server"] is True and pq["reversals"] == 0
    assert np.all(gp["cx"] >= gp["cw"] / 2) and np.all(gp["cx"] <= SW - gp["cw"] / 2)   # clamped to the source
    v = np.abs(np.diff(gp["cx"])) * FPS / gp["cw"][1:]
    assert v.max() <= C.CAMERA["pan_fast"] + 1e-6                       # pan caps hold


def test_safe_plan_flags_a_span_wider_than_the_crop():
    boxes, bxy, bx, ct = _serve_clip()
    for bb in boxes: bb[2] = (1240, 1270, 1, 200, 262, 1)               # far returner at the other side of the frame
    bxy = bxy.copy(); bxy[:, 0] = np.interp(np.arange(len(boxes)) / FPS, ct, [350, 1255, 540])
    gp = C.safe_plan(boxes, bxy, ct, 0.0, FPS, scale=S)
    assert not gp["feasible"] and gp["infeasible"]                      # honest: the 9:16 crop cannot hold both


def test_locked_wide_is_the_whole_frame_and_still():
    boxes, bxy, bx, ct = _serve_clip()
    gp = C.plan_clip_capped(boxes, bxy, bx, ct, 0.0, FPS, cfg=dict(C.CAMERA, locked_wide=True), scale=S)
    n = len(boxes)
    assert gp["planner"] == "locked_wide" and np.all(gp["cw"] == SW) and np.ptp(gp["cx"]) == 0
    x = np.zeros(n, int); y = np.zeros(n, int)
    assert C.role_clip_qa(gp["roles"], x, y, gp["cw"], gp["ch"], S, SW, FPS, 0.0)["pass"]
    assert C.preserve_qa(gp, x, y, gp["cw"], gp["ch"], S, FPS, 0.0)["opening_has_server"] is True


def test_l1_plan_holds_roles_server_caps_and_low_jerk():
    from opencourtvision.reel import ball
    boxes, bxy, bx, ct = _serve_clip()
    gp = C.plan_clip_capped(boxes, bxy, bx, ct, 0.0, FPS, cfg=dict(C.CAMERA, long_shot_mode="off"), scale=S)
    assert gp["planner"] == "l1_lp" and gp["feasible"] and gp["plan_s"] < 20
    n = len(boxes); x = _x(gp); y = np.zeros(n, int)
    assert C.role_clip_qa(gp["roles"], x, y, gp["cw"], gp["ch"], S, SW, FPS, 0.0)["pass"]
    pq = C.preserve_qa(gp, x, y, gp["cw"], gp["ch"], S, FPS, 0.0)
    assert pq["opening_has_server"] is True and pq["reversals"] == 0 and pq["server_out_frames"] == 0
    assert ball.jerk_p95(gp["cx"], SW)["px_per_frame3"] <= 0.05          # MD's A/B jerk threshold
    v = np.abs(np.diff(gp["cx"])) * FPS / gp["cw"][1:]
    assert v.max() <= C.CAMERA["pan_fast"] + 1e-6


def test_l1_plan_reports_geometry_it_cannot_hold():
    boxes, bxy, bx, ct = _serve_clip()
    for bb in boxes: bb[2] = (1240, 1270, 1, 200, 262, 1)
    bxy = bxy.copy(); bxy[:, 0] = np.interp(np.arange(len(boxes)) / FPS, ct, [350, 1255, 540])
    gp = C.l1_plan(boxes, bxy, bxy[:, 0] * S, ct, 0.0, FPS, scale=S)
    assert not gp["feasible"] and gp["infeasible"] and gp["slack_px"] > 0


def test_rts_fills_gaps_and_smooths():
    z = np.r_[np.linspace(0, 100, 30), np.full(20, np.nan), np.linspace(167, 200, 10)]
    zs = C.rts_smooth(z + np.random.default_rng(0).normal(0, 3, len(z)), 30)
    assert np.isfinite(zs).all() and abs(zs[40] - 133) < 15


def _modes():
    boxes, bxy, bx, ct = _serve_clip()
    out = {}
    for m in ("off", "cut", "fast_pan"):
        gp = C.plan_clip_capped(boxes, bxy, bx, ct, 0.0, FPS, cfg=dict(C.CAMERA, long_shot_mode=m), scale=S)
        x = _x(gp); out[m] = (gp, x, C.landing_qa(gp["shots"], x, gp["cw"], S, SW, FPS, 0.0))
    return boxes, out


def test_v49_cut_mode_cuts_to_the_receiving_side_before_the_bounce():
    boxes, out = _modes()
    gp, x, lq = out["cut"]
    assert gp["long_shot_mode"] == "cut" and gp["cuts"] and gp["long_shots"] >= 1
    assert lq["pass"] and lq["shots"] >= 1
    n = len(boxes)
    assert C.role_clip_qa(gp["roles"], x, np.zeros(n, int), gp["cw"], gp["ch"], S, SW, FPS, 0.0)["pass"]
    pq = C.preserve_qa(gp, x, np.zeros(n, int), gp["cw"], gp["ch"], S, FPS, 0.0)
    assert pq["opening_has_server"] is True and pq["reversals"] == 0
    assert C.jerk_p95_cuts(gp["cx"], SW, gp["cut_frames"])["px_per_frame3"] <= 0.05
    k = gp["cut_frames"][0]; s = next(s for s in gp["shots"] if s.get("cut"))
    assert s["i"] < k <= s["c0"]                                    # after the contact window, before the context window
    v = np.abs(np.diff(gp["cx"])) * FPS / gp["cw"][1:]; v[[c - 1 for c in gp["cut_frames"]]] = 0
    assert v.max() <= C.CAMERA["pan_fast"] + 1e-6                    # outside the cut, the normal caps hold


def test_v49_fast_pan_lifts_caps_only_in_flight_and_never_cuts():
    _, out = _modes()
    gp, x, lq = out["fast_pan"]
    assert not gp["cuts"] and lq["pass"]
    v = np.abs(np.diff(gp["cx"])) * FPS / gp["cw"][1:]
    assert v.max() <= C.CAMERA["pan_fast_flight"] + 1e-6
    assert v[~gp["fast"][1:]].max() <= C.CAMERA["pan_fast"] + 1e-6   # outside flight the normal caps hold


def test_landing_qa_wide_passes_and_reports_failures():
    _, out = _modes()
    gp = out["off"][0]
    assert C.landing_qa(gp["shots"], np.zeros(len(gp["cx"]), int), gp["cw"], S, SW, FPS, 0.0, wide=True)["pass"]
    assert not C.landing_qa(gp["shots"], np.zeros(len(gp["cx"]), int), gp["cw"], S, SW, FPS, 0.0)["pass"]
