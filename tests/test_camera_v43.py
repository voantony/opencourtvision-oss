"""Camera v4.3 (Media Director patch on 303246d) + OpenCourtVision integration: gate planner, infeasibility rule,
calm zoom, doubles crop-height gate, hitter / receiver clipped gates."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from opencourtvision.reel import camera as C  # noqa: E402
from opencourtvision.reel.ball import jerk_p95  # noqa: E402

FPS, S, SW = 60, 3.0, 3840


def _clip(t1, t2, xn=(200, 240), xf=(800, 820), dur=4.5):
    """Doubles: two near players (side 0), two far (side 1); the near hitter at t1, the far hitter at t2 (analysis px)."""
    n = int(dur * FPS)
    boxes = [[(xn[0], xn[1], 1, 500, 700, 0), (xn[0] + 300, xn[1] + 300, 1, 480, 690, 0),
              (xf[0], xf[1], 1, 200, 260, 1), (xf[0] - 250, xf[1] - 250, 1, 205, 262, 1)] for _ in range(n)]
    ts = np.arange(n) / FPS
    bx = np.interp(ts, [t1, t2], [sum(xn) / 2, sum(xf) / 2]); by = np.interp(ts, [t1, t2], [600, 230])
    return boxes, np.stack([bx, by], 1), bx * S


def _x(p):
    return np.clip(np.round(p["cx"] - p["cw"] / 2), 0, SW - p["cw"]).astype(int)


def test_gate_plan_holds_hitters_under_caps():
    boxes, bxy, bx = _clip(1.0, 3.0)
    p = C.plan_clip(boxes, bxy, bx, [1.0, 3.0], 0.0, FPS, scale=S)
    assert p["feasible"] and all(p["reached"])
    v, a, quick = p["caps"]
    assert v <= 1 + 1e-6 and a <= 1 + 1e-6 and not quick                     # 0.8 / 0.5 cw/s, 1.35 cw/s^2, no reversal < 0.5 s
    n = len(boxes)
    q = C.role_clip_qa(p["roles"], _x(p), np.zeros(n, int), p["cw"], np.full(n, 2160), S, SW, FPS, 0.0)
    assert q["pass"] and q["contacts"] == 2
    assert jerk_p95(p["cx"], SW)["px_per_frame3"] < 0.5 * 0.333333             # jerk gate (float path)
    hard = [g for g in p["gates"] if g[5]]
    lead = hard[1][0] - int(0.35 * FPS)                                         # on the next hitter 0.35 s early
    i, lo, hi, c, w, _ = hard[1]
    m = C.CAMERA["rel_margin"] * p["cw"][lead]
    assert c + w / 2 - p["cw"][lead] / 2 + m - 1 <= p["cx"][lead] <= c - w / 2 + p["cw"][lead] / 2 - m + 1


def test_infeasible_clip_goes_wide_or_is_flagged():
    boxes, bxy, bx = _clip(1.0, 1.15, xn=(20, 60), xf=(1200, 1240))         # 3500 px apart, 0.15 s: no pan can do it
    p = C.plan_clip(boxes, bxy, bx, [1.0, 1.15], 0.0, FPS, scale=S)
    assert not p["feasible"] and p["infeasible_pairs"]
    assert C.infeasible_decision(p, n_wide=0)[0] == "wide"
    assert C.infeasible_decision(p, n_wide=1)[0] == "reject"                   # wide limit reached -> flag for rejection
    assert C.infeasible_decision(p, 0, dict(C.CAMERA, infeasible_action="reject"))[0] == "reject"


def test_feasibility_rule_is_necessary_only():
    g = [(60, 0, 0, 700.0, 100.0, True), (130, 0, 0, 2400.0, 100.0, True)]  # 1.17 s apart, 1700 px between centres
    assert C.infeasible_contacts(g, 240, FPS, SW, contacts=[1.0, 2.17]) == []  # 0.8 cw/s x 0.97 s covers the gap
    g2 = [(60, 0, 0, 300.0, 100.0, True), (75, 0, 0, 3500.0, 100.0, True)]
    assert C.infeasible_contacts(g2, 240, FPS, SW, contacts=[1.0, 1.25])
    soft = [(60, 0, 0, 300.0, 100.0, True), (75, 0, 0, 3500.0, 100.0, False)]   # receivers without a contact are soft
    assert C.infeasible_contacts(soft, 240, FPS, SW, contacts=[1.0]) == []


def test_calm_zoom_rate_short_tightening_and_dips():
    n = 600
    h = np.full(n, 1.0); h[200:290] = 0.75                                      # 1.5 s tightening: skipped
    out = C.calm_height(h, FPS)
    assert out.min() > 0.99
    h2 = np.full(n, 1.0); h2[100:250] = 0.95                                    # 2.5 s, 5 % dip: filled (no breathing)
    assert C.calm_height(h2, FPS).min() > 0.99
    h3 = np.full(n, 1.0); h3[100:500] = 0.7                                     # long, deep tightening: kept, eased, <= 22 %/s
    o3 = C.calm_height(h3, FPS)
    assert o3.min() < 0.75 and np.abs(np.diff(np.log(o3))).max() <= np.log1p(0.22) / FPS + 1e-9


def test_crop_height_gate_by_format():
    h = np.r_[np.full(52, 0.9), np.full(48, 0.96)]                              # mean 0.929, 48 % at >= 95 %
    assert C.crop_height_qa(h, fmt="doubles")["warnings"] == []
    assert len(C.crop_height_qa(h)["warnings"]) == 2                            # singles: 0.75-0.85, < 25 %
    assert len(C.crop_height_qa(h, fmt="doubles", sport="padel")["warnings"]) == 2
    bad = C.crop_height_qa(np.r_[np.full(40, 0.9), np.full(60, 0.97)], fmt="doubles")
    assert len(bad["warnings"]) == 2


def test_role_clip_qa_flags_a_clipped_hitter():
    boxes, bxy, bx = _clip(1.0, 2.2)
    p = C.plan_clip(boxes, bxy, bx, [1.0, 2.2], 0.0, FPS, scale=S)
    n = len(boxes); x = np.full(n, SW - int(p["cw"].max()) - 10)                # crop parked on the right: near hitter out
    q = C.role_clip_qa(p["roles"], x, np.zeros(n, int), p["cw"], np.full(n, 2160), S, SW, FPS, 0.0)
    assert q["contacts_hitter_clipped"] >= 1 and not q["pass"]
    assert C.role_clip_qa(p["roles"], x, np.zeros(n, int), p["cw"], np.full(n, 2160), S, SW, FPS, 0.0, wide=True)["pass"]


def test_tilt_keeps_pre_v43_limits_and_pan_keeps_new():
    assert C.CAMERA["pan_acc"] == 1.35 and C.CAMERA["pan_max"] == 0.5                 # pan: v4.3 limits
    assert C.CAMERA["tilt_max"] == 0.5 and C.CAMERA["tilt_acc"] == 0.6                 # tilt: limits before 0c59fa6
    n, SH = 600, 2160
    ch = np.full(n, 1620.0)
    top = np.where(np.arange(n) < 200, 200.0, 1000.0); bot = top + 300                 # player box jumps down 800 px
    y = C.vertical_follow(top, bot, ch, FPS, SH).astype(float)
    cy = y + ch / 2
    v = np.abs(np.diff(cy))
    assert v.max() <= 0.5 * 1620 / FPS + 1.0                                            # 0.5 crop-heights/s (+ rounding)
    k = FPS // 2; vs = np.convolve(v, np.ones(k) / k, "valid")                          # 0.5 s mean speed (px/frame)
    assert np.abs(vs[k:] - vs[:-k]).max() <= 0.6 * 1620 / FPS ** 2 * k + 1.0            # 0.6 crop-heights/s^2 (+ rounding)
    assert np.ptp(cy) > 150                                                             # and it does follow
    y2 = C.vertical_follow(top, bot, ch, FPS, SH, dict(C.CAMERA, tilt_acc=1.35)).astype(float)
    assert np.abs(np.diff(y2)).max() > v.max()                                          # the tilt cap is what binds


def test_framing_risk_triggers_auto_wide_or_reject():
    boxes, bxy, bx = _clip(1.0, 3.0)
    p = C.plan_clip(boxes, bxy, bx, [1.0, 3.0], 0.0, FPS, scale=S)
    n = len(boxes)
    ok = C.role_clip_qa(p["roles"], _x(p), np.zeros(n, int), p["cw"], np.full(n, 2160), S, SW, FPS, 0.0)
    assert p["feasible"] and ok["pass"]
    assert C.infeasible_decision(p, 0, roles_qa=ok) == (None, "")                       # framing safe: no action
    xbad = np.full(n, SW - int(p["cw"].max()) - 10)                                     # near hitter out of the crop
    bad = C.role_clip_qa(p["roles"], xbad, np.zeros(n, int), p["cw"], np.full(n, 2160), S, SW, FPS, 0.0)
    act, why = C.infeasible_decision(p, 0, roles_qa=bad)
    assert act == "wide" and "contact framing" in why                                    # feasible pan plan, framing at risk
    assert C.infeasible_decision(p, 1, roles_qa=bad)[0] == "reject"                    # wide limit reached -> reject flag
    ybad = np.full(n, 2160 - 900)                                                       # vertical: crop below the far players
    vq = C.role_clip_qa(p["roles"], _x(p), ybad, p["cw"], np.full(n, 900), S, SW, FPS, 0.0)
    assert not vq["pass"] and C.infeasible_decision(p, 0, roles_qa=vq)[0] == "wide"
