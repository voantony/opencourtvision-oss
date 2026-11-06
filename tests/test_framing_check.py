"""Framing check on rendered courts (no footage): cut-off sides, team missing, camera moved, tilt, hints, debounce."""
import numpy as np
import pytest

from opencourtvision.framing import FramingChecker, court_corners_px, geometry_warnings, horizon_roll_deg, normalise_H, players_by_side
from opencourtvision.framing.synth import apply_warp, default_H, player_boxes, render_court, warp_matrix

SIZE = (640, 360)


@pytest.fixture(scope="module")
def scene():
    H = default_H(SIZE)
    return H, render_court(H, SIZE, seed=1), player_boxes(H, SIZE, seed=2)


def test_good_setup_is_ok(scene):
    H, img, boxes = scene
    v = FramingChecker(SIZE).lock(img, H).check(img, boxes, live=False)
    assert v.verdict == "ok" and v.warnings == [] and v.hint == "looks good"
    assert v.counts == {"near": 2, "far": 2} and v.drift_px < 1.0


def test_normalise_H_flips_negative_w():
    H = default_H(SIZE)
    assert geometry_warnings(-H, SIZE)[1] == geometry_warnings(H, SIZE)[1]
    assert (normalise_H(-H) @ [5.0, 12.0, 1.0])[2] > 0


@pytest.mark.parametrize("dx,dy,edge,hint", [(200, 0, "right", "turn phone left"), (-200, 0, "left", "turn phone right"),
                                             (0, 90, "bottom", "tilt down"), (0, -90, "top", "tilt up")])
def test_cut_off_side_and_hint(dx, dy, edge, hint):
    H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]) @ default_H(SIZE)
    warn, cut, _ = geometry_warnings(H, SIZE)
    assert "court_cut_off" in warn and edge in cut["edges"]
    fc = FramingChecker(SIZE)
    fc.H_lock = normalise_H(H)
    v = fc.check(None, player_boxes(default_H(SIZE), SIZE), live=False, track=False)
    assert v.verdict == "court_cut_off"
    # moving the image right pushes the right side out -> the fix is to turn the phone toward it (the image shifts back)
    assert v.hint in (hint, "turn phone right" if hint == "turn phone left" else "turn phone left", "tilt up", "tilt down")


def test_zoomed_in_says_move_back():
    A = np.diag([1.6, 1.6, 1.0]); A[:2, 2] = [-0.3 * SIZE[0], -0.3 * SIZE[1]]
    fc = FramingChecker(SIZE); fc.H_lock = normalise_H(A @ default_H(SIZE))
    v = fc.check(None, None, live=False, track=False)
    assert v.verdict == "court_cut_off" and v.hint.startswith("move phone back")


def test_team_missing(scene):
    H, img, boxes = scene
    near_only = boxes[:2]
    assert players_by_side(H, near_only) == {"near": 2, "far": 0}
    v = FramingChecker(SIZE).lock(img, H).check(img, near_only, live=False)
    assert v.verdict == "far_team_missing" and "far" in v.hint
    v = FramingChecker(SIZE).lock(img, H).check(img, boxes[2:], live=False)
    assert v.verdict == "near_team_missing"


def test_spectator_outside_zone_not_counted(scene):
    H, _, boxes = scene
    far_left_bench = np.array([[2, 100, 12, 130, 1.0]])            # left of the court, far outside the play zone
    assert players_by_side(H, np.vstack([boxes[:2], far_left_bench]))["far"] == 0


def test_camera_moved_detected_and_small_jitter_ignored(scene):
    H, img, boxes = scene
    rng = np.random.default_rng(0)
    fc = FramingChecker(SIZE).lock(img, H)
    big, Hb, bb = apply_warp(img, H, boxes, warp_matrix("shift", SIZE, 0.4, rng), SIZE)
    v = fc.check(big, bb, live=False)
    assert "camera_moved" in v.warnings and v.verdict == "camera_moved" and v.hint.startswith("phone moved")
    small, _, sb = apply_warp(img, H, boxes, warp_matrix("shift", SIZE, 0.01, rng), SIZE)
    assert "camera_moved" not in FramingChecker(SIZE).lock(img, H).check(small, sb, live=False).warnings


def test_rotation_reports_tilt_and_level_hint(scene):
    H, img, boxes = scene
    A = warp_matrix("rotate", SIZE, 0.8, np.random.default_rng(3))
    Hr = normalise_H(A @ H)
    assert abs(horizon_roll_deg(Hr)) == pytest.approx(8.0, abs=0.5)
    fc = FramingChecker(SIZE); fc.H_lock = Hr
    v = fc.check(None, None, live=False, track=False)
    assert "too_low_or_tilted" in v.warnings


def test_squash_reports_too_low():
    A = warp_matrix("squash", SIZE, 0.9, np.random.default_rng(0))
    warn, _, info = geometry_warnings(A @ default_H(SIZE), SIZE)
    assert info["too_low"] and "too_low_or_tilted" in warn


def test_live_debounce_and_window(scene):
    H, img, boxes = scene
    fc = FramingChecker(SIZE, p=dict(DEBOUNCE=3, WINDOW=10)).lock(img, H)
    for _ in range(5): fc.check(img, boxes, track=False)
    v = fc.check(img, boxes[:2], track=False)                          # one frame without far players: not missing yet
    assert "far_team_missing" not in v.warnings
    for _ in range(10): v = fc.check(img, boxes[:2], track=False)
    assert "far_team_missing" in v.shown


def test_verdict_is_json_safe(scene):
    import json
    H, img, boxes = scene
    json.dumps(FramingChecker(SIZE).lock(img, H).check(img, boxes).to_dict())


# --- app-facing checks: ok / hint / warn, one fix, quality signal (v3: "just make do") ----------------------------

def _fc_with(H):
    fc = FramingChecker(SIZE); fc.H_lock = normalise_H(H)
    return fc


def test_checks_all_ok_on_good_setup(scene):
    H, img, boxes = scene
    v = FramingChecker(SIZE).lock(img, H).check(img, boxes, live=False)
    assert set(v.checks) == {"steady", "baselines", "lines", "level", "players"}
    assert all(c["level"] == "ok" and c["fix"] is None for c in v.checks.values()) and v.fix is None and v.level == "ok"
    assert v.info["line_support"] > 0.6 and v.quality["court_visible"] == 1.0 and v.quality["vis_far"] == 1.0


@pytest.mark.parametrize("dy,which,action", [(-90, "far", "Tilt the phone up"), (90, "near", "Tilt the phone down")])
def test_one_cut_baseline_is_only_a_hint(dy, which, action):
    v = _fc_with(np.array([[1, 0, 0], [0, 1, dy], [0, 0, 1.0]]) @ default_H(SIZE)).check(None, None, live=False, track=False)
    assert v.checks["baselines"]["level"] == "hint" and v.level == "hint" and v.fix_check == "baselines"
    assert v.fix.startswith(action) and f"the {which} baseline is partly cut off" in v.fix
    assert v.checks["lines"]["level"] == "ok"                     # same cause, one hint


def test_dev_match_style_cut_corner_is_a_hint_not_a_warning():
    v = _fc_with(np.array([[1, 0, 120], [0, 1, 0], [0, 0, 1.0]]) @ default_H(SIZE)).check(None, None, live=False, track=False)
    assert "near_right" in v.cut["corners"] and v.level == "hint" and v.checks["baselines"]["level"] == "hint"


def test_both_baselines_missing_is_a_firm_warning():
    A = np.diag([3.0, 3.0, 1.0]); A[:2, 2] = [-SIZE[0], -SIZE[1]]          # zoomed far in on mid-court
    v = _fc_with(A @ default_H(SIZE)).check(None, None, live=False, track=False)
    assert v.quality["near_baseline"] < 0.5 and v.quality["far_baseline"] < 0.5
    assert v.checks["baselines"]["level"] == "warn" and v.level == "warn" and "neither baseline" in v.fix


def test_tilt_and_low_phone_are_hints(scene):
    H, img, boxes = scene
    v = _fc_with(warp_matrix("rotate", SIZE, 0.8, np.random.default_rng(3)) @ H).check(None, None, live=False, track=False)
    assert v.checks["level"] == dict(level="hint", fix="The phone is tilted about 8°. Level it if you can.", dismissed=False)
    v = _fc_with(warp_matrix("squash", SIZE, 0.9, np.random.default_rng(0)) @ H).check(None, None, live=False, track=False)
    assert v.checks["level"]["level"] == "hint"


def test_blur_is_an_unclear_lines_hint_not_a_move(scene):
    import cv2
    H, img, boxes = scene
    v = FramingChecker(SIZE).lock(img, H).check(cv2.GaussianBlur(img, (0, 0), 4), boxes, live=False, recording=True)
    assert "camera_moved" not in v.warnings and v.checks["steady"]["level"] == "ok"
    assert v.checks["lines"]["level"] == "hint" and v.level == "hint" and v.quality["blurred"]


def test_move_is_warn_only_while_recording(scene):
    H, img, boxes = scene
    big, _, bb = apply_warp(img, H, boxes, warp_matrix("shift", SIZE, 0.4, np.random.default_rng(0)), SIZE)
    assert FramingChecker(SIZE).lock(img, H).check(big, bb, live=False, recording=True).checks["steady"]["level"] == "warn"
    assert FramingChecker(SIZE).lock(img, H).check(big, bb, live=False, recording=False).checks["steady"]["level"] == "hint"


def test_team_missing_hint_then_warn_when_sustained(scene):
    H, img, boxes = scene
    fc = FramingChecker(SIZE, p=dict(WINDOW=5, DEBOUNCE=2, TEAM_WARN_CHECKS=8)).lock(img, H)
    for _ in range(5): fc.check(img, boxes, track=False, recording=True)
    levels = [fc.check(img, boxes[:2], track=False, recording=True).checks["players"]["level"] for _ in range(14)]
    assert "hint" in levels and levels[-1] == "warn" and levels.index("warn") > levels.index("hint")
    v = fc.check(img, boxes[:2], track=False, recording=True)
    assert v.level == "warn" and v.fix.startswith("The far players have been out of the picture")


def test_sustained_team_warning_survives_a_stray_detection(scene):
    H, img, boxes = scene
    fc = FramingChecker(SIZE, p=dict(WINDOW=3, DEBOUNCE=2, TEAM_WARN_CHECKS=10, PRESENT_MIN=0.5)).lock(img, H)
    for _ in range(3): fc.check(img, boxes, track=False, recording=True)
    lv = []
    for i in range(16):                                         # far team gone, one stray far box on check 9
        lv.append(fc.check(img, boxes if i == 9 else boxes[:2], track=False, recording=True).checks["players"]["level"])
    assert lv[-1] == "warn"


def test_featureless_setup_cannot_track_so_never_claims_moved(scene):
    H, img, boxes = scene
    flat = np.tile(np.linspace(60, 200, SIZE[0]).astype(img.dtype), (SIZE[1], 1)).reshape(img.shape[:2] + img.shape[2:]) \
        if img.ndim == 2 else np.repeat(np.tile(np.linspace(60, 200, SIZE[0]), (SIZE[1], 1))[..., None], 3, 2).astype(img.dtype)
    fc = FramingChecker(SIZE).lock(flat, H)
    v = fc.check(flat, None, live=False, recording=True)
    assert v.checks["steady"]["level"] == "ok" and v.quality["tracked"] is False


def test_both_teams_missing_is_a_break_not_a_message(scene):
    H, img, boxes = scene
    fc = FramingChecker(SIZE, p=dict(WINDOW=5, DEBOUNCE=2, TEAM_WARN_CHECKS=4)).lock(img, H)
    for _ in range(12): v = fc.check(img, np.zeros((0, 5)), track=False, recording=True)
    assert v.checks["players"]["level"] == "ok" and v.level == "ok"


def test_dismissed_hint_is_suppressed_but_warn_is_not():
    fc = _fc_with(np.array([[1, 0, 0], [0, 1, -90], [0, 0, 1.0]]) @ default_H(SIZE))
    assert fc.check(None, None, live=False, track=False).fix_check == "baselines"
    fc.dismiss("baselines")
    v = fc.check(None, None, live=False, track=False)
    assert v.checks["baselines"] == dict(level="hint", fix=v.checks["baselines"]["fix"], dismissed=True)
    assert v.level == "ok" and v.fix is None
    A = np.diag([3.0, 3.0, 1.0]); A[:2, 2] = [-SIZE[0], -SIZE[1]]
    fc2 = _fc_with(A @ default_H(SIZE)); fc2.dismiss("baselines")
    assert fc2.check(None, None, live=False, track=False).level == "warn"


def test_cut_sideline_gives_one_hint_not_two():
    v = _fc_with(np.array([[1, 0, 200], [0, 1, 0], [0, 0, 1.0]]) @ default_H(SIZE)).check(None, None, live=False, track=False)
    assert v.info["lines"]["right_sideline"]["in_frame"] < 0.98
    assert v.checks["baselines"]["level"] == "hint" and v.checks["lines"]["level"] == "ok" and v.fix.startswith("Turn the phone right")


def test_line_report_counts_cut_lines():
    from opencourtvision.framing.check import line_report
    H = np.array([[1, 0, 0], [0, 1, -90], [0, 0, 1.0]]) @ default_H(SIZE)
    r = line_report(H, SIZE)
    assert r["lines"]["far_baseline"]["in_frame"] == 0.0 and r["lines"]["near_baseline"]["in_frame"] == 1.0 and r["support"] is None


def test_bench_people_not_counted_as_players():
    H = default_H(SIZE)
    from opencourtvision.framing.check import NET_Y, _proj
    q, _ = _proj(H, [(-2.5, NET_Y - 1.0)])                    # sitting on the courtside bench by the net post
    b = np.array([[q[0, 0] - 4, q[0, 1] - 20, q[0, 0] + 4, q[0, 1], 0.9]])
    assert players_by_side(H, b) == {"near": 0, "far": 0}


def test_far_roi_covers_far_half():
    from opencourtvision.framing.check import far_roi
    H = default_H(SIZE)
    x0, y0, x1, y1 = far_roi(H, SIZE)
    far = player_boxes(H, SIZE, seed=2)[2:]
    assert all(x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= b[3] <= y1 for b in far)
    assert far_roi(np.array([[1, 0, 0], [0, 1, -900], [0, 0, 1.0]]) @ H, SIZE) is None


def test_session_api_setup_follow_record_move_relock_quality(scene):
    import json

    from opencourtvision.framing.api import FramingSession
    H, img, boxes = scene
    up = lambda a: np.kron(a, np.ones((2, 2), np.uint8))       # the app hands a bigger frame; corners in its pixels
    corners = [tuple(2 * np.array(c[0])) for c in court_corners_px(H).values()]
    s = FramingSession()
    r = s.setup(up(img), court_corners=corners)
    assert r["schema"] == "vamos.framing.v2" and r["status"] == "ok" and r["fix"] is None and r["blocks_recording"] is False
    moved, _, mb = apply_warp(img, H, boxes, warp_matrix("shift", SIZE, 0.15, np.random.default_rng(0)), SIZE)
    for _ in range(3): r = s.check(up(moved), 2 * mb)              # adjusting the phone during setup: followed, no message
    assert r["checks"]["steady"]["level"] == "ok"
    s.start_recording()
    for i in range(3): r = s.check(up(moved), 2 * mb, t=float(i))
    assert r["status"] == "ok"
    moved2, _, mb2 = apply_warp(moved, H, mb, warp_matrix("shift", SIZE, 0.4, np.random.default_rng(1)), SIZE)
    rs = [s.check(up(moved2), 2 * mb2, t=3.0 + i) for i in range(2)]
    assert rs[0]["checks"]["steady"]["level"] == "ok"            # debounced: one check is not enough
    assert rs[1]["status"] == "warn" and rs[1]["fix_check"] == "steady" and rs[1]["blocks_recording"] is False
    s.relock(up(moved2))
    assert s.check(up(moved2), 2 * mb2, t=6.0)["checks"]["steady"]["level"] == "ok"
    q = s.quality()
    assert q["schema"] == "vamos.framing_quality.v1" and len(q["samples"]) == 6 and q["summary"]["reliable"] < 1.0
    json.dumps(rs[1]); json.dumps(q)


def test_quality_signal_widens_uncertainty_never_refuses():
    from opencourtvision.framing.quality import attach_to_points, for_window, uncertainty_scale
    from opencourtvision.robust.framing import point_evidence
    full = dict(court_visible=1.0, vis_near=1.0, vis_far=1.0, moved=False, blurred=False)
    part = dict(court_visible=0.55, vis_near=0.9, vis_far=0.3, moved=False, blurred=False)
    doc = dict(schema="vamos.framing_quality.v1", samples=[dict(t=0.0, **full), dict(t=10.0, **full), dict(t=50.0, **part), dict(t=60.0, **part)])
    a, b = for_window(doc, 0, 20), for_window(doc, 45, 65)
    assert a["uncertainty_scale"] == 1.0 and a["vision_mode"] == "full"
    assert b["uncertainty_scale"] > 1.5 and b["vision_mode"] == "partial"
    assert for_window(doc, 30, 40)["court_visible"] == 1.0                 # no sample inside: the last one before
    pts = [dict(A=1.0, B=9.0), dict(A=48.0, B=61.0)]
    assert attach_to_points(pts, doc) == 2 and pts[1]["framing"]["vis_far"] == 0.3
    p_full, used_full = point_evidence(0.5, near=0.5, far=2.0, vis_near=a["vis_near"], vis_far=a["vis_far"])
    p_part, used_part = point_evidence(0.5, near=0.5, far=2.0, vis_near=b["vis_near"], vis_far=b["vis_far"])
    assert used_full == ["near", "far"] and used_part == ["near"] and 0.5 < p_part < p_full   # less sure, still scored
    assert uncertainty_scale(0.1, 0.1) == 4.0 and uncertainty_scale(1.0, 1.0) == 1.0


def test_overlay_draws_without_saving(scene):
    from opencourtvision.framing.overlay import draw
    H, img, boxes = scene
    v = FramingChecker(SIZE).lock(img, H).check(img, boxes[:2], live=False)
    out = draw(img, v, boxes[:2])
    assert out.shape == (SIZE[1], SIZE[0], 3) and out.dtype == np.uint8


def test_quality_lowers_confidence_never_refuses():
    from opencourtvision.framing.quality import attach_to_points, soften
    assert soften(0.9, 1.0) == 0.9 and soften(0.9, 2.0) == 0.7 and soften(0.9, 4.0) == 0.6
    s = lambda t, cv: dict(t=t, court_visible=cv, vis_near=1.0, vis_far=cv, moved=False, blurred=False)
    doc = dict(samples=[s(1, 1.0), s(30, 0.3)])
    pts = [dict(A=0, B=5, winner_confidence=0.9, bounce_confidence=0.8),
           dict(A=29, B=35, winner_confidence=0.9, bounce_confidence=0.8),
           dict(A=29, B=35, winner_confidence=1.0, winner_src="user")]
    assert attach_to_points(pts, doc) == 3
    assert pts[0]["winner_confidence"] == 0.9 and pts[0]["bounce_confidence"] == 0.8
    assert 0.5 < pts[1]["winner_confidence"] < 0.9 and pts[1]["bounce_confidence"] < 0.8
    assert pts[2]["winner_confidence"] == 1.0                     # a user's own call is ground truth
