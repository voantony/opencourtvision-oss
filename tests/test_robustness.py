"""Robustness to imperfect mounts: perturbations, camera-motion handling, partial-court policy, harness (synthetic)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from opencourtvision.editloop.motion import MotionCues, live_gate  # noqa: E402
from opencourtvision.robust import camera_motion as CM  # noqa: E402
from opencourtvision.robust import framing as FR  # noqa: E402
from opencourtvision.robust.perturb import Perturbation, perturb_frames, perturb_homography, perturb_points, suite  # noqa: E402


def _scene(seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    img = np.clip(gaussian_filter(rng.normal(size=(108, 192)), 1.5) * 60 + 110, 0, 255)
    img[40:42, 50:165] = 230; img[90:92, 5:190] = 230
    return img.astype(np.uint8)


def test_suite_covers_paths_and_crops():
    names = {p.name for p in suite()}
    assert {"clean", "shake", "drift", "bumps", "pan", "corner_far_left_20"} <= names
    assert all(f"crop_{s}_{f}" in names for s in ("left", "right", "top", "bottom") for f in (10, 20, 30))
    t = np.linspace(0, 100, 601)
    b = Perturbation("b", "bump", amp=0.05, bumps=(0.5,)).path(t)
    assert b[0, 0] == 0 and abs(b[-1, 0] - 0.05) < 1e-9 and abs(b[400, 0] - 0.05) < 1e-9     # a step that persists
    m = Perturbation("c", "crop", crop={"top": 0.2, "left": 0.1}).visible_mask((10, 20))
    assert not m[:2].any() and not m[:, :2].any() and m[2:, 2:].all()


def test_tracks_and_court_fit_move_with_the_camera():
    pert = Perturbation("b", "bump", amp=0.1, bumps=(0.5,), crop={"right": 0.3})
    t = np.array([0.0, 10.0]); px = np.array([[100.0, 50.0], [100.0, 50.0]])
    q = perturb_points(px, t, pert, (1920, 1080))
    assert np.allclose(q[0], [100, 50]) and np.allclose(q[1], [100 + 192, 50 + 54])
    assert np.isnan(perturb_points(np.array([[1800.0, 50.0]]), t[:1], pert, (1920, 1080))).all()  # cropped away
    H = np.array([[50.0, 0, 300], [0, 20.0, 100], [0, 0, 1]])
    H2 = perturb_homography(H, (0.1, 0.05), (1920, 1080))
    assert np.allclose(H2, CM.refit_homography(H, (192, 54)))


def test_bump_detected_refit_accepted_and_shake_marked_unreliable():
    img = _scene(); t = np.arange(0, 30, 1 / 6.0)
    bump = Perturbation("b", "bump", amp=0.04, bumps=(0.5,))
    fr = perturb_frames(img[None], np.array([0.0]), t, bump)
    A = CM.analyze(fr, t)
    assert [e[1] for e in A["events"]] == ["bump"] and 0 <= A["events"][0][0] - 15.0 < 1.5   # flagged once the shake window clears
    assert A["refits"][0][2] and abs(A["fit_offset"][-1][0] - 0.04 * 192) < 0.8           # re-fit carries the court
    assert A["reliable"].mean() > 0.9
    shake = Perturbation("s", "shake", amp=0.012)
    S = CM.analyze(perturb_frames(img[None], np.array([0.0]), t, shake), t)
    assert S["reliable"].mean() < 0.2                                                       # unstable phone


def test_low_confidence_keeps_the_audio_point():
    t = np.arange(0, 600, 1 / 6.0); E = np.full((len(t), 12, 24), 0.3, np.float32)        # idle court everywhere
    pts = [{"t0": 100.0, "t1": 110.0, "sources": {"audio"}}, {"t0": 300.0, "t1": 305.0, "sources": {"audio"}}]
    assert live_gate(pts, MotionCues(E, t)) == []                                          # confident idle: dropped
    rel = ~((t > 95) & (t < 115))                                                          # camera unstable there
    assert live_gate(pts, MotionCues(E, t, reliable=rel)) == [pts[0]]                      # no decision: kept (audio)


def test_partial_court_policy_and_framing_check():
    H = np.array([[60.0, 0, 300], [0, -30.0, 1000], [0, 0, 1]])                            # court metres -> px, far = up
    assert FR.court_visible_fraction(H, (1920, 1080)) > 0.95 and FR.framing_check(H, (1920, 1080)) == []
    Hc = CM.refit_homography(H, (0, -400))                                                 # camera tilted down: far cut
    w = FR.framing_check(Hc, (1920, 1080), offsets_px=np.random.default_rng(0).normal(0, 1, (30, 2)))
    assert any("far" in x for x in w) and any("unstable" in x for x in w)
    assert FR.side_visible_fraction(Hc, (1920, 1080), "near") > 0.95
    assert FR.vision_mode(0.9) == "full" and FR.vision_mode(0.5) == "partial" and FR.vision_mode(0.2) == "audio"
    assert FR.vision_mode(0.9, reliable=0.3) == "audio"
    p, used = FR.point_evidence(0.6, near=2.0, far=-5.0, vis_near=1.0, vis_far=0.2)        # far side out of view
    assert used == ["near"] and p > 0.6
    assert FR.point_evidence(0.6, near=None, far=None)[0] == pytest.approx(0.6)            # still scored on the prior


def test_harness_us_boundaries_hold_and_handling_recovers():
    import robustness as RB
    D = RB.synthetic(n_pts=10)
    P = [p for p in suite() if p.name in ("clean", "shake", "bumps", "crop_top_30", "corner_far_left_20")]
    res = RB.run(D, P, log=lambda *_: None)
    for k, r in res.items():
        assert r["US_audio"]["recall"] == 1.0 and r["US_audio"]["boundary_change_max"] == 0.0   # audio: never moves
        assert r["US"].get("boundary_change_max", 0.0) == 0.0                                   # gate keeps/drops only
    assert res["shake/naive"]["US"]["recall"] < 0.5 <= res["shake/handled"]["US"]["recall"]     # degrade to audio
    assert res["shake/handled"]["camera"]["mode"] == "audio"


def test_mount_occluder_detected_masked_and_fixed_to_the_phone():
    from opencourtvision.robust import occluders as OC
    img = _scene(); t = np.arange(0, 60, 1 / 6.0)
    fixed = Perturbation("m", "mount", mount={"side": "right", "w": 0.12, "h": 0.35, "y": 0.6})
    moving = Perturbation("mb", "bump", amp=0.06, bumps=(0.3, 0.6), mount=fixed.mount)
    truth = fixed.mount_mask((108, 192))
    for pert, conf in ((fixed, "low"), (moving, "high")):
        rng = np.random.default_rng(1)
        fr = perturb_frames(img[None], np.array([0.0]), t, pert)
        fr += rng.normal(0, 1.0, fr.shape).astype(np.float32)                              # sensor noise
        for k in range(0, len(fr), 3): fr[k, 20:40, rng.integers(10, 150):][:, :8] += 60    # a player walking past
        A = CM.analyze(fr, t)
        m, info = OC.detect_occluders(fr[::3], A["offset"][::3])
        iou = (m & truth).sum() / (m | truth).sum()
        assert iou > 0.6 and info["confidence"] == conf and info["regions"][0]["edge"]
    px = np.array([[1900.0, 650.0], [400.0, 300.0]])
    out = OC.drop_masked(px, truth, (1920, 1080))
    assert np.isnan(out[0]).all() and np.allclose(out[1], px[1])                           # no ball / player on the mount
    fm = OC.feature_mask(truth, (1920, 1080))
    assert fm.shape == (1080, 1920) and fm[650, 1915] == 0 and fm[300, 400] == 255          # court re-fit ignores it
    assert OC.cells_mask(truth).sum() >= 4


def test_framing_check_mount_in_view():
    from opencourtvision.robust.perturb import suite
    mount = [q for q in suite() if q.name == "mount_right"][0]
    m = mount.mount_mask((108, 192))
    H = np.array([[60.0, 0, 630], [0, -22, 900], [0, 0, 1]])      # court spans x 630..1288 px, y 900..377 px
    assert FR.framing_check(H, (1920, 1080)) == []
    w = FR.framing_check(H, (1920, 1080), occluder=m)
    assert len(w) == 1 and w[0].startswith("mount in view at the right edge")
    assert FR.court_covered_fraction(H, (1920, 1080), m) == 0.0
    Hr = np.array([[60.0, 0, 1300], [0, -22, 900], [0, 0, 1]])    # court pushed right, behind the mount
    assert FR.court_covered_fraction(Hr, (1920, 1080), m) > 0.05
    assert "hides" in FR.framing_check(Hr, (1920, 1080), occluder=m)[-1]


def test_continuous_pan_stays_reliable_and_tracked():
    """v4.5: a slow pan is tracked frame to frame (reliable, sub-pixel fit), not one re-fit per stop."""
    img = _scene(); t = np.arange(0, 180, 1 / 6.0)
    pan = Perturbation("p", "pan", amp=0.10, period_s=60.0)
    fr = perturb_frames(img[None], np.array([0.0]), t, pan)
    A = CM.analyze(fr, t)
    true = pan.path(t) * [192, 108]
    assert A["reliable"].mean() > 0.9
    assert np.median(np.abs(A["fit_offset"][:, 0] - true[:, 0])) < 0.5
    assert "pan" in {e[1] for e in A["events"]}
    old = CM.analyze(fr, t, p=dict(CONTINUOUS=False))
    assert np.median(np.abs(old["fit_offset"][:, 0] - true[:, 0])) > np.median(np.abs(A["fit_offset"][:, 0] - true[:, 0]))


def test_track_checks_undo_the_camera(tmp_path):
    import robustness as RB
    D = RB.synthetic(n_pts=6)
    rng = np.random.default_rng(0); n = 300
    np.savez(tmp_path / "seg001.npz", t0=100.0, fps=30.0, ball=rng.uniform(100, 500, (n, 2)).astype(np.float32),
             pose_k=np.arange(0, n, 3), boxes=np.tile(np.array([[200, 100, 240, 200, 0.9]] + [[np.nan] * 5] * 7, np.float32), (n // 3, 1, 1)))
    TR = RB.load_tracks(str(tmp_path))
    assert len(TR["ball"]) == n and len(TR["feet"]) == n // 3 and np.allclose(TR["feet"][0], (220, 200))
    P = [p for p in suite() if p.name in ("clean", "pan", "crop_left_30")]
    res = RB.run_tracks(D, TR, P, log=lambda *_: None)
    assert res["clean/naive"]["tracks"]["ball"]["err_med_px"] == 0.0
    assert res["pan/handled"]["tracks"]["ball"]["err_med_px"] < res["pan/naive"]["tracks"]["ball"]["err_med_px"]
    assert res["crop_left_30/naive"]["tracks"]["ball"]["observable"] < 1.0
