"""Physics ball tracker: camera from the court, closed-form flight, integrator, and synthetic rallies with known truth
(bounce position / timing, robustness to 50% false detections and gaps, rejection of non-physical tracks)."""
import warnings

import numpy as np
import pytest

from opencourtvision.ball import physics3d as P3
from opencourtvision.ball import physics3d_synth as S

warnings.filterwarnings("ignore")
CAM = P3.Camera.look_at([5.3, -5.9, 2.75], [5.5, 13.0, 0.0], 830.0, (1280, 720))


def test_camera_from_homography_round_trip():
    g = np.array([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77], [5.485, 11.885], [1.37, 5.485]], float)
    H = P3.fit_homography(g, CAM.ground_to_image(g))
    cam = P3.Camera.from_homography(H, CAM.image_size)
    assert abs(cam.f - CAM.f) < 1e-3 * CAM.f
    assert np.allclose(cam.centre, CAM.centre, atol=1e-3)
    X = np.array([[3.0, 8.0, 1.2], [7.0, 20.0, 2.5]])
    assert np.allclose(cam.project(X), CAM.project(X), atol=1e-2)


def test_ballistic_dlt_exact_without_noise():
    p0, v0 = np.array([4.0, 2.0, 1.0]), np.array([1.0, 25.0, 4.0])
    t = np.arange(-4, 5) / 30.0
    X = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, P3.G)
    th = P3.ballistic_dlt(CAM, t, CAM.project(X))
    assert np.allclose(th, np.r_[p0, v0], atol=1e-6)
    th3, ok = P3.batch_triplet_solve(CAM, t[[0, 4, 8]], CAM.project(X)[None, [0, 4, 8]])
    assert ok[0] and np.allclose(th3[0], np.r_[p0, v0], atol=1e-6)


def test_taylor_integrator_matches_rk4():
    th = np.array([[4, 0, 1.0, 2, 40, 3, -0.2], [5, 24, 1, -1, -25, 6, 0.3]])
    Xa, _ = P3.simulate(th, -10, 40, 30.0, P3.TENNIS.kd)
    Xb, _ = P3.simulate(th, -10, 40, 30.0, P3.TENNIS.kd, exact=True)
    assert np.abs(Xa - Xb).max() < 0.005


def _run(seed, **kw):
    rng = np.random.default_rng(seed)
    R = S.rally(rng)
    c, uv = S.detections(rng, CAM, R["X"], **kw)
    return R, uv, P3.track(CAM, c)


def _bounce_errors(R, out):
    errs = []
    for e in out["events"]:
        if e["kind"] != "bounce":
            continue
        d = [(abs(e["k"] - t * R["fps"]), np.hypot(e["xy"][0] - x, e["xy"][1] - y)) for t, x, y in R["bounces"]]
        errs.append(min(d, key=lambda q: q[1]))
    return errs


@pytest.mark.parametrize("seed", [1, 2])
def test_clean_rally_bounces_solved(seed):
    R, uv, out = _run(seed, false_share=0.0, gap=None)
    errs = _bounce_errors(R, out)
    assert errs, "no bounce solved"
    assert all(dm < 0.3 and dk < 1.5 for dk, dm in errs)
    rep = np.isfinite(out["pos"][:, 0])
    d = np.hypot(*(out["pos"] - uv).T)
    assert np.mean(d[rep] < 5.0) > 0.95


def test_false_detections_and_gap_robust():
    R, uv, out = _run(3, false_share=0.5, gap=(0.45, 8))
    rep = np.isfinite(out["pos"][:, 0]); vis = np.isfinite(uv[:, 0])
    d = np.hypot(*(out["pos"] - uv).T)
    assert np.mean(d[rep] < 5.0) > 0.95                      # what it reports is the ball
    assert np.sum(rep & vis & (d < 5.0)) / vis.sum() > 0.5    # and it still finds most of it
    assert all(dm < 1.0 for _, dm in _bounce_errors(R, out))  # no invented bounces


def test_clutter_only_reports_nothing():
    rng = np.random.default_rng(7)
    T = 150
    c = np.full((T, 4, 3), np.nan)
    st = rng.uniform([0, 200], [1280, 720], (3, 2))
    for i in range(T):
        c[i, :3, :2] = st + rng.normal(0, 0.7, (3, 2)); c[i, :3, 2] = 0.6
        c[i, 3] = (rng.uniform(0, 1280), rng.uniform(0, 720), 0.4)
    out = P3.track(CAM, c)
    assert out["stats"]["tracked_frames"] == 0


def test_non_physical_track_rejected():
    T = 60
    k = np.arange(T)
    uv = np.c_[640 + 200 * np.sin(k / 4.0), 300 + 80 * np.cos(k / 3.0)]          # a racket-like loop, not a flight
    out = P3.track(CAM, np.c_[uv, np.full(T, 0.9)][:, None, :])
    assert out["stats"]["inlier_frames"] < 12


def test_in_probability_from_bounce_covariance():
    R, uv, out = _run(1, false_share=0.0, gap=None)
    b = [e for e in out["events"] if e["kind"] == "bounce" and "p_in" in e]
    assert b
    for e in b:
        inside = 0.3 < e["xy"][0] < P3.COURT_W - 0.3 and 0.3 < e["xy"][1] < P3.COURT_L - 0.3
        if inside:
            assert e["p_in"] > 0.9


@pytest.mark.parametrize("seed", [2, 7])
def test_last_bounce_with_three_detections_after_it(seed):
    """a rally cut 3 detections after its last bounce: the terminal (short-tail) bounce fit finds that bounce."""
    rng = np.random.default_rng(100 + seed)
    R = S.rally(rng)
    c, uv = S.detections(rng, CAM, R["X"], false_share=0.0, gap=None)
    tb, xb, yb = R["bounces"][-1]
    kb = tb * R["fps"]
    post = [k for k in range(int(np.floor(kb)) + 1, len(c))
            if np.isfinite(c[k, :, 0]).any() and np.nanmin(np.hypot(*(c[k, :, :2] - uv[k]).T)) < 4]
    cut = c[:post[2] + 1]

    def found(terminal):
        out = P3.track(CAM, cut, P3.Config(terminal=terminal))
        return [e for e in out["events"] if e["kind"] == "bounce" and abs(e["k"] - kb) < 3
                and np.hypot(e["xy"][0] - xb, e["xy"][1] - yb) < 0.5]

    hit = found(True)
    assert len(hit) == 1 and hit[0].get("terminal")
    assert not found(False)
