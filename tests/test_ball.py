"""Ball engine v2: live-ball filter, size gate, court gate, bounce clean-up, tracker, proxy metrics (synthetic camera)."""
import numpy as np
import pytest

from opencourtvision.ball import bounce as B
from opencourtvision.ball import live as L
from opencourtvision.ball import proxy as P
from opencourtvision.ball.geometry import BALL_D_MAX, BALL_D_MIN, CourtGate
from opencourtvision.ball.tracker import BallEngine, Kalman, chain

FPS = 30.0
W, H = 960, 540


class Cam:
    """Pinhole camera 6 m behind the near baseline, 4 m up, looking down the court (OpenCourtVision court metres)."""

    def __init__(self, f=700.0, pos=(5.485, -6.0, 4.0), look=(5.485, 14.0, 0.0)):
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        c = np.array(pos, float); fwd = np.array(look, float) - c; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.vstack([right, down, fwd]); self.t = -self.R @ c

    def project(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        p = (self.K @ (self.R @ X.T + self.t[:, None])).T
        return p[:, :2] / p[:, 2:]

    def H(self):
        return self.K @ np.c_[self.R[:, 0], self.R[:, 1], self.t]


@pytest.fixture(scope="module")
def cam():
    return Cam()


@pytest.fixture(scope="module")
def gate(cam):
    return CourtGate(cam.H(), image_size=(W, H))


def arc(p0, v0, n=30, g=9.81):
    t = np.arange(n) / FPS
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    X = p0 + v0 * t[:, None]
    X[:, 2] = p0[2] + v0[2] * t - 0.5 * g * t ** 2
    return X


def as_pos(px, n_before=0, n=None):
    n = n or len(px) + n_before
    pos = np.full((n, 2), np.nan); pos[n_before:n_before + len(px)] = px
    return pos


def test_homography_consistent_with_camera(cam, gate):
    pts = np.array([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77], [5.485, 11.885]])
    np.testing.assert_allclose(gate.to_image(pts), cam.project(np.c_[pts, np.zeros(len(pts))]), atol=1e-6)


def test_live_filter_accepts_real_arc_after_hit(cam, gate):
    X = arc((5.0, 1.0, 1.0), (0.3, 18.0, 3.5), n=30)          # struck at the near baseline, travelling up the court
    pos = as_pos(cam.project(X), n_before=3, n=40)
    rng = np.random.default_rng(0); pos += rng.normal(0, 0.3, pos.shape)
    live, trs = L.filter_live(pos, FPS, gate, hits=[3 / FPS], t0=0.0)
    assert len(trs) == 1 and trs[0].live, trs[0].reasons
    assert trs[0].info["contact"]["kind"] == "audio_no_player"
    assert np.isfinite(live[3:33, 0]).all()


def test_live_filter_rejects_static_ball(cam, gate):
    px = np.repeat(cam.project([[3.0, 8.0, 0.0]]), 40, 0) + np.random.default_rng(1).normal(0, 0.3, (40, 2))
    _, trs = L.filter_live(as_pos(px), FPS, gate, hits=[0.5], t0=0.0)
    assert not trs[0].live and "static" in trs[0].reasons


def test_live_filter_rejects_rolling_ball(cam, gate):
    t = np.arange(45) / FPS
    X = np.c_[np.full(45, 2.0), 6.0 + 1.5 * t, np.zeros(45)]  # 1.5 m/s along the ground, no gravity curvature
    px = cam.project(X) + np.random.default_rng(2).normal(0, 0.3, (45, 2))
    _, trs = L.filter_live(as_pos(px), FPS, gate, hits=[0.2], t0=0.0)
    assert not trs[0].live and "slow_or_rolling" in trs[0].reasons


def _diam(cam, X):
    return np.linalg.norm(cam.project(X - [0.0335, 0, 0]) - cam.project(X + [0.0335, 0, 0]), axis=1)


def test_live_filter_rejects_ball_on_adjacent_court():
    cam = Cam(f=2200.0)                                        # ~4K-like resolution: ball sizes are measurable
    gate = CourtGate(cam.H(), image_size=(W, H))
    X = arc((22.0, 4.0, 1.0), (0.0, 16.0, 4.0), n=30)        # a real arc, but 7 m beyond our right sideline
    px = cam.project(X)
    # one camera: the same rays pass ~2.5 m above our side margin, so position alone is ambiguous; size settles it
    assert gate.contains(px).mean() > 0.5
    assert gate.contains(px, diam_px=_diam(cam, X)).mean() < 0.5
    _, trs = L.filter_live(as_pos(px), FPS, gate, hits=[0.0], t0=0.0, sizes=_diam(cam, X))
    assert not trs[0].live and "off_court" in trs[0].reasons
    X2 = arc((5.0, 1.0, 1.0), (0.3, 18.0, 3.5), n=30)         # our own court's arc keeps passing with sizes
    _, ok = L.filter_live(as_pos(cam.project(X2)), FPS, gate, hits=[0.0], t0=0.0, sizes=_diam(cam, X2))
    assert ok[0].live, ok[0].reasons
    t = np.arange(30) / FPS
    X3 = np.c_[np.full(30, 24.0), 18.0 - 3.0 * t, np.zeros(30)]  # a ball rolling across the far next court
    _, t3 = L.filter_live(as_pos(cam.project(X3)), FPS, gate, hits=[0.0], t0=0.0)
    assert not t3[0].live


def test_live_filter_needs_a_contact(cam, gate):
    X = arc((5.0, 1.0, 1.0), (0.3, 18.0, 3.5), n=30)
    _, trs = L.filter_live(as_pos(cam.project(X)), FPS, gate, hits=[], t0=0.0)
    assert not trs[0].live and trs[0].reasons == ["no_contact"]


def test_live_filter_contact_by_velocity_reversal(cam, gate):
    a = arc((5.0, 2.0, 1.0), (0.0, 14.0, 2.0), n=30)         # falls to the ground ...
    zc = a[:, 2]; k = int(np.argmax(zc < 0)) if (zc < 0).any() else len(zc)
    a = a[:k]
    b = arc(a[-1] * [1, 1, 0], (0.0, 10.0, 4.0), n=20)        # ... and bounces up (bounce = velocity reversal)
    X = np.r_[a, b[1:]]
    _, trs = L.filter_live(as_pos(cam.project(X)), FPS, gate, hits=[], t0=0.0)
    assert trs[0].info["reversals"], trs[0].info
    assert trs[0].info["contact"]["kind"] == "velocity_reversal"


def test_live_filter_contact_needs_player_when_boxes_given(cam, gate):
    X = arc((5.0, 1.0, 1.0), (0.3, 18.0, 3.5), n=30)
    px = cam.project(X)
    far_box = [(0, 0, 10, 10)]                                 # a player nowhere near the ball at the hit
    near = (px[0, 0] - 20, px[0, 1] - 60, px[0, 0] + 20, px[0, 1] + 10)
    _, t_far = L.filter_live(as_pos(px), FPS, gate, hits=[0.0], players=[far_box] * 30)
    _, t_near = L.filter_live(as_pos(px), FPS, gate, hits=[0.0], players=[[near]] * 30)
    assert not t_far[0].live and t_near[0].live and t_near[0].info["contact"]["kind"] == "audio_player"


def test_gate_prism_follows_height(cam, gate):
    assert gate.contains(cam.project([[5.0, 12.0, 6.0]])).all()          # a lob over our court
    assert gate.contains(cam.project([[12.0, 20.0, 3.0]])).all()         # a wide ball over the side margin
    assert not gate.contains(cam.project([[-10.0, -3.0, 0.3]])).any()    # low, well beside the court
    # monocular limit: this ball is 3 m beyond our side margin, but its ray crosses the margin ~1.3 m up
    assert gate.contains(cam.project([[-8.0, 3.0, 0.0]])).all()


def test_size_gate(cam, gate):
    p = cam.project([[5.0, 3.0, 0.0]])
    lo, hi = gate.expected_diameter_px(p)
    true_d = np.linalg.norm(cam.project([[5.0 - 0.0335, 3.0, 0.0]]) - cam.project([[5.0 + 0.0335, 3.0, 0.0]]))
    assert lo[0] <= true_d * 1.05 and true_d <= hi[0] * 1.05            # the 6.54-6.86 cm band brackets the true size
    assert np.isclose(hi[0] / lo[0], BALL_D_MAX / BALL_D_MIN)
    assert gate.size_ok(p, [true_d])[0]
    assert not gate.size_ok(p, [8 * hi[0] + 5])[0]                       # a shoe / head-sized blob
    far = cam.project([[5.0, 23.0, 0.0]])
    assert gate.expected_diameter_px(far)[0][0] < lo[0]                 # smaller with depth


def test_bounce_cleanup_dedup_and_court(gate):
    on = gate.to_image([[5.0, 15.0]])[0]; off = gate.to_image([[5.0, 30.0]])[0]
    bs = [dict(frame=10, x=on[0], y=on[1]), dict(frame=16, x=on[0], y=on[1]),           # 200 ms apart: 2nd dropped
          dict(frame=40, x=off[0], y=off[1]),                                           # 6 m behind the baseline
          dict(frame=60, x=None, y=None),                                               # no position: kept (upstream)
          dict(frame=80, x=on[0], y=on[1])]
    kept, dropped = B.clean_bounces(bs, FPS, gate)
    assert [b["frame"] for b in kept] == [10, 60, 80] and dropped == 2


def test_detect_bounces_skips_contacts(cam, gate):
    a = arc((5.0, 2.0, 1.0), (0.0, 14.0, 2.0), n=30)
    k = int(np.argmax(a[:, 2] < 0)); a = a[:k]
    b = arc(a[-1] * [1, 1, 0], (0.0, 10.0, 4.0), n=20)
    pos = as_pos(cam.project(np.r_[a, b[1:]]))
    _, trs = L.filter_live(pos, FPS, gate, hits=[0.0], t0=0.0)
    bo = B.detect_bounces(trs, FPS, hits=[0.0])
    assert len(bo) == 1 and abs(bo[0]["frame"] - (len(a) - 1)) <= 2
    rk = bo[0]["t"]
    assert B.detect_bounces(trs, FPS, hits=[rk]) == []                  # a reversal at a hit is a contact


def test_kalman_and_chain():
    kf = Kalman((0, 0))
    for i in range(1, 15):
        kf.predict(); kf.update((3.0 * i, 0.5 * i * i))
    p = kf.predict()
    assert abs(p[0] - 45) < 2 and abs(p[1] - 112.5) < 4
    rng = np.random.default_rng(3)
    blobs = [[(100 + 5 * k, 100 + 2 * k, 0.9)] + [tuple(rng.uniform(0, 900, 2)) + (0.6,) for _ in range(2)] for k in range(8)]
    ch = chain(blobs, max_step=20)
    assert len(ch) == 8 and all(abs(ch[k][0] - (100 + 5 * k)) < 1e-6 for k in ch)


def _render(cam, X, n, bg=(60, 90, 170)):
    """Clay-coloured frames with a yellow ball drawn at the projected 3D positions (None = not drawn)."""
    import cv2
    rng = np.random.default_rng(4)
    base = np.clip(np.array(bg, float) + rng.normal(0, 4, (H, W, 3)), 0, 255).astype(np.uint8)
    frames = []
    for i in range(n):
        f = base.copy()
        if X[i] is not None:
            p = cam.project([X[i]])[0]
            d = max(3.0, float(np.linalg.norm(cam.project([X[i] - [0.0335, 0, 0]]) - cam.project([X[i] + [0.0335, 0, 0]]))))
            cv2.circle(f, (int(round(p[0])), int(round(p[1]))), int(round(d / 2)), (40, 230, 220), -1)
        frames.append(f)
    return frames


def test_engine_tracks_rendered_arc_and_ignores_static_ball(cam, gate):
    Xa = arc((5.0, 1.0, 1.0), (0.3, 15.0, 4.0), n=36)
    X = [None] * 4 + list(Xa) + [None] * 4
    frames = _render(cam, X, len(X))
    import cv2
    s = cam.project([[2.0, 6.0, 0.0]])[0]                      # a ball lying on the court: no motion, never a candidate
    for f in frames: cv2.circle(f, (int(s[0]), int(s[1])), 3, (40, 230, 220), -1)
    r = BallEngine(gate, None, (), FPS, 0.0, hits=[4 / FPS], reacquire="hsv_full").run(frames)
    live = ~np.isnan(r["pos"][:, 0])
    assert live[8:38].mean() > 0.8, r["stats"]
    d = np.hypot(*(r["pos"][8:38][live[8:38]] - cam.project(Xa)[4:34][live[8:38]]).T)
    assert np.median(d) < 2.0
    assert not live[:3].any() and r["conf"][live].min() >= 0.6
    assert r["stats"]["roi_pixel_frac"] < 0.2


def test_proxy_metrics():
    pos = np.full((90, 2), np.nan); pos[10:20] = 5.0
    assert P.event_detection_rate(pos, 0.0, FPS, [0.4, 2.5]) == (1, 2)
    assert P.post_hit_coverage(pos, 0.0, FPS, [10 / FPS], span=0.3) == (10, 10)
    clips = [(0.0, 10.0), (10.0, 18.0)]
    points = [[100.0 + t for t in (1.0, 2.0, 3.5, 5.0)], [300.0 + t for t in (0.8, 2.2, 4.0)]]
    onsets = [1.0, 2.0, 3.5, 5.0, 10.8, 12.2, 14.0, 7.7]
    al = P.align_clips(clips, points, onsets, step=0.05, search=2.0)
    assert [a["point"] for a in al] == [0, 1]
    assert abs(al[0]["offset"] - 100.0) < 0.06 and abs(al[1]["offset"] - 290.0) < 0.06


def test_tracknet3_runs_if_weights_present():
    torch = pytest.importorskip("torch")
    import os

    from opencourtvision.ball import tracknet3 as T
    w = os.path.join(T.cache_dir(), "TrackNet_best.pt")
    if not os.path.exists(w): pytest.skip("TrackNetV3 weights not downloaded (ocv-fetch-weights tracknetv3)")
    tn = T.TrackNetV3(device="cpu", threads=max(1, (torch.get_num_threads() or 1)))
    fr = [np.full((270, 480, 3), 90, np.uint8) for _ in range(5)]
    hm = tn.heatmaps(fr)
    assert hm.shape == (5, T.H, T.W) and np.isfinite(hm).all()
    assert all(isinstance(b, list) for b in tn.detect(fr))


def test_label_csv_roundtrip(tmp_path):
    from opencourtvision.cli import label_ball as LB
    f = tmp_path / "l.csv"
    LB.write(str(f), [dict(clip="3", frame=0, t="10.0000", x="100", y="50", visibility="1", status="ok"),
                      dict(clip="3", frame=3, t="10.1000", x="", y="", visibility="0", status="ok"),
                      dict(clip="3", frame=6, t="10.2000", x="1", y="1", visibility="1", status="pred")])
    gt = LB.ground_truth(LB.read(str(f)))
    assert gt == {("3", 10.0): (100.0, 50.0), ("3", 10.1): None}
    assert LB.main(["check", "--labels", str(f)]) == 0


def test_label_metrics():
    from opencourtvision.cli.bench_ball import label_metrics
    pos = np.full((5, 2), np.nan); pos[0] = (10, 10); pos[1] = (50, 50); pos[3] = (7, 7)
    m = label_metrics(pos, {0: (11, 10), 1: (10, 10), 2: (5, 5), 3: None, 4: None}, tol_px=4)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 2)


def test_transfer_homography_recovers_digital_crop():
    import cv2

    from opencourtvision.ball.geometry import apply, transfer_homography
    rng = np.random.default_rng(5)
    ref = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (5, 5), 0)
    for _ in range(300):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.rectangle(ref, (int(x), int(y)), (int(x + rng.integers(5, 40)), int(y + rng.integers(5, 40))), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    S = np.array([[1.05, 0, -30.0], [0, 1.05, 12.0], [0, 0, 1]])
    frame = cv2.warpPerspective(ref, S, (W, H))
    H0 = Cam().H()
    Hk, info = transfer_homography(ref, frame, H0)
    assert Hk is not None and abs(info["scale"] - 1.05) < 0.01
    pts = np.array([[5.0, 5.0], [10.0, 20.0]])
    np.testing.assert_allclose(apply(Hk, pts), apply(S @ H0, pts), atol=1.5)
