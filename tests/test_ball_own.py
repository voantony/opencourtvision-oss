"""Own ball tracker: physics filter (numpy) and heatmap net (skipped without torch)."""
import numpy as np
import pytest

from opencourtvision.ball.own import physics as P
from opencourtvision.ball.own.heatnet import decode_heatmap, gaussian_target, macs


def _rally(seed=0, noise=0.7):
    """Synthetic image track at 1280x720/30 fps: far hit -> bounce near -> near hit -> far, plus clutter."""
    rng = np.random.default_rng(seed)
    T = 90
    pos = np.full((T, 2), np.nan)
    # arc 1 (frames 0-29): coming towards camera, falling (vy > 0, accelerating)
    t = np.arange(0, 30); pos[t] = np.c_[600 + 4 * t, 250 + 6 * t + 0.25 * t ** 2]
    # arc 2 (30-49): after the bounce, still coming in but rising (vy drops sharply), same x direction
    e1 = pos[29] + [4, 6 + 0.5 * 29]
    t2 = np.arange(0, 20); pos[30 + t2] = np.c_[e1[0] + 4 * t2, e1[1] + 1.0 * t2 + 0.2 * t2 ** 2]
    # arc 3 (50-89): hit by the near player, going away (vy < 0)
    e2 = pos[49]
    t3 = np.arange(1, 41); pos[49 + t3] = np.c_[e2[0] - 3 * t3, e2[1] - 14 * t3 + 0.22 * t3 ** 2]
    pos += rng.normal(0, noise, pos.shape)
    drop = rng.random(T) < 0.15
    pos[drop] = np.nan
    pos[[12, 63]] = [[200, 650], [1100, 80]]       # two false detections
    return pos


def test_arcs_reject_outliers_and_find_events():
    pos = _rally()
    r = P.filter_track(pos, 30.0, boxes_at=lambda i: [(770, 560, 860, 760)] if 45 <= i <= 53 else [], y_net=300)
    assert 12 in r["outliers"] and 63 in r["outliers"]
    assert 2 <= r["stats"]["arcs"] <= 4
    kinds = [e["kind"] for e in r["events"]]
    assert "bounce" in kinds and "hit" in kinds
    b = [e for e in r["events"] if e["kind"] == "bounce"][0]
    assert abs(b["i"] - 29.5) <= 2
    h = [e for e in r["events"] if e["kind"] == "hit"][0]
    assert abs(h["i"] - 49) <= 2 and h["near_player"]


def test_densify_fills_inside_arcs_only():
    pos = _rally(1)
    r = P.filter_track(pos, 30.0)
    w = r["conf"]
    assert (w == P.FILL_W).sum() > 0
    filled = np.nonzero(w == P.FILL_W)[0]
    assert all(any(a.i0 <= i <= a.i1 for a in r["arcs"]) for i in filled)
    assert np.isnan(r["pos"][12, 0]) or np.hypot(*(r["pos"][12] - [200, 650])) > 50


def test_static_ball_is_not_an_arc():
    pos = np.c_[np.full(30, 400.0), np.full(30, 500.0)] + np.random.default_rng(2).normal(0, 0.5, (30, 2))
    arcs, outl = P.segment_arcs(pos)
    assert arcs == [] and len(outl) == 30


def test_heatmap_roundtrip():
    hm = gaussian_target(432, 768, (100.3, 200.7))
    (x, y, p), = decode_heatmap(hm, 0.5)
    assert abs(x - 100.3) < 0.6 and abs(y - 200.7) < 0.6 and p > 0.9
    assert decode_heatmap(gaussian_target(432, 768, None), 0.5) == []
    assert 3e9 < macs(16) < 1e10


def test_model_shapes_and_loss():
    torch = pytest.importorskip("torch")
    from opencourtvision.ball.own.heatnet import build_model, focal_loss, param_count
    m = build_model(16)
    assert 2e5 < param_count(m) < 6e5
    y = m(torch.rand(2, 9, 64, 96))
    assert y.shape == (2, 1, 64, 96)
    tg = torch.from_numpy(np.stack([gaussian_target(64, 96, (40, 30))] * 2))[:, None]
    loss = focal_loss(y, tg)
    loss.backward()
    assert np.isfinite(loss.item())


def test_court_roi_and_fast_modes():
    torch = pytest.importorskip("torch")
    from opencourtvision.ball.own.fastpath import FastRunner, court_roi
    from opencourtvision.ball.own.heatnet import build_model
    assert court_roi(None, (768, 432)) == (0, 0, 768, 432)
    H = np.array([[30.0, 0, 100], [0, -8, 400], [0, 0, 1.0]])
    x0, y0, x1, y1 = court_roi(H, (768, 432))
    assert all(v % 8 == 0 for v in (x0, y0, x1, y1)) and 0 <= x0 < x1 <= 768 and 0 <= y0 < y1 <= 432

    class T:                                             # stand-in for HeatNetTracker (random weights, CPU)
        device, th = "cpu", 0.0
    T.torch = torch
    T.model = build_model(8).eval()
    small = [np.random.default_rng(i).random((3, 96, 160)).astype(np.float32) for i in range(5)]
    small[3] = small[1]                                  # frames 2's neighbours are identical -> no motion at frame 2
    for mode in ("plain", "court", "motion", "track"):
        pos, st = FastRunner(T).run(small, mode, roi=(0, 0, 160, 96))
        assert pos.shape == (5, 2) and 0 < st["px_share"] <= 1.0
    pos, st = FastRunner(T).run(small, "motion", roi=(0, 0, 160, 96))
    assert np.isnan(pos[2, 0]) and st["calls"] == 4
