import numpy as np

from opencourtvision.reel.ball import detect, path, track, zoom_at


def _frames(n=120, w=1280, h=720):
    """Grey court with a yellow ball moving left->right along a parabola."""
    for i in range(n):
        f = np.full((h, w, 3), 90, np.uint8)
        x = 100 + i * 9
        y = int(200 + 0.02 * (i - 60) ** 2)
        f[y - 3:y + 4, x - 3:x + 4] = (40, 230, 230)  # BGR yellow
        yield f


def test_detect_and_track_synthetic_ball():
    cands, act, _, _ = detect(_frames())
    pos = track(cands)
    ok = ~np.isnan(pos[:, 0])
    assert ok.mean() > 0.9
    i = np.where(ok)[0][-1]
    assert abs(pos[i, 0] - (100 + i * 9)) < 3


def test_path_bounded_and_smooth():
    n = 300
    pos = np.column_stack([np.linspace(100, 1200, n), np.full(n, 300.0)])
    pos[100:140] = np.nan
    cx, bx = path(pos, np.full(n, np.nan), 1920.0, 3.0)
    assert np.isfinite(cx).all()
    assert np.abs(np.diff(cx)).max() <= 20.0 + 1e-6  # velocity limit
    assert cx[-1] > cx[0]


def test_zoom_ramps():
    ts = np.arange(0, 3, 1 / 60)
    z = zoom_at(ts, [1.5], False)
    assert z.max() == np.float64(1.08) and z[0] == 1.0
    assert (zoom_at(ts, [], True) == 1.08).all()
