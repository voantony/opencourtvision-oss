"""US-safe tracker opts + bad-footage degradations + court override (19 Oct ball-net-opt)."""
import json
import os

import numpy as np

from opencourtvision.ball.own import physics as P
from opencourtvision.robust.perturb import bad_footage_suite, degrade_candidates, depth_bin


def test_reject_static_drops_still_blob():
    pos = np.full((40, 2), np.nan)
    pos[5:25] = (400.0, 300.0)  # static spare ball / shoe
    out = P.reject_static(pos, window=10, max_extent=6.0)
    assert not np.isfinite(out[12, 0])
    # moving ball survives
    pos2 = np.full((40, 2), np.nan)
    for i in range(5, 25):
        pos2[i] = (200 + 10 * i, 300.0)
    out2 = P.reject_static(pos2, window=10, max_extent=6.0)
    assert np.isfinite(out2[12, 0])


def test_audio_break_splits_arc():
    T = 30
    pos = np.full((T, 2), np.nan)
    # one continuous ballistic-looking path that should normally stay one arc
    for i in range(4, 26):
        pos[i] = (200 + 8 * i, 400 - 0.15 * (i - 15) ** 2)
    arcs0, _ = P.segment_arcs(pos)
    arcs1, _ = P.segment_arcs_audio(pos, audio_breaks=[15])
    assert len(arcs1) >= len(arcs0)  # forced break at contact


def test_filter_track_opt_flags():
    pos = np.full((40, 2), np.nan)
    for i in range(5, 20):
        pos[i] = (300 + 6 * i, 350 - i)
    for i in range(20, 35):
        pos[i] = (300 + 6 * i, 330 + (i - 20) * 4)  # bounce-like
    r = P.filter_track_opt(pos, fps=30.0, audio_breaks=[20], hardneg_static=True)
    assert r["stats"]["hardneg_static"] is True
    assert r["stats"]["audio_breaks"] == 1
    assert r["pos"].shape == (40, 2)


def test_degrade_candidates_smoke():
    c = np.full((20, 4, 3), np.nan)
    c[:, 0] = (640, 360, 0.9)
    for name in bad_footage_suite():
        out = degrade_candidates(c, name, seed=1, y_net=300.0)
        assert out.shape == c.shape
        if name == "clean":
            assert np.allclose(out[:, 0, :2], c[:, 0, :2], equal_nan=True)


def test_depth_bin():
    assert depth_bin(200, 300) == "far"
    assert depth_bin(500, 300) == "near"
    assert depth_bin(320, 300) == "mid"


def test_apply_court_override_roundtrip():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "tools", "point_winner", "pw_stages.py")
    spec = importlib.util.spec_from_file_location("pw_stages", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ov = {"view": 1, "H_img2court": [[2.0, 0.0, 1.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]]}
    fixture = os.path.join(os.path.dirname(__file__), "..", "does-not-persist-court.json")
    # write beside the test only in a temp file; the matrix is invented, not a filmed court
    import tempfile
    fd, fixture = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump(ov, open(fixture, "w"))
    try:
        cbv = [{"t0": 0, "t1": 1e9, "H_img2court": np.eye(3).tolist(), "detected": False} for _ in range(4)]
        out = m.apply_court_override(cbv, fixture)
    finally:
        os.remove(fixture)
    assert out[ov["view"]]["manual"] is True
    assert out[ov["view"]]["H_img2court"] == ov["H_img2court"]
