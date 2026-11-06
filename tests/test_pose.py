import json

import numpy as np
import pytest

from opencourtvision.pose import compare as C
from opencourtvision.pose import metrics as M
from opencourtvision.pose import strokes as S
from opencourtvision.pose import syllables as SY
from opencourtvision.pose import synthetic as syn
from opencourtvision.pose.analyze import analyze
from opencourtvision.pose.court import Court, homography
from opencourtvision.pose.estimate import PoseEstimator, extract, load_keypoints, pick_player, save_keypoints
from opencourtvision.pose.skeleton import COCO17, JOINTS, from_layout, interpolate_gaps


def _rally(**kw):
    return syn.rally([dict(t=1.5, kind="serve", lean=kw.get("lean", 25)), dict(t=4.0, kind="forehand", stance=kw.get("stance", 50),
                      contact=kw.get("contact", 55), finish=kw.get("finish", -15)), dict(t=6.5, kind="backhand", stance=20)],
                     noise_px=kw.get("noise", 0.0), seed=kw.get("seed", 0))


def test_homography_roundtrip_and_scale():
    court = Court(syn.synthetic_court())
    p = syn.court_to_px(300, 700)
    assert np.allclose(court.to_court(p), [3.0, 7.0], atol=1e-6)
    assert np.isclose(court.cm_per_px(p[None])[0], 1 / syn.S, rtol=1e-6)
    H = homography([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (2, 0), (2, 2), (0, 2)])
    assert np.allclose(H @ [1, 1, 1] / (H @ [1, 1, 1])[2], [2, 2, 1])
    with pytest.raises(ValueError):
        Court.parse("1,2,3")


def test_layout_mapping_and_gap_fill():
    coco = np.random.default_rng(0).random((5, 17, 2)); conf = np.ones((5, 17)); conf[2, 9] = 0.1
    kp, c = from_layout(coco, conf, "coco17")
    assert kp.shape == (5, len(JOINTS), 2) and np.isnan(kp[:, JOINTS.index("l_heel")]).all()
    assert np.allclose(kp[0, JOINTS.index("l_wrist")], coco[0, COCO17["l_wrist"]]) and np.isnan(kp[2, JOINTS.index("l_wrist")]).all()
    filled = interpolate_gaps(kp, 2)
    assert np.isfinite(filled[2, JOINTS.index("l_wrist")]).all()


def test_stroke_classification_with_audio_hits():
    r = _rally(noise=1.0)
    st = S.strokes(r["kp"], r["fps"], r["hits"] + [8.5])           # 8.5 s: a hit with no swing (the other player)
    assert [s["kind"] for s in st] == r["kinds"]
    assert all(s["confidence"] >= 0.6 for s in st)


def test_left_handed_mirror():
    r = syn.rally([dict(t=2.0, kind="forehand"), dict(t=4.5, kind="backhand")], hand="left")
    assert [s["kind"] for s in S.strokes(r["kp"], r["fps"], r["hits"], hand="left")] == ["forehand", "backhand"]


def test_metrics_recover_synthetic_parameters():
    r = _rally(stance=50, contact=60, finish=5, lean=30)
    court = Court(r["court"])
    st = {s["kind"]: M.stroke_metrics(r["kp"], r["fps"], s, court) for s in S.strokes(r["kp"], r["fps"], r["hits"])}
    fh = st["forehand"]
    assert abs(fh["stance_deg"] - 50) < 1 and abs(fh["contact_cm"] - 60) < 2 and abs(fh["finish_cm"] - 5) < 2
    assert abs(st["serve"]["lean_cm"] - 30) < 2 and fh["lean_cm"] is None
    fw = M.footwork(r["kp"], r["fps"], court)
    assert fw["distance_m"] > 1 and fw["coverage_m2"] >= 0
    # without a court, the torso-length fallback lands in the right ballpark
    fh2 = M.stroke_metrics(r["kp"], r["fps"], dict(t=4.0, kind="forehand"), None)
    assert 40 < fh2["contact_cm"] < 80


def test_compare_detects_era_change_and_ignores_noise():
    rng = np.random.default_rng(1); rows = []
    for g, mu in (("early", 54.0), ("late", 62.0)):
        for s in range(6):
            for _ in range(15):
                rows.append(dict(group=g, session=f"{g}{s}", contact_cm=rng.normal(mu, 6), finish_cm=rng.normal(0, 8), zoom=rng.random()))
    res = {r["metric"]: r for r in C.compare(rows, ["contact_cm", "finish_cm"], covariates=["zoom"])}
    assert res["contact_cm"]["changed"] and res["contact_cm"]["stars"]
    assert not res["finish_cm"]["changed"]
    assert np.allclose(C.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_residualize_removes_covariate():
    x = np.linspace(0, 1, 50); y = 3 * x + 10
    assert np.std(C.residualize(y, x)) < 1e-9


def test_syllables_find_movement_types():
    kp, truth, names = syn.movement_sequence(seed=0)
    r = SY.fit(kp, 30.0, k=10)
    assert SY.purity(r["labels"], truth) >= 0.7
    inst = r["instances"]
    assert len(inst) >= len(names) and all(0 <= d["x"] <= 1 and 0 <= d["y"] <= 1 for d in inst)
    assert abs(sum(d["dur"] for d in inst) - len(kp) / 30) < 0.1


class _FakeEstimator(PoseEstimator):
    """Mock backend: returns the synthetic player plus a decoy higher up the image (the far player)."""
    def __init__(self, kp): self.kp, self.i = kp, 0
    def __call__(self, frame):
        p = self.kp[self.i]; self.i += 1; decoy = p.copy(); decoy[..., 1] -= 600
        return [(decoy, np.ones(17)), (p, np.ones(17))]


def test_extract_with_mock_backend_tracks_near_player(tmp_path):
    r = _rally()
    frames = (np.zeros((4, 4, 3), np.uint8) for _ in range(len(r["kp"])))
    kp, conf, fps = extract(frames, _FakeEstimator(r["kp"]), 30.0)
    assert np.allclose(kp, r["kp"]) and fps == 30.0
    f = tmp_path / "kp.npz"; save_keypoints(f, kp, conf, fps, (1000, 1400))
    d = load_keypoints(f); assert d["kp"].shape == kp.shape and d["size"] == (1000, 1400)
    assert pick_player([], None) is None


def test_analyze_events_verified_only_with_audio(tmp_path):
    r = _rally()
    res = analyze(r["kp"], r["fps"], r["hits"], Court(r["court"]), do_syllables=False)
    assert len(res["events"]) == 3 and all(e["verified"] for e in res["events"])
    assert res["events"][0]["label"]["shot_type"] == "serve" and res["events"][1]["label"]["position"].startswith("near")
    res2 = analyze(r["kp"], r["fps"], None, None, do_syllables=False)        # pose-only: one source -> unverified
    assert res2["events"] and not any(e["verified"] for e in res2["events"])
    json.dumps(res, default=float)
