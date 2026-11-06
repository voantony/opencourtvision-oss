import numpy as np

from opencourtvision.editloop import learned as LN


def test_gbdt_and_lr_learn_and_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)); y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(float)
    for m in (LN.GBDT(n=30, depth=2, min_leaf=10), LN.LR()):
        m.fit(X, y)
        acc = ((m.predict(X) > 0.5) == y).mean()
        assert acc > 0.9
        m2 = LN.model_from_json(m.to_json())
        assert np.allclose(m2.predict(X), m.predict(X))


def _clips():
    return [dict(t0=10.0, t1=14.0, A=8.4, B=16.8, hits=4, parts=[(8.4, 16.8)], quiet_serve=False, reaction_db=0.0),
            dict(t0=30.0, t1=33.0, A=28.4, B=35.8, hits=3, parts=[(28.4, 35.8)], quiet_serve=False, reaction_db=0.0)]


def test_refine_uses_learned_serve_and_last_contact():
    t = np.array([8.8, 10.0, 11.0, 12.5, 14.0, 15.5, 30.0, 31.0, 33.0])
    p_in = np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])     # 15.5 is in play: the candidate ended early
    p_sv = np.array([0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1])     # 8.8 is a quiet serve before the first hit
    R = LN.refine(_clips(), t, t * 0, p_in, p_sv)
    assert abs(R[0]["t0"] - 8.8) < 1e-9 and abs(R[0]["t1"] - 15.5) < 1e-9
    assert R[0]["A"] < 8.8 - 0.5 and R[0]["B"] >= 15.5 + 1.5
    assert R[0]["B"] + 0.2 <= R[1]["A"] + 1e-9                          # never overlaps the next clip
    U = LN.refine(_clips(), t, t * 0, p_in * 0 + 0.9, p_sv * 0, s=dict(union=True))
    for c, r in zip(_clips(), U):                                       # union never starts later / ends earlier
        assert r["A"] <= c["A"] + 1e-9 and r["B"] >= c["B"] - 1e-9


def test_hybrid_filter_drops_and_keeps_order():
    t = np.array([10.0, 11.0, 14.0, 30.0, 31.0, 33.0]); p = np.full(6, 0.9)

    class Const:
        def predict(self, F):
            return np.array([0.9, 0.1])[: len(F)]
    keep, drop = LN.hybrid_points(_clips(), t, t * 0, p, p * 0, Const())
    assert len(keep) == 1 and len(drop) == 1 and keep[0]["t0"] == 10.0
    assert LN.cand_features(keep).shape == (1, len(LN.CAND_FEATURES))


def test_learned_module_is_audio_only():
    import inspect
    src = inspect.getsource(LN).lower()
    for w in ("motion", "track", "ball_", "player"):
        assert w not in src.replace("no ball, player or motion input", "")
