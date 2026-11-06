import json

import pytest

from opencourtvision import api
from opencourtvision.api import features as F
from opencourtvision.api.types import InvalidCorrection

# index map copied from feature/shared-training opencourtvision/shared/features.py (vamos.feat.v1)
SHARED_TAGS = ("low_light", "shaky", "behind_fence", "portrait_phone", "narrow_fov", "low_resolution")


def _m(**kw):
    return api.analyze(None, dict(engine="stub", n_points=20, seed=3, **kw))


def test_names_match_collector_contract():
    assert api.FEATURES_VERSION == "vamos.feat.v1" and api.FEATURE_DIM == 16 == len(api.FEATURE_NAMES)
    assert F.TAGS == SHARED_TAGS and api.FEATURE_NAMES[9] == "confidence" and api.FEATURE_NAMES[10:] == tuple(f"tag_{t}" for t in SHARED_TAGS)


def test_schema_and_vectors():
    m = _m()
    assert m.schema == "vamos.match_result.v1.1" and m.features_version == "vamos.feat.v1"
    assert m.uncertain_queue
    for it in m.uncertain_queue:
        assert F.valid(it["features"]) and it["features_version"] == "vamos.feat.v1"
        assert it["features"][9] == pytest.approx(it["confidence"], abs=1e-4)
    for p in m.points:
        assert F.valid(p["features"]) and len(p["features_src"]) == 16
        assert p["features"][0] == pytest.approx(p["hits"] / 20, abs=1e-4)
        assert p["features"][5] in (1.0, -1.0) and p["features"][6] in (1.0, -1.0)


def test_first_point_serve_side_and_server():
    p = min(_m().points, key=lambda p: p["first_hit"])
    assert p["features"][5] == 1.0 and p["features"][6] == 1.0   # deuce side, A serves first


def test_stub_deterministic():
    assert _m().to_dict() == _m().to_dict()
    assert all(s == "stub" for s in (_m().points[0]["features_src"][i] for i in F.STUB_INDICES))


def test_json_round_trip():
    d = _m().to_dict()
    s = json.dumps(d)
    m2 = api.MatchResult.from_dict(json.loads(s))
    assert m2.to_dict() == d


def test_reads_v1_and_reinfer_fills_features():
    d = _m().to_dict()
    d["schema"] = "vamos.match_result.v1"; d.pop("features_version")
    for it in d["uncertain_queue"]: it.pop("features"); it.pop("features_version")
    m = api.MatchResult.from_dict(d)
    assert m.schema == "vamos.match_result.v1.1"
    m2 = api.reinfer(m, [])
    assert all(F.valid(it["features"]) for it in m2.uncertain_queue)


def test_reinfer_updates_features_and_still_refuses_private_fields():
    m = _m()
    pid = m.points[0]["id"]
    m2 = api.reinfer(m, [dict(kind="point_winner", point=pid, value="B")])
    p = next(x for x in m2.points if x["id"] == pid)
    assert p["features"][9] == 1.0 and F.valid(p["features"])
    for bad in ("face", "embedding", "biometric", "player_name", "video", "image"):
        with pytest.raises(InvalidCorrection):
            api.reinfer(m, [{"kind": "point_winner", "point": pid, "value": "A", bad: 1}])
    with pytest.raises(InvalidCorrection):
        api.reinfer(m, [{"kind": "point_winner", "point": pid, "value": "A", "features": [0.0] * 16}])


def test_fix_example_is_share_v2_shape():
    it = _m().uncertain_queue[0]
    ex = api.fix_example(it, "A", "fix-00000001")
    assert set(ex) == {"client_id", "kind", "value", "features", "struggle_tags", "source"}
    assert len(ex["features"]) == 16 and ex["source"] == "user"


def test_struggle_tags_and_clamp():
    v, _ = F.point_features(dict(id=1, hits=10_000, react=float("nan")), None, tags=["shaky", "narrow_fov"])
    assert v[0] == 10.0 and v[4] == 0.0 and v[11] == 1.0 and v[14] == 1.0 and F.valid(v)
