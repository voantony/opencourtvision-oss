"""opencourtvision.api: analyze / reinfer / render contract, stub fixtures, real engine on a synthetic clip."""
import json

import pytest

from opencourtvision import api
from opencourtvision.api.engine import score_timeline


@pytest.fixture()
def stub():
    return api.analyze(None, {"engine": "stub", "n_points": 24, "seed": 7})


def test_stub_shape_and_honest_meta(stub):
    d = stub.to_dict()
    assert d["schema"] == api.SCHEMA and d["meta"]["engine"] == "stub"
    assert {"points", "players", "uncertain_queue_source"} <= set(d["meta"]["stubbed"])
    for k in ("points", "score_timeline", "uncertain_queue", "clips", "reel_plan", "players"):
        assert d[k], k
    assert len(d["score_timeline"]) == len(d["points"]) == 24
    assert d["reel_plan"]["schema"] == "vamos.reel_plan.v1"
    json.dumps(d)                                             # JSON-safe (ScoreState serialised)
    assert api.MatchResult.from_dict(json.loads(json.dumps(d))).to_dict() == d


def test_stub_is_deterministic():
    a = api.analyze(None, {"engine": "stub", "seed": 3}).to_dict()
    b = api.analyze(None, {"engine": "stub", "seed": 3}).to_dict()
    assert a == b


def test_players_are_roles_never_faces(stub):
    blob = json.dumps(stub.to_dict()).lower()
    for bad in ("face", "embedding", "biometric"):
        assert bad not in blob
    assert {p["identity"] for p in stub.players} == {"court_side_role"}


def test_score_timeline_is_rules_exact():
    pts = [dict(id=i, first_hit=float(i), last_hit=i + 1.0, winner="A") for i in range(1, 5)]
    tl, _ = score_timeline(pts, {"sport": "tennis"})
    assert [e["before"]["call"] for e in tl] == ["0-0", "15-0", "30-0", "40-0"]
    assert tl[-1]["after"]["games"] == [1, 0] and all(e["verified"] for e in tl)


def test_unknown_winner_marks_rest_unverified():
    pts = [dict(id=1, first_hit=1.0, last_hit=2.0, winner="A"), dict(id=2, first_hit=3.0, last_hit=4.0, winner=None),
           dict(id=3, first_hit=5.0, last_hit=6.0, winner="B")]
    tl, _ = score_timeline(pts, {"sport": "tennis"})
    assert [e["verified"] for e in tl] == [True, False, False] and tl[1]["note"] == "winner_unknown"


def test_reinfer_point_winner_updates_score_and_queue(stub):
    item = next(x for x in stub.uncertain_queue if x["kind"] == "point_winner")
    p = next(p for p in stub.points if p["id"] == item["point"])
    flip = "B" if p["winner"] == "A" else "A"
    before = json.dumps(stub.to_dict())
    m2 = api.reinfer(stub, [{"item_id": item["id"], "value": flip}])
    assert json.dumps(stub.to_dict()) == before                       # pure: input untouched
    p2 = next(p for p in m2.points if p["id"] == item["point"])
    assert p2["winner"] == flip and p2["winner_confidence"] == 1.0 and p2["winner_src"] == "user"
    assert item["id"] not in {x["id"] for x in m2.uncertain_queue}
    assert m2.meta["n_corrections"] == 1 and item["id"] in m2.meta["fixed"]
    i = [e["point"] for e in m2.score_timeline].index(item["point"])
    assert m2.score_timeline[i]["after"] != stub.score_timeline[i]["after"] or m2.score_timeline[i]["winner"] == flip


def test_reinfer_by_kind_and_point_and_dict_input(stub):
    m2 = api.reinfer(stub.to_dict(), [{"kind": "bounce", "point": 2, "value": {"x": 0.4, "y": 0.9}},
                                      {"kind": "cut_off", "point": 3, "value": True}])
    assert next(p for p in m2.points if p["id"] == 2)["bounce"] == {"x": 0.4, "y": 0.9}
    assert 3 not in {c["point"] for c in m2.clips}


@pytest.mark.parametrize("bad", [
    {"item_id": "pw-1", "value": "C"},
    {"item_id": "zz-1", "value": "A"},
    {"kind": "point_winner", "point": 999, "value": "A"},
    {"kind": "point_winner", "point": 1, "value": "A", "face_embedding": [0.1]},
    {"kind": "point_winner", "point": 1, "value": "A", "player_name": "x"},
    {"kind": "point_winner", "point": 1},
    {"kind": "point_winner", "point": 1, "value": "A", "source": "server"},
])
def test_reinfer_rejects_bad_corrections_atomically(stub, bad):
    with pytest.raises(api.InvalidCorrection):
        api.reinfer(stub, [{"kind": "point_winner", "point": 1, "value": "B"}, bad])


def test_render_stub_returns_plan_only_and_overlay_toggle(stub):
    on = api.render(stub, "vamos_clean_v1", {"score_overlay": True, "long_shot_mode": "receiver"})
    off = api.render(stub, "vamos_clean_v1", {"score_overlay": False})
    assert on["rendered"] is False and "encode" in on["stubbed"] and on["plan"]["long_shot_mode"] == "receiver"
    assert off["plan"]["bugs"] == [] and off["plan"]["captions"] == [] and off["plan"]["score_overlay"] is False
    with pytest.raises(ValueError):
        api.render(stub, "vamos_clean_v1", {"long_shot_mode": "drone"})


def test_auto_falls_back_to_stub_with_reason(tmp_path):
    m = api.analyze(str(tmp_path / "missing.mp4"), {"engine": "auto"})
    assert m.meta["engine"] == "stub" and m.meta["fallback_reason"] == "video not found"
    with pytest.raises(ValueError):
        api.analyze(None, {"engine": "cloud"})


def test_real_engine_on_synthetic_clip(clips, tmp_path):
    m = api.analyze(clips["match_sim"]["mp4"], {"engine": "real", "work": str(tmp_path / "w"), "mode": "rallies"})
    assert m.meta["engine"] == "real" and m.meta["stubbed"] == ["players"]
    assert m.points and len(m.score_timeline) == len(m.points)
    json.dumps(m.to_dict())
    if m.uncertain_queue:
        it = m.uncertain_queue[0]
        val = {"point_winner": "A", "bounce": "none", "player_id": {"near": "A"}}[it["kind"]]
        m2 = api.reinfer(m, [{"item_id": it["id"], "value": val}])
        assert it["id"] not in {x["id"] for x in m2.uncertain_queue}
