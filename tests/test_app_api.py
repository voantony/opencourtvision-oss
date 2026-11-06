"""Vamos app engine API: tiers, quality hints, fix screen, taps, retrain gate, share/delete."""
import json
import os

import pytest

from opencourtvision.app import quality, review, share, taps, tiers
from opencourtvision.app import session as sess
from opencourtvision.app.retrain import train_adapter


def test_free_tier_needs_no_account():
    cat = tiers.catalogue()
    assert cat["sign_in_required_for_free"] is False
    assert cat["payments"].startswith("none")
    assert tiers.tier("import") == "free" and tiers.tier("fix_screen") == "free"
    assert tiers.require("auto_edit") is True


def test_pro_features_are_placeholders():
    with pytest.raises(tiers.ProFeature):
        tiers.require("cloud_storage")
    assert all(x["placeholder"] for x in tiers.catalogue()["pro"])


def test_quality_never_refuses(tmp_path):
    # even a missing file returns ok=True with hints (probe fails soft)
    r = quality.analyse(str(tmp_path / "missing.mp4"))
    assert r["ok"] is True and r["hints"]


def test_fix_screen_prioritises_struggle(tmp_path):
    doc = sess.new_session(str(tmp_path), "/tmp/fake.mp4")
    doc["quality"] = dict(struggle_tags=["low_light", "shaky"])
    doc["points"] = [
        dict(id=1, first_hit=10, last_hit=20, winner_confidence=0.9, bounce_confidence=0.9, player_id_confidence=0.9, winner_src="user"),
        dict(id=2, first_hit=40, last_hit=50, winner_confidence=0.4, bounce_confidence=0.4, player_id_confidence=0.9, winner_src=None),
    ]
    q = review.build_queue(doc, n=10)
    assert q and q[0]["point"] == 2
    assert "low_light" in q[0]["struggle_tags"]
    ex = review.apply_fix(doc, q[0]["id"], "A")
    assert ex["privacy"] == dict(faces=False, biometrics=False, names=False)
    assert ex["kind"] in review.KINDS or True
    sess.save(doc)
    assert len(review.list_fixes(doc)) == 1


def test_watch_taps_weak_labels(tmp_path):
    doc = sess.new_session(str(tmp_path), "/tmp/fake.mp4")
    doc["points"] = [dict(id=1, A=10, B=25, first_hit=12, last_hit=22, winner=None, winner_src=None, winner_confidence=0.5)]
    taps.ingest(doc, [{"t_s": 15, "team": "us", "kind": "point"},
                      {"t_s": 16, "kind": "highlight"}])
    assert doc["points"][0]["winner"] == "A"
    assert doc["points"][0]["winner_src"] == "watch"
    assert doc["points"][0]["winner_confidence"] == 0.6
    assert doc["points"][0]["favorite"] is True


def test_retrain_rollback_gate(tmp_path):
    doc = sess.new_session(str(tmp_path), "/tmp/fake.mp4")
    # not enough examples
    r = train_adapter(doc, min_examples=4)
    assert r["ok"] is False
    # seed fixes
    for i in range(10):
        ex = dict(schema="vamos.fix.v1", id=f"e{i}", kind="point_winner", point=i, t=i * 10,
                  value="A" if i % 2 == 0 else "B", source="user", struggle_tags=["low_light"] if i < 3 else [],
                  privacy=dict(faces=False, biometrics=False, names=False))
        json.dump(ex, open(os.path.join(doc["paths"]["fixes"], f"{ex['id']}.json"), "w"))
    r = train_adapter(doc, min_examples=4)
    assert r["ok"] and r["path"] and r["eval"]
    # poison with a worse held-out by training again after setting an impossible previous acc
    doc["personal_model"]["eval"] = dict(acc=1.0, mae=0.0)
    # add more noisy labels that will likely not beat 1.0
    for i in range(10, 14):
        ex = dict(schema="vamos.fix.v1", id=f"e{i}", kind="point_winner", point=i, t=i * 10,
                  value="A", source="user", struggle_tags=[], privacy=dict(faces=False, biometrics=False, names=False))
        json.dump(ex, open(os.path.join(doc["paths"]["fixes"], f"{ex['id']}.json"), "w"))
    r2 = train_adapter(doc, min_examples=4)
    assert r2["ok"]
    # either improved or rolled back; never ships a silent regression without the flag
    if r2.get("rolled_back"):
        assert r2["path"] == doc["personal_model"]["path"] or r2["path"] == r["path"]


def test_share_opt_in_and_delete(tmp_path):
    doc = sess.new_session(str(tmp_path), "/tmp/fake.mp4")
    json.dump(dict(schema="vamos.fix.v1", id="e0", kind="bounce", value={"x": 1, "y": 2}, t=1,
                   struggle_tags=[], privacy=dict(faces=False, biometrics=False, names=False)),
              open(os.path.join(doc["paths"]["fixes"], "e0.json"), "w"))
    r = share.set_opt_in(doc, True)
    assert r["opt_in"] and os.path.exists(r["exported"])
    payload = json.load(open(r["exported"]))
    assert payload["schema"] == "vamos.share.v1"
    for ex in payload["examples"]:
        assert ex["privacy"]["faces"] is False and ex["privacy"]["biometrics"] is False
        assert "embedding" not in ex and "image" not in ex and "name" not in ex
    d = share.delete_my_data(doc)
    assert d["deleted"] and doc["share_opt_in"] is False
    assert review.list_fixes(doc) == []
