import json

import pytest

from opencourtvision.eval.events import evaluate, evaluate_dirs, match_events
from opencourtvision.events import Event, caption, captions, load, save, verify
from opencourtvision.labels import ShotLabel, allowed_directions, court_position, validate, validate_rally
from opencourtvision.labels.synthetic import ground_truth, predictions
from opencourtvision.stages import STAGES, StageCache
from opencourtvision.windows import coverage, frame_mask, hit_times, windows


# ---- schema
def test_enum_parsing_and_roundtrip():
    lab = ShotLabel(position="near_ad", side="forehand", shot_type="swing", direction="inside_out", formation="non_serve", hand="right")
    assert ShotLabel.from_dict(lab.to_dict()) == lab
    with pytest.raises(ValueError):
        ShotLabel(shot_type="tweener")


@pytest.mark.parametrize("hand,pos,side,ok", [
    ("right", "near_deuce", "forehand", {"cross_court", "down_the_line"}),
    ("right", "near_deuce", "backhand", {"inside_in", "inside_out"}),
    ("right", "far_ad", "backhand", {"cross_court", "down_the_line"}),
    ("right", "far_ad", "forehand", {"inside_in", "inside_out"}),
    ("left", "near_ad", "forehand", {"cross_court", "down_the_line"}),
    ("left", "near_ad", "backhand", {"inside_in", "inside_out"}),
    ("left", "far_deuce", "backhand", {"cross_court", "down_the_line"}),
    ("left", "far_deuce", "forehand", {"inside_in", "inside_out"}),
])
def test_handedness_rules(hand, pos, side, ok):
    assert {str(d) for d in allowed_directions(pos, side, hand)} == ok
    for d in ("cross_court", "down_the_line", "inside_in", "inside_out"):
        errs = validate(dict(position=pos, side=side, shot_type="swing", direction=d, hand=hand))
        assert (not errs) == (d in ok)


def test_serve_and_formation_rules():
    assert not validate(dict(shot_type="serve", direction="T", formation="i_formation"))
    assert validate(dict(shot_type="serve", direction="cross_court"))
    assert validate(dict(shot_type="second_serve", formation="non_serve"))
    assert validate(dict(shot_type="volley", direction="wide"))
    assert validate(dict(shot_type="volley", formation="australian"))
    assert not validate(dict(shot_type="lob"))                     # partial labels are fine


def test_rally_rules():
    good = [dict(shot_type="serve"), dict(shot_type="return"), dict(shot_type="volley", outcome="win")]
    assert validate_rally(good) == []
    bad = [dict(shot_type="volley"), dict(shot_type="return", outcome="error"), dict(shot_type="swing")]
    msgs = [m for _, m in validate_rally(bad)]
    assert any("starts with a serve" in m for m in msgs) and any("follow a serve" in m for m in msgs) and any("last shot" in m for m in msgs)


def test_court_position_quadrants():
    assert str(court_position(8, 3)) == "near_deuce" and str(court_position(2, 3)) == "near_ad"
    assert str(court_position(2, 20)) == "far_deuce" and str(court_position(8, 20)) == "far_ad"


# ---- events + verify gate
def _ev(conf=0.9, ev=("audio: hit", "pose: wrist"), **lab):
    return Event("contact", 1.0, "P1", ShotLabel(**lab) if lab else None, conf, list(ev))


def test_verify_gate_blocks_captions():
    good = _ev(side="forehand", shot_type="volley", direction="cross_court", position="near_deuce", hand="right")
    low = _ev(0.4, side="forehand", shot_type="volley")
    one_source = _ev(ev=("pose: wrist",), side="backhand", shot_type="swing")
    bad_rule = _ev(side="backhand", shot_type="swing", direction="cross_court", position="near_deuce", hand="right")
    verify([good, low, one_source, bad_rule])
    assert good.verified and caption(good) == "Forehand volley cross court"
    for e in (low, one_source, bad_rule):
        assert not e.verified and caption(e) is None and any(x.startswith("verify:") for x in e.evidence)
    assert captions([good, low, one_source, bad_rule]) == [(1.0, "Forehand volley cross court")]
    e = _ev(shot_type="serve", direction="T"); e.verified = False
    assert caption(e) is None                                      # never captioned without verify()
    verify([e]); assert caption(e) == "Serve T"


def test_event_json_roundtrip(tmp_path):
    ev = [_ev(side="forehand", shot_type="swing"), Event("bounce", 2.0)]
    save(tmp_path / "e.json", ev, video="x"); back, meta = load(tmp_path / "e.json")
    assert meta["video"] == "x" and back[0].label == ev[0].label and back[1].label is None


# ---- eval
def test_match_tolerances():
    gt = [Event("contact", 1.0), Event("point_end", 5.0)]
    pred = [Event("contact", 1.07, confidence=0.9), Event("point_end", 5.14, confidence=0.9)]
    pairs, up, ug = match_events(pred, gt); assert len(pairs) == 2
    pred = [Event("contact", 1.09, confidence=0.9), Event("point_end", 5.16, confidence=0.9)]
    pairs, up, ug = match_events(pred, gt); assert not pairs and len(up) == 2 and len(ug) == 2
    pairs, _, _ = match_events([Event("bounce", 1.0, confidence=0.9)], [Event("contact", 1.0)]); assert not pairs


def test_perfect_and_corrupted_eval():
    gt = ground_truth(seed=3)
    rep = evaluate(gt, gt)
    assert rep["overall"]["f1"] == 1.0 and all(v["f1"] == 1.0 for v in rep["per_label"].values())
    rep = evaluate(predictions(gt, seed=4), gt)
    assert 0.6 < rep["overall"]["f1"] < 1.0 and set(rep["per_type"]) == {"contact", "bounce", "point_end"}
    cm = rep["stroke_confusion"]
    assert cm["forehand"]["backhand"] + cm["backhand"]["forehand"] > 0 and sum(cm[r]["none"] for r in cm) > 0
    assert sum(r["n"] for r in rep["calibration"]) == rep["n_pred"] and 0 <= rep["ece"] <= 1
    hi = [r for r in rep["calibration"] if r["n"] and r["mean_conf"] >= 0.8][0]
    lo = [r for r in rep["calibration"] if r["n"] and r["mean_conf"] < 0.6][0]
    assert hi["accuracy"] > lo["accuracy"]                         # confidence carries signal


def test_synthetic_labels_obey_rules():
    gt = ground_truth(seed=5)
    assert all(not validate(e.label) for e in gt if e.label)


def test_eval_dirs_and_cli(tmp_path, capsys):
    from opencourtvision.cli.eval import main
    gt = ground_truth(n_rallies=3, seed=9); (tmp_path / "labels").mkdir(); (tmp_path / "pred").mkdir()
    save(tmp_path / "labels" / "m.json", gt); save(tmp_path / "pred" / "m.json", predictions(gt, seed=2))
    rep = evaluate_dirs(str(tmp_path / "pred"), str(tmp_path / "labels")); assert rep["files"] == ["m.json"]
    main(["--pred", str(tmp_path / "pred"), "--labels", str(tmp_path / "labels"), "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "per event type" in out and "stroke confusion" in out and json.load(open(tmp_path / "r.json"))["n_gt"] == len(gt)


# ---- windows + stage cache
def test_windows_merge_and_mask():
    w = windows([1.0, 1.5, 10.0], pre=0.5, post=0.5)
    assert w == [(0.5, 2.0), (9.5, 10.5)] and abs(coverage(w, 20) - 0.125) < 1e-9
    m = frame_mask(400, 30, w); assert m[30] and not m[200] and m[300]
    assert hit_times([dict(times=[2.0, 1.0]), dict(times=[5.0])]) == [1.0, 2.0, 5.0] and hit_times({"hits": [3]}) == [3.0]


def test_stage_cache(tmp_path):
    c = StageCache(str(tmp_path)); f = tmp_path / "in.txt"; f.write_text("a"); calls = []
    fn = lambda: calls.append(1) or {"v": 1}
    r1, k1, hit1 = c.run("pose", fn, [str(f)], {"k": 12})
    r2, k2, hit2 = c.run("pose", fn, [str(f)], {"k": 12})
    assert r1 == r2 == {"v": 1} and k1 == k2 and not hit1 and hit2 and len(calls) == 1
    f.write_text("b"); _, k3, hit3 = c.run("pose", fn, [str(f)], {"k": 12}); assert k3 != k1 and not hit3
    _, k4, _ = c.run("pose", fn, [str(f)], {"k": 8}); assert k4 != k3
    with pytest.raises(ValueError):
        c.key("nope")
    assert STAGES[0] == "ingest" and "verify" in STAGES
