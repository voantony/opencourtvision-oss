"""The Mac app's EngineBridge parses stdout of `python -m opencourtvision.app` as one JSON doc."""
import json

from opencourtvision.app import __main__ as cli
from opencourtvision.app import analyze as an


def test_points_from_rally_list(tmp_path):
    (tmp_path / "rally_list.json").write_text(json.dumps(dict(rallies=[
        dict(idx=1, start_s=0.0, end_s=13.55, hits=6, rally_s=5.75, reaction_db=0.3),
        dict(idx=2, start_s=16.4, end_s=26.55, hits=4),
    ])))
    pts = an.points_from_segments(str(tmp_path), {})
    assert [p["id"] for p in pts] == [1, 2]
    assert pts[0]["A"] == 0.0 and pts[0]["B"] == 13.55 and pts[0]["hits"] == 6
    assert pts[1]["first_hit"] == 16.4  # no rally_s -> whole window
    assert all(p["winner"] is None and p["winner_confidence"] == 0.5 for p in pts)


def test_analyze_stdout_is_pure_json(tmp_path, monkeypatch, capsys):
    def fake_pipeline(args):
        print("[1 audio] 0.04 s")
        print(json.dumps(dict(schema="ocv.rally_list.v1")))
        (tmp_path / "w" / "rally_list.json").write_text(json.dumps(dict(rallies=[dict(idx=1, start_s=1, end_s=9, hits=5)])))
        return dict(mode="rallies")

    monkeypatch.setattr(an, "pipeline_run", fake_pipeline)
    monkeypatch.setattr(an.quality, "analyse", lambda v: dict(ok=True, hints=[], struggle_tags=[], metrics={}))
    video = tmp_path / "v.mp4"; video.write_bytes(b"")
    assert cli.main(["analyze", str(video), "--work", str(tmp_path / "w")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_points"] == 1 and out["mode"] == "rallies" and out["sign_in_required"] is False


def test_analyze_takes_framing_quality_and_never_refuses(tmp_path, monkeypatch, capsys):
    def fake_pipeline(args):
        (tmp_path / "w" / "rally_list.json").write_text(json.dumps(dict(rallies=[dict(idx=1, start_s=1, end_s=9, hits=5),
                                                                                  dict(idx=2, start_s=40, end_s=52, hits=7)])))
        return dict(mode="rallies")

    monkeypatch.setattr(an, "pipeline_run", fake_pipeline)
    monkeypatch.setattr(an.quality, "analyse", lambda v: dict(ok=True, hints=[], struggle_tags=[], metrics={}))
    s = lambda t, cv, vf: dict(t=t, court_visible=cv, vis_near=1.0, vis_far=vf, moved=False, blurred=False)
    fq = tmp_path / "framing.json"
    fq.write_text(json.dumps(dict(schema="vamos.framing_quality.v1", samples=[s(2, 0.9, 1.0), s(41, 0.3, 0.2), s(50, 0.3, 0.2)])))
    video = tmp_path / "v.mp4"; video.write_bytes(b"")
    assert cli.main(["analyze", str(video), "--work", str(tmp_path / "w"), "--framing", str(fq)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_points"] == 2                                   # a badly framed point is still analysed
    doc = json.load(open(tmp_path / "w" / "session.json"))
    p1, p2 = doc["points"]
    assert p1["framing"]["vision_mode"] == "full" and p2["framing"]["vision_mode"] == "audio"
    assert p2["framing"]["uncertainty_scale"] > p1["framing"]["uncertainty_scale"]
    assert doc["framing_quality"]["schema"] == "vamos.framing_quality.v1"
