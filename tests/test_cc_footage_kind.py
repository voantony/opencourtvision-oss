"""CC footage pre-labeller: full_match vs partial_or_practice tag (label_mode match / rallies)."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def cc():
    p = os.path.join(HERE, "..", "tools", "cc_footage", "cc_footage.py")
    spec = importlib.util.spec_from_file_location("cc_footage", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _match(n=80, start=300.0, every=28.0):
    return [dict(t0=start + every * k, t1=start + every * k + 4.0, hits=4, pause_before=24.0) for k in range(n)]


def test_full_match_vs_partial(cc):
    full = cc.video_kind(_match(), 300.0, 3600.0, dict(fixed_camera=True))
    assert full["video_kind"] == "full_match" and full["label_mode"] == "match" and not full["mode_reasons"]
    short = cc.video_kind(_match(30), 300.0, 900.0)
    assert short["video_kind"] == "partial_or_practice" and short["label_mode"] == "rallies"
    edited = cc.video_kind(_match(), 300.0, 3600.0, dict(fixed_camera=False))
    assert edited["label_mode"] == "rallies" and any("camera" in r for r in edited["mode_reasons"])
    drill = [dict(t0=5.0 + 22 * k, t1=5.0 + 22 * k + 20.0, hits=25, pause_before=2.0) for k in range(100)]
    assert cc.video_kind(drill, 0.0, 2400.0)["label_mode"] == "rallies"


def test_retag_existing_prelabels(cc, tmp_path):
    d = tmp_path / "prelabels" / "tennis" / "vid1"; d.mkdir(parents=True)
    rallies = [dict(rally=k + 1, contacts=[dict(t=300.0 + 28 * k + i) for i in range(4)]) for k in range(80)]
    json.dump(dict(video_id="vid1", rallies=rallies), open(d / "prelabels.json", "w"))
    json.dump(dict(sport="tennis", video_id="vid1", duration_s=3600.0, fixed_camera=True), open(d / "qa.json", "w"))
    json.dump(dict(play_start_s=296.0), open(d / "trim.json", "w"))
    assert cc.retag(str(tmp_path)) == 1
    qa = json.load(open(d / "qa.json")); pl = json.load(open(d / "prelabels.json"))
    assert qa["video_kind"] == "full_match" and pl["label_mode"] == "match"
    assert "video_kind" in open(tmp_path / "prelabels" / "qa_summary.csv").read()
