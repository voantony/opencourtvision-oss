"""tools/ball_track_v2/extend_clips.py: clip-window rule and plan/merge on synthetic data (no footage, no labels)."""
import argparse
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def EC():
    p = os.path.join(HERE, "..", "tools", "ball_track_v2", "extend_clips.py")
    spec = importlib.util.spec_from_file_location("extend_clips", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def test_window_end(EC):
    assert EC.window_end(10.0, [0.0, 20.0]) == pytest.approx(13.0)          # +3 s
    assert EC.window_end(10.0, [0.0, 12.0]) == pytest.approx(11.7)          # next serve - 0.3 s
    assert EC.window_end(10.0, [0.0, 10.2]) == pytest.approx(13.0)          # a "next" start within 0.5 s is ignored
    assert EC.window_end(10.0, []) == pytest.approx(13.0)


def test_plan_and_merge(EC, tmp_path):
    data = tmp_path / "data"; (data / "clips").mkdir(parents=True)
    idx = {}
    for i, (t0, tl) in enumerate([(100.0, 104.0), (200.0, 206.9)], 1):       # last contact 4 s / 6.9 s into a 7.5 s clip
        nm = f"dev_match_seg{i:03d}"; np.savez(data / "clips" / f"{nm}.npz", t0=t0, fps=30.0)
        idx[nm] = dict(source="dev_match", idx=i, n=225, split="train")
    json.dump(dict(clips=idx), open(data / "index.json", "w"))
    labels = dict(points=[dict(idx=1, start=95.0, end=107.0), dict(idx=2, start=195.0, end=207.0), dict(idx=3, start=300.0, end=310.0)])
    json.dump(labels, open(tmp_path / "labels.json", "w"))
    sv = dict(points={"1": dict(shots=[dict(t=104.0)]), "2": dict(shots=[dict(t=206.9)])})
    json.dump(sv, open(tmp_path / "sv.json", "w"))
    plan = tmp_path / "plan.json"
    EC.plan(argparse.Namespace(data=str(data), labels=str(tmp_path / "labels.json"), sv=str(tmp_path / "sv.json"), prelabels=None,
                               source="dev_match", after=3.0, gap=0.3, out=str(plan)))
    P = json.load(open(plan))["clips"]
    assert [c["name"] for c in P] == ["dev_match_seg002"]                       # clip 1 already covers +3 s
    assert P[0]["n_new"] == int(np.floor((206.9 + 3.0 - 200.0) * 30)) + 1
    cands = tmp_path / "c"; ext = tmp_path / "ce"; cands.mkdir(); ext.mkdir()
    for nm in idx:
        np.savez(cands / f"{nm}.npz", cands=np.zeros((225, 1, 3)))
    np.savez(ext / "dev_match_seg002.npz", cands=np.zeros((P[0]["n_new"], 1, 3)))
    out = tmp_path / "m"
    EC.merge(argparse.Namespace(plan=str(plan), data=str(data), out=str(out), cands=str(cands), cands_ext=str(ext), cands_out=str(tmp_path / "cm")))
    m = json.load(open(out / "index.json"))["clips"]
    assert m["dev_match_seg002"]["n"] == P[0]["n_new"] and m["dev_match_seg001"]["n"] == 225
    assert np.load(tmp_path / "cm" / "dev_match_seg002.npz")["cands"].shape[0] == P[0]["n_new"]
    assert np.load(tmp_path / "cm" / "dev_match_seg001.npz")["cands"].shape[0] == 225
