"""Ops V6 (closed EU beta): re-ID embeddings and colour histograms stay in memory, per match; never written, exported or
carried across matches."""
import builtins
import copy
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencourtvision.players import APPEARANCE_KEYS, strip_appearance
from opencourtvision.players.rules import near_team
from opencourtvision.players.synthetic import make_frames, match_points
from opencourtvision.players.tracker import IDTracker

ROOT = Path(__file__).resolve().parents[1]
CUES = ("court", "motion", "colour_box", "osnet")


@pytest.fixture(scope="module")
def frames():
    P = match_points(seed=2); fr, _ = make_frames(P, seed=2, spectators=0)
    nt = dict(zip([p["idx"] for p in P], near_team(P, True)))
    for f in fr: f["near_team"] = nt[f["point"]]
    assert any(d.get("osnet") is not None and d.get("colour_box") is not None for f in fr for d in f["dets"])
    return fr


def _run(fr):
    tr = IDTracker(cues=CUES); f0 = fr[3]; tr.init(f0, {i: d["_true"] for i, d in enumerate(f0["dets"])})
    return tr, tr.run(fr)


def test_tracker_writes_nothing(frames, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def guarded_open(f, mode="r", *a, **k):
        if any(c in mode for c in "wax+"): pytest.fail(f"tracker wrote {f}")
        return real_open(f, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    for fn in ("save", "savez", "savez_compressed", "savetxt"):
        monkeypatch.setattr(np, fn, lambda *a, **k: pytest.fail("tracker saved an array"))
    monkeypatch.setattr(pickle, "dump", lambda *a, **k: pytest.fail("tracker pickled"))
    tr, preds = _run(frames)
    assert any(tr.tmpl[s] for s in tr.tmpl)                        # templates exist, in memory
    assert list(tmp_path.iterdir()) == []


def test_outputs_carry_ids_only(frames):
    tr, preds = _run(frames)
    for p in preds:
        assert all(isinstance(k, int) and v in ("A1", "A2", "B1", "B2") for k, v in p.items())
    _, margins = tr.point(frames[:5])
    assert all(isinstance(v, float) for m in margins for v in m.values())


def test_tracker_cannot_be_pickled_copied_or_serialised(frames):
    tr, _ = _run(frames)
    with pytest.raises(TypeError, match="never pickled"):
        pickle.dumps(tr)
    with pytest.raises(TypeError, match="never pickled"):
        copy.deepcopy(tr)
    with pytest.raises(TypeError, match="never pickled"):
        copy.copy(tr)


def test_templates_are_per_match(frames):
    tr, _ = _run(frames)
    assert any(tr.tmpl.values())
    tr.end_match()
    assert all(v == {} for v in tr.tmpl.values()) and tr.anchor == {} and tr.rot == {} and tr.rcv == {}
    with IDTracker(cues=CUES) as t2:
        t2.init(frames[3], {i: d["_true"] for i, d in enumerate(frames[3]["dets"])}); t2.run(frames)
        assert any(t2.tmpl.values())
    assert all(v == {} for v in t2.tmpl.values())
    assert all(v == {} for v in IDTracker().tmpl.values())             # a new match starts from nothing


def test_strip_appearance(frames):
    clean = strip_appearance(frames[:10])
    assert all(k not in d for f in clean for d in f["dets"] for k in APPEARANCE_KEYS)
    assert any(d.get("osnet") is not None for d in frames[3]["dets"])     # original untouched


def test_static_no_persistence_or_export_of_appearance():
    """The shipped players package has no file / serialisation writes, and nothing else in the engine reads its
    appearance features (so no session, export or upload path can carry them)."""
    bad = []
    for f in (ROOT / "opencourtvision" / "players").rglob("*.py"):
        if "_vendor" in f.parts: continue
        s = f.read_text()
        for needle in ("np.save", "savez", "savetxt", "json.dump", "import pickle", "pickle.dump", "imwrite", "torch.save", "to_csv", ', "w")', ", 'w')"):
            if needle in s: bad.append(f"{f.relative_to(ROOT)}: {needle}")
    for f in (ROOT / "opencourtvision").rglob("*.py"):
        if "players" in f.parts: continue
        s = f.read_text()
        for needle in ("players.reid", "players.cues", "OSNetEmbedder", "colour_hist", '"osnet"', "'osnet'"):
            if needle in s: bad.append(f"{f.relative_to(ROOT)}: {needle}")
    assert bad == []


def test_research_tool_refuses_in_beta():
    env = dict(os.environ, VAMOS_BETA="1")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "player_id" / "pid_run.py"), "eval", "--out", "/nonexistent"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "never runs in a beta build" in r.stderr
