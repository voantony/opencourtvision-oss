import argparse
import json
import os
import subprocess
import sys

from opencourtvision.ffmpeg import duration
from opencourtvision.pipeline import add_arguments, run
from opencourtvision.story import build


def _args(video, work, *extra):
    return add_arguments(argparse.ArgumentParser()).parse_args([video, "--final", "6-4 6-3", "--work", work, *extra])


def test_full_pipeline_with_story_and_render(clips, tmp_path):
    gt = json.load(open(clips["match_sim"]["gt"]))
    work, out = str(tmp_path / "work"), str(tmp_path / "out")
    rep = run(_args(clips["match_sim"]["mp4"], work, "--story", "--render", out, "--name", "Synthetic"))
    assert abs(rep["match_start_s"] - gt["match_start_s"]) <= 2.1  # snapped to a 2 s keyframe grid later; start itself is exact
    assert rep["checks"]["segments"] == len(gt["rallies"]) - sum(1 for r in gt["rallies"] if r[0] < gt["match_start_s"])
    for f in ("rallies.json", "segments.csv", "score_timeline.csv", "story.json", "swingcut_report.json"):
        assert os.path.exists(os.path.join(work, f))
    pts = os.path.join(out, "Synthetic - points only 4K.mp4")
    hl = os.path.join(out, "Synthetic - highlights 4K.mp4")
    assert duration(pts) > duration(hl) > 0
    st = json.load(open(os.path.join(work, "story.json")))
    assert {b["beat"] for b in st["beats"]} >= {"opener", "longest rally", "close"}


def test_story_empty():
    assert build([], [], 0.0) == {}


def test_cli_help():
    for mod in ("swingcut", "match_start", "reel"):
        r = subprocess.run([sys.executable, "-m", f"opencourtvision.cli.{mod}", "--help"], capture_output=True, text=True)
        assert r.returncode == 0 and "usage" in r.stdout


def test_rallies_only_mode_pipeline(clips, tmp_path):
    work = str(tmp_path / "work")
    a = add_arguments(argparse.ArgumentParser()).parse_args([clips["match_sim"]["mp4"], "--mode", "rallies", "--work", work,
                                                             "--reel-order", "rank", "--region", "US"])
    doc = run(a)
    assert doc["schema"] == "ocv.rally_list.v1" and doc["mode"] == "rallies" and doc["order"] == "rank"
    assert doc["n_rallies"] > 0 and sorted(doc["reel"]) == list(range(1, doc["n_rallies"] + 1))
    assert os.path.exists(os.path.join(work, "rally_list.json"))
    for f in ("score_timeline.csv", "swingcut_report.json"):       # no score engine output in rallies-only mode
        assert not os.path.exists(os.path.join(work, f))
    assert doc["suggestion"]["suggested"] in ("match", "rallies")


def test_match_mode_requires_final(tmp_path):
    import pytest
    a = add_arguments(argparse.ArgumentParser()).parse_args(["x.mp4", "--work", str(tmp_path)])
    with pytest.raises(SystemExit):
        run(a)
