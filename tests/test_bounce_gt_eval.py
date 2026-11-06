"""physball_eval: hand-marked last-bounce ground truth (bounce-marker labels) scoring. Synthetic events only."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools", "ball_tracker")


@pytest.fixture(scope="module")
def PE():
    sys.path.insert(0, TOOLS)
    try:
        spec = importlib.util.spec_from_file_location("physball_eval", os.path.join(TOOLS, "physball_eval.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        yield m
    finally:
        sys.path.remove(TOOLS)


def ev(t, x, y, kind="bounce"):
    return dict(kind=kind, t=t, x=x, y=y)


def test_hit_within_tolerance(PE):
    gt = dict(kind="bounce", k=60, u=500.0, v=400.0, t_last=101.0)      # clip t0 = 100 s, last contact at clip frame 30
    r = PE.score_last_bounce(gt, [ev(0.5, 10, 10), ev(61 / 30, 505, 398)], t0=100.0)
    assert r["found"] and r["ok"] and r["dk"] == 1 and r["px"] < 6


def test_wrong_frame_or_place_is_not_ok(PE):
    gt = dict(kind="bounce", k=60, u=500.0, v=400.0, t_last=101.0)
    assert not PE.score_last_bounce(gt, [ev(66 / 30, 500, 400)], t0=100.0)["ok"]          # 6 frames late
    assert not PE.score_last_bounce(gt, [ev(60 / 30, 560, 400)], t0=100.0)["ok"]          # 60 px off
    miss = PE.score_last_bounce(gt, [ev(0.9, 500, 400)], t0=100.0)                           # before the last contact
    assert not miss["found"] and miss["dk"] is None


def test_none_and_summary(PE):
    rows = [PE.score_last_bounce(dict(kind="none", t_last=101.0), [ev(62 / 30, 1, 1)], t0=100.0),
            PE.score_last_bounce(dict(kind="none", t_last=101.0), [], t0=100.0),
            PE.score_last_bounce(dict(kind="bounce", k=60, u=0, v=0, t_last=101.0), [ev(60 / 30, 3, 4)], t0=100.0),
            PE.score_last_bounce(dict(kind="bounce", k=60, u=0, v=0, t_last=101.0), [], t0=100.0)]
    s = PE.gt_summary(rows)
    assert s["gt_none_n"] == 2 and s["gt_none_false"] == 1
    assert s["gt_last_n"] == 2 and s["gt_last_ok"] == 1 and s["gt_last_recall"] == 0.5 and s["gt_last_precision"] == 1.0
    assert s["gt_last_px_med"] == 5.0


def test_load_bounce_gt_keeps_finished_only(PE, tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"items": {"3": dict(idx=3, done=True, kind="bounce", k=1, u=0, v=0, t_last=1.0),
                                       "4": dict(idx=4, done=False, kind="bounce"), "5": dict(idx=5, done=True, kind="none", t_last=2.0)}}))
    assert sorted(PE.load_bounce_gt(str(p))) == [3, 5]
