"""Highlight ranker (story/highlight.py): rule properties on a synthetic match. No recorded match is loaded."""
import os
import sys

import pytest

from opencourtvision.scoring.formats import M_CAP
from opencourtvision.story import build
from opencourtvision.story.config import MatchSettings, UserConfig
from opencourtvision.story.highlight import (
    PointInput,
    coaching_top,
    great_lost_points,
    highlights,
    interleave,
    match_top,
    personal_reel,
    points_from_swingvision,
    score_points,
)

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_match import MATCH  # noqa: E402


@pytest.fixture(scope="module")
def pts():
    return [PointInput.from_dict(d) for d in MATCH["points"]]


@pytest.fixture(scope="module")
def scored(pts):
    return score_points(pts)


@pytest.fixture(scope="module")
def by(scored):
    return {s["point"]: s for s in scored}


def test_synthetic_shape(pts):
    assert len(pts) == MATCH["n_points"] >= 40
    assert {p.server for p in pts} <= {"host", "guest"} and {p.winner for p in pts} == {"host", "guest"}


def test_rankers_are_permutations(scored):
    ids = {s["point"] for s in scored}
    assert set(match_top(scored, len(scored))) == ids
    assert len(set(coaching_top(scored, 5))) == 5
    r = personal_reel(scored, 0.5)
    assert sorted(r["order"]) == sorted(r["selected"]) and r["order"][-1] == r["climax"]


def test_long_rally_outranks_double_fault():
    pts = [
        PointInput(point=1, server="host", winner="guest", rally_shots=0, rally_span_s=0, ending="double_fault"),
        PointInput(point=2, server="guest", winner="host", rally_shots=16, rally_span_s=20, ending="winner", react_db=12),
    ]
    sc = {s["point"]: s for s in score_points(pts)}
    assert sc[2]["pq"] > sc[1]["pq"]


def test_o5_audio_reaction():
    pts = [
        PointInput(point=1, server="host", winner="host", rally_shots=6, rally_span_s=6, react_db=10.0),
        PointInput(point=2, server="host", winner="host", rally_shots=6, rally_span_s=6, react_db=6.3),
    ]
    scored = score_points(pts)
    assert scored[0]["reaction_extends_clip"] and not scored[1]["reaction_extends_clip"]
    assert not any(s["reaction_beat_ok"] for s in scored)
    conf = score_points(pts, MatchSettings(pose_confirmed_reactions={1, 2}))
    assert [s["point"] for s in conf if s["reaction_beat_ok"]] == [1]          # 6.3 dB is under the present threshold
    assert [s["pq"] for s in conf] == [s["pq"] for s in scored]


def test_highlights_wrapper_uses_user_config(pts, scored):
    res = highlights(pts, MatchSettings(), UserConfig(1.0))
    assert res["match_top"] == match_top(scored)
    assert res["coaching_top"] == coaching_top(scored)
    assert res["user_team"] == "host" and res["provisional"] and res["reaction_beats"] == []
    assert highlights(pts)["personal"]["loss_weight"] == 0.5


# ---------------------------------------------------------------- rule properties

def _lost(r, by):
    return [by[x]["lost"] for x in r["order"]]




def test_no_two_lost_in_a_row_every_slider(scored, by):
    """Coach 19 Oct: 'no two lost points in a row' applies at every loss_weight slider setting."""
    for lw in (0.0, 0.25, 0.5, 0.75, 1.0):
        lost = _lost(personal_reel(scored, lw), by)
        assert not any(a and b for a, b in zip(lost, lost[1:])), f"adjacent lost points at loss_weight={lw}"


@pytest.mark.parametrize("lw", [i / 10 for i in range(11)])
def test_slider_properties(scored, by, lw):
    r = personal_reel(scored, lw)
    lost = _lost(r, by)
    assert not any(a and b for a, b in zip(lost, lost[1:]))                    # O8/O14: no two lost points adjacent
    assert r["order"][-1] == r["climax"] and sorted(r["order"]) == sorted(r["selected"])
    assert set(r["great"]) <= set(r["selected"])                               # always included
    for g in r["great"]:
        assert r["scores"][g] == by[g]["pq"]                                   # never scaled
    ordinary = [x for x in r["selected"] if by[x]["lost"] and x not in r["great"]]
    assert len(ordinary) <= (0 if lw == 0.0 else 1)
    if lw < 1.0:
        assert not by[r["climax"]]["lost"]                                     # O9


def test_slider_only_touches_ordinary_lost(scored, by):
    a, b = personal_reel(scored, 0.2)["scores"], personal_reel(scored, 0.9)["scores"]
    for x, s in by.items():
        if a[x] is None: assert b[x] is None; continue
        if s["lost"] and x not in set(great_lost_points(scored)):
            assert a[x] == pytest.approx(s["pq"] * 0.2) and b[x] == pytest.approx(s["pq"] * 0.9)
        else:
            assert a[x] == b[x]


def test_lost_climax_allowed_only_at_one():
    pts = [PointInput(point=i, server="host", winner="guest" if i == 3 else "host", rally_shots=r, rally_span_s=r)
           for i, r in ((1, 2), (2, 3), (3, 20), (4, 1))]
    sc = score_points(pts)
    assert great_lost_points(sc) == [3]
    assert sc[2]["pq"] > 1.2 * max(s["pq"] for s in sc if not s["lost"])
    assert personal_reel(sc, 1.0, n=4)["climax"] == 3                         # O9: a lost point may be the climax only at 1.0
    for lw in (0.0, 0.5, 0.99):
        assert personal_reel(sc, lw, n=4)["climax"] != 3


def test_great_min_and_max():
    # every lost point is short: none in the top 5% -> the best lost point anyway (min 1)
    pts = [PointInput(point=i, server="host", winner="host" if i % 2 else "guest", rally_shots=i if i % 2 else 1, rally_span_s=i)
           for i in range(1, 41)]
    sc = score_points(pts)
    g = great_lost_points(sc)
    assert len(g) == 1 and sc[g[0] - 1]["lost"]
    # many long lost points: at most 3
    pts = [PointInput(point=i, server="host", winner="guest", rally_shots=10, rally_span_s=10) for i in range(1, 11)]
    assert len(great_lost_points(score_points(pts))) == 3


def test_interleave_reference_cases():
    t = interleave([1, 2, 3, 4, 5, 6], lambda x: x in (1, 2, 3, 4))   # infeasible: minimum is 2 adjacent lost pairs
    assert sum(a < 5 and b < 5 for a, b in zip(t, t[1:])) == 2 and t[-1] == 6
    for L, n in (((2, 3), 6), ((3, 4, 5), 8)):
        t = interleave(list(range(1, n + 1)), lambda x: x in L)
        assert not any(a in L and b in L for a, b in zip(t, t[1:])) and t[-1] == n and sorted(t) == list(range(1, n + 1))
    assert interleave([], bool) == []


def test_pressure_properties(pts):
    sc = score_points(pts)
    assert all(1.0 <= s["M"] <= M_CAP for s in sc)
    # (a) §3.3: a match point whose content is below 1/1.6 of a no-pressure point's content ranks below it
    mp = PointInput(point=1, server="guest", winner="guest", host_pts=3, guest_pts=1, games_before=(5, 3), sets_before=(1, 0), rally_shots=2)
    calm = PointInput(point=2, server="host", winner="host", host_pts=1, guest_pts=0, games_before=(0, 0), sets_before=(0, 0), rally_shots=12, rally_span_s=20)
    a, b = score_points([mp, calm])
    assert a["pressure"]["match_point"] and a["M"] == pytest.approx(1.6) and a["missed_big"]
    assert a["content"] < b["content"] / 1.6 and a["pq"] < b["pq"]


def test_user_team_is_a_match_setting(pts, by):
    g = score_points(pts, MatchSettings(user_team="guest"))
    assert all(s["lost"] != by[s["point"]]["lost"] for s in g)
    r = personal_reel(g, 0.5)
    assert sum(g[x - 1]["lost"] and x not in r["great"] for x in r["selected"]) <= 1
    assert r["great"] == great_lost_points(g)
    with pytest.raises(ValueError):
        MatchSettings(user_team="me")


def test_user_config(tmp_path, monkeypatch):
    assert UserConfig().loss_weight == 0.5
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            UserConfig(bad)
    p = tmp_path / "u.json"
    assert UserConfig.load(str(p)).loss_weight == 0.5                          # missing file -> default
    UserConfig(0.25).save(str(p))
    assert UserConfig.load(str(p)).loss_weight == 0.25
    monkeypatch.setenv("OCV_USER_CONFIG", str(p))
    assert UserConfig.load().loss_weight == 0.25
    with pytest.raises(ValueError):
        personal_reel([], 2.0)


# ---------------------------------------------------------------- adapters and callers

def test_points_from_swingvision_synthetic(tmp_path):
    pytest.importorskip("openpyxl")
    from test_swingvision import build as build_xlsx

    from opencourtvision.importers import swingvision as sv
    m = sv.load(build_xlsx(str(tmp_path / "s.xlsx")))
    p = points_from_swingvision(m)
    assert [x.ending for x in p] == ["unforced_error", "service_winner", "double_fault", None]
    assert [x.rally_shots for x in p] == [3, 2, 0, 2] and p[1].guest_pts == 1 and p[3].host_pts == 1 and p[3].guest_pts == 2
    sc = score_points(p)
    assert [s["lost"] for s in sc] == [True, False, True, False]
    assert sc[1]["g_out"] == pytest.approx(0.20)                               # service winner 0.40 x kappa 0.5


def _segs():
    import numpy as np
    rng = np.random.default_rng(0)
    segs, t = [], 10.0
    for i in range(12):
        hits = 2 + (i * 5) % 11
        times = list(t + 1 + np.cumsum(rng.uniform(0.7, 1.4, hits)))
        segs.append(dict(A=t, B=times[-1] + 3, rallies=[dict(t0=times[0], t1=times[-1], hits=hits, times=times, amax=0.5)], hits=hits,
                         fault_like=i == 5, serve_only=False))
        t = times[-1] + 20
    lb = np.full(int((t + 10) * 100), -40.0)
    lb[int((segs[3]["rallies"][0]["t1"] + 0.5) * 100):int((segs[3]["rallies"][0]["t1"] + 2.5) * 100)] = -10.0   # loud reaction after point 4
    return segs, lb


def test_story_build_uses_pq_and_o5():
    segs, lb = _segs()
    st = build(segs, lb, 0.0)
    assert st["n_points"] == 12 and {b["beat"] for b in st["beats"]} >= {"opener", "longest rally", "close"}
    assert "loudest reaction" not in {b["beat"] for b in st["beats"]}          # audio-only: no standalone reaction beat
    assert 4 in st["reaction_extends"]
    top = {p["point"]: p for p in st["top_points"]}
    assert 6 not in top or top[6]["excitement"] == 0.0                          # fault-like: non-play
    st2 = build(segs, lb, 0.0, pose_confirmed={4})
    assert any(b["beat"] == "loudest reaction" and b["point"] == 4 for b in st2["beats"])
    sc = [dict(winner="host" if i % 3 else "guest", break_point=i == 7) for i in range(12)]
    st3 = build(segs, lb, 0.0, score=sc, user=UserConfig(0.0))
    assert st3["personal"]["loss_weight"] == 0.0 and st3["personal"]["climax"] is not None
    p8 = next(p for p in st3["top_points"] if p["point"] == 8) if any(p["point"] == 8 for p in st3["top_points"]) else None
    assert p8 is None or p8["pq_terms"]["M"] == pytest.approx(1.3)
