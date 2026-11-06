"""Highlight style engine hard rules (Antony) + Media Director vamos_clean_v1 preset."""
import copy
import dataclasses

import pytest

from opencourtvision.scoring.formats import get_format
from opencourtvision.scoring.rules import initial_state
from opencourtvision.style.plan import PlanError, plan_reel, validate, whole_bounds
from opencourtvision.style.preset import HARD, PresetError, grade_filter, list_presets, load
from opencourtvision.style.preset import validate as validate_preset
from opencourtvision.style.stakes import caption_for, stakes


@pytest.fixture
def preset():
    return load("vamos_clean_v1")


@pytest.fixture
def fmt():
    return get_format("tennis", "advantage")


def _st(fmt, sets=(0, 0), games=(0, 0), pts=(0, 0), server="A"):
    return dataclasses.replace(initial_state(fmt, server), sets=sets, games=games, pts=pts, server=server)


def _pt(i, t0, hits=6, favorite=False, winner="A", verified=True, state=None, react=2.0):
    first, last = t0, t0 + hits * 1.2
    return dict(id=i, kind="point", A=first - 3.5, B=last + 2.5, first_hit=first, last_hit=last, hits=hits,
                react=react, favorite=favorite, winner=winner, winner_verified=verified,
                state=state, state_verified=verified and state is not None, sources=["audio", "user"])


# ---------------------------------------------------------------- preset loader

def test_loads_vamos_clean_v1(preset):
    assert preset["id"] == "vamos_clean_v1" and "vamos_clean_v1" in list_presets()
    assert preset["grade"]["enabled"] is True and preset["grade"]["lut_file"].endswith(".cube")
    assert preset["transitions"]["default"] == "hard_cut"
    assert preset["audio"]["bake_music"] is False
    assert preset["audio"]["prefer_platform_sound_at_share"] is True
    assert "MATCH POINT" in preset["captions"]["stakes_allowlist"]


def test_grade_on_off_flag(preset):
    assert grade_filter(preset) is not None
    off = load("vamos_clean_v1", grade=False)
    assert off["grade"]["enabled"] is False and grade_filter(off) is None


def test_preset_rejects_too_many_slow_mos(preset):
    bad = copy.deepcopy(preset); bad["slow_mo"]["max_per_reel"] = 3
    with pytest.raises(PresetError, match="slow_mo"):
        validate_preset(bad)


def test_preset_rejects_slower_than_0_45(preset):
    bad = copy.deepcopy(preset); bad["slow_mo"]["max_factor"] = 0.3
    with pytest.raises(PresetError, match="0.45"):
        validate_preset(bad)


def test_preset_rejects_letterbox_enabled(preset):
    bad = copy.deepcopy(preset); bad["hard_constraints"]["nine_sixteen_letterbox_forbidden"] = False
    with pytest.raises(PresetError, match="letterbox"):
        validate_preset(bad)


def test_preset_rejects_baked_music(preset):
    bad = copy.deepcopy(preset); bad["audio"]["bake_music"] = True
    with pytest.raises(PresetError, match="baked"):
        validate_preset(bad)


# ---------------------------------------------------------------- stakes captions (real score state; mock frame 01 is wrong)

def test_mock_frame_01_score_is_not_match_point(fmt, preset):
    """Mock frame 01 shows MATCH POINT on 2-1 30-15 — that is wrong. Caption logic must come from the real score state."""
    st = _st(fmt, games=(2, 1), pts=(2, 1), server="A")   # 2-1, 30-15
    lab, team = stakes(st, fmt, verified=True)
    assert lab is None and team is None
    assert caption_for(st, fmt, True, preset["captions"]["stakes_allowlist"]) == (None, None)


def test_match_point_only_when_one_point_from_match(fmt, preset):
    st = _st(fmt, sets=(1, 0), games=(5, 2), pts=(3, 1), server="B")  # A one point from 6-2 and the match
    lab, team = caption_for(st, fmt, True, preset["captions"]["stakes_allowlist"])
    assert lab == "MATCH POINT" and team == "A"


def test_set_point_not_match_point(fmt, preset):
    st = _st(fmt, sets=(0, 0), games=(5, 4), pts=(3, 2), server="B")
    lab, team = caption_for(st, fmt, True, preset["captions"]["stakes_allowlist"])
    assert lab == "SET POINT" and team == "A"


def test_break_point(fmt, preset):
    st = _st(fmt, games=(3, 4), pts=(1, 3), server="A")  # 15-40, receiver B has break point
    lab, team = caption_for(st, fmt, True, preset["captions"]["stakes_allowlist"])
    assert lab == "BREAK POINT" and team == "B"


def test_unverified_score_never_captions(fmt, preset):
    st = _st(fmt, sets=(1, 0), games=(5, 2), pts=(3, 1), server="B")
    assert caption_for(st, fmt, False, preset["captions"]["stakes_allowlist"]) == (None, None)


def test_hype_labels_not_on_allowlist(preset):
    assert "INSANE" not in preset["captions"]["stakes_allowlist"]
    assert "CRAZY" not in preset["captions"]["stakes_allowlist"]


# ---------------------------------------------------------------- whole points, slow-mo, hard cuts, no letterbox

def test_whole_bounds_floor():
    p = _pt(1, 100.0, hits=5)
    a, b = whole_bounds(p)
    assert a <= p["first_hit"] - HARD["pre_roll_s_min"] + 1e-6
    assert b >= p["last_hit"] + HARD["post_roll_s_min"] - 1e-6


def test_plan_whole_points_only(preset, fmt):
    pts = [_pt(i, 100 + i * 40, hits=5 + i, favorite=(i == 2), state=_st(fmt), verified=True) for i in range(5)]
    plan = plan_reel(pts, preset, match=dict(sport="tennis", format=fmt, format_name="doubles",
                                             teams=["Host", "Guest"], date_label="17 OCT 2026",
                                             final="6-1 6-2", final_replayed="6-1 6-2", winner="Host"))
    for c in plan["clips"]:
        p = next(x for x in pts if x["id"] == c["id"])
        assert c["a"] <= p["first_hit"] - 3.0 + 1e-3
        assert c["b"] >= p["last_hit"] + 2.0 - 1e-3
    assert plan["transitions"] == "hard_cut"
    assert plan["letterbox"] is False and plan["wide_mode"] is False


def test_slow_mo_budget_and_floor(preset, fmt):
    pts = [_pt(i, 100 + i * 40, hits=8, favorite=True, state=_st(fmt)) for i in range(6)]
    plan = plan_reel(pts, preset, max_points=6, match=dict(sport="tennis", format=fmt, teams=["A", "B"]))
    assert plan["slow_mo_count"] <= 2
    for c in plan["clips"]:
        assert min(q[2] for q in c["parts"]) >= 0.45 - 1e-9


def test_slow_mo_only_on_favorites(preset, fmt):
    pts = [_pt(i, 100 + i * 40, hits=8, favorite=(i == 0), state=_st(fmt)) for i in range(4)]
    plan = plan_reel(pts, preset, match=dict(sport="tennis", format=fmt, teams=["A", "B"]))
    for c in plan["clips"]:
        if c["slow"]:
            assert next(p for p in pts if p["id"] == c["id"])["favorite"]


def test_captions_clear_before_cut(preset, fmt):
    st_mp = _st(fmt, sets=(1, 0), games=(5, 2), pts=(3, 1), server="B")
    pts = [_pt(1, 100, hits=6, favorite=True, state=st_mp),
           _pt(2, 140, hits=6, favorite=False, state=_st(fmt))]
    plan = plan_reel(pts, preset, match=dict(sport="tennis", format=fmt, teams=["A", "B"]))
    cuts = [c["out_start"] for c in plan["clips"][1:]]
    for o in plan["captions"]:
        assert not any(o["start"] + 1e-9 < t < o["end"] - 1e-9 for t in cuts)
        assert o["text"] in preset["captions"]["stakes_allowlist"]


def test_validate_rejects_mid_point_cut(preset, fmt):
    pts = [_pt(1, 100, hits=6, state=_st(fmt))]
    plan = plan_reel(pts, preset, match=dict(sport="tennis", format=fmt, teams=["A", "B"]))
    plan["clips"][0]["a"] = pts[0]["first_hit"] - 1.0          # too short pre-roll
    with pytest.raises(PlanError, match="R-WHOLE"):
        validate(plan, pts, preset)


def test_validate_rejects_third_slow_mo(preset, fmt):
    pts = [_pt(i, 100 + i * 40, hits=8, favorite=True, state=_st(fmt)) for i in range(3)]
    plan = plan_reel(pts, preset, max_points=3, match=dict(sport="tennis", format=fmt, teams=["A", "B"]))
    # force a third slow-mo past the planner
    for c in plan["clips"]:
        c["slow"] = True
    with pytest.raises(PlanError, match="R-SLOW"):
        validate(plan, pts, preset)


def test_music_defaults_to_platform_picker(preset, fmt):
    pts = [_pt(1, 100, hits=5, state=_st(fmt))]
    plan = plan_reel(pts, preset, match=dict(sport="tennis", format=fmt, teams=["A", "B"]))
    assert plan["music"] == "platform_sound_picker_at_share"
    assert preset["audio"]["bake_music"] is False


# ---------------------------------------------------------------- v4.9 score toggle (Antony): confirmed / high-confidence only

def test_score_toggle_and_confidence(preset, fmt):
    pts = [_pt(i, 100 + i * 40, hits=5, state=_st(fmt), verified=(i < 2)) for i in range(5)]
    pts[2]["state_conf"] = 0.95; pts[3]["state_conf"] = 0.6           # 4: unverified, no confidence
    m = dict(sport="tennis", format=fmt, teams=("Us", "Them"), final=None)
    plan = plan_reel(pts, preset, max_points=5, max_s=200, match=m)
    shown = {b["point"] for b in plan["bugs"]}
    assert shown == {p["id"] for p in pts if p["id"] in {c["id"] for c in plan["clips"]} and p["id"] in (0, 1, 2)}
    off = plan_reel(pts, preset, max_points=5, max_s=200, match=m, show_score=False)
    assert off["bugs"] == [] and off["outro"] is None
