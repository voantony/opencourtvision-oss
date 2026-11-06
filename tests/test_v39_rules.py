"""v3.9 rules: forward-only scorer by default, global decoder switchable (full only), motion never sets US clip
boundaries, filter types from score/audio/user only, player reels from setup tags + score, score-pressure switch."""
import numpy as np
import pytest

from opencourtvision.editloop.build import flags
from opencourtvision.editloop.provenance import ProvenanceError
from opencourtvision.editloop.select import PRESSURE_KEYS, player_reel, story_score_arg, strip_pressure, typed_filter
from opencourtvision.scoring.engine import make_scorer
from opencourtvision.scoring.formats import get_format
from opencourtvision.scoring.forward import ForwardScorer
from opencourtvision.scoring.rules import advance, initial_state

FMT = get_format("tennis", "advantage")


def test_forward_scorer_is_default_everywhere_and_global_is_a_full_switch():
    for r in ("US", "UK", "ES", "ROW"):
        f = flags(r)
        assert f.scorer == "forward" and not f.pending_commit and f.voice_edits and f.score_checks
        sc = make_scorer(f, FMT)
        assert isinstance(sc, ForwardScorer) and sc.policy == "point_end" and sc.voice_edits and sc.score_checks
    g = make_scorer(flags("ROW", scorer="global"), FMT)
    assert callable(g) and not isinstance(g, ForwardScorer)
    assert flags("US", scorer="global").scorer == "forward"
    assert isinstance(make_scorer(flags("US", scorer="global"), FMT), ForwardScorer)


def test_profiles_v39():
    assert flags("UK").no_3d_trajectory and flags("US").no_3d_trajectory and not flags("ROW").no_3d_trajectory
    assert not flags("ES").auto_refereeing
    assert flags("ROW").motion_boundaries and not flags("US").motion_boundaries
    assert flags("ROW").score_pressure and not flags("US").score_pressure
    assert not flags("ROW", score_pressure=False).score_pressure


class _Cues:
    p = {"idle_min": 9, "fill_min": 15, "fill_single_amp": 14}

    def __init__(self, idle):
        self.idle = idle

    def activity(self, a, b):
        return 0 if any(a <= t <= b for t in self.idle) else 20


def test_us_motion_only_gates_whole_clips_never_boundaries():
    from opencourtvision.editloop.motion import live_gate
    pts = [{"t0": 10.0, "t1": 14.0, "sources": {"audio"}}, {"t0": 30.0, "t1": 33.0, "sources": {"audio"}}]
    out = live_gate(pts, _Cues(idle=[31.0]))
    assert out == [pts[0]] and out[0] is pts[0] and (out[0]["t0"], out[0]["t1"]) == (10.0, 14.0)


def test_segment_match_us_never_uses_motion_boundaries(monkeypatch):
    import opencourtvision.editloop.motion as motion
    from opencourtvision.editloop import segment
    called = {}
    monkeypatch.setattr(motion, "apply_cues", lambda *a, **k: (called.__setitem__("apply", True), [])[1])
    le = np.full(12000, -60.0); bg = np.full(12000, -60.0)
    for t in (10, 11.2, 12.4, 14.0):
        le[int(t * 100)] = -40.0
    segment.segment_match(le, bg, 0.0, None, flags("US", vision_rally_cues=True), cues=_Cues(idle=[]))
    assert "apply" not in called
    segment.segment_match(le, bg, 0.0, None, flags("ROW", vision_rally_cues=True), cues=_Cues(idle=[]))
    assert called.get("apply")


def test_filter_types_never_from_tracking():
    segs = [{"types": {"ace": "score"}}, {"types": {"ace": "user"}}, {"types": {"loud_rally": "audio"}}, {"types": {}}]
    assert typed_filter(segs, "ace") == [0, 1]
    with pytest.raises(ProvenanceError):
        typed_filter([{"types": {"winner": "ball_track"}}], "winner")
    with pytest.raises(ProvenanceError):
        typed_filter([{"types": {"forehand": "pose"}}], "forehand")


def test_full_variant_has_no_tracking_typed_playback():
    import opencourtvision.variants.full.selection as fs
    assert not hasattr(fs, "typed_filter_by_tracking") and not hasattr(fs, "TRACKING_TYPES")


def test_player_reel_from_setup_tags_and_score():
    st = initial_state(FMT, "A", 0); states = []; pts = []
    for w in "AAAA" + "BBBB" + "AAAA":            # game 1 A serves (p1), game 2 B serves (p3), game 3 A serves (p2)
        states.append(st); pts.append({"winner": w}); st = advance(st, w, FMT)
    setup = {"teams": {"p1": "A", "p2": "A", "p3": "B", "p4": "B"}, "serve_order": {"A": ["p1", "p2"], "B": ["p3", "p4"]}}
    assert player_reel(pts, "p3", setup, states) == [4, 5, 6, 7]
    assert player_reel(pts, "p1", setup, states) == [0, 1, 2, 3, 8, 9, 10, 11]   # served game 1 + team won game 3


def test_score_pressure_switch():
    score = [{"winner": "host", "break_point": True, "match_point": True}]
    assert story_score_arg(score, flags("ROW")) == score
    assert story_score_arg(score, flags("ROW", score_pressure=False)) == [{"winner": "host"}]
    assert story_score_arg(score, flags("US")) is None
    from opencourtvision.story.highlight import PointInput
    p = PointInput(point=1, server=None, winner="host", pressure_flags={"match_point": True})
    out = strip_pressure([p], flags("ROW", score_pressure=False))
    assert not any(out[0].pressure_flags.values()) and out[0].winner == "host"
    assert "break_point" in PRESSURE_KEYS


def test_no_pose_category_play_trigger():
    """Sony EP 4212218 design intent: nothing starts/stops recording or play detection from pose categories."""
    import ast
    import os
    root = os.path.join(os.path.dirname(__file__), "..", "opencourtvision")
    bad = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                src = open(os.path.join(d, f)).read()
                for n in ast.walk(ast.parse(src)):
                    if isinstance(n, ast.FunctionDef) and "pose" in n.name and any(k in n.name for k in ("start", "stop", "trigger")):
                        bad.append(f"{f}:{n.name}")
    assert not bad
