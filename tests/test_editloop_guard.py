"""Edit loop (eng notes v3.7 §2.8.7/§2.8.8): tracked-ball events must never feed clip selection, enforced structurally."""
import ast
import os
import subprocess
import sys

import numpy as np
import pytest

from opencourtvision.editloop import ProvenanceError, check
from opencourtvision.editloop.build import flags
from opencourtvision.editloop.segment import attach_faults, audio_points, clip_bounds, gap_fill, segment_match
from opencourtvision.editloop.select import select_clips, selection_features, story_score_arg

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "opencourtvision")
FORBIDDEN_MODULES = ("opencourtvision.ball", "opencourtvision.importers", "opencourtvision.scoring.decoder", "opencourtvision.pose",
                     "opencourtvision.events", "opencourtvision.reel", "opencourtvision.variants.full")


def _imports(path, pkg):
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            yield from (a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                base = pkg.split(".")[: len(pkg.split(".")) - n.level + 1]
                yield ".".join(base + ([n.module] if n.module else []))
            else:
                yield n.module


def test_editloop_never_imports_tracking_modules():
    d = os.path.join(ROOT, "editloop")
    for f in os.listdir(d):
        if f.endswith(".py"):
            for m in _imports(os.path.join(d, f), "opencourtvision.editloop"):
                assert not any(m == b or m.startswith(b + ".") for b in FORBIDDEN_MODULES), f"{f} imports {m}"


def test_segmentation_path_loads_no_ball_code():
    code = ("import sys, numpy as np\n"
            "from opencourtvision.editloop.segment import segment_match\n"
            "from opencourtvision.editloop.build import flags\n"
            "le = np.full(6000, -60.0); bg = np.full(6000, -60.0)\n"
            "for t in (10.0, 11.2, 12.5, 30.0, 31.1): le[int(t*100)] = -30.0\n"
            "segment_match(le, bg, 0.0, [(6, 0)], flags('US'))\n"
            "bad = [m for m in sys.modules if m.startswith(('opencourtvision.ball', 'opencourtvision.importers', 'opencourtvision.scoring.decoder', 'torch'))]\n"
            "assert not bad, bad\n")
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "..")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_provenance_rejects_ball_sources():
    check([{"sources": {"audio"}}, {"sources": {"audio", "structure", "motion_energy"}}])
    for bad in ({"ball_track"}, {"audio", "bounce"}, {"swingvision"}, set()):
        with pytest.raises(ProvenanceError):
            check([{"sources": bad}])


def test_selection_rejects_ball_features_and_common_drops_score():
    seg = {"A": 0, "B": 10, "sources": {"audio"}, "features": {"hits": 5, "react": 3, "break_point": 1}}
    with pytest.raises(ProvenanceError):
        selection_features({**seg, "features": {"hits": 3, "t_dead": 4.2}})
    assert "break_point" not in selection_features(seg)
    assert story_score_arg([{"break_point": True}], flags("US")) is None
    with pytest.raises(ProvenanceError):
        select_clips([{**seg, "sources": {"ball"}}], 1)


def test_full_variant_keeps_ball_rule():
    from opencourtvision.variants.full.selection import score_aware_select
    seg = {"A": 0, "B": 10, "sources": {"audio"}, "features": {"hits": 5, "break_point": 1}}
    assert score_aware_select([seg], flags("ROW"), 1) == [0]
    with pytest.raises(ProvenanceError):
        score_aware_select([{**seg, "features": {"bounce": 1}}], flags("ROW"), 1)


def test_us_profile_cannot_be_overridden():
    f = flags("US", score_in_selection=True, typed_filters_tracking=True, multi_phone=True, cloud_processing=True)
    assert not (f.score_in_selection or f.typed_filters_tracking or f.multi_phone or f.cloud_processing)
    assert not flags("ROW").vision_rally_cues                       # vision fallback is off by default
    # v3.9: tracking-typed filters off in every build; multi-phone held in every full-app profile (PlayReplay check)
    assert not any(flags(r).typed_filters_tracking or flags(r).multi_phone or flags(r).vision_rank_features
                   for r in ("ROW", "UK", "ES", "US"))


def _env(hits, n=12000):
    le = np.full(n, -60.0); bg = np.full(n, -60.0)
    for t, db in hits: le[int(round(t * 100))] = -60.0 + db
    return le, bg


def test_e1_never_merges_two_points_and_attaches_faults():
    # point 1: 10-14 s; point 2 serves 6.6 s after (would have merged under padding); fault at 40 s, second serve 48 s
    hits = [(10, 20), (11.2, 20), (12.4, 20), (14.0, 20), (20.6, 20), (21.8, 20), (23.1, 20), (40, 22), (48, 22), (49.2, 20), (50.5, 20)]
    le, bg = _env(hits)
    P, R = audio_points(le, bg, 0.0)
    assert len(P) == 4 and all(p["sources"] == {"audio"} for p in P)
    P2 = attach_faults(P)
    assert len(P2) == 3 and P2[-1]["t0"] == 40.0
    S = clip_bounds(P2, None, 120.0)
    assert all(a["B"] <= b["A"] for a, b in zip(S, S[1:]))           # no overlaps, no merges
    assert S[0]["B"] <= 20.6 - 0.5


def test_e3_gap_fill_admits_isolated_serve_in_game_gap_only():
    hits = [(10, 20), (11.2, 20), (12.4, 20), (40, 19), (70, 20), (71.2, 20), (72.4, 20)]   # 40 s: lone serve (ace?)
    le, bg = _env(hits)
    P, R = audio_points(le, bg, 0.0, {"single_pause": 99})
    assert len(P) == 2
    P2, add = gap_fill(P, R, 0.0, sets=None)
    assert len(add) == 1 and add[0]["sources"] == {"audio", "structure"}
    segs = segment_match(le, bg, 0.0, None, flags("US"))
    assert len(segs) == 3 and all(s["sources"] <= {"audio", "structure"} for s in segs)


def test_motion_cues_drop_idle_court_and_tag_provenance():
    from opencourtvision.editloop.motion import MotionCues, apply_cues
    t = np.arange(1, 6 * 200) / 6.0
    E = np.full((len(t), 12, 24), 0.2, np.float32)
    E[(t > 9) & (t < 16), 3:9, 4:20] = 5.0               # court active during the first point only
    cues = MotionCues(E, t)
    hits = [(10, 20), (11.2, 20), (12.4, 20), (70, 20), (71.2, 20), (72.4, 20)]
    le, bg = _env(hits, 20000)
    P, R = audio_points(le, bg, 0.0)
    out = apply_cues(P, cues, R, 0.0, None)
    assert len(out) == 1 and out[0]["t0"] == 10.0
    assert flags("US").vision_rally_cues is False and flags("US", vision_rally_cues=True).vision_rally_cues
