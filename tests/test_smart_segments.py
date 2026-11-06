"""Smart dead space (editloop/smart.py) and rallies-only mode (editloop/modes.py) on synthetic audio envelopes."""
import json

import numpy as np
import pytest

from opencourtvision.editloop import segment as SG
from opencourtvision.editloop.build import flags
from opencourtvision.editloop.modes import rallies_only, rally_list, suggest_mode
from opencourtvision.editloop.smart import SMART, anatomy, smart_points

FR = 100


def env(dur, hits, quiet=(), react=()):
    """le/bg/lb per 10 ms: background 0 dB; hits are 25 dB onsets, quiet onsets 11 dB; react = (t0, t1, dB) broadband."""
    n = int(dur * FR)
    le = np.zeros(n); bg = np.zeros(n); lb = np.zeros(n)
    for t, a in [(t, 25.0) for t in hits] + [(t, 11.0) for t in quiet]:
        i = int(t * FR); le[i:i + 3] = a
    for a, b, db in react:
        lb[int(a * FR):int(b * FR)] = db
    return le, bg, lb


def rally(t0, n, ioi=1.2):
    return [t0 + k * ioi for k in range(n)]


def cand(times):
    return dict(t0=times[0], t1=times[-1], hits=len(times), amax=25.0, times=list(times), sources={"audio"})


def test_anatomy_fault_then_rally_and_pickup_dropped():
    fault = [100.0]
    played = rally(108.0, 6)
    pickup = [played[-1] + 6.0]
    f, run, dropped = anatomy(fault + played + pickup, SMART)
    assert f == fault and run == played and dropped == [pickup]


def test_boundaries_are_evidence_based_and_reaction_capped():
    hits = rally(50.0, 5) + rally(90.0, 5)
    le, bg, lb = env(140, hits, quiet=[48.6], react=[(96.0, 99.0, 20.0)])   # quiet far serve at 48.6; cheering after point 2
    out, _ = smart_points([cand(rally(50.0, 5)), cand(rally(90.0, 5))], le, bg, lb)
    a, b = out
    assert a["quiet_serve"] and a["A"] <= 48.6 - 0.5 and a["A"] >= 48.6 - 2.0      # starts just before the quiet serve
    assert abs(a["B"] - (a["t1"] + SMART["TAIL"])) < 1e-6 and not a["reaction_ext"]
    assert b["reaction_ext"] and b["B"] <= b["t1"] + SMART["CAP"] + 1e-6 and b["B"] > b["t1"] + SMART["TAIL"]


def test_first_serve_fault_merged_into_point_not_its_own_clip():
    le, bg, lb = env(100, [20.0] + rally(30.0, 6))
    out, rej = smart_points([cand([20.0]), cand(rally(30.0, 6))], le, bg, lb)
    assert len(out) == 1 and out[0]["fault_first"] and len(out[0]["parts"]) == 2
    assert [w for _, w in rej] == ["fault->next"]


def test_pickup_after_point_and_break_hits_dropped():
    pts = [rally(100.0, 6), [109.0, 110.0], rally(140.0, 5), [250.0], rally(330.0, 5)]
    le, bg, lb = env(400, [t for r in pts for t in r])
    out, rej = smart_points([cand(r) for r in pts], le, bg, lb)
    assert [round(c["t0"]) for c in out] == [100, 140, 330]
    assert sorted(w for _, w in rej) == ["break", "pickup"]


def test_us_smart_segment_match_is_audio_only():
    hits = rally(50.0, 5) + rally(90.0, 5) + rally(130.0, 5)
    le, bg, lb = env(170, hits)
    segs = SG.segment_match(le, bg, 0.0, None, flags("US"), lb=lb, smart=True)
    assert len(segs) == 3 and all(s["sources"] <= {"audio", "structure"} for s in segs)
    with pytest.raises(ValueError):
        SG.segment_match(le, bg, 0.0, None, flags("US"), smart=True)


class _Cues:
    """Records use: rallies-only in the US build must not touch motion."""
    p = dict(idle_min=9.0)

    def __init__(self): self.used = False

    def activity(self, *a, **k):
        self.used = True
        return 0.0


def test_rallies_only_us_ignores_motion_and_has_no_score_fields():
    hits = [t for k in range(6) for t in rally(20.0 + 25 * k, 14, 0.9)]
    le, bg, lb = env(190, hits)
    cues = _Cues()
    clips = rallies_only(le, bg, lb, 0.0, flags("US", vision_rally_cues=True), cues=cues)
    assert not cues.used and len(clips) == 6
    doc = rally_list(clips, "drill.mp4", "US", order="length")
    assert doc["schema"] == "ocv.rally_list.v1" and doc["mode"] == "rallies"
    banned = {"winner", "score", "score_before", "server", "games", "sets", "won", "lost"}
    assert not banned & set(json.dumps(doc).replace('"', " ").split())
    assert sorted(doc["reel"]) == list(range(1, 7))
    r2 = rally_list(clips, order="rank")
    assert r2["order"] == "rank" and len(r2["reel"]) == 6
    with pytest.raises(ValueError):
        rally_list(clips, order="won")


def _groups(starts, n, ioi=1.2, pause=20.0):
    from opencourtvision.detection.audio import group_rallies
    t = np.array(sorted(x for s in starts for x in rally(s, n, ioi))); return group_rallies(t, np.full(len(t), 25.0))


def test_suggest_mode_match_vs_practice():
    match = _groups([60.0 + 25 * k for k in range(60)], 4)
    s = suggest_mode(match, 50.0)
    assert s["suggested"] == "match" and not s["reasons"]
    drill = _groups([5.0 + 30 * k for k in range(10)], 25, 1.0)
    d = suggest_mode(drill, 0.0)
    assert d["suggested"] == "rallies"
    txt = " ".join(d["reasons"])
    assert "too few points" in txt and "practice-like" in txt and "mid-set" in txt
    assert suggest_mode([], 0.0)["suggested"] == "rallies"


def test_fault_lead_in_is_a_short_jump_cut():
    le, bg, lb = env(100, [20.0] + rally(30.0, 6))
    out, _ = smart_points([cand([20.0]), cand(rally(30.0, 6))], le, bg, lb)
    fa, fb = out[0]["parts"][0]
    assert fb - fa <= 0.8 + 1e-9 and fa <= 20.0 - 0.5                          # toss + serve + call, <= 0.8 s


def test_rally_chaining_keeps_the_rest_of_a_paused_rally():
    first, rest = rally(100.0, 5), rally(110.0, 4)                              # 5.2 s pause (lob / quiet far hits)
    le, bg, lb = env(160, first + rest)
    off, _ = smart_points([cand(first + rest)], le, bg, lb, p=dict(KEEP_HITS=None))
    assert off[0]["t1"] == first[-1]                                            # chaining off: the rest is dropped
    on, _ = smart_points([cand(first + rest)], le, bg, lb, p=dict(KEEP_HITS=3))
    assert on[0]["t1"] == rest[-1] and on[0]["t0"] == first[0]
    (p0, p1), = on[0]["pauses"]                                                 # the pause is a jump cut, not dead time
    assert abs(p0 - (first[-1] + 1.0)) < 1e-9 and abs(p1 - (rest[0] - 1.0)) < 1e-9
    assert on[0]["parts"] == [(on[0]["A"], p0), (p1, on[0]["B"])]
    fin = SG.__dict__.get("finish") or __import__("opencourtvision.editloop.smart", fromlist=["finish"]).finish
    f = fin(on, KF=[0.0, 50.0, 99.0, 108.0, 150.0], dur=160.0)[0]
    assert f["pauses"][0][1] == 108.0 and f["parts"][-1][0] == 108.0           # resume snapped back to a keyframe
    from opencourtvision.render.cuts import pause_spans
    assert pause_spans(f) == [(f["A"], f["pauses"][0][0]), (108.0, f["B"])]
    assert smart_points([cand(first + rest)], le, bg, lb)[0][0]["t1"] == rest[-1]   # on by default (KEEP_HITS 4)
    far, _ = smart_points([cand(first + rest)], le, bg, lb, p=dict(KEEP_HITS=3, CHAIN_MAX=4.0))
    assert far[0]["t1"] == first[-1]                                            # beyond CHAIN_MAX: not the same rally
    two, _ = smart_points([cand(first + rally(130.0, 5))], le, bg, lb, p=dict(POINT_GAP=12.0))
    assert len(two) == 2                                                        # an inter-point gap splits the candidate


def test_us_live_gate_on_by_default_keep_drop_only():
    class Cues:
        p = {"idle_min": 5}

        def activity(self, a, b):
            return 0 if a > 60 else 20                                          # court idle for the second clip
    assert flags("US").live_gate and not flags("US").vision_rally_cues
    hits = rally(10.0, 5) + rally(80.0, 5)
    le, bg, lb = env(140, hits)
    plain = SG.segment_match(le, bg, 0.0, None, flags("US"), lb=lb, smart=True)
    gated = SG.segment_match(le, bg, 0.0, None, flags("US"), cues=Cues(), lb=lb, smart=True)
    assert len(gated) == len(plain) - 1
    assert (gated[0]["A"], gated[0]["B"]) == (plain[0]["A"], plain[0]["B"])     # boundaries untouched (audio only)
    off = SG.segment_match(le, bg, 0.0, None, flags("US", live_gate=False), cues=Cues(), lb=lb, smart=True)
    assert len(off) == len(plain)
