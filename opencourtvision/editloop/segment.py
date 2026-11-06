"""E1 audio point segmentation and E3 structure-guided gap fill (no ball tracking anywhere in this module).

A *point* here is a run of audio hit onsets. Point segmentation is decoupled from clip padding: padded clips never merge
two points (the old `render.cuts.build_segments` merged any two rallies whose padded clips touched, which made 10 of the
101 points on dev_match share a segment). Where padding overlaps, the cut goes between the last hit of one point and the
first hit of the next.
"""
import bisect

from ..detection.audio import detect_hits, group_rallies
from ..render.cuts import point_end
from ..scoring.games import changeover_games

E1 = dict(th=14.0, min_rise=6.0, gap=6.0, amin=16.0, single_amp=20.0, single_pause=8.0)
E3 = dict(min_gap=15.0, single_amp=18.0, quiet=3.0, after_fault=25.0, fault_hits=2)


def _point(r, src):
    return dict(t0=r["t0"], t1=r["t1"], hits=r["hits"], amax=r["amax"], times=list(r["times"]), sources={*src},
                pause_before=r.get("pause_before", 99.0), pause_after=r.get("pause_after", 99.0))


def audio_points(le, bg, start, p=None):
    """E1: hit onsets -> rallies -> points. Returns (points, all_rallies)."""
    p = {**E1, **(p or {})}
    t, amp = detect_hits(le, bg, th=p["th"], min_rise=p["min_rise"])
    R = group_rallies(t, amp, gap=p["gap"])
    keep = [r for r in R if r["t0"] >= start - 0.01 and r["amax"] >= p["amin"]
            and (r["hits"] >= 2 or (r["amax"] >= p["single_amp"] and r["pause_before"] >= p["single_pause"]))]
    return [_point(r, ("audio",)) for r in keep], R


def gap_fill(points, rallies, start, sets, p=None):
    """E3: admit isolated loud single onsets (serve-only points: aces, service winners, double faults) where the match
    structure says a point is likely missing: inside an in-game gap >= min_gap s (the len(changeovers) longest gaps,
    known from the final score, are excluded), or within after_fault s of a fault-like point (<= fault_hits hits)."""
    p = {**E3, **(p or {})}
    pts = sorted(points, key=lambda x: x["t0"])
    if len(pts) < 2: return pts, []
    nco = len(changeover_games(sets)) if sets else 0
    gaps = sorted(((pts[i + 1]["t0"] - pts[i]["t1"], i) for i in range(len(pts) - 1)), reverse=True)
    co = {i for _, i in gaps[:nco]}
    t1s = [x["t1"] for x in pts]
    have = {(round(x["t0"], 2), round(x["t1"], 2)) for x in pts}
    add = []
    for r in rallies:
        if r["t0"] < start or (round(r["t0"], 2), round(r["t1"], 2)) in have: continue
        j = bisect.bisect_left(t1s, r["t0"]) - 1
        if j < 0 or j + 1 >= len(pts) or j in co: continue
        prev, nxt = pts[j], pts[j + 1]
        if not (prev["t1"] < r["t0"] and r["t1"] < nxt["t0"]): continue
        fault_prev = prev["hits"] <= p["fault_hits"] and r["t0"] - prev["t1"] <= p["after_fault"]
        quiet = r["pause_before"] >= p["quiet"] and r["pause_after"] >= p["quiet"]
        if r["amax"] >= p["single_amp"] and quiet and (nxt["t0"] - prev["t1"] >= p["min_gap"] or fault_prev):
            add.append(_point(r, ("audio", "structure")))
    return sorted(pts + add, key=lambda x: x["t0"]), add


def clip_bounds(points, KF=None, dur=None, lead=4.0, min_lead=1.5, tail=None):
    """Clip [A, B] per point. Padding never merges points: overlapping pads are split between the two points."""
    out = []
    for q in points:
        a, b = q["t0"] - lead, point_end(q["t1"], None, tail, clip_type="point")
        if dur is not None: b = min(b, dur)
        if out and a < out[-1]["B"]:
            prev_last = out[-1]["t1"]
            cut = max(prev_last + 1.0, q["t0"] - max(min_lead, (q["t0"] - prev_last) / 2.0))
            cut = min(cut, q["t0"] - 0.5)
            out[-1]["B"] = max(out[-1]["A"] + 0.5, cut); a = out[-1]["B"]
        out.append({**q, "A": max(0.0, a), "B": b})
    if KF:   # snap outward to keyframes for stream copy, without crossing a neighbour
        for i, s in enumerate(out):
            k = bisect.bisect_right(KF, s["A"]) - 1
            A = KF[max(0, k)]
            if i == 0 or A >= out[i - 1]["B"]: s["A"] = A
            j = bisect.bisect_left(KF, s["B"])
            B = KF[j] if j < len(KF) else (dur if dur is not None else s["B"])
            if i + 1 == len(out) or B <= out[i + 1]["A"]: s["B"] = B
    for i, s in enumerate(out):
        s["fault_like"] = s["hits"] <= 2 and (out[i + 1]["t0"] - s["t1"] < 20 if i + 1 < len(out) else False)
        s["serve_only"] = s["hits"] == 1
        s["gap_after"] = out[i + 1]["A"] - s["B"] if i + 1 < len(out) else 0.0
        s["rallies"] = [dict(t0=s["t0"], t1=s["t1"], hits=s["hits"], amax=s["amax"], times=s["times"],
                             pause_after=s["gap_after"])]
    return out


FAULT = dict(max_gap=15.0, max_hits=2)


def attach_faults(points, p=None):
    """A short audio point (<= max_hits onsets) followed by the next serve within max_gap s is a first-serve fault: it is
    shown as the start of the next point rather than as its own clip (audio rhythm only)."""
    p = {**FAULT, **(p or {})}
    pts = sorted(points, key=lambda x: x["t0"]); out = []; k = 0
    while k < len(pts):
        q = pts[k]
        if q["hits"] <= p["max_hits"] and k + 1 < len(pts) and pts[k + 1]["t0"] - q["t1"] <= p["max_gap"]:
            n = pts[k + 1]
            out.append({**n, "t0": q["t0"], "hits": q["hits"] + n["hits"], "times": q["times"] + n["times"],
                        "amax": max(q["amax"], n["amax"]), "sources": set(q["sources"]) | set(n["sources"]), "fault_first": True})
            k += 2
        else:
            out.append(q); k += 1
    return out


def segment_match(le, bg, start, sets, flags, KF=None, dur=None, cues=None, lead=4.0, steps=None, lb=None, smart=False):
    """E1 (+E3 if flags.gap_fill) (+fault attach) (+motion cues if flags.vision_rally_cues and cues given) -> clips.
    smart=True (needs lb): evidence-based boundaries and non-point rejection (editloop/smart.py) instead of fixed
    padding; same motion rules (EU admission, US live gate only). `steps`, if a dict, receives the intermediate point
    lists by step name (for evaluation)."""
    from .provenance import check
    P, R = audio_points(le, bg, start)
    if steps is not None: steps["E1"] = P
    if flags.gap_fill:
        P, _ = gap_fill(P, R, start, sets)
        if steps is not None: steps["E1+E3"] = P
    if smart:
        if lb is None: raise ValueError("smart boundaries need the broadband loudness lb")
        from .smart import finish, smart_points
        if flags.vision_rally_cues and cues is not None and getattr(flags, "motion_boundaries", False):
            from .motion import admit_candidates
            P = admit_candidates(P, cues, R, start, sets)
            if steps is not None: steps["+motion_admit"] = P
        S, rej = smart_points(P, le, bg, lb, sets)
        if (flags.vision_rally_cues or getattr(flags, "live_gate", False)) and cues is not None \
                and not getattr(flags, "motion_boundaries", False):
            from .motion import live_gate
            S = live_gate(S, cues)                  # US: drops whole idle-court clips only; boundaries stay audio
        if steps is not None: steps["+smart"] = S; steps["rejected"] = rej
        return check(finish(S, KF, dur), flags.allowed_sources)
    if flags.vision_rally_cues and cues is not None and getattr(flags, "motion_boundaries", False):
        from .motion import apply_cues
        P = apply_cues(P, cues, R, start, sets)
        if steps is not None: steps["+motion"] = P
    elif flags.vision_rally_cues and cues is not None:        # US: motion gates live play only, never boundaries
        from .motion import live_gate
        P = live_gate(attach_faults(P), cues)
        if steps is not None: steps["+live_gate"] = P
    else:
        P = attach_faults(P)
        if steps is not None: steps["+fault_attach"] = P
    return check(clip_bounds(P, KF, dur, lead), flags.allowed_sources)
