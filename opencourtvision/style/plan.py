"""Reel plan (pure, testable): which whole points, where the 0-2 slow-mos go, which verified captions, the score bug,
intro and result card, all on the OUTPUT timeline. render.py executes a plan; validate() is run before every export.

Hard rules (Antony; enforced here whatever the preset says, covered by tests/test_style_engine.py):
  R-WHOLE    every clip is a whole point: from >= 3.0 s before the first contact to >= 2.0 s after the last
  R-SLOW     0-2 slow-mo moments per reel, never slower than 0.45x, cosine ramps (no hard speed step), only on
             user-verified exceptional points (favourite / marked highlight)
  R-CAPTION  stakes captions only from the stakes allowlist, only from a verified score state (rules engine), and only
             on the clip that shows that whole point; they clear before the next cut
  R-CUT      hard cuts between points (audio gets a short crossfade)
  R-9x16     no letterboxing, no wide mode: 9:16 is always a true tracked crop (render.py)
Selection uses audio, structure and user input only (never tracked-ball evidence: editloop/provenance.py).
"""
import math

import numpy as np

from ..editloop.provenance import check as provenance_check
from ..render.overlays import clear_on_cuts
from .preset import HARD
from .stakes import caption_for, saved

EPS = 1e-6
RAMP_STEPS = 5  # cosine samples per ramp side


class PlanError(ValueError):
    pass


def clip_score(p):
    """Audio + structure + user score of one point (no tracking terms). A caller-ranked point (Highlights only:
    app/highlights.rank, audio + rules) carries its own rank_score, used as is."""
    if p.get("rank_score") is not None: return round(float(p["rank_score"]) + (4.0 if p.get("favorite") else 0.0), 3)
    dur = p["last_hit"] - p["first_hit"]
    return round(p["hits"] + 0.5 * dur + 0.3 * max(0.0, p.get("react", 0.0)) + (4.0 if p.get("favorite") else 0.0), 3)


def whole_bounds(p, pre=3.5, post=2.5, pre_min=3.0, pre_max=4.0, post_min=2.0, post_cap=6.0):
    """Clip [a, b] for a whole point: a in [first-pre_max, first-pre_min] (prefers the engine's own start when it is in
    that window), b from last + post up to the engine's end (celebration kept) capped at last + post_cap."""
    f, L = p["first_hit"], p["last_hit"]
    a = min(max(p.get("A", f - pre), f - pre_max), f - pre_min)
    if p.get("A") is None: a = f - pre
    b = min(max(p.get("B", L + post), L + max(post, post_min)), L + post_cap)
    return max(0.0, a), b


def ramp_parts(t_c, factor, ramp_s, hold_s):
    """Speed parts (src_t0, src_t1, speed) for a slow-mo centred on t_c: cosine ramp 1 -> factor, plateau, factor -> 1."""
    h0, h1 = t_c - hold_s / 2, t_c + hold_s / 2
    parts = []
    for k in range(RAMP_STEPS):          # ramp in, sampled at step mid-points
        u = (k + 0.5) / RAMP_STEPS
        s = 1 - (1 - factor) * 0.5 * (1 - math.cos(math.pi * u))
        parts.append((h0 - ramp_s + k * ramp_s / RAMP_STEPS, h0 - ramp_s + (k + 1) * ramp_s / RAMP_STEPS, s))
    parts.append((h0, h1, factor))
    for k in range(RAMP_STEPS):
        u = (k + 0.5) / RAMP_STEPS
        s = factor + (1 - factor) * 0.5 * (1 - math.cos(math.pi * u))
        parts.append((h1 + k * ramp_s / RAMP_STEPS, h1 + (k + 1) * ramp_s / RAMP_STEPS, s))
    return parts


def clip_parts(a, b, slow=None):
    """Whole clip [a, b] as speed parts; slow = list of ramp parts inside it (or None)."""
    if not slow: return [(a, b, 1.0)]
    out = [(a, slow[0][0], 1.0)] + list(slow) + [(slow[-1][1], b, 1.0)]
    return [q for q in out if q[1] - q[0] > EPS]


def out_len(parts):
    return sum((q[1] - q[0]) / q[2] for q in parts)


def to_out(parts, t):
    """Source time t (inside the clip) -> output time from the clip start."""
    o = 0.0
    for t0, t1, s in parts:
        if t <= t1 + EPS: return o + max(0.0, t - t0) / s
        o += (t1 - t0) / s
    return o


def _verified_final(final, replay_final):
    return bool(final) and final == replay_final


SCORE_MIN_CONF = 0.90   # v4.9 (Antony): with the score on, a point's score bug shows only when confirmed or >= this


def score_shown(p, min_conf=SCORE_MIN_CONF):
    """A point's score may be shown: there is a score state and it is confirmed (verified / user anchor) or the
    engine's confidence for it is >= min_conf. Never guessed."""
    if p.get("state") is None: return False
    if p.get("state_verified"): return True
    c = p.get("state_conf")
    return c is not None and float(c) >= min_conf


def plan_reel(points, preset, aspect="9:16", match=None, max_points=8, max_s=80.0, music=False, keep_pct=60,
              show_score=True, score_min_conf=SCORE_MIN_CONF):
    """points: session points (dicts: id, A, B, first_hit, last_hit, hits, react, favorite, winner, winner_verified,
    state (ScoreState before the point or None), state_verified, sources). match: dict(sport, format (ScoringFormat),
    teams (user first), date, final (user-entered), final_replayed (rules engine), user_team)."""
    hc = preset.get("hard_constraints", {})
    pre_min = max(HARD["pre_roll_s_min"], float(hc.get("pre_roll_s_min", 3.0)))
    post_min = max(HARD["post_roll_s_min"], float(hc.get("post_roll_s_min", 2.0)))
    pts = [p for p in points if p.get("kind", "point") == "point"]
    provenance_check([{"sources": p.get("sources", ["audio"])} for p in pts])
    if not pts: raise PlanError("no points to plan")
    sc = {p["id"]: clip_score(p) for p in pts}
    # keep_pct: quality floor among the given points (Highlights only pre-selects its best clips and passes 0)
    p60 = float(np.percentile(list(sc.values()), keep_pct)) if len(pts) >= 5 and keep_pct > 0 else -1e9
    keep = [p for p in pts if sc[p["id"]] >= p60]            # quality over quantity (spec §0.1)
    last = max(pts, key=lambda p: p["first_hit"])
    show_result = show_score and match is not None and _verified_final(match.get("final"), match.get("final_replayed"))
    ranked = sorted(keep, key=lambda p: -sc[p["id"]])
    chosen = []
    if show_result: chosen.append(last)                      # the result card needs the final point, whole
    for p in ranked:
        if p in chosen or len(chosen) >= max_points: continue
        a, b = whole_bounds(p, pre_min=pre_min, post_min=post_min)
        if sum(whole_bounds(q)[1] - whole_bounds(q)[0] for q in chosen) + (b - a) > max_s: continue
        chosen.append(p)
    chosen.sort(key=lambda p: p["first_hit"])                # chronological (Antony)
    sm = preset["slow_mo"]
    n_slow = min(int(sm.get("max_per_reel", 0)), HARD["slow_mo_max_per_reel"])
    factor = max(float(sm.get("max_factor", 0.45)), HARD["slow_mo_min_factor"])
    ramp_s, hold_s = sm.get("ramp_ms", 280) / 1000.0, min(sm.get("hold_ms_max", 700), 700) / 1000.0
    eligible = sorted([p for p in chosen if p.get("favorite")], key=lambda p: -sc[p["id"]])[:n_slow]
    slow_ids = {p["id"] for p in eligible}
    fmt = (match or {}).get("format")
    capcfg = preset["captions"]; allow = capcfg["stakes_allowlist"]
    cap_d = float(np.clip(1.0, capcfg["duration_s"]["min"], capcfg["duration_s"]["max"]))
    clips, overlays, t_out, last_cap = [], [], 0.0, -1e9
    n_caps = 0
    for p in chosen:
        a, b = whole_bounds(p, pre_min=pre_min, post_min=post_min)
        slow = ramp_parts(p["last_hit"], factor, ramp_s, hold_s) if p["id"] in slow_ids else None
        if slow and (slow[0][0] < a or slow[-1][1] > b): slow = None
        parts = clip_parts(a, b, slow)
        L = out_len(parts)
        c = dict(id=p["id"], a=round(a, 3), b=round(b, 3), first_hit=p["first_hit"], last_hit=p["last_hit"], parts=parts,
                 out_start=round(t_out, 4), out_len=round(L, 4), slow=bool(slow), score=sc[p["id"]], transition_in="hard_cut")
        verified = bool(p.get("state_verified")) and p.get("state") is not None and fmt is not None
        if verified:
            lab, team = caption_for(p["state"], fmt, True, allow)
            t_c = t_out + 0.3
            if lab and n_caps < capcfg.get("max_stakes_per_reel", 3) and t_c - last_cap >= 8.0:
                overlays.append(dict(kind="caption", text=lab, team=team, point=p["id"], start=t_c, end=t_c + cap_d))
                n_caps += 1; last_cap = t_c
                sv = saved(lab, team, p.get("winner") if p.get("winner_verified") else None)
                if sv and sv in allow and n_caps < capcfg.get("max_stakes_per_reel", 3):
                    ts = t_out + to_out(parts, p["last_hit"] + 1.0)
                    overlays.append(dict(kind="caption", text=sv, team=team, point=p["id"], start=ts, end=ts + cap_d)); n_caps += 1
        if show_score and fmt is not None and score_shown(p, score_min_conf):   # v4.9 toggle; confirmed / >= 0.9 only
            upd = t_out + to_out(parts, p["last_hit"] + 1.0)     # ~t_dec + 0.5 s; never mid-rally
            overlays.append(dict(kind="score_bug", point=p["id"], state=p["state"],
                                 winner=p.get("winner") if p.get("winner_verified") else None,
                                 start=t_out, end=t_out + L, update_at=upd, persistent_in_clip=True))
        clips.append(c); t_out += L
    cuts = [c["out_start"] for c in clips[1:]]
    caps = clear_on_cuts([o for o in overlays if o["kind"] == "caption"], cuts)
    bugs = [dict(o, end=o["end"]) for o in overlays if o["kind"] == "score_bug"]
    intro = None
    if match:
        lines = [f"{match.get('sport', 'tennis').upper()} · {match.get('format_name', 'DOUBLES').upper()}"]
        if match.get("date_label"): lines.append(match["date_label"])
        if match.get("teams"): lines.append(f"{match['teams'][0]}  vs  {match['teams'][1]}")
        intro = dict(kind="intro", lines=lines, start=0.0, end=min(1.5, clips[0]["out_len"]))
    outro = None
    if show_result:
        dur = float(np.clip(2.5, preset["outro"]["duration_s"]["min"], preset["outro"]["duration_s"]["max"]))
        outro = dict(kind="result", winner=match.get("winner"), teams=match.get("teams"), score=match["final"], start=t_out, end=t_out + dur, hold_last_frame=True)
    plan = dict(schema="vamos.reel_plan.v1", preset=preset["id"], aspect=aspect, clips=clips, captions=caps, bugs=bugs,
                intro=intro, outro=outro, transitions="hard_cut", audio_crossfade_s=preset["transitions"].get("audio_crossfade_s", 0.2),
                music=("platform_sound_picker_at_share" if not music else "user_selected"), grade=preset["grade"].get("enabled", True),
                slow_mo_count=sum(c["slow"] for c in clips), letterbox=False, wide_mode=False,
                duration_s=round(t_out + (outro["end"] - outro["start"] if outro else 0.0), 3),
                dropped_below_p60=len(pts) - len(keep))
    validate(plan, points, preset)
    return plan


def validate(plan, points, preset):
    """Raise PlanError unless every hard rule holds. Called by plan_reel and again by render before encoding."""
    by = {p["id"]: p for p in points}
    clips = plan["clips"]
    for c in clips:
        p = by[c["id"]]
        if c["a"] > p["first_hit"] - HARD["pre_roll_s_min"] + 1e-3 or c["b"] < p["last_hit"] + HARD["post_roll_s_min"] - 1e-3:
            raise PlanError(f"R-WHOLE: clip {c['id']} [{c['a']}, {c['b']}] cuts point {p['first_hit']}-{p['last_hit']}")
        if abs(c["parts"][0][0] - c["a"]) > 1e-6 or abs(c["parts"][-1][1] - c["b"]) > 1e-6:
            raise PlanError(f"R-WHOLE: clip {c['id']} parts do not span the clip")
        for (x0, x1, _), (y0, _, _) in zip(c["parts"], c["parts"][1:]):
            if abs(x1 - y0) > 1e-6: raise PlanError(f"R-WHOLE: gap inside clip {c['id']}")
        speeds = [q[2] for q in c["parts"]]
        if min(speeds) < HARD["slow_mo_min_factor"] - 1e-9: raise PlanError(f"R-SLOW: {min(speeds)}x slower than 0.45x")
        # Cosine ramps change gradually; a hard step is a single jump bigger than half the (1 - floor) range.
        if any(abs(s1 - s0) > max(0.35, (1.0 - HARD["slow_mo_min_factor"]) / 2 + 0.05) for s0, s1 in zip(speeds, speeds[1:])):
            raise PlanError("R-SLOW: hard speed step")
        if c.get("transition_in", "hard_cut") != "hard_cut": raise PlanError("R-CUT: only hard cuts")
    for x, y in zip(clips, clips[1:]):
        if y["a"] < x["b"] - 1e-6: raise PlanError("clips overlap or are out of time order")
    n_slow = sum(c["slow"] for c in clips)
    if n_slow > HARD["slow_mo_max_per_reel"]: raise PlanError(f"R-SLOW: {n_slow} slow-mos > 2")
    for c in clips:
        if c["slow"] and not by[c["id"]].get("favorite"): raise PlanError("R-SLOW: slow-mo on an unverified point")
    allow = set(preset["captions"]["stakes_allowlist"])
    cuts = [c["out_start"] for c in clips[1:]]
    for o in plan["captions"]:
        if o["text"] not in allow: raise PlanError(f"R-CAPTION: {o['text']!r} not on the stakes allowlist")
        p = by[o["point"]]
        if not p.get("state_verified"): raise PlanError(f"R-CAPTION: caption on unverified point {o['point']}")
        c = next(c for c in clips if c["id"] == o["point"])
        if o["start"] < c["out_start"] - 1e-6 or o["end"] > c["out_start"] + c["out_len"] + 1e-6:
            raise PlanError("R-CAPTION: caption outside the whole point it names")
        if any(o["start"] + 1e-9 < t < o["end"] - 1e-9 for t in cuts): raise PlanError("R-CAPTION: caption spans a cut")
    if len([o for o in plan["captions"] if o["text"] != "MATCH POINT SAVED"]) > preset["captions"].get("max_stakes_per_reel", 3):
        raise PlanError("R-CAPTION: too many stakes captions")
    if plan.get("letterbox") or plan.get("wide_mode"): raise PlanError("R-9x16: letterbox / wide mode")
    return True
