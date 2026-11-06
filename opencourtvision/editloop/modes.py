"""Processing modes (v4.0): "match" (score engine, point winner, game/set state, score prompts/taps) and "rallies"
(rallies-only: practice, drills, partial clips, highlight uploads; no score keeping at all).

Rallies-only runs the same rally boundaries and dead-space trimming as match mode (editloop/smart.py) and skips the
score engine. Output (schema ocv.rally_list.v1): a rally list (start/end, hits, duration) plus a reel order by rally
length or by a highlight rank that has no won/lost term. Builds: US uses audio only (no motion cues at all in this
mode); the full app may live-gate idle-court clips with motion. Ball tracking is never used.

`suggest_mode` only recommends: when the score engine cannot build a consistent match (no serve pattern, too few
points, a clip starting mid-set, practice-like rallies) it suggests "rallies" with the reasons; the user decides.
"""
import numpy as np

MODES = ("match", "rallies")
RALLY_SCHEMA = "ocv.rally_list.v1"
SUGGEST = dict(min_points=24, serve_pause=12.0, min_serve_share=0.5, practice_hits=12, long_rally_hits=20,
               long_rally_share=0.3, play_share=0.5, mid_set_s=10.0)


def suggest_mode(rallies, start=0.0, duration=None, p=None):
    """rallies: hit groups (t0, t1, hits) after `start` (the warm-up cutoff). Returns dict(suggested, reasons, evidence)."""
    p = {**SUGGEST, **(p or {})}
    R = [r for r in rallies if r["t0"] >= start - 0.01]
    ev = dict(candidates=len(R))
    reasons = []
    if not R:
        return dict(suggested="rallies", reasons=["no rallies found"], evidence=ev)
    hits = np.array([r["hits"] for r in R], float)
    S = [r for r in R if r["hits"] >= 3]                        # strong rallies; resets are measured between them
    pb = np.array([b["t0"] - a["t1"] for a, b in zip(S, S[1:])] or [0.0], float)
    span = max(1e-6, R[-1]["t1"] - R[0]["t0"])
    play = float(sum(r["t1"] - r["t0"] for r in R)) / span
    serve_share = float(np.mean(pb >= p["serve_pause"]))
    ev.update(serve_share=round(serve_share, 2), median_hits=float(np.median(hits)),
              long_rally_share=round(float(np.mean(hits >= p["long_rally_hits"])), 2), play_share=round(play, 2),
              first_rally_s=round(R[0]["t0"], 1))
    if len(R) < p["min_points"]:
        reasons.append(f"too few points for a match ({len(R)} < {p['min_points']})")
    if serve_share < p["min_serve_share"]:
        reasons.append(f"no serve pattern (only {serve_share:.0%} of rallies follow a >= {p['serve_pause']:.0f} s reset)")
    if ev["median_hits"] > p["practice_hits"] or ev["long_rally_share"] > p["long_rally_share"] or play > p["play_share"]:
        reasons.append("practice-like rallies (long, continuous hitting)")
    if start <= p["mid_set_s"] and R[0]["t0"] <= p["mid_set_s"]:
        reasons.append("video starts in play (clip may start mid-set)")
    return dict(suggested="rallies" if reasons else "match", reasons=reasons, evidence=ev)


def rally_rank(c):
    """Highlight rank for rallies-only: rally length and audio reaction; no won/lost or score term."""
    dur = c["t1"] - c["t0"]
    return round(c["hits"] + 0.5 * dur + 0.3 * max(0.0, float(c.get("reaction_db") or 0.0)), 2)


def rallies_only(le, bg, lb, start, flags, KF=None, dur=None, cues=None):
    """Rally clips with smart boundaries; no score state anywhere (no final score, no changeovers, no fault->point
    semantics beyond attaching a short lead-in run). US: audio only, motion cues ignored."""
    from .provenance import check
    from .segment import audio_points, gap_fill
    from .smart import finish, smart_points
    P, R = audio_points(le, bg, start)
    if flags.gap_fill:
        P, _ = gap_fill(P, R, start, None)
    S, _ = smart_points(P, le, bg, lb, None)
    if flags.variant != "us" and flags.vision_rally_cues and cues is not None:
        from .motion import live_gate
        S = live_gate(S, cues)
    return check(finish(S, KF, dur), flags.allowed_sources)


def rally_list(clips, video=None, build=None, order="length", suggestion=None):
    """ocv.rally_list.v1 document; `reel` is the rally order for the reel (by length or by rally_rank)."""
    if order not in ("length", "rank"):
        raise ValueError("order must be 'length' or 'rank'")
    rows = []
    for i, c in enumerate(clips):
        rows.append(dict(idx=i + 1, start_s=round(c["A"], 2), end_s=round(c["B"], 2), duration_s=round(c["B"] - c["A"], 2),
                         hits=c["hits"], rally_s=round(c["t1"] - c["t0"], 2), reaction_db=c.get("reaction_db"),
                         rank_score=rally_rank(c)))
    key = (lambda r: (-r["rally_s"], r["idx"])) if order == "length" else (lambda r: (-r["rank_score"], r["idx"]))
    return dict(schema=RALLY_SCHEMA, mode="rallies", video=video, build=build, order=order, n_rallies=len(rows),
                rallies=rows, reel=[r["idx"] for r in sorted(rows, key=key)], suggestion=suggestion)
