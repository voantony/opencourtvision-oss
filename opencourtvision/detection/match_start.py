"""Warmup -> match transition.

Two detectors:
* `warmup_cutoff` (used by the main pipeline): find the warmup core (near-continuous hitting), then the first
  strong rally after a reset pause with rally rhythm, stepping back to a first-serve fault if present.
* `estimate_match_start` (standalone CLI): rolling 5-minute rhythm table plus, optionally, the first cluster of
  spoken score callouts in a whisper transcript.
"""
import json
import os
import re
import subprocess
import tempfile

import numpy as np

from ..ffmpeg import default_font, ffmpeg_bin, fontfile_opt
from .audio import detect_hits, load_audio, loudness, time_groups


def warmup_cutoff(R, override=None):
    """Match start = first serve after the continuous-hitting warmup ends.
    1) warmup core: last 2-min window (first half of video) whose hit coverage >= 0.45 (near-continuous hitting).
    2) after that, the first strong rally (>= 3 hits) that follows a reset pause >= 20 s and has rally rhythm
       (median inter-hit <= 2.5 s; practice serves are spaced 3-6 s apart and are rejected).
    3) step back to a serve-only hit <= 15 s before it (first-serve fault), if any.
    Does not rely on changeover gaps (early games are often quick)."""
    if override is not None:
        return override, {"method": "override"}
    S = [r for r in R if r["amax"] >= 16]
    for i, r in enumerate(S):
        r["pb"] = r["t0"] - S[i - 1]["t1"] if i else r["t0"]
    half = R[-1]["t1"] / 2
    core = None
    for t0 in np.arange(0, half, 10):
        gs = [r for r in S if t0 <= r["t0"] < t0 + 120 and r["hits"] >= 2]
        cov = sum(min(r["t1"], t0 + 120) - r["t0"] for r in gs) / 120
        if cov >= 0.45:
            core = float(t0)
        elif core is not None and t0 > core + 180:
            break  # warmup = the opening run of near-continuous hitting
    lo = core if core is not None else 0

    def med_ih(r):
        d = np.diff(r["times"])
        return float(np.median(d)) if len(d) else 99

    rejected, start, why = [], None, None
    for r in S:
        if r["t0"] <= lo or r["hits"] < 3:
            continue
        if r["pb"] >= 20 and med_ih(r) <= 2.5:
            start = r["t0"]
            why = r
            break
        if r["pb"] >= 20:
            rejected.append((round(r["t0"], 1), f"median inter-hit {med_ih(r):.1f}s (practice serves?)"))
    if start is None:
        return lo, {"warmup_core_window_start_s": core, "confidence": "low"}
    prev = [r for r in S if start - 15 <= r["t0"] < start and r["hits"] == 1]
    if prev:
        start = prev[0]["t0"]
    ev = {"warmup_core_window_start_s": core, "first_rally_s": why["t0"], "pause_before_s": round(why["pb"], 1),
          "median_interhit_s": round(med_ih(why), 2), "hits": why["hits"], "rejected_before": rejected,
          "confidence": "high" if why["pb"] >= 30 else "medium"}
    return start, ev


def rhythm(groups, dur, win):
    """Per 30 s step, over the following `win` seconds: hit-coverage fraction, median pause, longest rally."""
    rows = []
    for t0 in np.arange(0, max(dur - win, 1), 30):
        gs = [g for g in groups if t0 <= g[0] < t0 + win]
        if len(gs) < 3:
            rows.append((t0, 0, 0, 0, len(gs)))
            continue
        cover = sum(g[-1] - g[0] for g in gs) / win
        pauses = [b[0] - a[-1] for a, b in zip(gs, gs[1:])]
        rows.append((t0, cover, float(np.median(pauses)), max(g[-1] - g[0] for g in gs), len(gs)))
    return rows


def match_like(r):  # tuned on a fixed-camera doubles recording: warmup cover ~0.5-0.9, match ~0.1-0.3
    t0, cover, med_pause, longest, n = r
    return n >= 4 and cover < 0.40 and med_pause >= 9 and longest < 60


def rhythm_start(rows, groups):
    ok = [match_like(r) for r in rows]
    for i in range(len(ok)):
        if all(ok[i:i + 6]) and len(ok[i:i + 6]) == 6:  # match-like for >= 3 consecutive minutes of window starts
            t0 = rows[i][0]
            # first rally start at/after t0 that is preceded by a pause >= 9 s
            for a, b in zip(groups, groups[1:]):
                if b[0] >= t0 and b[0] - a[-1] >= 9:
                    return b[0], i
            return t0, i
    return None, None


SCORE = re.compile(r"\b(love|fifteen|thirty|forty|deuce|ad(vantage)? (in|out)|game|set point|break point|\d+\s*-\s*\d+|15|30|40)\b", re.I)


def first_callouts(transcript_path):
    """(start_s, text) of whisper transcript segments that look like score callouts."""
    segs = json.load(open(transcript_path))["segments"]
    found = []
    for s in segs:
        txt = s["text"]
        if SCORE.findall(txt) and not re.search(r"(minutes?|years?|euros?|dollars?|times)", txt, re.I):
            found.append((s["start"], txt.strip()))
    return found


def estimate_match_start(src, transcript=None, gap=6.0, win=300.0):
    sr, x = load_audio(src)
    dur = len(x) / sr
    le, bg, _ = loudness(sr, x)
    t, _ = detect_hits(le, bg)
    groups = time_groups(t, gap)
    rows = rhythm(groups, dur, win)
    rs, _ = rhythm_start(rows, groups)
    res = {"duration_s": dur, "n_hits": int(len(t)), "n_rallies": len(groups), "rhythm_start_s": rs}
    res["rhythm_table"] = [dict(t=float(r[0]), cover=round(r[1], 2), med_pause=round(r[2], 1), longest=round(r[3], 1),
                                n=r[4], match_like=match_like(r)) for r in rows[:40]]
    if transcript and os.path.exists(transcript):
        c = first_callouts(transcript)
        res["callouts_first10"] = c[:10]
        # first callout cluster: >= 3 score callouts within 5 min
        cs = [c_[0] for c_ in c]
        res["callout_start_s"] = next((v for i, v in enumerate(cs) if len([w for w in cs[i:] if w - v < 300]) >= 3), None)
    est = [v for v in (res.get("rhythm_start_s"), res.get("callout_start_s")) if v is not None]
    if est:
        # callouts come just BEFORE the serve of a point; take the first rally start at or after (min estimate - 15 s)
        lo = min(est) - 15
        res["estimate_s"] = next((g[0] for g in groups if g[0] >= lo), None)
        res["spread_s"] = (max(est) - min(est)) if len(est) > 1 else None
        res["confidence"] = "high" if len(est) > 1 and res["spread_s"] < 90 else (
            "medium" if len(est) > 1 and res["spread_s"] < 240 else "low")
    res["_groups"] = groups
    return res


def contact_sheet(src, groups, estimate, out, font=None):
    """6x4 tile of frames at rally starts within +-4 min of the estimate, labelled with their time."""
    font = font or default_font()
    starts = [g[0] for g in groups if estimate - 240 <= g[0] <= estimate + 240][:24]
    tmpd = tempfile.mkdtemp()
    for i, s in enumerate(starts):
        subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-ss", f"{max(0, s - 0.3):.2f}", "-i", src, "-frames:v", "1", "-vf",
                        f"scale=400:-2,drawtext={fontfile_opt(font)}text='{int(s // 60)}m{int(s % 60):02d}s':x=4:y=4:fontsize=18:fontcolor=yellow:box=1:boxcolor=black",
                        f"{tmpd}/{i:02d}.jpg"])
    subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-i", f"{tmpd}/%02d.jpg", "-vf", "tile=6x4", "-frames:v", "1", "-q:v", "4", out])
