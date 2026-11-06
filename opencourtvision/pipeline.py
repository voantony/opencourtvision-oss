"""Full match pipeline (the `ocv-swingcut` command):
audio -> features -> hits/rallies -> warmup cutoff -> (STT callouts) -> points/games -> highlights
(-> story) -> score timeline + report (-> lossless render).
All stages cache in WORK; delete a cache file to recompute that stage."""
import json
import os
import time

import numpy as np

from .detection.audio import detect_hits, extract_wav, group_rallies, loudness, read_wav
from .detection.match_start import warmup_cutoff
from .ffmpeg import duration, keyframes
from .render.cuts import build_segments, render
from .scoring.callouts import DEFAULT_MODEL, found_callouts, score_timeline, transcribe_callouts, write_score_csv
from .scoring.games import assign_changeovers, parse_final
from .utils import StageTimer, hms


def add_arguments(ap):
    ap.add_argument("video", help="match video (fixed camera, with audio)")
    ap.add_argument("--final", help='final score, e.g. "6-4 6-3" (required in match mode)')
    ap.add_argument("--mode", choices=("match", "rallies"), default="match",
                    help="match: score engine + points; rallies: rallies-only (practice, drills, partial clips), no score keeping")
    ap.add_argument("--reel-order", choices=("length", "rank"), default="length", help="rallies mode: reel order (no won/lost weighting)")
    ap.add_argument("--work", help="cache/output dir (default: <video dir>/../work)")
    ap.add_argument("--teams", nargs=2, default=["Team A", "Team B"], help="team labels (reserved for reports)")
    ap.add_argument("--stt", action="store_true", help="transcribe score callouts with mlx-whisper (Apple Silicon)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="mlx-whisper model dir or HF repo (env OCV_WHISPER_MODEL)")
    ap.add_argument("--render", metavar="OUTDIR", help="write lossless points-only and highlights videos here")
    ap.add_argument("--name", default="Match", help="output file name prefix")
    ap.add_argument("--start", type=float, help="override match start (s)")
    ap.add_argument("--lead", type=float, default=4.0, help="lead-in before first hit (s), snapped back to keyframe")
    ap.add_argument("--tail", type=float, default=2.5, help="tail after last hit (s; never below 2.0, capped at 6.0), snapped forward to keyframe")
    ap.add_argument("--merge_gap", type=float, default=1.0, help="merge adjacent points only if padded gap < this (s)")
    ap.add_argument("--story", action="store_true", help="rank points into story beats (work/story.json) and use them for highlights")
    ap.add_argument("--hl", type=int, default=12, help="max number of highlight points")
    ap.add_argument("--hl_max", type=float, default=240, help="max highlights duration (s)")
    ap.add_argument("--segmenter", choices=("smart", "e1", "legacy"), default="smart",
                    help="e1: audio-only point segmentation (+E3 gap fill), padding never merges points (eng notes §2.8.7); legacy: old merge rule")
    ap.add_argument("--region", choices=("US", "UK", "ES", "ROW"), help="region profile (default env OCV_BUILD_REGION or ROW; a US-variant build is always US)")
    ap.add_argument("--motion", help="motion-energy npz: the live-play keep/drop gate (US smart segmenter, on by default) and "
                    "the vision rally-cue fallback (OCV_VISION_RALLY_CUES=1)")
    return ap


def checks(segs, games):
    ng = games["n_games"]
    real = [s for s in segs if not s["fault_like"]]
    return {"segments": len(segs), "fault_like": len(segs) - len(real), "point_like": len(real),
            "expected_points_range": (4 * ng, 10 * ng), "count_ok": 4 * ng <= len(real) <= 10 * ng}


def pick_highlights(segs, n_hl):
    ok = [i for i, s in enumerate(segs) if s["B"] - s["A"] <= 40 and not s["fault_like"] and not s["serve_only"]]
    hl = sorted(ok, key=lambda i: -segs[i]["hits"])[:n_hl * 2 // 3]
    why = {i: f"long rally ({segs[i]['hits']} hits)" for i in hl}
    for i in sorted(ok, key=lambda i: -segs[i]["react"]):
        if len(hl) >= n_hl:
            break
        if i not in hl and segs[i]["hits"] >= 3:
            hl.append(i)
            why[i] = f"loud reaction (+{segs[i]['react']:.0f} dB)"
    hl.sort()
    return hl, why


def story_highlights(segs, st, n_hl):
    ranked = [p["point"] - 1 for p in st["top_points"] if (segs[p["point"] - 1]["B"] - segs[p["point"] - 1]["A"]) <= 40]
    ds = [d["point"] - 1 for d in st["difficult_shots"]]
    hl, why = [], {}
    for b in st["beats"]:
        i = b["point"] - 1
        if i not in hl:
            hl.append(i)
            why[i] = f"story: {b['beat']}" + (f" ({b['tag']})" if b["tag"] else "")
    for i in ds[:3]:
        if i not in hl:
            hl.append(i)
            why[i] = "difficult shot"
    for i in ranked:
        if len(hl) >= n_hl:
            break
        if i not in hl:
            hl.append(i)
            why[i] = "high excitement"
    hl.sort()
    return hl, why


def run(A):
    W = A.work or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(A.video))), "work")
    os.makedirs(W, exist_ok=True)
    P = lambda f: os.path.join(W, f)  # noqa: E731
    T = StageTimer()
    mode = getattr(A, "mode", "match") or "match"
    if mode == "match" and not A.final:
        raise SystemExit("--final is required in match mode (or use --mode rallies)")
    sets = parse_final(A.final) if mode == "match" else None

    with T("1 audio"):
        w = P("audio_16k.wav")
        if not os.path.exists(w):
            extract_wav(A.video, w)
        sr, x = read_wav(w)
    with T("2 features"):
        f = P("features.npz")
        if os.path.exists(f):
            z = np.load(f)
            le, bg, lb = z["le"], z["bg"], z["lb"]
        else:
            le, bg, lb = loudness(sr, x)
            np.savez(f, le=le, bg=bg, lb=lb)
    with T("3 hits+rallies"):
        t, amp = detect_hits(le, bg)
        R = group_rallies(t, amp)
        json.dump(R, open(P("rallies.json"), "w"))
    with T("4 warmup cutoff"):
        start, ev = warmup_cutoff(R, A.start)
    from .editloop.modes import suggest_mode
    suggestion = suggest_mode(R, start, duration(A.video))
    if suggestion["suggested"] != mode:
        print(f"[mode] {mode} selected; suggested: {suggestion['suggested']} ({'; '.join(suggestion['reasons']) or 'consistent match'})")
    if mode == "rallies":
        return run_rallies(A, W, P, T, le, bg, lb, start, suggestion)
    with T("5 callouts (STT)"):
        C = transcribe_callouts(R, start, w, W, A.model, run_stt=A.stt)
    with T("6 points+games"):
        KF = keyframes(A.video, P("keyframes.txt"))
        if getattr(A, "segmenter", "legacy") in ("e1", "smart"):
            from .editloop.build import flags as edit_flags
            from .editloop.segment import segment_match
            fl = edit_flags(getattr(A, "region", None))
            cues = None
            smart_seg = getattr(A, "segmenter", "e1") == "smart"
            if getattr(A, "motion", None) and (fl.vision_rally_cues or (smart_seg and fl.live_gate and not fl.motion_boundaries)):
                from .editloop.motion import MotionCues
                cues = MotionCues.load(A.motion)          # US smart: keep/drop gate only (live_gate, on by default)
            segs = segment_match(le, bg, start, sets, fl, KF, duration(A.video), cues, A.lead, lb=lb,
                                 smart=getattr(A, "segmenter", "e1") == "smart")
        else:
            segs = build_segments(R, start, KF, duration(A.video), A.lead, A.tail, A.merge_gap)
        games = assign_changeovers(segs, sets, hms)

    # reaction loudness after the last hit, relative to the 3 s before the first hit
    for s in segs:
        last = s["rallies"][-1]["t1"]
        first = s["rallies"][0]["t0"]
        win = lb[int((last + .4) * 100):int((last + 3) * 100)]
        pre = lb[max(0, int(first * 100) - 300):int(first * 100)]
        s["react"] = float(np.percentile(win, 90) - np.median(pre)) if len(win) and len(pre) else 0

    hl, why = pick_highlights(segs, A.hl)
    if A.story:
        from .story import build
        t0 = time.time()
        co_idx = set(c[0] - 1 for c in games["changeover_gaps"])
        st = build(segs, lb, start, co_idx)
        json.dump(st, open(P("story.json"), "w"), indent=1, default=float)
        hl, why = story_highlights(segs, st, A.hl)
        T.times["6b story"] = round(time.time() - t0, 3)
    while sum(segs[i]["B"] - segs[i]["A"] for i in hl) > A.hl_max and len(hl) > 1:
        hl.remove(min((i for i in hl if not why[i].startswith("story: close") and not why[i].startswith("story: peak")),
                      key=lambda i: (segs[i]["hits"], segs[i]["react"])))

    co = set(c[0] - 1 for c in games["changeover_gaps"])
    with open(P("segments.csv"), "w") as fh:
        fh.write("idx,start_s,end_s,dur_s,start_hms,end_hms,first_hit_s,lead_in_s,hits,rallies,fault_like,serve_only,gap_after_s,changeover_after,react_db,highlight\n")
        for i, s in enumerate(segs):
            fh.write(f"{i + 1},{s['A']:.6f},{s['B']:.6f},{s['B'] - s['A']:.3f},{hms(s['A'])},{hms(s['B'])},{s['rallies'][0]['t0']:.2f},"
                     f"{s['rallies'][0]['t0'] - s['A']:.2f},{s['hits']},{len(s['rallies'])},{int(s['fault_like'])},{int(s['serve_only'])},"
                     f"{s['gap_after']:.1f},{int(i in co)},{s['react']:.1f},{why.get(i, '') if i in hl else ''}\n")

    chk = checks(segs, games)
    tl = score_timeline(segs, C, start)
    write_score_csv(tl, P("score_timeline.csv"))
    chk["callouts_matched"] = sum(r["conf"] == "callout" for r in tl)
    chk["discrepancies"] = [(r["point"], hms(r["start"]), r["flag"]) for r in tl if r["flag"]]
    ends = [segs[c[0] - 1]["B"] for c in games["changeover_gaps"]]
    spans = [b - a for a, b in zip([start] + ends, ends)]
    chk["changeover_spans_s"] = [round(v) for v in spans]
    chk["changeover_pattern_ok"] = len(ends) == len(games["changeover_after_games"]) and all(60 < v < 720 for v in spans)
    chk["scoreboard_ok"] = chk["callouts_matched"] >= 0.6 * len(tl) and not chk["discrepancies"]
    cf = found_callouts(C, start)
    rep = dict(mode="match", mode_suggestion=suggestion, match_start_s=start, match_start_hms=hms(start), warmup_evidence=ev, games=games, checks=chk, n_callouts=len(cf),
               callouts=cf[:200], highlights=[(i + 1, hms(segs[i]["A"]), round(segs[i]["B"] - segs[i]["A"], 1), why[i]) for i in hl],
               cut_total_s=round(sum(s["B"] - s["A"] for s in segs), 1), stage_times_s=T.times)
    json.dump(rep, open(P("swingcut_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "callouts"}, indent=1, default=str))
    if A.render:
        T.times["7 render (lossless)"] = render(A.video, segs, hl, W, A.render, A.name)
        print("[7 render]", T.times["7 render (lossless)"], "s")
    return rep


def run_rallies(A, W, P, T, le, bg, lb, start, suggestion):
    """Rallies-only mode: smart rally boundaries + dead-space trim, rally list + reel; no score engine, no point winner,
    no game/set state, no score prompts or taps. US build: audio only (motion cues ignored)."""
    from .editloop.build import flags as edit_flags
    from .editloop.modes import rallies_only, rally_list
    fl = edit_flags(getattr(A, "region", None))
    cues = None
    if fl.variant != "us" and fl.vision_rally_cues and getattr(A, "motion", None):
        from .editloop.motion import MotionCues
        cues = MotionCues.load(A.motion)
    with T("6 rallies"):
        KF = keyframes(A.video, P("keyframes.txt"))
        segs = rallies_only(le, bg, lb, start, fl, KF, duration(A.video), cues)
    doc = rally_list(segs, os.path.basename(A.video), fl.name, getattr(A, "reel_order", "length"), suggestion)
    doc.update(start_s=start, stage_times_s=T.times)
    json.dump(doc, open(P("rally_list.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in doc.items() if k != "rallies"}, indent=1, default=float))
    if A.render:
        order = [i - 1 for i in doc["reel"]]
        T.times["7 render (lossless)"] = render(A.video, [segs[i] for i in order], [], W, A.render, A.name + " - rallies")
    return doc
