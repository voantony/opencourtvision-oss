"""One-click "Highlights only" mode: one video in -> a styled reel (9:16 + 16:9) and the best individual clips out.

No score engine, no point winner, no player ID, no ball tracking. Works the same for matches, coaching sessions and
drills (no match structure needed). Cheapest cues first:

  1. audio (16 kHz mono, one demux pass): hit transients -> rallies (hit runs with short gaps); long drill runs are
     split at their largest internal pauses so a 3-minute basket drill becomes several ranked windows.
  2. rank from audio + rules only (Legal v3.9: no vision rank features): rally length (hits, seconds), intensity
     (hits per second, hit loudness) and the crowd/player reaction after the last hit.
  3. motion (optional, full build only, flags.live_gate): a 128x72 / 3 fps decode of only the top candidates, used as a
     keep/drop live-play gate (an idle court drops a clip; it never moves a boundary or changes the order) and, for
     --focus, to place the 9:16 crop on the near/far player's side of the court.
  4. camera (social): Grand Slam broadcast grammar in reel/camera.py (broadcast_plan): 16:9 locked wide during the
     rally, a tight reaction crop after the last hit, a 2.5x slow-mo replay of the last shot only on starred points
     (R-SLOW), back to wide for the next serve; 9:16 follows the hitter under the existing pan/jerk caps.
  5. styled export with the existing style engine (style/plan.py + style/render.py, Media Director's vamos_clean_v1).

Presets:
  social      9:16 + 16:9 reel (<= 60 s, chronological, whole rallies) + individual clips in both aspects.
  recruiting  one 16:9 cut of 3-5 min for college coaches: title card (name, class year, UTR/ranking, contact),
              sections SERVES / NET PLAY / RALLIES-or-POINTS, a name lower-third, court audio (no music) or --mute,
              no grade (true colour). Stroke types come from audio only: serves (first hit after a reset) and net
              play (runs of fast exchanges). Forehand/backhand would need pose, which is both not cheap and, as a
              tracking-derived type filter, off in every build (flags.typed_filters_tracking), so it is not used.
"""
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..detection.audio import detect_hits, extract_wav, loudness, read_wav
from ..ffmpeg import ffmpeg_bin, ffprobe_bin, is_macos

SPORT = {  # max gap between hits inside one rally (s); padel rallies have wall bounces and slower lobs
    "tennis": dict(gap=4.0, th=14.0), "padel": dict(gap=4.5, th=11.0), "pickleball": dict(gap=3.5, th=14.0)}
# padel th: softer hits (no strings ping, glass/turf court) measured on CC padel clips: 14 dB missed most rallies
HL = dict(min_hits=3, split_over_s=35.0, piece_s=18.0, pre=3.5, post=2.5, reel_s=75.0, reel_clips=8, social_max_rally_s=12.0, n_clips=10,
          motion_topk=14, motion_fps=3, motion_w=128, motion_h=72, idle_ratio=0.30, serve_reset_s=6.0,
          fast_gap_s=0.95, recruit_min_s=180.0, recruit_target_s=240.0, recruit_max_s=300.0)


# ---------------------------------------------------------------- probe
def probe(video):
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,channels:format=duration",
                          "-of", "json", video], capture_output=True, text=True).stdout
    meta = json.loads(out or "{}")
    v = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = 30.0
    try:
        num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
        fps = float(num) / float(den)
    except Exception:
        pass
    return dict(width=int(v.get("width") or 1920), height=int(v.get("height") or 1080), fps=fps,
                duration=float((meta.get("format") or {}).get("duration") or 0.0), has_audio=a is not None,
                channels=int((a or {}).get("channels") or 2))


# ---------------------------------------------------------------- 1. audio -> rallies
def group_hits(t, amp, gap):
    """Runs of hits with gaps <= gap -> dicts(t0, t1, hits, times, amps, pause_before)."""
    out = []
    for i in range(len(t)):
        if out and t[i] - out[-1]["times"][-1] <= gap:
            out[-1]["times"].append(float(t[i])); out[-1]["amps"].append(float(amp[i]))
        else:
            out.append(dict(times=[float(t[i])], amps=[float(amp[i])]))
    prev = None
    for r in out:
        r["t0"], r["t1"], r["hits"] = r["times"][0], r["times"][-1], len(r["times"])
        r["pause_before"] = r["t0"] - prev if prev is not None else r["t0"]
        prev = r["t1"]
    return out


def split_long(r, max_s=HL["split_over_s"], piece_s=HL["piece_s"]):
    """A run longer than max_s (a drill, a basket feed, continuous hitting) is split at its largest internal pauses
    until every piece is <= piece_s. Real rallies (< max_s) are never split: clips stay whole rallies."""
    if r["t1"] - r["t0"] <= max_s:
        return [r]
    T, A = r["times"], r["amps"]
    pieces, stack = [], [(0, len(T) - 1)]
    while stack:
        i, j = stack.pop()
        if T[j] - T[i] <= piece_s or j - i < 2:
            pieces.append((i, j)); continue
        gaps = np.diff(T[i:j + 1])
        k = i + int(np.argmax(gaps))                      # split after hit k
        stack += [(i, k), (k + 1, j)]
    out = []
    for i, j in sorted(pieces):
        out.append(dict(times=T[i:j + 1], amps=A[i:j + 1], t0=T[i], t1=T[j], hits=j - i + 1,
                        pause_before=(T[i] - T[i - 1]) if i else r["pause_before"], split_from_drill=True))
    return out


def rallies_from_audio(le, bg, gap, th=14.0):
    t, amp = detect_hits(le, bg, th=th)
    R = []
    for r in group_hits(t, amp, gap):
        R += split_long(r)
    return R, len(t)


def reaction_db(lb, r):
    win = lb[int((r["t1"] + .4) * 100):int((r["t1"] + 3) * 100)]
    pre = lb[max(0, int(r["t0"] * 100) - 300):int(r["t0"] * 100)]
    return float(np.percentile(win, 90) - np.median(pre)) if len(win) and len(pre) else 0.0


# ---------------------------------------------------------------- 2. rank (audio + rules only)
def _z(x):
    x = np.asarray(x, float)
    if len(x) < 2: return np.zeros_like(x)
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return np.clip((x - med) / mad, -3, 3)


def rank(R, lb, min_hits=HL["min_hits"]):
    """Adds dur, pace, loud, react, rank_score; returns rallies with >= min_hits, best first."""
    C = [r for r in R if r["hits"] >= min_hits]
    if not C: return []
    for r in C:
        r["dur"] = r["t1"] - r["t0"]
        r["pace"] = (r["hits"] - 1) / max(1.0, r["dur"])
        r["loud"] = float(np.median(r["amps"]))
        r["react"] = reaction_db(lb, r)
        g = np.diff(r["times"])
        r["fast_run"] = int(_longest_run(g < HL["fast_gap_s"]))
    zh, zd = _z([r["hits"] for r in C]), _z([r["dur"] for r in C])
    zp, zl = _z([r["pace"] for r in C]), _z([r["loud"] for r in C])
    zr = _z([max(0.0, r["react"]) for r in C])
    for i, r in enumerate(C):
        r["rank_score"] = round(float(1.0 * zh[i] + 0.6 * zd[i] + 0.5 * zp[i] + 0.4 * zl[i] + 0.5 * zr[i]), 3)
        r["why"] = max((("long rally", zh[i] + zd[i]), ("high pace", zp[i]), ("big hits", zl[i]), ("reaction", zr[i])),
                       key=lambda kv: kv[1])[0]
    return sorted(C, key=lambda r: -r["rank_score"])


def _longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0; best = max(best, cur)
    return best


# ---------------------------------------------------------------- 3. motion (low-res, top-K windows only)
def keyframe_interval(video, at=None, span=12.0):
    """Mean keyframe spacing (s) over a short window (one cheap packet scan of `span` seconds)."""
    iv = f"{at:.1f}%+{span:.0f}" if at else f"%+{span:.0f}"
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-read_intervals", iv, "-select_streams", "v", "-show_entries",
                          "packet=flags", "-of", "csv=p=0", video], capture_output=True, text=True).stdout
    k = sum(1 for line in out.splitlines() if "K" in line)
    return span / k if k else 99.0


def _decode_gray(video, a, b, fps=HL["motion_fps"], w=HL["motion_w"], h=HL["motion_h"], keyframes_only=False):
    """-> (frames (n, h, w) float32, sample times). keyframes_only: decode only I-frames (a 4K HEVC phone/GoPro file
    with ~1 s GOP: ~1 Hz samples at ~3x less decode work than a full decode)."""
    nice = ["nice", "-n", "20", ffmpeg_bin(), "-nostdin", "-v", "error"]
    seek = ["-ss", f"{max(0, a):.3f}", "-t", f"{b - a:.3f}"]
    if keyframes_only:
        cmd = [*nice, "-skip_frame", "nokey", *seek, "-i", video, "-an", "-vf", f"scale={w}:{h},format=gray",
               "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
    else:
        raw = b""
        if is_macos():
            raw = subprocess.run([*nice, "-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld", *seek, "-i", video,
                                  "-an", "-vf", f"fps={fps},scale_vt=w={w}:h={h},hwdownload,format=nv12,format=gray",
                                  "-f", "rawvideo", "-"], capture_output=True).stdout
        if not raw:                                        # no VideoToolbox path for this file: CPU decode
            raw = subprocess.run([*nice, *seek, "-i", video, "-an", "-vf", f"fps={fps},scale={w}:{h},format=gray",
                                  "-f", "rawvideo", "-"], capture_output=True).stdout
    n = len(raw) // (w * h)
    F = np.frombuffer(raw[:n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)
    ts = a + (np.arange(n) + 0.5) * (b - a) / max(1, n)
    return F, ts


def motion_stats(F, ts=None):
    """Per window: activity (75th pct of the moving-cell share), motion centroid x (0-1) of the near (bottom) and far
    (top) halves, and per-sample energy maps + hot-spot x for the broadcast camera (reel/camera.py). Regional frame
    differences only; no person detection or tracking."""
    from ..reel.camera import hot_spot
    if len(F) < 2: return dict(activity=None)
    D = np.abs(np.diff(F, axis=0))
    thr = max(6.0, float(np.percentile(D, 90)))
    mov = (D > thr)
    act = float(np.percentile(mov.reshape(len(D), -1).mean(1), 75))
    h, w = F.shape[1:]
    xs = (np.arange(w) + 0.5) / w
    E = D * mov
    def cx(rows):
        e = E[:, rows, :].sum(axis=(0, 1))
        return float((e * xs).sum() / e.sum()) if e.sum() > 0 else 0.5
    ts = (np.asarray(ts, float)[1:] if ts is not None and len(ts) == len(F) else np.arange(1, len(F), dtype=float))
    # hot spot per sample over a 3-sample window (smoother than one diff)
    hx = []
    for i in range(len(E)):
        sp = hot_spot(E[max(0, i - 1):i + 2])
        hx.append(sp[0] if sp else (hx[-1] if hx else 0.5))
    return dict(activity=act, cx_near=cx(slice(h // 2, h)), cx_far=cx(slice(int(h * 0.15), h // 2)), cx_all=cx(slice(0, h)),
                t=ts, x=np.array(hx), maps=E.astype(np.float16))


def motion_pass(video, cands, topk=HL["motion_topk"], workers=3, duration=None, meta=None):
    todo = cands[:topk]
    heavy = bool(meta) and meta["width"] * meta["height"] * meta["fps"] > 1920 * 1080 * 60
    kf = heavy and keyframe_interval(video, at=todo[0]["t0"] if todo else None) <= 1.2
    def one(r):
        a, b = _bounds(r, duration)
        return motion_stats(*_decode_gray(video, a, b, keyframes_only=kf))
    with ThreadPoolExecutor(workers) as ex:
        stats = list(ex.map(one, todo))
    for r, s in zip(todo, stats):
        r["motion"] = s
    return len(todo)


def live_gate(cands, ratio=HL["idle_ratio"]):
    """Drop a candidate whose court activity is < ratio x the median of the checked ones (ball pick-up noise, chatter,
    a racket bag). Unchecked candidates and windows with no decision are kept (no decision never drops a clip)."""
    acts = [r["motion"]["activity"] for r in cands if r.get("motion", {}).get("activity") is not None]
    if len(acts) < 4: return cands, 0
    med = float(np.median(acts)); keep, drop = [], 0
    for r in cands:
        a = r.get("motion", {}).get("activity")
        if a is not None and a < ratio * med: r["dropped"] = "idle court"; drop += 1
        else: keep.append(r)
    return keep, drop


def focus_cx(r, focus):
    if focus in (None, "", "center", "centre"): return 0.5
    try:
        return float(np.clip(float(focus), 0.0, 1.0))
    except ValueError:
        pass
    m = r.get("motion") or {}
    key = {"near": "cx_near", "far": "cx_far", "auto": "cx_all"}.get(focus)
    if not key or m.get(key) is None: return 0.5
    return float(np.clip(0.5 + 1.0 * (m[key] - 0.5), 0.0, 1.0))


# ---------------------------------------------------------------- selection helpers
def _bounds(r, duration=None, pre=HL["pre"], post=HL["post"]):
    a = max(0.0, r["t0"] - pre); b = r["t1"] + post
    if duration: b = min(b, duration)
    return a, b


def pick(cands, n, max_s, used=(), duration=None):
    out, tot = [], 0.0
    used = list(used)
    for r in cands:
        if len(out) >= n: break
        a, b = _bounds(r, duration)
        if any(a < ub and ua < b for ua, ub in used): continue
        if tot + (b - a) > max_s: continue
        out.append(r); used.append((a, b)); tot += b - a
    return out


def to_points(sel, duration=None):
    pts = []
    for i, r in enumerate(sel):
        a, b = _bounds(r, duration)
        pts.append(dict(id=i + 1, kind="point", A=a, B=b, first_hit=r["t0"], last_hit=r["t1"], hits=r["hits"],
                        react=r["react"], favorite=False, rank_score=r["rank_score"], sources=["audio"],
                        cx_frac=r.get("cx_frac", 0.5), hits_t=list(r["times"]), motion=r.get("motion")))
    return pts


# ---------------------------------------------------------------- 4a. social preset
def render_social(video, meta, sel, out, work, preset_name, title, sport, aspects, timer, camera="broadcast", star=(),
                  long_shot="cut", score=True):
    from ..style.plan import plan_reel
    from ..style.preset import load as load_preset
    from ..style.render import render_reel
    preset = load_preset(preset_name)
    pts = to_points(sel, meta["duration"])
    cx = {p["id"]: p["cx_frac"] for p in pts}
    match = dict(sport=sport, format=None, format_name="highlights", teams=None, final=None, date_label=title)
    fps = int(min(30, max(24, round(meta["fps"]))))

    def one(asp):
        t0 = time.time()
        plan = plan_reel(pts, preset, aspect=asp, match=match, max_points=HL["reel_clips"], max_s=HL["reel_s"],
                         keep_pct=0, show_score=score)
        for c in plan["clips"]: c["cx_frac"] = cx.get(c["id"], 0.5)
        if camera == "broadcast":
            from ..reel.camera import CAMERA, broadcast_check, broadcast_plan
            ccfg = dict(CAMERA, long_shot_mode=long_shot)
            by = {p["id"]: p for p in pts}
            for c in plan["clips"]:
                p = by[c["id"]]
                m = p.get("motion")
                mo = dict(t=m["t"], x=m["x"], maps=m["maps"].astype(np.float32)) if m and m.get("t") is not None else None
                c["starred"] = p["id"] in star
                c["framing"] = broadcast_plan(dict(c, hits_t=p["hits_t"]), mo, asp, meta["width"], meta["height"], fps,
                                              starred=c["starred"], camera_cfg=ccfg)
            broadcast_check(plan["clips"])
        tag = asp.replace(":", "x")
        path = os.path.join(out, f"reel_{tag}.mp4")
        w = os.path.join(work, f"reel_{tag}")
        r = render_reel(video, plan, preset, path, work=w, SW=meta["width"], SH=meta["height"], fps=fps)
        # individual clips = the reel's own per-clip renders (graded, reframed), remuxed; no extra decode
        cdir = os.path.join(out, "clips", tag); os.makedirs(cdir, exist_ok=True)
        by_rank = sorted(plan["clips"], key=lambda c: -c["score"])
        clips = []
        for k, c in enumerate(by_rank):
            src = os.path.join(w, f"seg_{c['id']}.mov")
            dst = os.path.join(cdir, f"clip_{k + 1:02d}_{int(c['a']) // 60:02d}m{int(c['a']) % 60:02d}s.mp4")
            subprocess.run([ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", src, "-c", "copy", "-movflags",
                            "+faststart", dst], check=True)
            clips.append(dst)
        res = dict(reel=path, duration_s=_dur(path) or r["duration_s"], n_clips=len(plan["clips"]), clips=clips, camera=camera,
                        replays=sum(1 for c in plan["clips"] for f in c.get("framing", ()) if f["kind"] == "replay"),
                   long_shot_mode=long_shot if asp == "9:16" else None,
                   cuts=sum(len(f.get("long_shot", {}).get("cuts", ())) for c in plan["clips"] for f in c.get("framing", ())),
                   long_shots=sum(f.get("long_shot", {}).get("long_shots", 0) for c in plan["clips"] for f in c.get("framing", ())))
        timer[f"render {asp}"] = round(time.time() - t0, 1)
        return res

    with ThreadPoolExecutor(len(aspects)) as ex:
        out_ = list(ex.map(one, aspects))
    return {a: r for a, r in zip(aspects, out_)}


def _dur(path):
    try:
        return round(float(subprocess.run([ffprobe_bin(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                          capture_output=True, text=True).stdout), 2)
    except ValueError:
        return None


# ---------------------------------------------------------------- 4b. recruiting preset
def recruiting_sections(cands, all_rallies, duration=None):
    """-> list of (title, [rally]) for a 3-5 min coach cut. Serves and net play are audio-typed only."""
    used, secs, total = [], [], 0.0
    match_like = sum(r["pause_before"] >= 12 for r in cands) >= 0.5 * max(1, len(cands))
    serves = [r for r in all_rallies if r["pause_before"] >= HL["serve_reset_s"] and r["hits"] >= 2]
    serves = sorted(serves, key=lambda r: -r["amps"][0])
    sv = []
    for r in serves:
        if len(sv) >= 6: break
        a, b = max(0.0, r["t0"] - 2.0), r["t0"] + 2.5      # the serve and the return, not the whole rally
        if any(a < ub and ua < b for ua, ub in used): continue
        sv.append(dict(r, clip=(a, b))); used.append((a, b))
    if len(sv) >= 3:
        secs.append(("SERVES", sv)); total += sum(x["clip"][1] - x["clip"][0] for x in sv)
    net = sorted([r for r in cands if r["fast_run"] >= 3], key=lambda r: (-r["fast_run"], -r["rank_score"]))
    nsel = pick(net, 6, 60.0, used, duration)
    for r in nsel: used.append(_bounds(r, duration))
    if len(nsel) >= 2:
        secs.append(("NET PLAY", nsel)); total += sum(np.subtract(*_bounds(r, duration)[::-1]) for r in nsel)
    rest = pick(cands, 40, max(60.0, HL["recruit_target_s"] - total), used, duration)
    if rest:
        secs.append(("POINTS" if match_like else "RALLIES", rest))
    return secs


def _card_cmd(path, lines, dur, fps, ch, preset, size=(1920, 1080)):
    from ..style.preset import font
    pal = preset["palette"]; W, H = size
    night = pal["night"].lstrip("#"); chalk = pal["chalk"].lstrip("#"); mist = pal["mist"].lstrip("#")
    volt = pal["volt"].lstrip("#")
    fh, fn = font(preset, "headlines"), font(preset, "names")
    vf = []
    n = len(lines); y0 = H // 2 - (n * 70) // 2
    for i, (txt, big) in enumerate(lines):
        f = fh if big else fn
        fo = f"fontfile={f}:" if f else ""
        vf.append(f"drawtext={fo}text='{_esc(txt)}':fontsize={84 if big else 44}:fontcolor=0x{(chalk if big else mist).upper()}:"
                  f"x=(w-text_w)/2:y={y0 + i * 78 + (0 if big else 14)}")
    vf.append(f"drawbox=x=(w-160)/2:y={y0 + n * 78 + 20}:w=160:h=4:color=0x{volt.upper()}@1:t=fill")
    layout = "mono" if ch == 1 else "stereo"
    vcodec = ["-c:v", "h264_videotoolbox", "-b:v", "8M"] if is_macos() else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    return [ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{night}:s={W}x{H}:r={fps}:d={dur}",
            "-f", "lavfi", "-i", f"anullsrc=r=48000:cl={layout}", "-t", f"{dur}", "-vf", ",".join(vf), *vcodec,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", path]


def _esc(s):
    return (s or "").replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019").replace("%", "\\%")


def render_recruiting(video, meta, secs, out, work, preset_name, profile, mute, timer):
    from ..style.preset import font
    from ..style.preset import load as load_preset
    from ..style.render import render_clip
    t0 = time.time()
    preset = load_preset(preset_name, grade=False)          # clean: true colour for coaches
    fps = int(min(30, max(24, round(meta["fps"]))))
    w = os.path.join(work, "recruiting"); os.makedirs(w, exist_ok=True)
    ch = meta["channels"] if meta["has_audio"] else 2
    name = profile.get("name") or "Player"
    info = " · ".join(x for x in (f"Class of {profile['class_year']}" if profile.get("class_year") else "",
                                  profile.get("ranking") or "") if x)
    contact = profile.get("contact") or ""
    files, timeline, t_out = [], [], 0.0
    def card(tag, lines, dur):
        nonlocal t_out
        p = os.path.join(w, f"card_{tag}.mov")
        subprocess.run(_card_cmd(p, lines, dur, fps, ch, preset), check=True)
        files.append(p); timeline.append(("card", t_out, t_out + dur)); t_out += dur
    card("title", [(name.upper(), True)] + ([(info, False)] if info else []) + ([(contact, False)] if contact else []), 5.0)
    jobs = []
    for si, (title, rs) in enumerate(secs):
        jobs.append(("card", si, title))
        for k, r in enumerate(rs):
            a, b = r.get("clip") or _bounds(r, meta["duration"])
            jobs.append(("clip", f"{si}_{k}", dict(id=f"r{si}_{k}", a=round(a, 3), b=round(b, 3), parts=[(a, b, 1.0)])))
    # render clips in parallel (2 at a time; the encoder is hardware on macOS)
    clip_jobs = [j for j in jobs if j[0] == "clip"]
    with ThreadPoolExecutor(2) as ex:
        paths = dict(zip([j[1] for j in clip_jobs], ex.map(
            lambda j: render_clip(video, j[2], preset, "16:9", w, meta["width"], meta["height"], fps), clip_jobs)))
    for kind, key, val in jobs:
        if kind == "card":
            card(f"s{key}", [(val, True)], 1.6)
        else:
            d = val["b"] - val["a"]
            files.append(paths[key]); timeline.append(("clip", t_out, t_out + d)); t_out += d
    card("end", [(name.upper(), True)] + ([(info, False)] if info else []) + ([(contact, False)] if contact else []), 4.0)
    lst = os.path.join(w, "concat.txt")
    with open(lst, "w") as f:
        for p in files: f.write(f"file '{p}'\n")
    pal = preset["palette"]; fn = font(preset, "names")
    fo = f"fontfile={fn}:" if fn else ""
    lower = " · ".join(x for x in (name, info) if x)
    en = "+".join(f"between(t,{a:.2f},{b:.2f})" for k, a, b in timeline if k == "clip") or "0"
    vf = (f"drawtext={fo}text='{_esc(lower)}':fontsize=34:fontcolor=0x{pal['chalk'].lstrip('#').upper()}:"
          f"box=1:boxcolor=0x{pal['night'].lstrip('#').upper()}@0.70:boxborderw=14:x=48:y=h-96:enable='{en}'")
    af = "volume=0" if mute else "loudnorm=I=-16:TP=-1.0:LRA=11"
    path = os.path.join(out, "recruiting_16x9.mp4")
    vcodec = ["-c:v", "h264_videotoolbox", "-b:v", "10M"] if is_macos() else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    subprocess.run(["nice", "-n", "20", ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", vf, "-af", af, *vcodec, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", path], check=True)
    timer["render recruiting"] = round(time.time() - t0, 1)
    return dict(reel=path, duration_s=round(t_out, 1), muted=bool(mute),
                sections=[dict(title=t, n=len(rs)) for t, rs in secs], short=t_out < HL["recruit_min_s"])


# ---------------------------------------------------------------- entry point
SCORE_DEFAULT = dict(social=True, recruiting=False)   # score overlay per preset (shown only when confirmed / >= 0.9)


def run(video, out=None, sport="tennis", preset="social", style="vamos_clean_v1", focus=None, title=None,
        aspects=("9:16", "16:9"), motion=True, region=None, profile=None, mute=False, work=None, camera="broadcast",
        star=(), long_shot="cut", score=None):
    from ..editloop.build import flags
    if score is None: score = SCORE_DEFAULT.get(preset, False)          # per reel preset: social on, recruiting off
    t_all = time.time(); T = {}
    video = os.path.abspath(video)
    out = os.path.abspath(out or os.path.join(os.path.dirname(video), os.path.splitext(os.path.basename(video))[0] + "_highlights"))
    work = work or os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    meta = probe(video)
    if not meta["has_audio"]:
        raise RuntimeError("this video has no audio track; Highlights only finds rallies from the sound of the hits")
    t = time.time()
    wav = os.path.join(work, "audio_16k.wav")
    if not os.path.exists(wav): extract_wav(video, wav)
    sr, x = read_wav(wav)
    le, bg, lb = loudness(sr, x)
    T["audio"] = round(time.time() - t, 1)
    t = time.time()
    sp = SPORT.get(sport, SPORT["tennis"])
    R, n_hits = rallies_from_audio(le, bg, sp["gap"], sp["th"])
    # whole clips only (R-WHOLE): 3 s before the first hit and 2 s after the last must exist in the video
    R = [r for r in R if r["t0"] >= 3.0 and r["t1"] + 2.0 <= meta["duration"]]
    # a match (serve pattern) skips its warm-up; lessons / drills / practice keep everything (no match structure)
    from ..detection.audio import group_rallies
    from ..detection.match_start import warmup_cutoff
    from ..editloop.modes import suggest_mode
    t_, a_ = detect_hits(le, bg, th=sp["th"]); R6 = group_rallies(t_, a_)
    start, _ = warmup_cutoff(R6, None) if R6 else (0.0, None)
    kind = suggest_mode(R6, start, meta["duration"])["suggested"] if R6 else "rallies"
    if kind == "match": R = [r for r in R if r["t0"] >= start - 0.5]
    cands = rank(R, lb)
    T["rallies+rank"] = round(time.time() - t, 2)
    if not cands:
        raise RuntimeError("no rallies found (no runs of 3+ ball hits in the audio)")
    fl = flags(region)
    use_motion = bool(motion) and fl.variant != "us" and fl.live_gate
    n_checked = n_drop = 0
    want_cam = camera == "broadcast" and preset == "social" and fl.variant != "us"
    if use_motion or want_cam or (focus not in (None, "", "center", "centre")):
        t = time.time()
        n_checked = motion_pass(video, cands, duration=meta["duration"], meta=meta)
        if use_motion: cands, n_drop = live_gate(cands)
        T["motion (top-K, low-res)"] = round(time.time() - t, 1)
    for r in cands: r["cx_frac"] = focus_cx(r, focus)
    for r in cands: r.pop("maps", None)
    result = dict(schema="vamos.highlights.v1", footage=("match (warm-up skipped from %.0f s)" % start) if kind == "match" else "practice / drills / partial", camera=camera if preset == "social" else "wide", video=video, out=out, sport=sport, preset=preset, score_overlay=("on (confirmed or >= 90 % confidence only)" if score else "off"), focus=focus or "center",
                  duration_s=round(meta["duration"], 1), source=f"{meta['width']}x{meta['height']}@{meta['fps']:.0f}",
                  hits=n_hits, rallies=len(cands), motion_checked=n_checked, dropped_idle=n_drop,
                  ranked=[dict(start_s=round(r["t0"], 1), end_s=round(r["t1"], 1), hits=r["hits"], score=r["rank_score"],
                               why=r["why"], drill_piece=bool(r.get("split_from_drill"))) for r in cands[:20]])
    if preset == "social":
        good = [r for r in cands if r["rank_score"] > -2.0]
        good = good if len(good) >= 4 else cands
        short = [r for r in good if r["dur"] <= HL["social_max_rally_s"]]      # a reel moment, not a 40 s rally
        sel = pick(short if len(short) >= 4 else good, HL["n_clips"], 1e9, duration=meta["duration"])
        # star: 1-based ranks of the clips the user starred (R-SLOW: replays only on those, at most 2)
        result["exports"] = render_social(video, meta, sel, out, work, style, title, sport, aspects, T, camera=camera,
                                          star=set(int(k) for k in star or ()), long_shot=long_shot, score=score)
        poster = os.path.join(out, "poster_9x16.jpg")
        reel = result["exports"].get("9:16", next(iter(result["exports"].values())))["reel"]
        subprocess.run([ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-ss", "2.5", "-i", reel, "-frames:v", "1", "-q:v", "3",
                        poster], check=False)
        result["poster"] = poster
    elif preset == "recruiting":
        secs = recruiting_sections(cands, [r for r in R if r["hits"] >= 1], meta["duration"])
        result["exports"] = {"16:9": render_recruiting(video, meta, secs, out, work, style, profile or {}, mute, T)}
    else:
        raise ValueError("preset must be 'social' or 'recruiting'")
    total = time.time() - t_all
    T["total"] = round(total, 1)
    result["stage_times_s"] = T
    result["minutes_per_hour_of_footage"] = round(total / 60.0 / max(1e-6, meta["duration"] / 3600.0), 2)
    json.dump(result, open(os.path.join(out, "highlights.json"), "w"), indent=1, default=float)
    if os.environ.get("VAMOS_KEEP_WORK") != "1":
        shutil.rmtree(os.path.join(work, "recruiting"), ignore_errors=True)
    return result
