"""Execute a reel plan: per clip reframe -> grade (optional) -> overlays, then hard-cut concat.

Camera / framing stays in reel/camera.py (and reel/ball.py); this module only crops with that code, then applies the
preset grade and overlays. Music is never baked: the export is court audio only; the platform sound picker is used at
share time (preset audio.prefer_platform_sound_at_share)."""
import json
import os
import subprocess
import tempfile

import numpy as np

from ..ffmpeg import ffmpeg_bin, is_macos
from .plan import PlanError, validate
from .preset import asset, font, grade_filter

OUT = {"9:16": (1080, 1920), "16:9": (1920, 1080)}


def _atempo(speed):
    """ffmpeg atempo accepts 0.5-100 per filter; chain for speeds in [0.45, 0.5)."""
    speed = float(speed)
    if abs(speed - 1.0) < 1e-6:
        return "anull"
    parts, s = [], speed
    while s < 0.5 - 1e-9:
        parts.append("atempo=0.5"); s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)



def _escape_drawtext(s):
    return (s or "").replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")


def _score_call(st):
    """Compact score string for the bug: finished sets Mist-style via plain text, live pts bold."""
    if st is None: return ""
    sets = f"{st.sets[0]}-{st.sets[1]}"
    games = f"{st.games[0]}-{st.games[1]}"
    pts = f"{st.pts[0]}-{st.pts[1]}"
    return f"{sets}  {games}  {pts}"


def _overlay_filters(plan, preset, ow, oh, aspect):
    """ffmpeg drawtext / drawbox chain for intro, captions, score bug, outro. Applied AFTER the grade."""
    pal = preset["palette"]
    chalk, mist, volt, night = pal["chalk"], pal["mist"], pal["volt"], pal["night"]
    fc = font(preset, "captions"); fn = font(preset, "names"); fh = font(preset, "headlines"); fnum = font(preset, "numbers_live")
    filters = []
    # intro band
    if plan.get("intro"):
        intro = plan["intro"]
        y0 = int(oh * 0.70) if aspect == "9:16" else int(oh * 0.72)
        filters.append(f"drawbox=x=0:y={y0}:w={ow}:h={oh - y0}:color=0x{night.lstrip('#').upper()}@0.80:t=fill:enable='between(t,{intro['start']:.3f},{intro['end']:.3f})'")
        for i, line in enumerate(intro["lines"][:3]):
            size = (28 if i < 2 else 40) if aspect == "9:16" else (24 if i < 2 else 36)
            col = mist if i < 2 else chalk
            ff = fn if i < 2 else fh
            y = y0 + 24 + i * (size + 10)
            ff_opt = f"fontfile={ff}:" if ff else ""
            filters.append(f"drawtext={ff_opt}text='{_escape_drawtext(line)}':fontsize={size}:fontcolor=0x{col.lstrip('#').upper()}:"
                           f"x=(w-text_w)/2:y={y}:enable='between(t,{intro['start']:.3f},{intro['end']:.3f})'")
    # stakes captions
    zone = preset["captions"]["zone"]["9x16" if aspect == "9:16" else "16x9"]
    cy = int(oh * (zone["y_frac_min"] + zone["y_frac_max"]) / 2)
    cx_bias = zone.get("center_x_bias_px", 0) if aspect == "9:16" else 0
    cap_size = preset["typography"]["sizes_px"]["1080x1920" if aspect == "9:16" else "1920x1080"]["caption"]
    for o in plan["captions"]:
        ff_opt = f"fontfile={fc}:" if fc else ""
        filters.append(f"drawtext={ff_opt}text='{_escape_drawtext(o['text'])}':fontsize={cap_size}:fontcolor=0x{chalk.lstrip('#').upper()}:"
                       f"x=(w-text_w)/2+{cx_bias}:y={cy}:enable='between(t,{o['start']:.3f},{o['end']:.3f})'")
        # Volt underline (approx: a thin box under the caption centre)
        uw = min(ow - 80, max(200, int(cap_size * len(o["text"]) * 0.45)))
        filters.append(f"drawbox=x=(w-{uw})/2+{cx_bias}:y={cy + cap_size + 6}:w={uw}:h=3:color=0x{volt.lstrip('#').upper()}@1:t=fill:"
                       f"enable='between(t,{o['start']:.3f},{o['end']:.3f})'")
    # score bug (story reel: only on beats where we planned one)
    bug_y = 320 if aspect == "9:16" else 54
    bug_h = int(oh * (0.054 if aspect == "9:16" else 0.065))
    bug_w = int(ow * (0.70 if aspect == "9:16" else 0.26))
    bug_x = (ow - bug_w) // 2 if aspect == "9:16" else 96
    for o in plan["bugs"]:
        call = _score_call(o.get("state"))
        if not call: continue
        filters.append(f"drawbox=x={bug_x}:y={bug_y}:w={bug_w}:h={bug_h}:color=0x{night.lstrip('#').upper()}@0.90:t=fill:"
                       f"enable='between(t,{o['start']:.3f},{o['end']:.3f})'")
        ff_opt = f"fontfile={fnum}:" if fnum else ""
        filters.append(f"drawtext={ff_opt}text='{_escape_drawtext(call)}':fontsize={26 if aspect == '9:16' else 20}:"
                       f"fontcolor=0x{chalk.lstrip('#').upper()}:x={bug_x + 16}:y={bug_y + bug_h // 2 - 10}:"
                       f"enable='between(t,{o['start']:.3f},{o['end']:.3f})'")
    # result / outro (hold last frame + card)
    if plan.get("outro"):
        o = plan["outro"]
        t0, t1 = o["start"], o["end"]
        filters.append(f"drawbox=x=0:y=0:w={ow}:h={oh}:color=0x{night.lstrip('#').upper()}@0.85:t=fill:enable='between(t,{t0:.3f},{t1:.3f})'")
        winner = o.get("winner") or (o.get("teams") or ["", ""])[0]
        score = o.get("score") or ""
        ff_opt = f"fontfile={fh}:" if fh else ""
        filters.append(f"drawtext={ff_opt}text='{_escape_drawtext(winner)}':fontsize={48 if aspect == '9:16' else 56}:"
                       f"fontcolor=0x{chalk.lstrip('#').upper()}:x=(w-text_w)/2:y=h*0.42:enable='between(t,{t0:.3f},{t1:.3f})'")
        ff_opt = f"fontfile={fnum}:" if fnum else ""
        filters.append(f"drawtext={ff_opt}text='{_escape_drawtext(score)}':fontsize={52 if aspect == '9:16' else 64}:"
                       f"fontcolor=0x{chalk.lstrip('#').upper()}:x=(w-text_w)/2:y=h*0.52:enable='between(t,{t0:.3f},{t1:.3f})'")
        logo = asset(preset, preset.get("assets", {}).get("logo_c", ""))
        if logo and os.path.exists(logo):
            # logo is overlaid in a second input pass; mark for the caller
            plan.setdefault("_logo", logo)
    return filters


def _reframe_filter(aspect, ow, oh, SW, SH, cx_frac=0.5):
    """True 9:16 tracked crop (no letterbox) or a centred 16:9 scale. Camera path is Engineer's; this is a simple
    centre-weighted crop used when no per-frame plan is available (v1 sample path)."""
    if aspect == "9:16":
        # full-height 9:16 window on the source, then scale to delivery — never letterbox
        cw = int(SH * 9 / 16) // 2 * 2
        x = max(0, min(SW - cw, int(SW * cx_frac - cw / 2)))
        return f"crop={cw}:{SH}:{x}:0,scale={ow}:{oh}:flags=lanczos,setsar=1"
    return f"scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},setsar=1"


def render_clip(src, clip, preset, aspect, work, SW=3840, SH=2160, fps=30, hwaccel=None, cx_frac=0.5):
    """Render one whole-point clip to work/seg_XX.mov: reframe -> grade -> (overlays applied on the concat timeline)."""
    ow, oh = OUT[aspect]
    a, b = clip["a"], clip["b"]
    reframe = _reframe_filter(aspect, ow, oh, SW, SH, cx_frac)
    grade = grade_filter(preset)
    # speed parts (slow-mo with cosine-sampled steps already in the plan)
    parts = clip["parts"]
    n = len(parts)
    fc = [f"[0:v]fps={fps},format=yuv420p,{reframe}{(',' + grade) if grade else ''},split={n}"  # fps first: drop frames before the 4K crop
          + "".join(f"[vs{i}]" for i in range(n)),
          f"[0:a]aresample=48000,asplit={n}" + "".join(f"[as{i}]" for i in range(n))]
    cat = ""
    for i, (t0, t1, sp) in enumerate(parts):
        # times are absolute source; trim relative to clip start
        r0, r1 = t0 - a, t1 - a
        if abs(sp - 1.0) < 1e-6:
            fc.append(f"[vs{i}]trim={r0:.4f}:{r1:.4f},setpts=PTS-STARTPTS[v{i}]")
            fc.append(f"[as{i}]atrim={r0:.4f}:{r1:.4f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            # never slower than 0.45x (validated upstream); optical-flow when fps high enough else blend
            mi = "mci" if fps >= 50 else "blend"
            fc.append(f"[vs{i}]trim={r0:.4f}:{r1:.4f},setpts=(PTS-STARTPTS)/{sp:.4f},minterpolate=fps={fps}:mi_mode={mi}[v{i}]")
            fc.append(f"[as{i}]atrim={r0:.4f}:{r1:.4f},asetpts=PTS-STARTPTS,{_atempo(sp)}[a{i}]")
        cat += f"[v{i}][a{i}]"
    fc.append(f"{cat}concat=n={n}:v=1:a=1[outv][outa]")
    out = os.path.join(work, f"seg_{clip['id']}.mov")
    vcodec = "h264_videotoolbox" if is_macos() else "libx264"
    hwa = ["-hwaccel", hwaccel] if hwaccel else ([] if not is_macos() else ["-hwaccel", "videotoolbox"])
    cmd = ["nice", "-n", "20", ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", *hwa,
           "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src,
           "-filter_complex", ";".join(fc), "-map", "[outv]", "-map", "[outa]",
           "-r", str(fps), "-c:v", vcodec, "-b:v", "8M", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return out


def _final_enc():
    """Final overlay pass encoder: VideoToolbox on macOS (hardware, ~5x faster than libx264 on a busy Mac), x264 elsewhere."""
    if is_macos(): return ["-c:v", "h264_videotoolbox", "-b:v", "12M"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def _enc():
    if is_macos():
        return ["-c:v", "h264_videotoolbox", "-b:v", "8M"], ["-hwaccel", "videotoolbox"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"], []


def render_broadcast_clip(src, clip, preset, aspect, work, SW=3840, SH=2160, fps=30):
    """Broadcast-grammar clip (reel/camera.py broadcast_plan): each framing segment (wide / follow / reaction / replay)
    is rendered from the source with its own crop, graded, then the segments are hard-cut into work/seg_<id>.mov.
    The live segments are contiguous in source time, so the whole point plays once; a replay is appended after it."""
    from ..reel.camera import crop_render, crop_render_cmds, crop_render_head
    ow, oh = OUT[aspect]
    grade = grade_filter(preset)
    g = ("," + grade) if grade else ""
    vcodec, hwa = _enc()
    parts = []
    fh = font(preset, "names")
    for k, s in enumerate(clip["framing"]):
        t0, t1, sp = s["t0"], s["t1"], float(s.get("speed", 1.0))
        if t1 - t0 < 0.05: continue
        out = os.path.join(work, f"seg_{clip['id']}_{k}.mov")
        if s["kind"] == "wide":
            fc = f"[0:v]fps={fps},format=yuv420p,{_reframe_filter('16:9', ow, oh, SW, SH)}{g}[v]"
        elif s["kind"] == "follow":
            path = s["path"]; n = len(path)
            cr = crop_render(path - s["cw"] / 2, np.zeros(n), np.full(n, s["cw"]), np.full(n, SH), SW, SH, ow, oh)
            cmd = os.path.join(work, f"seg_{clip['id']}_{k}.cmd")
            open(cmd, "w").write("\n".join(crop_render_cmds(cr, fps)))
            fc = crop_render_head(cr, cmd, SH, fps, 1, ow, oh).replace("[0:v]", f"[0:v]fps={fps},format=yuv420p,", 1) + f"[vh];[vh]null{g}[v]"
        else:
            # v4.9 fix: hardware-decoded frames arrive as semi-planar NV12, where an odd crop offset swaps U and V (the
            # clay court turned blue in reaction shots). Convert to planar yuv420p first and keep the window even.
            x, y, w, h = (int(v) // 2 * 2 for v in s["window"])
            fc = f"[0:v]fps={fps},format=yuv420p,crop={w}:{h}:{x}:{y},scale={ow}:{oh}:flags=lanczos,setsar=1{g}"
            if s["kind"] == "replay":
                tag = f"drawtext={'fontfile=' + fh + ':' if fh else ''}text='REPLAY':fontsize={40 if aspect == '9:16' else 34}:" \
                      f"fontcolor=0x{preset['palette']['chalk'].lstrip('#').upper()}:x={60 if aspect == '16:9' else 64}:" \
                      f"y={60 if aspect == '16:9' else 260}:box=1:boxcolor=0x{preset['palette']['night'].lstrip('#').upper()}@0.75:boxborderw=12"
                mi = "mci" if fps >= 50 else "blend"
                fc += f",setpts=PTS/{sp:.4f},minterpolate=fps={fps}:mi_mode={mi},{tag}"
            fc += "[v]"
        af = "aresample=48000,aformat=channel_layouts=stereo"
        if abs(sp - 1.0) > 1e-6: af += f",asetpts=PTS-STARTPTS,{_atempo(sp)},volume={float(s.get('gain', 1.0)):.2f}"
        cmd = ["nice", "-n", "20", ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", *hwa,
               "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", src, "-filter_complex", fc + f";[0:a]{af}[a]",
               "-map", "[v]", "-map", "[a]", "-r", str(fps), *vcodec, "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out]
        subprocess.run(cmd, check=True)
        parts.append(out)
    lst = os.path.join(work, f"seg_{clip['id']}.txt")
    with open(lst, "w") as f:
        for p_ in parts: f.write(f"file '{p_}'\n")
    out = os.path.join(work, f"seg_{clip['id']}.mov")
    subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", out], check=True)
    return out


def render_reel(src, plan, preset, out_path, work=None, SW=3840, SH=2160, fps=30, probe=None):
    """Validate the plan, render each whole-point clip, hard-cut concat, burn overlays on the output timeline.
    Returns dict(out, duration_s, slow_mo_count, captions, grade)."""
    # points stub for re-validation: reconstruct minimal point records from the plan + original times
    points = [dict(id=c["id"], first_hit=c["first_hit"], last_hit=c["last_hit"], A=c["a"], B=c["b"],
                   favorite=c["slow"], kind="point", sources=["audio", "user"], state_verified=False)
              for c in plan["clips"]]
    # re-attach verification from bugs/captions
    for o in plan["captions"] + plan["bugs"]:
        for p in points:
            if p["id"] == o.get("point"):
                p["state_verified"] = True
                if o.get("state") is not None: p["state"] = o["state"]
    validate(plan, points, preset)
    if plan.get("letterbox") or plan.get("wide_mode"):
        raise PlanError("R-9x16: letterbox / wide mode forbidden")
    if plan.get("music") not in (None, "platform_sound_picker_at_share", "court_only", "user_selected"):
        raise PlanError("music must not be baked; use the platform sound picker at share")
    work = work or tempfile.mkdtemp(prefix="vamos_reel_")
    os.makedirs(work, exist_ok=True)
    json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1, default=str)
    segs = []
    for c in plan["clips"]:
        if c.get("framing"):
            segs.append(render_broadcast_clip(src, c, preset, plan["aspect"], work, SW, SH, fps))
        else:
            segs.append(render_clip(src, c, preset, plan["aspect"], work, SW, SH, fps, cx_frac=float(c.get("cx_frac", 0.5))))
    # concat (hard cuts; short audio crossfade)
    concat = os.path.join(work, "concat.txt")
    with open(concat, "w") as f:
        for s in segs: f.write(f"file '{s}'\n")
    bare = os.path.join(work, "bare.mov")
    # audio_crossfade_s reserved for acrossfade graph (v1: hard-cut via concat demuxer)
    # acrossfade between clips would need a more complex graph; for v1 we hard-cut video and soft-pad audio via concat demuxer
    subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", concat,
                    "-c", "copy", bare], check=True)
    # burn overlays on the output timeline
    ow, oh = OUT[plan["aspect"]]
    overlays = _overlay_filters(plan, preset, ow, oh, plan["aspect"])
    # outro: freeze last frame then draw the card
    hold = 0.0
    if plan.get("outro"):
        hold = plan["outro"]["end"] - plan["outro"]["start"]
    vchain = ",".join(overlays) if overlays else "null"
    if hold > 0:
        # tpad clones the last frame for the outro window; overlays already include the result card enable times
        vchain = f"tpad=stop_mode=clone:stop_duration={hold:.3f}," + vchain
    af = "loudnorm=I=-14:TP=-1.0:LRA=11" if plan["aspect"] == "9:16" else "loudnorm=I=-16:TP=-1.0:LRA=11"
    cmd = [ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-i", bare,
           "-vf", vchain, "-af", af,
           *_final_enc(), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    # logo as second input if present
    logo = plan.get("_logo") or asset(preset, preset.get("assets", {}).get("logo_c", ""))
    if plan.get("outro") and logo and os.path.exists(logo):
        # rebuild with logo overlay during outro only
        t0, t1 = plan["outro"]["start"], plan["outro"]["end"]
        lw = int(ow * (0.28 if plan["aspect"] == "9:16" else 0.12))
        cmd = [ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-i", bare, "-i", logo,
               "-filter_complex",
               f"[0:v]{vchain}[base];[1:v]scale={lw}:-1[lg];"
               f"[base][lg]overlay=(W-w)/2:H-h-{int(oh * 0.08)}:enable='between(t,{t0:.3f},{t1:.3f})'[vout]",
               "-map", "[vout]", "-map", "0:a", "-af", af,
               *_final_enc(), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)
    return dict(out=out_path, duration_s=plan["duration_s"], slow_mo_count=plan["slow_mo_count"],
                captions=[o["text"] for o in plan["captions"]], grade=bool(preset["grade"].get("enabled", True)),
                letterbox=False, music="platform_sound_picker_at_share", work=work)
