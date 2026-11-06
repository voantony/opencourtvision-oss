"""Point segments and lossless rendering.

Segments are padded (lead before the first hit, tail after the last) and snapped outward to video keyframes,
so the video stream can be stream-copied (no re-encode, full 4K quality). Audio is re-encoded to AAC with
short fades to avoid clicks at the cuts.
"""
import bisect
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from ..ffmpeg import ffmpeg_bin, video_codec

MIN_TAIL_S = 2.0          # no clip ends sooner than last contact + this (SwingVision cuts ~1 s after it; too short); reel beats excepted
DEAD_BALL_BUFFER_S = 1.5  # point clips keep this long after the decisive bounce / miss (editorial spec §2.1)

# Editorial spec (Media Director, draft v1, §2.1): end rule per clip type. t_last = last contact, t_dec = decisive bounce or miss.
CLIP_TYPES = {
    "point": dict(tail=2.5, dec=1.5, reaction_tail=5.0, cap=6.0),   # max(t_last + 2.5, t_dec + 1.5); +5 s on reaction; cap t_last + 6
    "rally": dict(dec=1.5),                                        # t_dec + 1.5
    "highlight_shot": dict(dec=1.2),                               # t_dec + 1.2
    "reel_beat": dict(dec=0.8, floor_last=0.6, floor_outcome=0.4),  # t_dec + 0.6-1.0 (0.8); own floor, exempt from MIN_TAIL_S
    "reaction": dict(after_onset=2.5, max_after_onset=3.5),        # onset + 2.5, max 3.5
    "celebration": dict(after_end=0.5, max_reel=4.0, max_highlights=8.0),   # celebration end + 0.5; length cap from t_dec - 0.5
}
REACTION_THRESHOLD = 0.6


def point_end(last_contact, ball_dead=None, tail=None, buffer=None, clip_type="point", reaction=None, reaction_onset=None,
              reaction_end=None, celebration_end=None, output="highlights"):
    """Clip end time for `clip_type` (see CLIP_TYPES). ball_dead = decisive bounce or miss (t_dec), when known.

    For reel beats ball_dead is "outcome visible": the decisive bounce, or the ball going into the net or fence.

    tail / buffer override the type's t_last / t_dec offsets. reaction is a 0-1 score (> 0.6 extends point clips to
    t_last + 5 s; reaction_end, if known, lets reel beats roll into the reaction). Every type except reel_beat ends no
    earlier than last_contact + MIN_TAIL_S and t_dec + 1.0 s (cut-off ending killer). reel_beat has its own floor,
    max(t_last + 0.6, outcome visible + 0.4); with no outcome time it uses the full point post-roll, since it cannot
    show that the result is on screen."""
    if clip_type not in CLIP_TYPES: raise ValueError(f"unknown clip_type {clip_type!r}; one of {', '.join(CLIP_TYPES)}")
    c = CLIP_TYPES[clip_type]; t = last_contact
    dec = ball_dead if ball_dead is not None and ball_dead >= last_contact else None
    if clip_type == "point":
        b = t + (c["tail"] if tail is None else tail)
        if dec is not None: b = max(b, dec + (c["dec"] if buffer is None else buffer))
        if reaction is not None and reaction > REACTION_THRESHOLD: b = max(b, t + c["reaction_tail"])
        b = min(b, t + c["cap"])
    elif clip_type == "reel_beat":
        if dec is None: return point_end(last_contact, None, tail, None, "point", reaction)
        b = dec + (c["dec"] if buffer is None else buffer)
        if tail is not None: b = max(b, t + tail)
        if reaction_end is not None and reaction is not None and reaction > REACTION_THRESHOLD: b = max(b, reaction_end)
        return max(b, t + c["floor_last"], dec + c["floor_outcome"])
    elif clip_type in ("rally", "highlight_shot"):
        b = (dec if dec is not None else t) + (c["dec"] if buffer is None else buffer)
        if tail is not None: b = max(b, t + tail)
    elif clip_type == "reaction":
        on = reaction_onset if reaction_onset is not None else t
        b = on + c["after_onset"]
        if reaction_end is not None: b = min(max(b, reaction_end), on + c["max_after_onset"])
    else:   # celebration
        start = (dec if dec is not None else t) - 0.5
        b = (celebration_end + c["after_end"]) if celebration_end is not None else t
        b = min(b, start + (c["max_reel"] if output == "reel" else c["max_highlights"]))
    if dec is not None: b = max(b, dec + 1.0)
    return max(b, t + MIN_TAIL_S)


def build_segments(R, start, KF, dur, lead=4.0, tail=None, merge_gap=1.0):
    """Rallies (>= 2 hits) plus isolated loud single hits (possible aces / serve winners, flagged serve_only).
    Segment end = point_end(last hit, rally "t_dead", clip_type="point", reaction=rally "reaction") (default last hit + 2.5 s,
    never < + 2 s), snapped forward to a keyframe (never earlier);
    only the end of the file can cut it shorter."""
    G = [r for r in R if r["t0"] >= start - 0.01 and r["amax"] >= 16 and (r["hits"] >= 2 or (r["amax"] >= 20 and r["pause_before"] >= 8))]
    segs = []
    for r in G:
        a, b = r["t0"] - lead, point_end(r["t1"], r.get("t_dead"), tail, clip_type="point", reaction=r.get("reaction"))
        if segs and a < segs[-1]["b"] + merge_gap:
            segs[-1]["b"] = max(segs[-1]["b"], b)
            segs[-1]["rallies"].append(r)
        else:
            segs.append(dict(a=a, b=b, rallies=[r]))
    for s in segs:
        s["A"] = KF[max(0, bisect.bisect_right(KF, s["a"]) - 1)]
        i = bisect.bisect_left(KF, s["b"])
        s["B"] = KF[i] if i < len(KF) else dur
        s["hits"] = sum(r["hits"] for r in s["rallies"])
        s["fault_like"] = s["hits"] <= 2 and s["rallies"][-1]["pause_after"] < 20
        s["serve_only"] = s["hits"] == 1
    for i, s in enumerate(segs):
        s["gap_after"] = segs[i + 1]["A"] - s["B"] if i + 1 < len(segs) else 0
    return segs


def _tag(video):
    # QuickTime-friendly tag for HEVC sources (e.g. 4K phone/SwingVision footage); left alone for other codecs.
    return ["-tag:v", "hvc1"] if video_codec(video) == "hevc" else []


def cut_segment(video, s, out, tag=None):
    d = s["B"] - s["A"]
    tag = _tag(video) if tag is None else tag
    subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-ss", f"{s['A']:.6f}", "-i", video, "-t", f"{d:.6f}",
                    "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", *tag, "-c:a", "aac", "-b:a", "256k",
                    "-af", f"afade=t=in:d=0.2,afade=t=out:st={max(0, d - 0.2):.3f}:d=0.2",
                    "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", out], check=True)
    return out


def pause_spans(s):
    """[A, B] minus the segment's chained pauses (editloop/smart.py); one span when there are none."""
    out, a = [], s["A"]
    for p0, p1 in sorted(s.get("pauses") or ()):
        if a < p0 < p1 < s["B"]: out.append((a, p0)); a = p1
    return out + [(a, s["B"])]


def concat(files, listfile, out, tag=()):
    with open(listfile, "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in files)
    subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                    "-c", "copy", *tag, "-movflags", "+faststart", out], check=True)


def render(video, segs, hl, workdir, outdir, name="Match", threads=3):
    """Cut every segment (cached in workdir/segs) and write '<name> - points only 4K.mp4' and '<name> - highlights 4K.mp4'."""
    sd = os.path.join(workdir, "segs")
    os.makedirs(sd, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    tag = _tag(video)

    def cut(i):
        s = segs[i]
        o = os.path.join(sd, f"p{i + 1:03d}_{s['A']:.3f}.mp4")
        if not os.path.exists(o):
            spans = pause_spans(s)
            if len(spans) == 1:
                cut_segment(video, s, o, tag)
            else:                                  # chained rally: jump cut over the mid-rally pause(s)
                sub = [cut_segment(video, dict(A=a, B=b), f"{o[:-4]}_{k}.mp4", tag) for k, (a, b) in enumerate(spans)]
                concat(sub, f"{o[:-4]}.txt", o, tag)
        return o

    with ThreadPoolExecutor(threads) as ex:
        files = list(ex.map(cut, range(len(segs))))
    for title, idx in ((f"{name} - points only 4K.mp4", range(len(segs))), (f"{name} - highlights 4K.mp4", hl)):
        concat([files[i] for i in idx], os.path.join(workdir, "concat.txt"), os.path.join(outdir, title), tag)
    return round(time.time() - t0, 1)
