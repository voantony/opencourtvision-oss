"""Capture-quality hints (Antony, 19 Oct): odd angles, behind-fence, shaky / handheld, low light.

Never refuse to process. Always return hints the UI can show (e.g. "raise the phone", "include the whole court").
The fix-screen flywheel (`review.prioritise`) uses `struggle_tags` so corrections from hard conditions are trained first.
"""
import json
import subprocess

from ..ffmpeg import ffprobe_bin

HINTS = {
    "low_resolution": "Video is under 720p. Vamos still works, but a higher resolution camera helps ball tracking.",
    "low_light": "The picture looks dark. More light, or turning on the court lights, will help.",
    "portrait_phone": "Looks like a portrait phone recording. Landscape with the whole court in frame works better.",
    "narrow_fov": "The court looks cropped. Try stepping back or zooming out so both baselines are visible.",
    "no_audio": "No usable audio track. Scoring still runs from vision, but hit detection is weaker without sound.",
    "short_clip": "Clip is under a minute. Import still works; match mode needs a fuller recording.",
    "shaky": "The camera looks handheld or shaky. A tripod or leaning the phone against the fence helps a lot.",
    "behind_fence": "Looks like a behind-fence angle. We will try; a higher vantage (raised phone / bleacher) is clearer.",
}


def probe(path):
    """Lightweight ffprobe summary. Missing tools or corrupt files -> empty dict (never raises to the caller)."""
    try:
        out = subprocess.run([ffprobe_bin(), "-v", "error", "-show_entries",
                              "format=duration,size:stream=codec_type,width,height,avg_frame_rate,bit_rate,pix_fmt,nb_frames",
                              "-of", "json", path], capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def analyse(path, motion_energy=None):
    """-> dict(ok=True, hints=[...], struggle_tags=[...], metrics={...}). ok is always True (never refuse)."""
    info = probe(path)
    streams = info.get("streams") or []
    fmt = info.get("format") or {}
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    dur = float(fmt.get("duration") or 0)
    hints, tags, metrics = [], [], dict(width=w, height=h, duration_s=round(dur, 1), has_audio=bool(a))
    if w and h and min(w, h) < 720:
        hints.append(HINTS["low_resolution"]); tags.append("low_resolution")
    if w and h and h > w * 1.2:
        hints.append(HINTS["portrait_phone"]); tags.append("portrait_phone")
    if w and h and w / max(h, 1) > 2.2:
        hints.append(HINTS["narrow_fov"]); tags.append("narrow_fov")
    if not a:
        hints.append(HINTS["no_audio"]); tags.append("no_audio")
    if 0 < dur < 60:
        hints.append(HINTS["short_clip"]); tags.append("short_clip")
    # motion_energy optional: high frame-to-frame energy with low subject motion ~ shake
    if motion_energy is not None and float(motion_energy) > 0.35:
        hints.append(HINTS["shaky"]); tags.append("shaky")
    # always append a gentle raise-the-phone tip when we only have a low / side angle signal
    if "portrait_phone" in tags or "narrow_fov" in tags:
        hints.append(HINTS["behind_fence"]); tags.append("behind_fence")
    return dict(ok=True, hints=hints or ["Looks usable. Tip: raise the phone so the whole court is in frame."],
                struggle_tags=tags, metrics=metrics)


def never_refuse(path):
    """Public guarantee used by the app: processing is always attempted."""
    return analyse(path)["ok"] is True
