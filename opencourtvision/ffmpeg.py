"""Locating ffmpeg/ffprobe/fonts and small ffprobe helpers.

Binary lookup order: $OCV_FFMPEG / $OCV_FFPROBE, then ~/bin/<name> (where a static build is
usually dropped), then PATH.
"""
import os
import shutil
import subprocess
import sys

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu
)


def _find(name, env):
    v = os.environ.get(env)
    if v:
        return v
    local = os.path.expanduser(f"~/bin/{name}")
    if os.path.exists(local):
        return local
    return shutil.which(name) or name


def ffmpeg_bin():
    return _find("ffmpeg", "OCV_FFMPEG")


def ffprobe_bin():
    return _find("ffprobe", "OCV_FFPROBE")


def is_macos():
    return sys.platform == "darwin"


def default_font():
    """Font for drawtext: $OCV_FONT, else a known system bold font, else None (fontconfig default)."""
    v = os.environ.get("OCV_FONT")
    if v:
        return v
    return next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)


def fontfile_opt(font):
    return f"fontfile='{font}':" if font else ""


def duration(path):
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out)


def video_codec(path):
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return out.strip()


def keyframes(path, cache=None):
    """Keyframe (packet) timestamps of the first video stream; cached to `cache` if given."""
    if cache and os.path.exists(cache):
        return [float(v) for v in open(cache) if v.strip()]
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts_time,flags",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    kf = [line.split(",")[0] for line in out.splitlines() if ",K" in line]
    if cache:
        open(cache, "w").write("\n".join(kf))
    return [float(v) for v in kf if v.strip()]
