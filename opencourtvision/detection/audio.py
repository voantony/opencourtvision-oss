"""Audio-based ball-hit detection.

Ball hits are short transients in the 1.5-6 kHz band. We compute band loudness per 10 ms frame,
subtract a 3 s running median background, and keep peaks that stand >= `th` dB above it and rise
sharply (> `min_rise` dB over 30-50 ms).
"""
import os
import subprocess
import tempfile

import numpy as np
import scipy.io.wavfile as wf
import scipy.ndimage as nd
import scipy.signal as ss

from ..ffmpeg import ffmpeg_bin

FRAME_RATE = 100  # loudness frames per second (10 ms hop)


def extract_wav(src, out, sr=16000):
    subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", out], check=True)


def read_wav(path):
    sr, x = wf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    return sr, x.astype(np.float32) / 32768


def load_audio(src):
    """WAV is read directly; anything else is decoded to 16 kHz mono via ffmpeg."""
    if src.lower().endswith(".wav"):
        return read_wav(src)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        extract_wav(src, tmp)
        return read_wav(tmp)
    finally:
        os.remove(tmp)


def loudness(sr, x, band=(1500, 6000)):
    """Returns (le, bg, lb): band loudness dB, its 3 s median background, broadband loudness dB (all per 10 ms)."""
    b, a = ss.butter(4, list(band), btype="band", fs=sr)
    y = ss.filtfilt(b, a, x)
    hop = sr // FRAME_RATE
    n = len(y) // hop
    le = 20 * np.log10(np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-6)
    lb = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-6)
    bg = nd.median_filter(le, size=301)
    return le, bg, lb


def detect_hits(le, bg, th=14.0, min_rise=6.0, min_dist=30):
    """Hit times (s) and their prominence above background (dB)."""
    d = le - bg
    rise = le - np.maximum.reduce([np.roll(le, k) for k in (3, 4, 5)])
    pk, _ = ss.find_peaks(d, height=th, distance=min_dist)
    pk = pk[rise[pk] > min_rise]
    return pk / float(FRAME_RATE), d[pk]


def group_rallies(t, amp, gap=6.0):
    """Group hits separated by <= gap s into rallies: dicts t0, t1, hits, amax, times, pause_before, pause_after."""
    if len(t) == 0:
        return []
    G = [[0]]
    for i in range(1, len(t)):
        (G.append([i]) if t[i] - t[G[-1][-1]] > gap else G[-1].append(i))
    out = [dict(t0=float(t[g[0]]), t1=float(t[g[-1]]), hits=len(g), amax=float(amp[g].max()), times=[float(t[i]) for i in g])
           for g in G]
    for i, r in enumerate(out):
        r["pause_before"] = r["t0"] - out[i - 1]["t1"] if i else r["t0"]
        r["pause_after"] = out[i + 1]["t0"] - r["t1"] if i + 1 < len(out) else 99
    return out


def time_groups(t, gap):
    """Plain lists of hit times per rally, keeping only groups with >= 2 hits (used by the rhythm detector)."""
    if len(t) == 0:
        return []
    g = [[t[0]]]
    for v in t[1:]:
        (g.append([v]) if v - g[-1][-1] > gap else g[-1].append(v))
    return [x for x in g if len(x) >= 2]
