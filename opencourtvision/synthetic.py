"""Synthetic tennis-like audio/video for tests and benchmarks (no real footage).

Hits are short decaying 2-5 kHz tone bursts ("pops") at known times over a noise floor; optional
distractors (low-frequency thumps, mid-band chatter) should NOT be detected as hits. Video is ffmpeg's
`testsrc` pattern, muxed with the audio.
"""
import json
import os
import subprocess

import numpy as np
import scipy.io.wavfile as wf

from .ffmpeg import ffmpeg_bin

SR = 16000


def pop(rng, sr=SR, amp=0.4):
    t = np.arange(int(sr * 0.02)) / sr
    f = rng.uniform(2200, 4800)
    return amp * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.004)


def thump(rng, sr=SR, amp=0.6):  # e.g. footstep / racket bag, outside the hit band
    t = np.arange(int(sr * 0.08)) / sr
    return amp * np.sin(2 * np.pi * rng.uniform(80, 180) * t) * np.exp(-t / 0.02)


def chatter(rng, sr=SR, amp=0.05, dur=1.5):  # speech-like 300-1200 Hz noise
    import scipy.signal as ss
    n = rng.standard_normal(int(sr * dur))
    b, a = ss.butter(4, [300, 1200], btype="band", fs=sr)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(n)) / len(n)))
    return amp * ss.lfilter(b, a, n) * env * 6


def render_audio(hit_times, total_s, seed=0, noise_db=-50.0, hit_db=(-14.0, -6.0), distractors=()):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total_s * SR)) * 10 ** (noise_db / 20)
    def add(sig, t):
        i = int(t * SR)
        j = min(len(x), i + len(sig))
        if 0 <= i < len(x):
            x[i:j] += sig[:j - i]
    for t in hit_times:
        add(pop(rng, amp=10 ** (rng.uniform(*hit_db) / 20)), t)
    for t, kind in distractors:
        add(thump(rng) if kind == "thump" else chatter(rng), t)
    return np.clip(x, -1, 1)


def _rally(rng, t, n, lo=0.9, hi=1.6):
    out = []
    for _ in range(n):
        out.append(round(t, 3))
        t += rng.uniform(lo, hi)
    return out


def scenario(name, seed=0):
    """Return a ground-truth dict for a named scenario: hits_clean, hits_noisy, match_sim."""
    rng = np.random.default_rng(seed)
    gt = dict(name=name, seed=seed, sample_rate=SR, rallies=[], distractors=[])
    if name in ("hits_clean", "hits_noisy"):
        total, t = 120.0, 3.0
        while t < total - 15:
            r = _rally(rng, t, int(rng.integers(3, 11)))
            gt["rallies"].append(r)
            t = r[-1] + rng.uniform(10, 18)
        gt.update(total_s=total, noise_db=-50.0 if name == "hits_clean" else -36.0,
                  hit_db=[-14.0, -6.0] if name == "hits_clean" else [-20.0, -10.0])
        if name == "hits_noisy":
            gt["distractors"] = [[round(float(v), 2), k] for v, k in
                                 zip(sorted(rng.uniform(1, total - 2, 12)), ["thump", "chatter"] * 6)]
    elif name == "match_sim":
        t = 5.0
        while t < 240:  # warmup: long near-continuous hitting with short ball-pickup breaks
            r = _rally(rng, t, int(rng.integers(10, 25)), 1.0, 1.8)
            gt["rallies"].append(r)
            t = r[-1] + rng.uniform(6.5, 11)
        t += 40  # reset pause before the first point
        first = True
        while t < 780:
            if first:  # first point opens with a fault; second serve 4.5 s later merges into the rally
                r = [round(t, 3)] + _rally(rng, t + 4.5, int(rng.integers(4, 9)))
                gt["match_start_s"] = round(t, 3)
                first = False
            else:
                r = _rally(rng, t, int(rng.integers(2, 10)), 1.0, 1.6)
            gt["rallies"].append(r)
            t = r[-1] + rng.uniform(18, 35)
        gt.update(total_s=round(t + 10, 1), noise_db=-48.0, hit_db=[-12.0, -6.0])
    else:
        raise ValueError(name)
    gt["hits"] = [h for r in gt["rallies"] for h in r]
    return gt


SCENARIOS = ("hits_clean", "hits_noisy", "match_sim")


def make_clip(gt, out_dir, video=True):
    """Write <name>.wav (and <name>.mp4 with testsrc video, 2 s GOP) + <name>.json ground truth; returns paths."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, gt["name"])
    x = render_audio(gt["hits"], gt["total_s"], gt["seed"], gt["noise_db"], tuple(gt["hit_db"]), [tuple(d) for d in gt["distractors"]])
    wf.write(base + ".wav", SR, (x * 32767).astype(np.int16))
    json.dump(gt, open(base + ".json", "w"), indent=1)
    paths = dict(wav=base + ".wav", gt=base + ".json")
    if video:
        subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc=size=320x180:rate=10:duration={gt['total_s']}", "-i", base + ".wav",
                        "-c:v", "libx264", "-preset", "ultrafast", "-g", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", base + ".mp4"], check=True)
        paths["mp4"] = base + ".mp4"
    return paths
