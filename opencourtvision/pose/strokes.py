"""Stroke detection and serve/forehand/backhand labelling from one player's keypoints.

Contact times come from the audio stage (preferred, ms-accurate) or from hitting-wrist speed peaks. A hit is the
tracked player's when their hitting wrist has a speed peak near it. Labels are rule-based on body geometry, so they
work without training data; a nearest-centroid model can be fitted later from labelled strokes.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .skeleton import J, hip_center, interpolate_gaps, shoulder_center

KINDS = ("forehand", "backhand", "serve")


def body_scale(kp):
    """Median torso length (shoulder centre -> hip centre) in px; the unit for scale-free thresholds."""
    return float(np.nanmedian(np.hypot(*(shoulder_center(kp) - hip_center(kp)).T)))


def wrist_speed(kp, fps, hand="right"):
    """Hitting-wrist speed relative to the hips, in torso lengths per second (T,)."""
    w = kp[:, J["r_wrist" if hand == "right" else "l_wrist"]] - hip_center(kp)
    w = uniform_filter1d(interpolate_gaps(w[:, None])[:, 0], 3, axis=0)
    v = np.gradient(w, axis=0) * fps
    return np.hypot(v[:, 0], v[:, 1]) / body_scale(kp)


def contacts(kp, fps, hit_times=None, hand="right", win=0.25, min_peak=4.0):
    """Contact times (s) for this player. With hit_times: keep the audio hits that have a wrist-speed peak within
    ±win s (contact = the audio time). Without: wrist-speed peaks above min_peak torso/s at least 0.6 s apart."""
    sp = wrist_speed(kp, fps, hand); sp = np.nan_to_num(sp)
    if hit_times is None:
        pk, _ = find_peaks(sp, height=min_peak, distance=max(1, int(0.6 * fps)))
        return [float(p / fps) for p in pk]
    out = []
    for t in hit_times:
        a, b = int(max(0, (t - win) * fps)), int(min(len(sp), (t + win) * fps + 1))
        if b > a and sp[a:b].max() >= min_peak: out.append(float(t))
    return out


def _at(x, t, fps):
    return x[int(np.clip(round(t * fps), 0, len(x) - 1))]


def classify(kp, fps, t, hand="right"):
    """-> (kind, confidence 0-1, evidence list). Image convention: near-end player with their back to the camera,
    so the player's right is image-right."""
    sc = body_scale(kp); hw = J["r_wrist" if hand == "right" else "l_wrist"]
    w = _at(kp[:, hw], t, fps); hc = _at(hip_center(kp), t, fps); head = _at(kp[:, J["head"]], t, fps)
    ev = []
    if np.isnan(w).any() or np.isnan(hc).any(): return "unknown", 0.0, ["hitting wrist not visible at contact"]
    above = (head[1] - w[1]) / sc if not np.isnan(head).any() else -9
    if above > 0.15:
        ev.append(f"wrist {above:.2f} torso above head at contact")
        return "serve", float(min(1, 0.6 + above)), ev
    side = (w[0] - hc[0]) / sc * (1 if hand == "right" else -1)   # + = hitting-hand side
    ev.append(f"wrist {side:+.2f} torso to the hitting-hand side of the hips")
    conf = float(min(1, 0.5 + abs(side) / 2))
    return ("forehand" if side > 0 else "backhand"), conf, ev


def strokes(kp, fps, hit_times=None, hand="right"):
    """-> list of dict(t, kind, confidence, evidence)."""
    return [dict(t=t, kind=k, confidence=round(c, 2), evidence=e) for t in contacts(kp, fps, hit_times, hand)
            for k, c, e in [classify(kp, fps, t, hand)]]
