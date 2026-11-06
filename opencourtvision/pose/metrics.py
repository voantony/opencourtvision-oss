"""Stroke mechanics and footwork, in cm / degrees / m/s.

Per stroke (at contact unless noted):
- stance_deg      angle between the feet line and the baseline (0 = square/open-parallel, 90 = one foot behind the other)
- contact_cm      hip centre -> hitting wrist distance
- finish_cm       highest hitting-wrist point in the 0.6 s after contact, relative to the shoulders (+ = above)
- lean_cm         serves: shoulder centre lateral offset from the feet centre (+ = to the player's left)
- court_x/court_y the player's court position (m) when a court calibration is given
Scale: from the court homography at the player's feet when `court` is given, else assume a torso of torso_cm.
"""
import numpy as np

from .skeleton import J, foot_point, ground_point, hip_center, shoulder_center
from .strokes import body_scale

STROKE_METRICS = ("stance_deg", "contact_cm", "finish_cm", "lean_cm")


def _i(t, fps, n):
    return int(np.clip(round(t * fps), 0, n - 1))


def scale_cm(kp, court=None, torso_cm=52.0):
    """cm per px per frame (T,)."""
    if court is not None:
        return court.cm_per_px(ground_point(kp))
    return np.full(len(kp), torso_cm / body_scale(kp))


def stroke_metrics(kp, fps, st, court=None, hand="right", torso_cm=52.0, finish_win=0.6):
    """kp (T,17,2); st: dict(t, kind, ...) from strokes.strokes. Returns a new dict with the metrics added."""
    n = len(kp); i = _i(st["t"], fps, n); cm = scale_cm(kp, court, torso_cm)[i]
    hw = J["r_wrist" if hand == "right" else "l_wrist"]
    hc, sc, g = hip_center(kp), shoulder_center(kp), ground_point(kp)
    lf, rf = foot_point(kp, "l"), foot_point(kp, "r")
    out = dict(st)
    if court is not None:
        out["stance_deg"] = round(float(court.baseline_angle(lf[i], rf[i])), 1)
        cx, cy = court.to_court(g[i]); out["court_x"], out["court_y"] = round(float(cx), 2), round(float(cy), 2)
    else:
        d = rf[i] - lf[i]; out["stance_deg"] = round(float(np.degrees(np.arctan2(abs(d[1]), abs(d[0])))), 1)
    out["contact_cm"] = round(float(np.hypot(*(kp[i, hw] - hc[i])) * cm), 1)
    j = _i(st["t"] + finish_win, fps, n)
    rise = (sc[i:j + 1, 1] - kp[i:j + 1, hw, 1])               # image y grows downward
    out["finish_cm"] = round(float(np.nanmax(rise) * cm), 1) if np.isfinite(rise).any() else None
    out["lean_cm"] = round(float((g[i, 0] - sc[i, 0]) * cm), 1) if st.get("kind") == "serve" else None   # image-left = player's left
    return out


def footwork(kp, fps, court=None, torso_cm=52.0, min_speed=0.3):
    """Movement summary: distance (m), mean/peak speed (m/s), time moving, court coverage (convex-hull m^2 with a
    court, else None). Uses the feet centre, smoothed over 0.2 s."""
    from scipy.ndimage import uniform_filter1d
    g = ground_point(kp); ok = ~np.isnan(g).any(1)
    if ok.sum() < 3: return {}
    if court is not None:
        p = court.to_court(g[ok])
    else:
        p = g[ok] * scale_cm(kp, None, torso_cm)[ok, None] / 100
    p = uniform_filter1d(p, max(1, int(0.2 * fps)), axis=0)
    v = np.hypot(*np.diff(p, axis=0).T) * fps
    res = dict(distance_m=round(float(v.sum() / fps), 2), mean_speed=round(float(v.mean()), 2),
               peak_speed=round(float(np.percentile(v, 99)), 2), moving_s=round(float((v > min_speed).sum() / fps), 2))
    if court is not None and len(p) >= 3:
        from scipy.spatial import ConvexHull
        try: res["coverage_m2"] = round(float(ConvexHull(p).volume), 2)
        except Exception: res["coverage_m2"] = 0.0
    return res
