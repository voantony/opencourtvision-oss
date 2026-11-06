"""Speed-label plausibility filter: a ball speed is captioned only if it is plausible for the sport and stroke and
consistent with the measured flight. Otherwise it is suppressed (no label), never "corrected".

SwingVision takeaway: implausible numbers (a 35 mph tennis serve, a 140 mph padel bandeja, repeated capped values) cost
more trust than a missing label. Two checks:
1. Range per sport and stroke (mph, peak speed off the racket/paddle). Bounds are generous outer limits, not typical values:
   tennis serve 50-165 (fastest recorded serves are ~163 mph), padel smash <= 125, pickleball drive <= 75.
2. Physics: with a measured contact->first-bounce distance d (m) and flight time t (s), the mean flight speed is d/t.
   A ball slows down in flight (drag), so the peak speed must be >= the mean (minus tolerance) and can't exceed it by more
   than MAX_PEAK_OVER_MEAN (tennis serve ~1.3-1.5x; padel/pickleball balls are slower and decay less).
"""
MPH_PER_MS = 2.2369362920544

RANGES = {   # (min, max) mph
    "tennis": {"serve": (50, 165), "second_serve": (40, 140), "forehand": (15, 125), "backhand": (15, 115), "volley": (8, 110),
               "overhead": (25, 135), "return": (15, 125), "lob": (8, 80), "default": (8, 130)},
    "padel": {"serve": (20, 75), "forehand": (10, 95), "backhand": (10, 90), "volley": (8, 90), "bandeja": (15, 85),
              "vibora": (20, 95), "smash": (25, 125), "lob": (6, 60), "default": (6, 110)},
    "pickleball": {"serve": (15, 60), "drive": (15, 75), "dink": (2, 25), "drop": (5, 35), "volley": (5, 70), "reset": (3, 35),
                   "erne": (10, 70), "speed_up": (15, 75), "overhead": (20, 75), "default": (2, 75)},
}
MAX_PEAK_OVER_MEAN = {"tennis": 1.8, "padel": 1.6, "pickleball": 1.6}
MEAN_TOL = 0.9                 # peak >= 0.9 × mean flight speed (measurement slack)


def check_speed(sport, stroke, mph, distance_m=None, flight_s=None):
    """-> (ok: bool, reason: str). reason is '' when ok."""
    sp = sport.lower(); tab = RANGES.get(sp)
    if tab is None:
        return False, f"unknown sport {sport!r}"
    if mph is None or mph != mph or mph <= 0:
        return False, "no speed"
    lo, hi = tab.get((stroke or "default").lower(), tab["default"])
    if mph < lo:
        return False, f"{mph:.0f} mph below the {sp} {stroke} floor ({lo})"
    if mph > hi:
        return False, f"{mph:.0f} mph above the {sp} {stroke} ceiling ({hi})"
    if distance_m is not None and flight_s is not None:
        if flight_s <= 0 or distance_m <= 0:
            return False, "bad flight measurement"
        mean = distance_m / flight_s * MPH_PER_MS
        if mph < MEAN_TOL * mean:
            return False, f"{mph:.0f} mph is slower than the ball's mean flight speed ({mean:.0f} mph)"
        if mph > MAX_PEAK_OVER_MEAN[sp] * mean:
            return False, f"{mph:.0f} mph is faster than {d_str(distance_m)} in {flight_s:.2f} s allows (mean {mean:.0f} mph)"
    return True, ""


def d_str(d):
    return f"{d:.1f} m"


def speed_label(sport, stroke, mph, distance_m=None, flight_s=None, verified=True, units="mph"):
    """Caption text ('118 MPH' / '190 KM/H') or None when suppressed or unverified."""
    if not verified:
        return None
    ok, _ = check_speed(sport, stroke, mph, distance_m, flight_s)
    if not ok:
        return None
    return f"{mph:.0f} MPH" if units == "mph" else f"{mph * 1.609344:.0f} KM/H"
