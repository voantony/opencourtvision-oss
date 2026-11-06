"""Framing quality signal for the engine: how much of the court the camera saw, and how reliably.

The record screen never refuses imperfect framing, and the engine never refuses to analyse it. Instead each framing
check records a small quality sample, the session keeps them with timestamps, and the engine reads them per point:

    q = frame_quality(H, size, info, moved=..., blurred=...)        # per check (FramingChecker does this)
    track = QualityTrack(); track.add(t, q)                          # per session (FramingSession does this)
    doc = track.to_json()                                            # vamos.framing_quality.v1, saved with the video
    w = for_window(doc, A, B)                                        # per point -> engine
    w["vis_near"], w["vis_far"]       -> robust.framing.point_evidence(..., vis_near=, vis_far=) (side evidence weight)
    w["vision_mode"]                  -> robust.framing.vision_mode: full / partial / audio
    w["uncertainty_scale"]            -> later stages multiply their spread (or divide evidence weight) by this; 1 = as built;
                                         attach_to_points already pulls each point's call confidences toward 0.5 by it

Numbers only: court fractions, line support, drift and sharpness. No pixels, boxes, faces or identities.
"""
import numpy as np

SCHEMA = "vamos.framing_quality.v1"


def frame_quality(H, size, info, moved=False, blurred=False, tracked=True):
    from ..robust.framing import court_visible_fraction
    lines = info.get("lines", {})
    hv = info.get("half_visible", {})
    return dict(court_visible=round(court_visible_fraction(H, size, margin=(0.0, 0.0)), 3),
                vis_near=hv.get("near"), vis_far=hv.get("far"),
                near_baseline=lines.get("near_baseline", {}).get("in_frame"), far_baseline=lines.get("far_baseline", {}).get("in_frame"),
                line_support=info.get("line_support"), roll_deg=info.get("roll_deg"), elevation_deg=info.get("elevation_deg"),
                moved=bool(moved), blurred=bool(blurred), tracked=bool(tracked))


class QualityTrack:
    """Timestamped quality samples for one recording (seconds from recording start)."""

    def __init__(self): self.samples = []

    def add(self, t, q):
        self.samples.append(dict(t=round(float(t), 2), **q))

    def to_json(self):
        return dict(schema=SCHEMA, samples=list(self.samples), summary=summarise_samples(self.samples))


def uncertainty_scale(court_visible, reliable):
    """1.0 for a fully visible, steady court; grows as the seen share of court x reliability drops (capped at 4)."""
    return float(np.clip(1.0 / max(0.25, float(court_visible) * float(reliable)), 1.0, 4.0))


def summarise_samples(samples):
    """Aggregate samples -> engine signal. Empty -> None (engine runs as built, no framing info)."""
    from ..robust.framing import vision_mode
    if not samples: return None
    med = lambda k: float(np.median([s[k] for s in samples if s.get(k) is not None] or [1.0]))
    reliable = float(np.mean([not (s.get("moved") or s.get("blurred")) for s in samples]))
    cv = med("court_visible")
    return dict(court_visible=round(cv, 3), vis_near=round(med("vis_near"), 3), vis_far=round(med("vis_far"), 3),
                reliable=round(reliable, 3), vision_mode=vision_mode(cv, reliable), uncertainty_scale=round(uncertainty_scale(cv, reliable), 3),
                samples=len(samples))


def for_window(doc, t0, t1):
    """Engine signal for [t0, t1] (one point). No sample inside -> the last sample before t0 (framing changes slowly)."""
    if not doc: return None
    ss = doc.get("samples", []) if isinstance(doc, dict) else doc
    win = [s for s in ss if t0 <= s["t"] <= t1]
    if not win:
        before = [s for s in ss if s["t"] < t0]
        win = before[-1:] if before else ss[:1]
    return summarise_samples(win)


CONF_KEYS = ("winner_confidence", "bounce_confidence", "player_id_confidence")


def soften(conf, scale):
    """Pull a call's confidence toward 0.5 (a coin flip) by the framing uncertainty scale: 1 -> unchanged, 2 -> halfway."""
    return round(0.5 + (float(conf) - 0.5) / max(1.0, float(scale)), 4)


def attach_to_points(points, doc):
    """Add point["framing"] (the engine signal for that point's time span) to every point and lower the point's call
    confidences by its uncertainty scale (user-confirmed calls are left alone), so weakly framed points rise in the
    fix-screen queue instead of being refused. Returns the count attached."""
    n = 0
    for p in points:
        w = for_window(doc, float(p.get("A", 0.0)), float(p.get("B", p.get("A", 0.0))))
        if w is None: continue
        p["framing"] = w; n += 1
        for k in CONF_KEYS:
            if k in p and p[k] is not None and not (k == "winner_confidence" and p.get("winner_src") == "user"):
                p[k] = soften(p[k], w["uncertainty_scale"])
    return n
