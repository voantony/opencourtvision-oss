"""Pose stage entry point: keypoints (+ audio hits, court) -> strokes with mechanics, contact events, footwork, syllables."""
import numpy as np

from ..events import Event, verify
from ..labels.schema import ShotLabel, court_position
from . import metrics as M
from . import strokes as S
from .skeleton import ground_point


def _syllables():
    """Movement syllables are a research / dev-build module (Legal 05:55: not packaged into Vamos US). None if absent."""
    try:
        from . import syllables
    except ImportError:
        return None
    return syllables


def analyze(kp, fps, hit_times=None, court=None, hand="right", k=12, player="near", do_syllables=True, min_confidence=0.6):
    st = [M.stroke_metrics(kp, fps, s, court, hand) for s in S.strokes(kp, fps, hit_times, hand)]
    g = ground_point(kp); ev = []
    for s in st:
        pos = None
        if court is not None:
            i = int(np.clip(round(s["t"] * fps), 0, len(kp) - 1))
            if np.isfinite(g[i]).all():
                cx, cy = court.to_court(g[i]); pos = court_position(cx, cy, court.width, court.length)
        kind = s["kind"]
        lab = ShotLabel(position=pos, hand=hand, shot_type="serve" if kind == "serve" else None,
                        side=kind if kind in ("forehand", "backhand") else None)
        evid = [f"pose: {x}" for x in s["evidence"]]
        if hit_times is not None: evid.insert(0, f"audio: ball hit at {s['t']:.3f} s")
        ev.append(Event("contact", s["t"], player, lab, s["confidence"], evid))
    verify(ev, min_confidence=min_confidence)
    out = dict(fps=fps, hand=hand, strokes=st, events=[e.to_dict() for e in ev], footwork=M.footwork(kp, fps, court))
    SY = _syllables() if do_syllables else None
    if SY is not None and len(kp) > fps * 3:
        sy = SY.fit(kp, fps, k=k)
        out["syllables"] = dict(k=sy["k"], instances=sy["instances"])
    return out


def stroke_rows(result, session, group=None):
    """Flatten strokes for compare.compare(): one row per stroke with session/group tags."""
    return [dict(s, session=session, group=group if group is not None else session) for s in result["strokes"]]
