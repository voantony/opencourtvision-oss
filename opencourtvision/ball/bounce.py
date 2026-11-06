"""Bounce events from live ball tracks, plus the bounce clean-up ported from tennis-vision (MIT).

Detection (ours): inside a live tracklet, a velocity reversal where image-y velocity turns from down to up at a local
lowest point, with no audio hit within +-80 ms (a reversal with a hit is a contact, not a bounce).

Clean-up, ported from philippdubach/tennis-vision `events/player_tracker.py::mask_ball_track_to_broadcast` (MIT,
Copyright (c) 2026 Philipp Dubach; see THIRD_PARTY_NOTICES.md), adapted to our CourtGate:
  1. court locality: drop a bounce whose ball pixel projects (ground homography; exact for a ball on the ground) more
     than `buffer_m` (2 m, upstream's 200 canvas px) outside the court rectangle. A bounce with no ball position cannot
     be verified and is kept, as upstream does.
  2. temporal dedup: keep the earliest of any bounces closer than `min_sep_s` (~400 ms; the real bounce-to-next-bounce
     minimum in tennis is ~500 ms).
"""
import numpy as np

from .live import CONTACT_S, reversals


def detect_bounces(trs, fps, hits=(), t0=0.0):
    """-> [dict(frame, t, x, y)] for live tracklets."""
    hits = np.asarray(sorted(hits), float)
    out = []
    for tr in trs:
        if not tr.live: continue
        for k in reversals(tr.idx, tr.xy):
            if not (1 <= k < len(tr.idx) - 1): continue
            v_in, v_out = tr.xy[k, 1] - tr.xy[k - 1, 1], tr.xy[k + 1, 1] - tr.xy[k, 1]
            if not (v_in > 0 and v_out < 0): continue
            t = t0 + tr.idx[k] / fps
            if len(hits) and np.min(np.abs(hits - t)) <= CONTACT_S: continue
            out.append(dict(frame=int(tr.idx[k]), t=float(t), x=float(tr.xy[k, 0]), y=float(tr.xy[k, 1])))
    return sorted(out, key=lambda b: b["frame"])


def clean_bounces(bounces, fps, gate=None, min_sep_s=0.4, buffer_m=2.0):
    """tennis-vision clean-up: court locality, then ~400 ms temporal dedup (earliest kept). -> (kept, dropped_count).
    bounces: [dict(frame, x, y, ...)]; x/y may be None/NaN (kept unverified)."""
    on = []
    for b in bounces:
        x, y = b.get("x"), b.get("y")
        if gate is None or x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            on.append(b); continue
        if bool(gate.on_court(np.array([[x, y]]), buffer_m)[0]): on.append(b)
    min_sep = max(1, round(fps * min_sep_s))
    kept, last = [], None
    for b in sorted(on, key=lambda b: b["frame"]):
        if last is None or b["frame"] - last >= min_sep:
            kept.append(b); last = b["frame"]
    return kept, len(bounces) - len(kept)
