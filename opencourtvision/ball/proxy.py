"""Proxy evaluation of ball tracks when no ground-truth labels exist yet.

Everything here is a PROXY: it measures agreement with event times (SwingVision shot times, our audio hits) and the
shape of the tracks, not pixel accuracy against labelled ball positions. Use labels/ball/*.csv (ocv-label-ball) for
the real numbers once frames are labelled.

* `align_clips`: a SwingVision "Filtered Points" export is a concatenation of point clips, so its clock is not the raw
  match clock. Each clip is matched to an SV point (monotone dynamic-programming alignment, clips/points may be skipped) and an offset raw = proxy + o,
  chosen so the most SV contacts (Video Time + lag) land on our audio onsets within `tol`. The alignment uses the
  audio onsets, so "detected near an SV shot" and "detected near an audio hit" are not independent references.
* `event_detection_rate`: fraction of reference event times with a ball detection within +-win seconds.
* `post_hit_coverage`: fraction of frames in the 600 ms after each audio hit that carry a detection.
* `false_track_stats`: tracklets in a variant's output that are static, slow/rolling, or off our court, by the same
  rules as the live filter (so for the engine this is a consistency check, for the baselines a real comparison).
"""
import numpy as np

from .live import MIN_SPEED_MPS, STATIC_EXTENT_D, STATIC_MIN_PX, tracklets


def _clip_point_score(a, b, contacts, u, tol, step, search):
    c = np.sort(np.asarray(contacts, float))
    if not len(c) or not len(u): return 0, None
    best, key = (0, None), (0, 0.0)
    for o in np.arange(c[0] - a - search, c[0] - a + search + step / 2, step):
        p = c - o
        p = p[(p >= a - tol) & (p <= b + tol)]
        if not len(p): continue
        r = np.min(np.abs(p[:, None] - u[None, :]), axis=1); hit = r <= tol
        k = (int(hit.sum()), -float(r[hit].sum()))       # most matches, then the smallest residual
        if k > key: key, best = k, (k[0], float(o))
    return best


def align_clips(clips, points, onsets_proxy, tol=0.15, step=0.01, search=8.0, band=(-3, 12), min_matches=2):
    """clips [(start, end)] in proxy seconds, in export order; points [list of SV contact times on the raw clock (lag
    applied)] in point order; onsets_proxy audio hits on the proxy clock. Monotone alignment (dynamic programming,
    clips and points may be skipped since the export can drop or merge points) of clip k to point j with an offset
    raw = proxy + o searched within +-search s of "clip start = the point's first contact"; a pair's score is the
    number of its contacts that land on an audio onset within tol (pairs under min_matches score 0).
    -> list of dict(clip, start, end, point, offset or None, matches, n_sv)."""
    on = np.sort(np.asarray(onsets_proxy, float))
    K, J = len(clips), len(points)
    sc = {}
    for k, (a, b) in enumerate(clips):
        u = on[(on >= a - tol) & (on <= b + tol)]
        for j in range(max(0, k + band[0]), min(J, k + band[1] + 1)):
            m, o = _clip_point_score(a, b, points[j], u, tol, step, search)
            if m >= min_matches: sc[k, j] = (m, o)
    D = np.zeros((K + 1, J + 1)); back = {}
    for k in range(1, K + 1):
        for j in range(1, J + 1):
            opts = [(D[k - 1, j], "skip_clip"), (D[k, j - 1], "skip_point")]
            if (k - 1, j - 1) in sc: opts.append((D[k - 1, j - 1] + sc[k - 1, j - 1][0], "pair"))
            D[k, j], back[k, j] = max(opts, key=lambda t: t[0])
    pair = {}
    k, j = K, J
    while k > 0 and j > 0:
        mv = back[k, j]
        if mv == "pair": pair[k - 1] = j - 1; k -= 1; j -= 1
        elif mv == "skip_clip": k -= 1
        else: j -= 1
    out = []
    for k, (a, b) in enumerate(clips):
        if k in pair:
            j = pair[k]; m, o = sc[k, j]
            c = np.asarray(points[j], float) - o
            out.append(dict(clip=k, start=float(a), end=float(b), point=j, offset=round(o, 3), matches=m, n_sv=int(np.sum((c >= a) & (c <= b)))))
        else:
            out.append(dict(clip=k, start=float(a), end=float(b), point=None, offset=None, matches=0, n_sv=0))
    return out


def sv_times_in_clip(al, sv_times_raw):
    """SV contact times of one aligned clip, on the proxy clock."""
    if al["offset"] is None: return np.zeros(0)
    p = np.asarray(sv_times_raw, float) - al["offset"]
    return p[(p >= al["start"]) & (p <= al["end"])]


def event_detection_rate(pos, t0, fps, events, win=0.150):
    """pos (N, 2) per frame starting at time t0; events: times (same clock). -> (detected, total)."""
    ok = ~np.isnan(np.asarray(pos)[:, 0])
    n = len(ok); hit = 0; tot = 0
    for e in events:
        i0, i1 = int(np.floor((e - win - t0) * fps)), int(np.ceil((e + win - t0) * fps))
        if i1 < 0 or i0 >= n: continue
        tot += 1
        hit += bool(ok[max(i0, 0):min(i1, n - 1) + 1].any())
    return hit, tot


def false_track_stats(pos, fps, gate=None, min_len=5):
    """-> dict(tracks, static, slow, off_court, false, frames_in_tracks, frames_false)."""
    trs = [t for t in tracklets(np.asarray(pos, float)) if len(t.idx) >= min_len]
    st = dict(tracks=len(trs), static=0, slow=0, off_court=0, false=0, frames_in_tracks=0, frames_false=0)
    for tr in trs:
        xy, idx = tr.xy, tr.idx
        d_px = 0.067 * float(np.median(gate.ball_scale(xy))) if gate is not None else 3.0
        extent = float(np.hypot(*(xy.max(0) - xy.min(0))))
        k3 = min(3, len(idx) - 1)
        if gate is not None:                       # ground-plane speed, as in the live filter
            c = gate.ground_speed_points(xy)
            gs = np.hypot(*(c[k3:] - c[:-k3]).T) / (idx[k3:] - idx[:-k3]) * fps
            sp_mps = float(np.median(np.where(np.isfinite(gs), gs, np.inf)))
        else:
            sp_mps = None
        static = extent <= max(STATIC_MIN_PX, STATIC_EXTENT_D * d_px)
        slow = (not static) and sp_mps is not None and sp_mps < MIN_SPEED_MPS
        off = gate is not None and float(np.mean(gate.contains(xy))) < 0.8
        st["static"] += static; st["slow"] += slow; st["off_court"] += off
        bad = static or slow or off
        st["false"] += bad; st["frames_in_tracks"] += len(idx); st["frames_false"] += len(idx) * bad
    return st


def read_track_csv(path, t0, fps, n, scale=(1.0, 1.0)):
    """Baseline track CSV with columns t,x,y (proxy seconds, pixels; empty x = no ball) -> (n, 2) array on our frame grid."""
    import csv
    pos = np.full((n, 2), np.nan)
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("x") or r["x"] in ("nan", "None"): continue
            i = int(round((float(r["t"]) - t0) * fps))
            if 0 <= i < n: pos[i] = (float(r["x"]) * scale[0], float(r["y"]) * scale[1])
    return pos


def post_hit_coverage(pos, t0, fps, hits, span=0.6):
    """Fraction of frames in [hit, hit + span] (the ball has just been struck and must be in flight) that carry a
    detection. -> (frames_with_ball, frames)."""
    ok = ~np.isnan(np.asarray(pos)[:, 0]); n = len(ok); sel = np.zeros(n, bool)
    for h in hits:
        i0, i1 = int(np.ceil((h - t0) * fps)), int(np.floor((h + span - t0) * fps))
        if i1 >= 0 and i0 < n: sel[max(i0, 0):min(i1, n - 1) + 1] = True
    return int((ok & sel).sum()), int(sel.sum())
