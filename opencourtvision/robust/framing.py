"""Partial-court tolerance and the setup framing check (spec in eng notes §2.9.x; UI later).

`court_visible_fraction(H, size)`: share of the court (singles/doubles rectangle plus a run-off margin, metres) whose
image under H falls inside the frame. `vision_mode(visible, reliable)`: full / partial / audio:
* full    (>= FULL_MIN of the court visible and stable): all vision cues as built;
* partial (>= PARTIAL_MIN): cues use the visible part only; thresholds that count active court cells scale with the
  visible share; scoring keeps the score prior and audio; per-side vision evidence only for the visible side;
* audio   (below, or camera unreliable): no vision decision at all; edits and scoring run on audio + score prior.
A point is never dropped because vision is missing.
"""
import numpy as np

FRAME = dict(FULL_MIN=0.8, PARTIAL_MIN=0.35, RELIABLE_MIN=0.5)
COURT_W, COURT_L = 10.97, 23.77


def court_visible_fraction(H, size, margin=(1.0, 2.0), n=24):
    w, h = size
    xs = np.linspace(-margin[0], COURT_W + margin[0], n); ys = np.linspace(-margin[1], COURT_L + margin[1], 2 * n)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    q = np.asarray(H, float) @ P
    u, v = q[0] / q[2], q[1] / q[2]
    inside = (q[2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return float(inside.mean())


def side_visible_fraction(H, size, side, n=24):
    """Visible share of one half: side 'far' (y in [L/2, L]) or 'near' (y in [0, L/2]) in court metres."""
    w, h = size
    ys = np.linspace(COURT_L / 2, COURT_L, n) if side == "far" else np.linspace(0, COURT_L / 2, n)
    X, Y = np.meshgrid(np.linspace(0, COURT_W, n), ys)
    q = np.asarray(H, float) @ np.stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    u, v = q[0] / q[2], q[1] / q[2]
    return float(((q[2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).mean())


def vision_mode(visible, reliable=1.0, p=None):
    p = {**FRAME, **(p or {})}
    if reliable < p["RELIABLE_MIN"] or visible < p["PARTIAL_MIN"]: return "audio"
    return "full" if visible >= p["FULL_MIN"] else "partial"


def court_covered_fraction(H, size, mask, n=24):
    """Share of the court (lines rectangle) whose image under H lands on a True pixel of mask (any resolution)."""
    w, h = size; mh, mw = mask.shape
    X, Y = np.meshgrid(np.linspace(0, COURT_W, n), np.linspace(0, COURT_L, 2 * n))
    q = np.asarray(H, float) @ np.stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    u, v = q[0] / q[2], q[1] / q[2]
    ok = (q[2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    hit = np.zeros(u.shape, bool)
    hit[ok] = mask[(v[ok] * mh / h).astype(int), (u[ok] * mw / w).astype(int)]
    return float(hit.mean())


def framing_check(H, size, offsets_px=None, players_px=None, occluder=None, p=None):
    """Setup framing check (spec §2.9.x): -> list of warnings. Court cut off (which side), far baseline out of frame,
    players' boxes touching the frame edge, phone unstable (offset RMS over the last 2 s), mount in view (occluder:
    the bool mask from robust.occluders.detect_occluders, any resolution)."""
    w, h = size; warn = []
    vis = court_visible_fraction(H, size, margin=(0.0, 0.0))
    if vis < 0.98: warn.append(f"court cut off: {100 * (1 - vis):.0f}% of the court outside the frame")
    far = side_visible_fraction(H, size, "far")
    if far < 0.98: warn.append("far side cut off" + (" (far baseline out of frame)" if far < 0.9 else ""))
    near = side_visible_fraction(H, size, "near")
    if near < 0.98: warn.append("near side cut off")
    if players_px is not None:
        for b in players_px:
            if b[0] <= 2 or b[1] <= 2 or b[2] >= w - 2 or b[3] >= h - 2: warn.append("a player is at the frame edge"); break
    if offsets_px is not None and len(offsets_px) > 2:
        st = np.hypot(*np.diff(np.asarray(offsets_px, float), axis=0).T)
        if np.sqrt((st ** 2).mean()) > 0.35: warn.append("phone unstable: mount it or rest it on something")
    if occluder is not None and occluder.mean() >= 0.004:
        mh, mw = occluder.shape; ys, xs = np.nonzero(occluder)
        edge = min((xs.min(), "left"), (mw - 1 - xs.max(), "right"), (ys.min(), "top"), (mh - 1 - ys.max(), "bottom"))[1]
        cov = court_covered_fraction(H, size, occluder)
        warn.append(f"mount in view at the {edge} edge ({100 * occluder.mean():.0f}% of the frame"
                    + (f", hides {100 * cov:.0f}% of the court" if cov >= 0.01 else "") + "): rotate or slide the phone")
    return warn


def point_evidence(prior, near=None, far=None, vis_near=1.0, vis_far=1.0, min_vis=0.5):
    """Partial-court scoring hook (point winner v1, §2.9): combine the score/audio prior P(A wins) with per-side vision
    evidence (log-odds, + favours A), using a side only when at least min_vis of it is in frame, weighted by how much
    is. With both sides out of view this returns the prior: the point is still scored, never dropped.
    -> (p_a, sides_used)."""
    lo = np.log(prior / (1 - prior)); used = []
    for ev, vis, name in ((near, vis_near, "near"), (far, vis_far, "far")):
        if ev is not None and vis >= min_vis:
            lo += float(ev) * vis; used.append(name)
    return float(1 / (1 + np.exp(-lo))), used
