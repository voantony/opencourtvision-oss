"""Ball physics filter (own code, numpy only).

A ball in flight between two contacts (racket, ground, wall) follows a smooth arc. In the image that arc is close to a
low-order polynomial in time over the <= ~1.5 s spans we see (gravity shows up as image-y acceleration, perspective
bends it a little, hence quadratic for short arcs and cubic for long ones). So:

  segment_arcs  greedy incremental fit over per-frame detections: a point that fits the running arc joins it; a single
                point off the arc whose successor is back on it is an outlier; two in a row start a new arc (the break is
                an event). Arcs that are too short, static (ball lying on court) or too slow are dropped.
  densify       fills frames inside an arc span from the fit (never extrapolates past its ends) -> pseudo-labels with a
                weight (1.0 observed inlier, FILL_W interpolated) and a cleaner per-frame track.
  events        consecutive arcs joined within a few frames: direction change -> 'hit' (near a player box, or a depth
                reversal), 'bounce' (sharp drop of image-y velocity with the horizontal direction kept: the ground pushes
                the ball up), else 'turn'.
  filter_track  all of the above in one call, returning the cleaned track, its confidence, arcs, events.

Coordinates are image px of whatever frame size the detections came in; thresholds scale with `scale` (frame width /
1280) so the same defaults work at 768, 1280 or 3840 px wide.
"""
import math

import numpy as np

FILL_W = 0.6            # weight of an interpolated (filled) frame as a pseudo-label / track confidence
MIN_PTS = 4             # observed points for an arc to count
MAX_GAP = 5             # frames without a detection inside one arc
TOL_PX = 3.0            # base fit tolerance at 1280 px wide
TOL_SPEED = 0.12        # extra tolerance per px/frame of ball speed (blur, timing jitter)
MAX_STEP = 90.0         # px/frame at 1280 wide: faster than any ball we film at 30 fps
STATIC_PX = 6.0         # arc extent below this over its span = ball lying still
MIN_SPEED = 1.0         # median px/frame at 1280 wide
TURN_DEG = 25.0         # direction change that makes an event
DVY_FRAC = 0.35         # bounce: image-y velocity drops by >= this share of the incoming speed


class Arc:
    """Polynomial fit x(i), y(i) over frame indices idx (observed inliers)."""

    def __init__(self, idx, xy):
        self.idx = np.asarray(idx, int)
        self.xy = np.asarray(xy, float)
        self.i0, self.i1 = int(self.idx[0]), int(self.idx[-1])
        self.deg = 1 if len(self.idx) < 5 else (2 if len(self.idx) < 14 else 3)
        t = self.idx - self.i0
        self.cx = np.polyfit(t, self.xy[:, 0], self.deg)
        self.cy = np.polyfit(t, self.xy[:, 1], self.deg)
        r = self.xy - self.at(self.idx)
        self.res = np.hypot(r[:, 0], r[:, 1])
        self.rmse = float(np.sqrt(np.mean(self.res ** 2)))

    def at(self, i):
        t = np.asarray(i, float) - self.i0
        return np.stack([np.polyval(self.cx, t), np.polyval(self.cy, t)], -1)

    def vel(self, i):
        t = float(i) - self.i0
        return np.array([np.polyval(np.polyder(self.cx), t), np.polyval(np.polyder(self.cy), t)])

    @property
    def speed(self):
        return float(np.median(np.hypot(*np.diff(self.at(np.arange(self.i0, self.i1 + 1)), axis=0).T))) if self.i1 > self.i0 else 0.0

    @property
    def extent(self):
        return float(np.hypot(*(self.xy.max(0) - self.xy.min(0))))


def _tol(v, scale):
    return TOL_PX * scale + TOL_SPEED * float(np.hypot(*v))


def segment_arcs(pos, conf=None, min_conf=0.0, scale=1.0, min_pts=MIN_PTS, max_gap=MAX_GAP):
    """pos (T, 2) with NaN where no detection -> (arcs, outlier_frames)."""
    pos = np.asarray(pos, float)
    ok = np.isfinite(pos[:, 0])
    if conf is not None:
        ok &= np.asarray(conf, float) >= min_conf
    pts = np.nonzero(ok)[0]
    arcs, outliers, cur = [], [], []
    max_step = MAX_STEP * scale

    def close(c):
        if len(c) < min_pts:
            outliers.extend(c); return
        a = Arc(c, pos[c])
        tol = _tol((a.extent / max(1, a.i1 - a.i0), 0), scale)
        keep = a.res <= 2.5 * tol
        if keep.sum() >= min_pts and not keep.all():
            outliers.extend(np.asarray(c)[~keep].tolist()); a = Arc(np.asarray(c)[keep], pos[np.asarray(c)[keep]])
        if a.rmse > 1.5 * tol or a.extent < STATIC_PX * scale or a.speed < MIN_SPEED * scale:
            outliers.extend(a.idx.tolist()); return
        arcs.append(a)

    k = 0
    while k < len(pts):
        i = int(pts[k])
        if cur and i - cur[-1] > max_gap:
            close(cur); cur = [i]; k += 1; continue
        jump = bool(cur) and np.hypot(*(pos[i] - pos[cur[-1]])) > max_step * (i - cur[-1])
        if not jump and len(cur) < 3:
            cur.append(i); k += 1; continue
        if not jump:
            a = Arc(cur[-12:], pos[cur[-12:]])
            if float(np.hypot(*(a.at(i) - pos[i]))) <= _tol(a.vel(cur[-1]), scale):
                cur.append(i); k += 1; continue
        # off the arc (or an impossible jump): a lone outlier if the next detection continues the arc, else an event
        if k + 1 < len(pts) and pts[k + 1] - cur[-1] <= max_gap:
            j = int(pts[k + 1])
            if len(cur) >= 3:
                a = Arc(cur[-12:], pos[cur[-12:]])
                back = float(np.hypot(*(a.at(j) - pos[j]))) <= _tol(a.vel(cur[-1]), scale)
            else:
                back = np.hypot(*(pos[j] - pos[cur[-1]])) <= max_step * (j - cur[-1]) and np.hypot(*(pos[j] - pos[i])) > max_step * (j - i)
            if back:
                outliers.append(i); k += 1; continue
        last = cur[-1]
        close(cur)
        cur = [last, i] if (i - last <= 2 and not jump) else [i]
        k += 1
    if cur:
        close(cur)
    return arcs, sorted(set(outliers) - {int(x) for a in arcs for x in a.idx})


def densify(arcs, T, max_fill=4):
    """-> pos (T, 2) NaN-filled, w (T,): 1.0 at observed inliers, FILL_W at frames filled from the arc fit when the
    nearest observed inlier is <= max_fill frames away and the frame lies inside the arc span."""
    pos = np.full((T, 2), np.nan); w = np.zeros(T)
    for a in arcs:
        span = np.arange(a.i0, min(a.i1, T - 1) + 1)
        if not len(span): continue
        d = np.min(np.abs(span[:, None] - a.idx[None, :]), axis=1)
        fill = span[(d <= max_fill) & (w[span] < FILL_W)]
        pos[fill] = a.at(fill); w[fill] = FILL_W
        obs = a.idx[a.idx < T]
        pos[obs] = a.xy[a.idx < T]; w[obs] = 1.0
    return pos, w


def _near_box(xy, boxes):
    """boxes: [(x0, y0, x1, y1)] -> True when xy is where a racket can be: box widened by 0.6 w, from 0.4 h above the
    top down to the bottom."""
    for b in boxes or ():
        x0, y0, x1, y1 = b[:4]
        w, h = x1 - x0, y1 - y0
        if x0 - 0.6 * w <= xy[0] <= x1 + 0.6 * w and y0 - 0.4 * h <= xy[1] <= y1:
            return True
    return False


def events(arcs, fps, boxes_at=None, y_net=None, max_join=6, scale=1.0):
    """Events between consecutive arcs. boxes_at(i) -> player boxes at frame i (same px) or None.
    y_net: image y of the net line (optional; a depth reversal below it is a near-side hit)."""
    out = []
    arcs = sorted(arcs, key=lambda a: a.i0)
    for a, b in zip(arcs, arcs[1:]):
        if b.i0 - a.i1 > max_join:
            continue
        ie = a.i1 if b.i0 <= a.i1 else int(round((a.i1 + b.i0) / 2))
        va, vb = a.vel(a.i1), b.vel(b.i0)
        sa, sb = float(np.hypot(*va)), float(np.hypot(*vb))
        if sa < 0.5 * scale or sb < 0.5 * scale:
            continue
        turn = math.degrees(math.acos(float(np.clip(np.dot(va, vb) / (sa * sb), -1, 1))))
        dvy = float(vb[1] - va[1])
        if turn < TURN_DEG and abs(dvy) < DVY_FRAC * sa:
            continue
        xy = a.at(ie) if b.i0 <= a.i1 else 0.5 * (a.at(a.i1) + b.at(b.i0))
        near = _near_box(xy, boxes_at(ie) if boxes_at else None)
        depth_rev = va[1] * vb[1] < 0
        x_keep = va[0] * vb[0] >= 0 or min(abs(va[0]), abs(vb[0])) < 0.5 * scale
        if near:
            kind = "hit"
        elif depth_rev and va[1] < 0 < vb[1]:
            kind = "hit"                                  # going away -> coming back: a far-side contact
        elif depth_rev and va[1] > 0 > vb[1] and y_net is not None and xy[1] > y_net and sb > 1.3 * sa:
            kind = "hit"                                  # coming in -> going away, faster: a near-side contact
        elif dvy < -DVY_FRAC * sa and x_keep:
            kind = "bounce"
        else:
            kind = "turn"
        out.append(dict(i=int(ie), t=float(ie / fps), x=float(xy[0]), y=float(xy[1]), kind=kind, turn_deg=round(turn, 1),
                        dvy=round(dvy, 2), near_player=bool(near)))
    return out


def filter_track(pos, fps, conf=None, min_conf=0.0, boxes_at=None, y_net=None, frame_w=1280, max_fill=4, arc_ok=None):
    """One call: arcs -> cleaned/filled track + events. -> dict(pos, conf, arcs, events, outliers, stats).
    arc_ok(arc) -> bool: optional extra check per arc (e.g. live-ball speed / on-court through the court homography)."""
    pos = np.asarray(pos, float)
    scale = frame_w / 1280.0
    arcs, outl = segment_arcs(pos, conf, min_conf, scale=scale)
    if arc_ok is not None:
        bad = [a for a in arcs if not arc_ok(a)]
        arcs = [a for a in arcs if arc_ok(a)]
        outl = sorted(set(outl) | {int(i) for a in bad for i in a.idx})
    p, w = densify(arcs, len(pos), max_fill=max_fill)
    ev = events(arcs, fps, boxes_at, y_net, scale=scale)
    n_in = int(np.isfinite(pos[:, 0]).sum())
    stats = dict(frames=len(pos), detections=n_in, inliers=int((w == 1.0).sum()), filled=int((w == FILL_W).sum()),
                 outliers=len(outl), arcs=len(arcs), arc_share=float((w == 1.0).sum() / n_in) if n_in else 0.0,
                 bounces=sum(e["kind"] == "bounce" for e in ev), hits=sum(e["kind"] == "hit" for e in ev))
    return dict(pos=p, conf=w, arcs=arcs, events=ev, outliers=outl, stats=stats)


# ----------------------------------------------------------------------------------------------------------- opts (19 Oct)
def reject_static(pos, conf=None, min_conf=0.0, window=15, max_extent=STATIC_PX, scale=1.0):
    """Hard-negative gate: zero out detections that barely move over `window` frames (spare balls, shoes, line glints).

    Keeps any detection whose neighbourhood span exceeds max_extent*scale, or that has fewer than 3 peers in-window.
    """
    pos = np.asarray(pos, float).copy()
    T = len(pos)
    ok = np.isfinite(pos[:, 0])
    if conf is not None:
        ok &= np.asarray(conf, float) >= min_conf
    thr = max_extent * scale
    for i in np.where(ok)[0]:
        lo, hi = max(0, i - window), min(T, i + window + 1)
        pts = pos[lo:hi][np.isfinite(pos[lo:hi, 0])]
        if len(pts) < 3:
            continue
        if float(np.ptp(pts[:, 0])) <= thr and float(np.ptp(pts[:, 1])) <= thr:
            pos[i] = np.nan
    return pos


def segment_arcs_audio(pos, conf=None, min_conf=0.0, max_gap=MAX_GAP, scale=1.0, audio_breaks=None):
    """Like segment_arcs, but frames in `audio_breaks` (iterable of ints) always start a new arc.

    Hits and padel glass rebounds break smooth motion; audio onsets mark those moments so the 2D filter may turn.
    """
    breaks = set(int(i) for i in (audio_breaks or ()) if i is not None)
    pos = np.asarray(pos, float)
    ok = np.isfinite(pos[:, 0])
    if conf is not None:
        ok &= np.asarray(conf, float) >= min_conf
    pts = np.where(ok)[0]
    if len(pts) < MIN_PTS:
        return [], list(map(int, pts))
    max_step = MAX_STEP * scale
    arcs, outliers, cur = [], [], []

    def close(c):
        if len(c) >= MIN_PTS:
            a = Arc(c, pos[c])
            extent = float(np.hypot(np.ptp(a.xy[:, 0]), np.ptp(a.xy[:, 1])))
            if extent >= STATIC_PX * scale and float(np.median(np.linalg.norm(np.diff(a.xy, axis=0), axis=1))) >= MIN_SPEED * scale:
                arcs.append(a)
            else:
                outliers.extend(map(int, c))
        else:
            outliers.extend(map(int, c))

    k = 0
    while k < len(pts):
        i = int(pts[k])
        if not cur:
            cur = [i]; k += 1; continue
        gap = i - cur[-1]
        jump = gap > max_gap or np.hypot(*(pos[i] - pos[cur[-1]])) > max_step * max(gap, 1)
        force = i in breaks or cur[-1] in breaks
        if force:
            close(cur)
            cur = [i]; k += 1; continue
        if not jump and len(cur) < 3:
            cur.append(i); k += 1; continue
        if not jump:
            a = Arc(cur[-12:], pos[cur[-12:]])
            if float(np.hypot(*(a.at(i) - pos[i]))) <= _tol(a.vel(cur[-1]), scale):
                cur.append(i); k += 1; continue
        if k + 1 < len(pts) and pts[k + 1] - cur[-1] <= max_gap:
            j = int(pts[k + 1])
            if len(cur) >= 3:
                a = Arc(cur[-12:], pos[cur[-12:]])
                back = float(np.hypot(*(a.at(j) - pos[j]))) <= _tol(a.vel(cur[-1]), scale)
            else:
                back = np.hypot(*(pos[j] - pos[cur[-1]])) <= max_step * (j - cur[-1]) and np.hypot(*(pos[j] - pos[i])) > max_step * (j - i)
            if back:
                outliers.append(i); k += 1; continue
        last = cur[-1]
        close(cur)
        cur = [last, i] if (i - last <= 2 and not jump) else [i]
        k += 1
    if cur:
        close(cur)
    return arcs, sorted(set(outliers) - {int(x) for a in arcs for x in a.idx})


def filter_track_opt(pos, fps, conf=None, min_conf=0.0, boxes_at=None, y_net=None, frame_w=1280,
                     max_fill=4, arc_ok=None, audio_breaks=None, hardneg_static=False):
    """filter_track + optional static hard-negative reject and audio-anchored arc breaks (US-safe, image-space)."""
    pos = np.asarray(pos, float)
    scale = frame_w / 1280.0
    if hardneg_static:
        pos = reject_static(pos, conf, min_conf, scale=scale)
    if audio_breaks:
        arcs, outl = segment_arcs_audio(pos, conf, min_conf, scale=scale, audio_breaks=audio_breaks)
    else:
        arcs, outl = segment_arcs(pos, conf, min_conf, scale=scale)
    if arc_ok is not None:
        bad = [a for a in arcs if not arc_ok(a)]
        arcs = [a for a in arcs if arc_ok(a)]
        outl = sorted(set(outl) | {int(i) for a in bad for i in a.idx})
    p, w = densify(arcs, len(pos), max_fill=max_fill)
    ev = events(arcs, fps, boxes_at, y_net, scale=scale)
    n_in = int(np.isfinite(pos[:, 0]).sum())
    stats = dict(frames=len(pos), detections=n_in, inliers=int((w == 1.0).sum()), filled=int((w == FILL_W).sum()),
                 outliers=len(outl), arcs=len(arcs), arc_share=float((w == 1.0).sum() / n_in) if n_in else 0.0,
                 bounces=sum(e["kind"] == "bounce" for e in ev), hits=sum(e["kind"] == "hit" for e in ev),
                 audio_breaks=len(audio_breaks or ()), hardneg_static=bool(hardneg_static))
    return dict(pos=p, conf=w, arcs=arcs, events=ev, outliers=outl, stats=stats)
