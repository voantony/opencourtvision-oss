"""Ball engine v2: predicted-window (ROI) search with a ballistic Kalman filter, full-frame TrackNetV3 only to
re-acquire, HSV + TrackNet fusion with the reel camera's per-frame confidence, size gate, live-ball filter, bounces.

Per frame:
  tracking  -> predict with a constant-acceleration Kalman filter (image px; gravity shows up as image-y acceleration),
               search HSV candidates only inside the predicted window (radius grows with misses and speed, widened
               around an audio hit where the ball changes direction), size-gate them, take the one nearest the
               prediction.
  lost      -> after LOST_FRAMES misses (or before the first lock) run full-frame TrackNetV3 on the last 8 frames,
               seed a new track from a temporally consistent chain of its blobs (back-filling those frames), at most
               once per 8 frames.
At the end: per-frame HSV and TrackNet positions -> `reel.ball.frame_confidence` (1.0 agree / 0.7 TN / 0.6 HSV / 0.3
disagree / 0) so the 9:16 camera keeps its scheme; then the live-ball filter blanks every non-live tracklet, and
bounces are detected and cleaned (tennis-vision rules).
"""
import time
from collections import Counter, deque

import numpy as np

from ..reel.ball import frame_confidence
from . import hsv as H
from .bounce import clean_bounces, detect_bounces
from .live import filter_live

LOST_FRAMES = 4
ROI_BASE, ROI_GROW, ROI_MAX = 28, 10, 160
HIT_WIDEN = 2.0


class Kalman:
    """6-state (x, y, vx, vy, ax, ay) constant-acceleration filter, dt = 1 frame."""

    def __init__(self, xy, q=4.0, r=2.0):
        self.x = np.array([xy[0], xy[1], 0, 0, 0, 0], float)
        self.P = np.diag([r * r, r * r, 400, 400, 25, 25]).astype(float)
        F = np.eye(6)
        F[0, 2] = F[1, 3] = 1; F[2, 4] = F[3, 5] = 1; F[0, 4] = F[1, 5] = 0.5
        self.F = F
        self.Q = np.diag([1, 1, q, q, q / 4, q / 4]).astype(float)
        self.Hm = np.zeros((2, 6)); self.Hm[0, 0] = self.Hm[1, 1] = 1
        self.R = np.eye(2) * r * r

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        y = np.asarray(z, float) - self.Hm @ self.x
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.Hm) @ self.P

    @property
    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))


def chain(blobs_per_frame, max_step=60.0, min_len=3):
    """Longest temporally consistent chain through per-frame blob lists [(x, y, score)] -> {k: (x, y)}."""
    best = {}
    n = len(blobs_per_frame)
    for s in range(n):
        for b in blobs_per_frame[s]:
            cur = {s: (b[0], b[1])}; last = np.array(b[:2]); lk = s
            for k in range(s + 1, n):
                c = [q for q in blobs_per_frame[k] if np.hypot(q[0] - last[0], q[1] - last[1]) <= max_step * (k - lk)]
                if not c: continue
                q = min(c, key=lambda q: np.hypot(q[0] - last[0], q[1] - last[1]))
                cur[k] = (q[0], q[1]); last = np.array(q[:2]); lk = k
            if len(cur) > len(best): best = cur
    return best if len(best) >= min_len else {}


class BallEngine:
    def __init__(self, gate=None, tracknet=None, overlays=(), fps=30.0, t0=0.0, hits=(), players=None,
                 use_size_gate=True, use_live_filter=True, reacquire="tracknet"):
        self.gate, self.tn, self.fps, self.t0 = gate, tracknet, float(fps), float(t0)
        self.overlays = list(overlays)
        self.hits = np.asarray(sorted(hits), float)
        self.players = players
        self.use_size_gate, self.use_live_filter, self.reacquire = use_size_gate, use_live_filter, reacquire
        self.stats = dict(frames=0, roi_frames=0, full_hsv_frames=0, tn_windows=0, size_rejected=0, seeds=0, roi_px=0, frame_px=0)

    def _near_hit(self, i):
        if not len(self.hits): return False
        return bool(np.min(np.abs(self.hits - (self.t0 + i / self.fps))) <= 0.08)

    def _gate_size(self, cands):
        if self.gate is None or not self.use_size_gate or not cands: return cands
        pts = np.array([c[:2] for c in cands]); ok = self.gate.size_ok(pts, [c[2] for c in cands])
        self.stats["size_rejected"] += int((~ok).sum())
        return [c for c, k in zip(cands, ok) if k]

    def run(self, frames):
        """frames: iterable of BGR frames (same size). Returns dict with per-frame arrays and events."""
        T0 = time.perf_counter()
        buf = deque(maxlen=8); greys = deque(maxlen=3)
        hsv_pos, tn_pos, hsv_d = [], [], []
        kf, miss, last_tn = None, 99, -99
        full = deque(maxlen=3)
        mask = None
        bg_set = self.tn is None or getattr(self.tn, "bg", None) is not None
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            if mask is None: mask = H.overlay_mask(h, w, self.overlays)
            self.stats["frames"] += 1; self.stats["frame_px"] += h * w
            g = H.grey(f); greys.append(g); buf.append(f)
            prev2 = greys[0] if len(greys) == 3 else None
            hsv_pos.append((np.nan, np.nan)); tn_pos.append((np.nan, np.nan)); hsv_d.append(np.nan)
            got = False
            if kf is not None and miss <= LOST_FRAMES:
                p = kf.predict()
                r = min(ROI_MAX, ROI_BASE + ROI_GROW * miss + 0.5 * kf.speed) * (HIT_WIDEN if self._near_hit(i) else 1.0)
                roi = (p[0] - r, p[1] - r, p[0] + r, p[1] + r)
                self.stats["roi_frames"] += 1; self.stats["roi_px"] += int(min(2 * r, w) * min(2 * r, h))
                c = self._gate_size(H.candidates(f, prev2, roi, mask))
                if c:
                    best = min(c, key=lambda q: np.hypot(q[0] - p[0], q[1] - p[1]))
                    kf.update(best[:2]); hsv_pos[i] = best[:2]; hsv_d[i] = best[2]; miss = 0; got = True
                else:
                    miss += 1
            if not got and (kf is None or miss > LOST_FRAMES):
                kf = None
                if self.reacquire == "tracknet" and self.tn is not None and len(buf) == 8 and i - last_tn >= 8:
                    if not bg_set: self.tn.set_background(list(buf)); bg_set = True
                    blobs = self.tn.detect(list(buf)); last_tn = i; self.stats["tn_windows"] += 1
                    # blobs inside overlays are drawn graphics, not the ball
                    blobs = [[b for b in bl if mask[min(h - 1, int(b[1])), min(w - 1, int(b[0]))] > 0] for bl in blobs]
                    ch = chain(blobs)
                    if ch:
                        base = i - 7
                        ks = sorted(ch)
                        for k in ks: tn_pos[base + k] = ch[k]
                        kf = Kalman(ch[ks[0]])
                        for prev, k in zip(ks, ks[1:]):
                            for _ in range(k - prev): kf.predict()
                            kf.update(ch[k])
                        for _ in range(7 - max(ch)): kf.predict()
                        miss = 7 - max(ch); self.stats["seeds"] += 1
                elif self.reacquire == "hsv_full":
                    # full-frame HSV, seeded only from a temporally consistent 3-frame chain of candidates
                    self.stats["full_hsv_frames"] += 1
                    full.append([c[:2] for c in self._gate_size(H.candidates(f, prev2, None, mask))])
                    if len(full) == 3:
                        ch = chain(list(full), max_step=40.0, min_len=3)
                        if ch:
                            kf = Kalman(ch[0])
                            for k in (1, 2): kf.predict(); kf.update(ch[k])
                            for k in (0, 1, 2): hsv_pos[i - 2 + k] = ch[k]
                            miss = 0; self.stats["seeds"] += 1; full.clear()
            if got: full.clear()
        hp, tp = np.array(hsv_pos, float).reshape(-1, 2), np.array(tn_pos, float).reshape(-1, 2)
        conf = frame_confidence(hp, tp)
        fused = np.where(~np.isnan(tp[:, :1]), tp, hp)             # TrackNet first, HSV fills (as the reel does)
        sizes = np.where(np.isnan(tp[:, 0]), np.array(hsv_d, float), np.nan)   # TrackNet heatmap blobs carry no ball size
        if self.use_live_filter:
            live, trs = filter_live(fused, self.fps, self.gate, self.hits, self.t0, self.players, sizes=sizes)
        else:
            from .live import tracklets
            live, trs = fused, tracklets(fused)
        conf = np.where(np.isnan(live[:, 0]), 0.0, conf)
        bounces, dropped = clean_bounces(detect_bounces(trs, self.fps, self.hits, self.t0), self.fps, self.gate)
        el = time.perf_counter() - T0
        st = dict(self.stats, seconds=round(el, 3), fps=round(self.stats["frames"] / el, 2) if el else None,
                  roi_pixel_frac=round(self.stats["roi_px"] / max(1, self.stats["frame_px"]), 4),
                  tn_frame_frac=round(8 * self.stats["tn_windows"] / max(1, self.stats["frames"]), 3),
                  bounces_dropped=dropped,
                  tracklets_live=sum(t.live for t in trs if len(t.idx) >= 5),
                  rejected=dict(Counter(r for t in trs if len(t.idx) >= 5 and not t.live for r in t.reasons)))
        return dict(pos=live, raw=fused, sizes=sizes, hsv=hp, tracknet=tp, conf=conf, tracklets=trs, bounces=bounces, stats=st)
