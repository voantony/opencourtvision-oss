"""Efficiency tricks for the own ball net on limited hardware (each one is A/B-benchmarked against the plain full-frame
pass by tools/ball_tracker/eval_heldout.py and kept only if it wins on accuracy and ms/frame).

  plain    every frame, whole 768x432 frame, batch 8.
  court    homography-constrained search: the net runs only on the court ROI (court + run-off projected through the
           court homography, extended upward for lobs), same batch.
  motion   temporal sparsity: per frame, the 3-frame grey difference inside the court ROI is tiled (64 px tiles); the
           net runs on the bounding box of the moving tiles only, and frames with no moving tile are skipped.
  track    physics-predicted ROI tracking: once two recent detections exist, a parabolic (constant-acceleration)
           predictor places a small crop (192x128) where the ball should be; full court-ROI re-detect when the crop
           misses twice, near an audio contact (direction change expected) or every `redetect_every` frames.
  fp16     inference under float16 autocast (orthogonal to the above).
All return per-frame best blobs in 768-wide px plus the seconds spent and the share of pixels the net processed.
"""
import time

import numpy as np

from .heatnet import decode_heatmap


def _round8(v, lo, hi):
    return int(np.clip(int(v) // 8 * 8, lo, hi))


def court_roi(Hc2i_net, size, margin=(3.5, 5.5), lift=0.45):
    """(x0, y0, x1, y1) in network px (multiples of 8) covering the court + margin, extended upward by `lift` x its
    height (balls above the far court). Hc2i_net: court metres -> network px. None -> whole frame."""
    W, H = size
    if Hc2i_net is None:
        return (0, 0, W, H)
    mx, my = margin
    pts = []
    for x in (-mx, 10.97 + mx):
        for y in (0.0, 23.77 + my):
            p = Hc2i_net @ np.array([x, y, 1.0])
            if p[2] > 0:
                pts.append(p[:2] / p[2])
    if len(pts) < 3:
        return (0, 0, W, H)
    pts = np.array(pts)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    y0 -= lift * (y1 - y0)
    x0, y0 = _round8(x0, 0, W - 64), _round8(y0, 0, H - 64)
    x1, y1 = _round8(x1 + 7, x0 + 64, W), _round8(y1 + 7, y0 + 64, H)
    return (x0, y0, x1, y1)


class FastRunner:
    def __init__(self, tracker, fp16=False):
        self.t = tracker
        self.fp16 = fp16 and tracker.device in ("mps", "cuda")
        self.spent = 0.0; self.px = 0; self.calls = 0

    def _net(self, x):
        torch = self.t.torch
        st = time.perf_counter()
        with torch.no_grad(), torch.autocast(self.t.device, dtype=torch.float16, enabled=self.fp16):
            pr = torch.sigmoid(self.t.model(torch.from_numpy(np.ascontiguousarray(x)).to(self.t.device)).float()).cpu().numpy()[:, 0]
        self.spent += time.perf_counter() - st
        self.px += int(np.prod(x.shape[2:])) * x.shape[0]; self.calls += 1
        return pr

    @staticmethod
    def _stack(small, i, box):
        n = len(small); x0, y0, x1, y1 = box
        return np.concatenate([small[j][:, y0:y1, x0:x1] for j in (max(0, i - 1), i, min(n - 1, i + 1))], 0)

    def _best(self, pr, box):
        c = decode_heatmap(pr, self.t.th, 1)
        return (c[0][0] + box[0], c[0][1] + box[1], c[0][2]) if c else None

    def run(self, small, mode="plain", roi=None, hits_idx=(), batch=8, redetect_every=30):
        """small: list of (9? no: 3, H, W) float32 RGB 0..1 frames at the network size. -> (pos (n, 2) NaN, stats)."""
        n = len(small); H, W = small[0].shape[1:]
        full = (0, 0, W, H)
        roi = roi or full
        pos = np.full((n, 2), np.nan)
        self.spent = 0.0; self.px = 0; self.calls = 0
        t_all = time.perf_counter()
        if mode in ("plain", "court"):
            box = full if mode == "plain" else roi
            for s in range(0, n, batch):
                idx = list(range(s, min(n, s + batch)))
                pr = self._net(np.stack([self._stack(small, i, box) for i in idx]))
                for i, hm in zip(idx, pr):
                    b = self._best(hm, box)
                    if b: pos[i] = b[:2]
        elif mode == "motion":
            x0, y0, x1, y1 = roi; T = 64
            for i in range(n):
                a, c = small[max(0, i - 1)][:, y0:y1, x0:x1].mean(0), small[min(n - 1, i + 1)][:, y0:y1, x0:x1].mean(0)
                d = np.abs(c - a) > 0.06
                hh, ww = d.shape
                ty, tx = np.nonzero(d[:hh // T * T, :ww // T * T].reshape(hh // T, T, ww // T, T).sum((1, 3)) >= 2)
                if not len(ty):
                    continue                                   # nothing moves in the court: skip the frame
                bx = (x0 + tx.min() * T, y0 + ty.min() * T, min(x1, x0 + (tx.max() + 1) * T), min(y1, y0 + (ty.max() + 1) * T))
                b = self._best(self._net(self._stack(small, i, bx)[None])[0], bx)
                if b: pos[i] = b[:2]
        elif mode == "track":
            cw, ch = 192, 128
            hits_idx = set(int(h) + k for h in hits_idx for k in range(-3, 4))
            miss, last_full = 99, -999
            for i in range(n):
                obs = [j for j in range(max(0, i - 4), i) if np.isfinite(pos[j, 0])]
                locked = len(obs) >= 2 and miss <= 1 and i not in hits_idx and i - last_full < redetect_every
                if locked:
                    if len(obs) >= 3:                          # parabolic (constant-acceleration) predictor
                        tt = np.array(obs[-3:], float); cx = np.polyfit(tt, pos[obs[-3:], 0], 2); cy = np.polyfit(tt, pos[obs[-3:], 1], 2)
                        px, py = np.polyval(cx, i), np.polyval(cy, i)
                    else:
                        j0, j1 = obs[-2:]; v = (pos[j1] - pos[j0]) / (j1 - j0); px, py = pos[j1] + v * (i - j1)
                    bx0 = _round8(px - cw / 2, 0, W - cw); by0 = _round8(py - ch / 2, 0, H - ch)
                    box = (bx0, by0, bx0 + cw, by0 + ch)
                else:
                    box = roi; last_full = i
                b = self._best(self._net(self._stack(small, i, box)[None])[0], box)
                if b:
                    pos[i] = b[:2]; miss = 0
                else:
                    miss += 1
        else:
            raise ValueError(mode)
        wall = time.perf_counter() - t_all
        return pos, dict(net_s=self.spent, wall_s=wall, px_share=self.px / float(n * H * W), calls=self.calls)
