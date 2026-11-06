"""Rule-based fusion of ball candidates from several detectors (ball tracker v2, research).

Each detector gives (T, K, 3) [u, v, peak] candidates per frame. Candidates from different detectors within d_px are
the same blob; its fused peak is a weighted noisy-OR of the detectors' peaks, so agreement raises confidence and a blob
seen by one detector only keeps (a weighted share of) its own peak. The physics tracker then picks among the fused
candidates. Also returns an agreement count per fused candidate (used by the learned scorer and the cascade).
"""
import numpy as np


def fuse(cand_list, weights=None, d_px=8.0, k_out=8):
    """-> (fused (T, k_out, 3), agree (T, k_out) number of detectors that saw each fused blob)."""
    weights = weights or [1.0] * len(cand_list)
    T = min(len(c) for c in cand_list)
    out = np.full((T, k_out, 3), np.nan); agree = np.zeros((T, k_out), int)
    for i in range(T):
        blobs = []          # [x, y, sum_w, 1 - prod(1 - w p), set(detectors)]
        for di, (c, w) in enumerate(zip(cand_list, weights)):
            for r in c[i]:
                if not np.isfinite(r[0]):
                    continue
                p = float(np.clip(r[2], 0, 1)) * w
                best = None
                for b in blobs:
                    if di not in b[4] and np.hypot(b[0] - r[0], b[1] - r[1]) <= d_px and (best is None or np.hypot(b[0] - r[0], b[1] - r[1]) < np.hypot(best[0] - r[0], best[1] - r[1])):
                        best = b
                if best is None:
                    blobs.append([float(r[0]), float(r[1]), p, p, {di}])
                else:
                    s = best[2] + p
                    if s > 0:
                        best[0] = (best[0] * best[2] + r[0] * p) / s; best[1] = (best[1] * best[2] + r[1] * p) / s
                    best[2] = s; best[3] = 1 - (1 - best[3]) * (1 - p); best[4].add(di)
        blobs.sort(key=lambda b: -b[3])
        for j, b in enumerate(blobs[:k_out]):
            out[i, j] = (b[0], b[1], b[3]); agree[i, j] = len(b[4])
    return out, agree
