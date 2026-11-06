"""Merge native-resolution far-half player detections into the 640x360 per-frame arrays (pure numpy, no model deps).

Kept separate from pw_tracks.py so it can be unit-tested. Bug fixed 18 Oct (S0 v2 tracks): the old in-place merge
held numpy row *views* of the frame it was rewriting, so wiping the frame before writing it back destroyed or
duplicated every low-res (near-half) detection. In the v2 tracks 76% of pose frames had no near-side player box
(v1 tracks, no merge: 25%).
"""
import numpy as np


def merge_far(B, P, FB, FP, roi, sc):
    """B (n,8,5) / P (n,8,17,3): low-res boxes/keypoints in 640-px coords, modified in place and returned.
    FB / FP: far-crop detections in SOURCE px. roi = (x0, y0, w, h) far crop in source px; sc = source px per 640 px.
    Low-res detections whose foot point lies inside the far crop are replaced by the far-crop ones; the rest are kept.
    Up to 8 per frame, highest confidence first."""
    x0, y0, w, h = roi
    for i in range(len(B)):
        keep = []
        for j in range(B.shape[1]):
            b = B[i, j]
            if not np.isfinite(b[0]):
                continue
            fx, fy = (b[0] + b[2]) / 2 * sc, b[3] * sc
            if x0 <= fx < x0 + w and y0 <= fy < y0 + h:
                continue
            keep.append((b.copy(), P[i, j].copy()))            # copies: B[i] is overwritten below
        for j in range(FB.shape[1]):
            if np.isfinite(FB[i, j, 0]):
                b = FB[i, j].copy(); b[:4] /= sc
                k = FP[i, j].astype(np.float32).copy(); k[:, :2] /= sc
                keep.append((b, k))
        keep.sort(key=lambda q: -q[0][4])
        B[i] = np.nan; P[i] = np.nan
        for j, (b, k) in enumerate(keep[:B.shape[1]]):
            B[i, j] = b; P[i, j] = k
    return B, P
