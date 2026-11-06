"""Per-detection identity cues and their distances. No faces, no biometrics: colour is measured on clothing regions,
pose is used for body-part regions and for the serve *role* (arm up at the toss), never for body proportions or gait.

COCO-17 keypoints, (x, y, conf) with x, y normalised to the crop (0..1, top-left origin).
"""
from __future__ import annotations

import numpy as np

L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_WR, R_WR, NOSE = 5, 6, 11, 12, 13, 14, 9, 10, 0
AB_BINS, L_BINS = 6, 4
COLOUR_DIM = 2 * (AB_BINS * AB_BINS + L_BINS)


def _bgr2lab(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def body_bands(h, kp=None, kp_min=0.3):
    """(upper, lower) row ranges in a crop of height h: shirt = shoulders..hips, shorts/skirt = hips..knees.
    Falls back to fixed box fractions when the pose is missing or weak."""
    up, lo = (int(0.18 * h), int(0.50 * h)), (int(0.50 * h), int(0.72 * h))
    if kp is None:
        return up, lo, False
    kp = np.asarray(kp, float)
    def y(i, j):
        v = [kp[k, 1] for k in (i, j) if kp[k, 2] >= kp_min]
        return float(np.mean(v)) * h if v else None
    ys, yh, yk = y(L_SH, R_SH), y(L_HIP, R_HIP), y(L_KNEE, R_KNEE)
    if ys is None or yh is None or yh - ys < 0.12 * h:
        return up, lo, False
    yk = yk if yk is not None and yk > yh + 0.05 * h else yh + 0.6 * (yh - ys)
    return (int(ys), int(yh)), (int(yh), int(min(h, yk))), True


def colour_hist(crop_bgr, mask=None, kp=None) -> np.ndarray:
    """Hellinger-normalised CIELAB histogram of the shirt band and the shorts band (2 x (6x6 a*b* + 4 L) bins).
    mask: bool HxW person mask (Apple Vision); None = whole box (background leaks in)."""
    lab = _bgr2lab(crop_bgr).reshape(-1, 3).astype(np.int32)
    h, w = crop_bgr.shape[:2]
    m = np.ones((h, w), bool) if mask is None else np.asarray(mask, bool)
    if mask is None:                                            # box-only: drop the outer 20 % columns (background)
        m[:, : int(0.2 * w)] = False; m[:, int(0.8 * w):] = False
    up, lo, _ = body_bands(h, kp)
    out = []
    for r0, r1 in (up, lo):
        sel = np.zeros((h, w), bool); sel[max(0, r0):max(r0 + 1, r1)] = True; sel &= m
        px = lab[sel.reshape(-1)]
        if len(px) < 8:
            out.append(np.full(AB_BINS * AB_BINS + L_BINS, 1.0 / np.sqrt(AB_BINS * AB_BINS + L_BINS))); continue
        a = np.clip((px[:, 1] - 128 + 48) * AB_BINS // 96, 0, AB_BINS - 1); b = np.clip((px[:, 2] - 128 + 48) * AB_BINS // 96, 0, AB_BINS - 1)
        hab = np.bincount(a * AB_BINS + b, minlength=AB_BINS * AB_BINS).astype(float)
        hl = np.bincount(np.clip(px[:, 0] * L_BINS // 256, 0, L_BINS - 1), minlength=L_BINS).astype(float)
        v = np.r_[hab / hab.sum(), hl / hl.sum()]
        out.append(np.sqrt(v) / np.sqrt(2.0))                   # each band has unit norm
    return np.concatenate(out).astype(np.float32)


def hellinger(a, b) -> float:
    """distance in [0, ~1] between two colour_hist vectors (mean over the 2 bands)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / 2.0)


def cosine(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def pose_norm(kp, kp_min=0.2) -> np.ndarray | None:
    """34-d pose vector centred on the hip midpoint and scaled by the torso length (shape-free use only: role
    classification). None when the torso is not visible."""
    kp = np.asarray(kp, float)
    if kp.shape != (17, 3) or min(kp[[L_SH, R_SH, L_HIP, R_HIP], 2]) < kp_min:
        return None
    hip = kp[[L_HIP, R_HIP], :2].mean(0); sh = kp[[L_SH, R_SH], :2].mean(0); s = np.linalg.norm(sh - hip)
    if s < 1e-6:
        return None
    v = (kp[:, :2] - hip) / s
    v[kp[:, 2] < kp_min] = 0.0
    return np.c_[v, (kp[:, 2] >= kp_min)].astype(np.float32).reshape(-1)


def serve_pose_score(kp, kp_min=0.2) -> float:
    """hand-built serve cue: how far the higher wrist is above the shoulders, in torso lengths (toss / trophy
    position > ~0.8). nan when not measurable."""
    kp = np.asarray(kp, float)
    if kp.shape != (17, 3):
        return float("nan")
    sh = [kp[i, 1] for i in (L_SH, R_SH) if kp[i, 2] >= kp_min]; hp = [kp[i, 1] for i in (L_HIP, R_HIP) if kp[i, 2] >= kp_min]
    wr = [kp[i, 1] for i in (L_WR, R_WR) if kp[i, 2] >= kp_min]
    if not sh or not hp or not wr:
        return float("nan")
    torso = max(np.mean(hp) - np.mean(sh), 1e-6)
    return float((np.mean(sh) - min(wr)) / torso)
