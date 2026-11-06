"""Failure-case synthesis for the framing check: warp a real (or rendered) frame with a known 3x3 A, so the true court
fit after the change is exactly A @ H. Used by benchmarks/framing_check_dev_match.py and the tests. No footage here."""
import numpy as np

from ..pose.court import COURT_L, COURT_W
from .check import normalise_H

LINES_M = [((0, 0), (COURT_W, 0)), ((0, COURT_L), (COURT_W, COURT_L)), ((0, 0), (0, COURT_L)), ((COURT_W, 0), (COURT_W, COURT_L)),
           ((1.37, 0), (1.37, COURT_L)), ((COURT_W - 1.37, 0), (COURT_W - 1.37, COURT_L)), ((1.37, 5.485), (COURT_W - 1.37, 5.485)),
           ((1.37, COURT_L - 5.485), (COURT_W - 1.37, COURT_L - 5.485)), ((COURT_W / 2, 5.485), (COURT_W / 2, COURT_L - 5.485)),
           ((-0.9, COURT_L / 2), (COURT_W + 0.9, COURT_L / 2))]


def default_H(size=(640, 360)):
    """A plausible behind-the-baseline phone view of a doubles court (court -> image)."""
    from ..pose.court import homography
    W, H = size
    img = [(0.12 * W, 0.86 * H), (0.88 * W, 0.86 * H), (0.70 * W, 0.20 * H), (0.30 * W, 0.20 * H)]
    return normalise_H(homography([(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)], img))


def render_court(H, size=(640, 360), seed=0):
    """Gray frame: textured clay-like ground + fence/trees texture above the far baseline + white court lines under H."""
    import cv2
    rng = np.random.default_rng(seed)
    W, Hh = size
    base = rng.normal(0, 1, (Hh // 8 + 1, W // 8 + 1)).astype(np.float32)
    tex = cv2.resize(base, (W, Hh), interpolation=cv2.INTER_CUBIC)
    img = np.clip(110 + 18 * tex + rng.normal(0, 6, (Hh, W)), 0, 255).astype(np.uint8)
    for _ in range(60):                                         # background blobs (fence posts, trees, signs) for ORB
        x, y = int(rng.integers(0, W)), int(rng.integers(0, max(1, Hh // 3)))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(4, 30)), y + int(rng.integers(4, 40))), int(rng.integers(20, 240)), -1)
    Hn = normalise_H(H)
    for a, b in LINES_M:
        P = np.array([[*a, 1.0], [*b, 1.0]]) @ Hn.T
        if (P[:, 2] <= 0).any(): continue
        p = P[:, :2] / P[:, 2:]
        cv2.line(img, tuple(int(round(v)) for v in p[0]), tuple(int(round(v)) for v in p[1]), 235, 2, cv2.LINE_AA)
    return img


def player_boxes(H, size, per_side=2, seed=0):
    """Person boxes (x0, y0, x1, y1, 1) standing on the court under H (feet = box bottom centre)."""
    rng = np.random.default_rng(seed)
    Hn = normalise_H(H); out = []
    for side in ("near", "far"):
        for _ in range(per_side):
            x = rng.uniform(1.5, COURT_W - 1.5); y = rng.uniform(1.0, 9.0) if side == "near" else rng.uniform(COURT_L - 9.0, COURT_L - 1.0)
            q = Hn @ [x, y, 1.0]; u, v = q[:2] / q[2]
            q2 = Hn @ [x, y + 0.5, 1.0]; scale = abs((q2[:2] / q2[2])[1] - v) * 2 + 1e-6   # px per metre, roughly
            h = 1.8 * scale * 1.6; w = 0.45 * h
            out.append([u - w / 2, v - h, u + w / 2, v, 1.0])
    return np.array(out, float)


def warp_matrix(kind, size, mag, rng):
    """3x3 image warp for one failure kind at magnitude mag in [0, 1]."""
    W, Hh = size; cx, cy = W / 2, Hh / 2
    T = lambda dx, dy: np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    sgn = lambda: rng.choice([-1.0, 1.0])
    if kind == "shift":
        a = rng.uniform(0, 2 * np.pi); r = mag * 0.12 * np.hypot(W, Hh)
        return T(r * np.cos(a), r * np.sin(a))
    if kind == "rotate":
        th = np.radians(sgn() * mag * 10.0); c, s = np.cos(th), np.sin(th)
        return T(cx, cy) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ T(-cx, -cy)
    if kind == "zoom":
        k = 1.0 + mag * 0.5; ox, oy = rng.uniform(-0.15, 0.15) * W, rng.uniform(-0.15, 0.15) * Hh
        return T(cx + ox, cy + oy) @ np.diag([k, k, 1.0]) @ T(-cx - ox, -cy - oy)
    if kind == "zoom_out":                                    # wider view (phone moved back / 0.5x): whole court in frame
        k = 1.0 - mag * 0.45
        return T(cx, cy) @ np.diag([k, k, 1.0]) @ T(-cx, -cy)
    if kind == "tilt":                                        # pitch approximated by a vertical shift (small angles)
        return T(0, sgn() * mag * 0.3 * Hh)
    if kind == "squash":                                      # lower phone ~ stronger foreshortening of the far half
        k = 1.0 - mag * 0.85
        return T(0, Hh) @ np.diag([1.0, k, 1.0]) @ T(0, -Hh)
    raise ValueError(kind)


def apply_warp(img, H, boxes, A, size):
    """-> (img', H' = A @ H, boxes' (boxes whose feet leave the frame are dropped, as a detector would not see them))."""
    import cv2
    W, Hh = size
    out = cv2.warpPerspective(img, A, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    nb = []
    for b in (boxes if boxes is not None else []):
        pts = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[0], b[3], 1], [b[2], b[3], 1]], float) @ A.T
        pts = pts[:, :2] / pts[:, 2:]
        x0, y0 = pts.min(0); x1, y1 = pts.max(0); fx = (x0 + x1) / 2
        if 0 <= fx < W and 0 <= y1 < Hh: nb.append([x0, y0, x1, y1, *b[4:]])
    return out, normalise_H(A @ normalise_H(H)), np.array(nb, float).reshape(-1, len(boxes[0]) if len(boxes) else 5)
