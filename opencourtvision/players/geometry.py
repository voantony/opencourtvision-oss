"""Court geometry for the player tracker: image foot point -> court metres, play zone, sides, far-court tiles.

Court frame (same as pw_tracks / detbench): x across the court 0..10.97 m (doubles width), y along it from the
near baseline (0) to the far baseline (23.77 m); net at y = 11.885 m. Near = camera end.
"""
from __future__ import annotations

import numpy as np

NET_Y, CW, CL = 11.885, 10.97, 23.77
MID_X = CW / 2
ZX, ZY = (-4.0, CW + 4.0), (-6.5, CL + 6.5)         # play zone (court + run-off) in metres
NEAR, FAR = 0, 1


def to_court(H_img2court, pts) -> np.ndarray:
    """(N,2) image px -> (N,2) court metres."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return np.zeros((0, 2))
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(H_img2court, float).T
    return p[:, :2] / p[:, 2:3]


def feet(boxes) -> np.ndarray:
    """foot point (bottom centre) of xyxy boxes."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]


def side_of(y) -> np.ndarray:
    return (np.asarray(y, float) >= NET_Y).astype(int)


def in_zone(xy, zx=ZX, zy=ZY) -> np.ndarray:
    xy = np.asarray(xy, float).reshape(-1, 2)
    return (xy[:, 0] >= zx[0]) & (xy[:, 0] <= zx[1]) & (xy[:, 1] >= zy[0]) & (xy[:, 1] <= zy[1])


def outside_dist(xy) -> np.ndarray:
    """metres outside the doubles court (0 inside)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    dx = np.maximum(0, np.maximum(-xy[:, 0], xy[:, 0] - CW)); dy = np.maximum(0, np.maximum(-xy[:, 1], xy[:, 1] - CL))
    return np.hypot(dx, dy)


def far_roi(H_img2court, sw=3840, sh=2160):
    """far-half crop (x0, y0, w, h) in source px (same rule as tools/point_winner/pw_tracks.far_roi), or None."""
    Hi = np.linalg.inv(np.asarray(H_img2court, float))

    def c2i(x, y):
        v = Hi @ [x, y, 1.0]
        return v[:2] / v[2]
    pts = np.array([c2i(x, y) for x in (-3.5, CW + 3.5) for y in (NET_Y - 1.0, CL + 5.5)])
    nb = c2i(CW / 2, 0.0)
    if not np.isfinite(pts).all() or not (-2000 < nb[0] < 6000):
        return None
    xs = np.clip(pts[:, 0], 0, sw); ys = pts[:, 1]
    ynet = max(ys[0], ys[2]); yback = min(ys[1], ys[3]); hpx = max(ynet - yback, 150.0)
    y0 = int(np.clip(yback - 0.9 * hpx, 0, sh)); y1 = int(np.clip(ynet + 0.15 * hpx, 0, sh))
    x0, x1 = int(xs.min()), int(xs.max())
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2; w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    return (x0, y0, w, h) if w >= 320 and h >= 160 else None


def tiles(roi):
    """square tiles (x, y, s, s) covering a wide far-court roi with 25 % overlap (detbench rfn384_tiles)."""
    x0, y0, w, h = roi
    s = min(h, w); step = max(1, int(0.75 * s))
    xs = list(range(0, max(1, w - s), step)) + [w - s]
    return [(x0 + x, y0, s, s) for x in sorted(set(xs))]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def contained(a, b) -> float:
    """intersection over the smaller box's area (box-in-box duplicates have low IoU but high containment)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / m if m > 0 else 0.0


def nms(dets, iou=0.5, contain=None):
    """greedy NMS on rows [x0, y0, x1, y1, conf, ...]; `contain` also drops boxes mostly inside a kept one."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(box_iou(d, k) < iou and (contain is None or contained(d, k) < contain) for k in keep):
            keep.append(d)
    return keep


def merge_far(full, far, roi, edge_px=6):
    """full-frame + far-court tile detections. Full-frame boxes whose foot is inside the far roi are replaced by the
    tile boxes (small far players are found better on native-res tiles), except that a tile box cut by the roi's
    bottom edge (a near-court player at the net standing just below the roi) is dropped when a full-frame box covers
    it, and that full-frame box is kept: its foot point is the real one."""
    x0, y0, w, h = roi
    inroi = lambda d: x0 <= (d[0] + d[2]) / 2 < x0 + w and y0 <= d[3] < y0 + h
    cut = [t for t in far if t[3] >= y0 + h - edge_px]
    keep_full = [d for d in full if not inroi(d) or any(contained(t, d) > 0.5 for t in cut)]
    far_ok = [t for t in far if t not in cut or not any(contained(t, d) > 0.5 for d in full)]
    return keep_full + far_ok
