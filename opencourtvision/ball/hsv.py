"""Yellow-ball candidates from colour + motion, with blob size, optionally restricted to a search window (ROI).

Colour/motion rule of `reel.ball.detect` (HSV (22, 80, 130)-(45, 255, 255); here the S floor is 50 AND frame difference > 18 against the
frame two steps back), but per call on one frame, returning each blob's minor/major axis so the size gate can use the
minor axis (motion blur only stretches the major one). Overlay masks (e.g. a SwingVision export's score bug and
mini-map) are zeroed out before detection.
"""
import cv2
import numpy as np

HSV_LO, HSV_HI = (22, 50, 130), (45, 255, 255)   # S floor 50 (reel uses 80): blurred far balls desaturate
MOTION_TH = 18


def blob_axes(mask_k):
    """Minor/major axis (px) of a binary blob from its second moments (a uniform disc of diameter d gives d, d)."""
    ys, xs = np.nonzero(mask_k)
    if len(xs) < 2: return 1.0, 1.0
    cov = np.cov(np.vstack([xs, ys]).astype(float)) + np.eye(2) / 12.0   # +1/12: pixel quantisation
    ev = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 1e-6, None)) * 4.0
    return float(ev[0]), float(ev[1])


def candidates(frame, prev2, roi=None, mask=None, min_area=1, max_area=400, max_aspect=4.0):
    """-> [(x, y, minor_px, major_px, area)] in full-frame px. prev2: grey frame two steps back (None -> no candidates).
    roi: (x0, y0, x1, y1) search window; mask: uint8 (H, W), 0 where detection is disabled (overlays)."""
    if prev2 is None: return []
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if roi is None else (max(0, int(roi[0])), max(0, int(roi[1])), min(w, int(roi[2])), min(h, int(roi[3])))
    if x1 - x0 < 3 or y1 - y0 < 3: return []
    f = frame[y0:y1, x0:x1]
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    mot = (cv2.absdiff(g, prev2[y0:y1, x0:x1]) > MOTION_TH).astype(np.uint8)
    ym = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), HSV_LO, HSV_HI)
    m = cv2.dilate(mot, np.ones((3, 3), np.uint8)) & (ym > 0).astype(np.uint8)
    if mask is not None: m &= (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for k in range(1, n):
        bx, by, bw, bh, area = st[k]
        if not (min_area <= area <= max_area): continue
        mi, ma = blob_axes(lab[by:by + bh, bx:bx + bw] == k)
        if ma > max_aspect * mi + 1: continue
        out.append((float(cen[k][0] + x0), float(cen[k][1] + y0), mi, ma, int(area)))
    return out


def grey(frame):
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def overlay_mask(h, w, boxes):
    """uint8 mask, 0 inside each (x0, y0, x1, y1) box."""
    m = np.full((h, w), 255, np.uint8)
    for x0, y0, x1, y1 in boxes: m[int(y0):int(y1), int(x0):int(x1)] = 0
    return m


# SwingVision "Filtered Points" export at 960x540: score bug (top-left), mini-map + shot label (top-right)
SWINGVISION_OVERLAYS_960 = [(0, 0, 230, 80), (812, 0, 960, 240)]
