"""Static occluders from the mount itself (18 Oct; e.g. the AcePad suction cup on the right edge of Antony's 9 Oct padel
video): find image regions that stay fixed across the match, and mask them out of ball, player and court detection.

Cues, on proxy frames (gray, any size) sampled across the match:
1. Still: per-pixel temporal std is low (players never pass in front of a mount part that is in front of them).
2. Fixed to the phone: when the camera moves (drift, bumps, pans: robust/camera_motion.py offsets), the scene shifts
   but a mount part does not. After undoing the camera offset, scene pixels line up and the mount part smears, so
   compensated std >> raw std. This is the decisive cue whenever the camera moved at all.
3. Without camera motion (perfectly fixed mount): still + touching the frame border + defocused (a close object is out
   of focus: low local gradient vs the scene) + a connected region of at least MIN_AREA of the frame. Sky / far trees
   can be still and smooth too, so this branch needs all four and is reported as `confidence="low"`.
-> (mask bool (h, w), info). Downstream: `cells_mask` for the motion-energy grid, `drop_masked` for ball / player points,
and the uint8 `feature_mask` for the court re-fit (geometry.transfer_homography takes it).
"""
import numpy as np

OCC = dict(STILL_STD=2.0, COMP_RATIO=3.0, GRAD_MAX=0.25, MIN_AREA=0.004, MOVE_PX=1.5, MAX_FRAMES=400)


def _shift(img, dx, dy):
    from scipy.ndimage import shift
    return shift(np.asarray(img, np.float32), (dy, dx), order=1, mode="nearest")


def detect_occluders(frames, offsets_px=None, p=None):
    from scipy import ndimage as ndi
    p = {**OCC, **(p or {})}
    F = np.asarray(frames, np.float32)
    if len(F) > p["MAX_FRAMES"]:
        idx = np.linspace(0, len(F) - 1, p["MAX_FRAMES"]).astype(int); F = F[idx]
        offsets_px = None if offsets_px is None else np.asarray(offsets_px, float)[idx]
    h, w = F.shape[1:]
    raw = F.std(0)
    still = raw < p["STILL_STD"]
    moved = offsets_px is not None and np.ptp(np.asarray(offsets_px, float), axis=0).max() >= p["MOVE_PX"]
    g = ndi.gaussian_gradient_magnitude(F.mean(0), 1.0)
    smooth = g < p["GRAD_MAX"] * np.median(g)
    smear = np.zeros((h, w), bool)
    if moved:
        # scene pixels line up after undoing the camera offset; a mount part (fixed to the phone) smears at its edges
        comp = np.stack([_shift(f, -o[0], -o[1]) for f, o in zip(F, offsets_px)]).std(0)
        smear = still & (comp > p["COMP_RATIO"] * np.maximum(raw, 0.5))
    cand = still & (smooth | smear)
    cand = ndi.binary_closing(ndi.binary_opening(cand, iterations=1), iterations=2)
    lab, n = ndi.label(cand)
    mask = np.zeros((h, w), bool); regions = []
    for k in range(1, n + 1):
        r = lab == k
        if r.mean() < p["MIN_AREA"]: continue
        ys, xs = np.nonzero(r)
        e = 3                                          # morphology erodes the outermost pixels
        border = ys.min() <= e or xs.min() <= e or ys.max() >= h - 1 - e or xs.max() >= w - 1 - e
        sm = float(smear[ndi.binary_dilation(r, iterations=2)].mean()) if moved else 0.0
        if not border and sm < 0.05: continue           # without smear evidence it must touch the frame edge
        mask |= r
        regions.append(dict(bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), area=round(float(r.mean()), 4),
                            edge=bool(border), smear=round(sm, 3), confidence="high" if sm >= 0.05 else "low"))
    mask = ndi.binary_dilation(mask, iterations=1) if mask.any() else mask
    conf = (max(regions, key=lambda r: r["area"])["confidence"]) if regions else None
    return mask, dict(regions=regions, confidence=conf, area=round(float(mask.mean()), 4))


def cells_mask(mask, R=12, C=24, min_cover=0.3):
    """Motion-grid cells that are mostly occluded (excluded from activity and from the visible-court share)."""
    h, w = mask.shape
    m = mask[: R * (h // R), : C * (w // C)].reshape(R, h // R, C, w // C).mean((1, 3))
    return m >= min_cover


def feature_mask(mask, size):
    """uint8 mask at frame size, 0 on occluders: for ORB / court-line search (transfer_homography(mask=...))."""
    from scipy.ndimage import zoom
    w, h = size
    z = zoom(mask.astype(np.float32), (h / mask.shape[0], w / mask.shape[1]), order=0) > 0.5
    return np.where(z, 0, 255).astype(np.uint8)


def drop_masked(px, mask, size):
    """Ball / player points (n, 2) frame px inside an occluder -> NaN (never a detection on the mount)."""
    px = np.asarray(px, float).copy(); w, h = size; mh, mw = mask.shape
    ok = ~np.isnan(px).any(1)
    i = np.clip((px[ok, 1] * mh / h).astype(int), 0, mh - 1); j = np.clip((px[ok, 0] * mw / w).astype(int), 0, mw - 1)
    bad = np.zeros(len(px), bool); bad[np.nonzero(ok)[0][mask[i, j]]] = True
    px[bad] = np.nan
    return px
