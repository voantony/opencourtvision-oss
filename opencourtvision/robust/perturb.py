"""Geometric perturbations applied on the analysis side (no 4K re-encode): a camera path d(t) (translation, in
fractions of the frame width / height) plus an optional static crop, applied to

* proxy frames (gray thumbnails),
* the regional motion-energy grid E[t, rows, cols] (content moves with the camera; camera motion adds ego-motion energy
  |grad(background) . delta d| per cell, which is what a frame difference of a moving static scene measures),
* point tracks (image px) and a court -> image homography.

Audio is untouched by construction. Paths: shake (hand-held jitter), drift (slow), bump (sudden steps that persist),
pan (slow sinusoid). Crops: 10/20/30 % from any side ("top" cuts the far baseline) or a corner.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Perturbation:
    name: str
    kind: str = "clean"                          # clean | shake | drift | bump | pan | crop
    amp: float = 0.0                             # fraction of the frame (shake: RMS; drift: total; bump: step; pan: peak)
    period_s: float = 60.0                       # pan period
    bumps: tuple = ()                            # bump times as fractions of the match
    crop: dict = field(default_factory=dict)     # {"left"|"right"|"top"|"bottom": fraction}
    seed: int = 0
    mount: dict = field(default_factory=dict)    # static occluder fixed to the phone: {"side": "right", "w": .1, "h": .3}

    def mount_mask(self, shape):
        """Bool mask of the mount part in view (fixed in image coordinates: it moves with the phone, not the scene)."""
        h, w = shape; m = np.zeros(shape, bool)
        if not self.mount: return m
        mw, mh = int(round(w * self.mount.get("w", 0.1))), int(round(h * self.mount.get("h", 0.3)))
        y0 = int(round(h * self.mount.get("y", 0.5))) - mh // 2
        yy, xx = np.mgrid[0:h, 0:w]
        cx = w - 1 if self.mount.get("side", "right") == "right" else 0
        m = ((xx - cx) / max(1, mw)) ** 2 + ((yy - (y0 + mh / 2)) / max(1, mh / 2)) ** 2 <= 1.0   # a half-disc (suction cup)
        return m

    def path(self, t):
        """-> (n, 2) camera offset (dx, dy) in fractions of width / height at times t (s)."""
        t = np.asarray(t, float); n = len(t); d = np.zeros((n, 2))
        if n == 0 or self.kind in ("clean", "crop", "mount"):
            return d
        T = max(1e-6, t[-1] - t[0]); u = (t - t[0]) / T
        if self.kind == "shake":                 # band-limited noise, ~3-8 Hz, RMS amp
            rng = np.random.default_rng(self.seed)
            hz = 1.0 / max(1e-6, float(np.median(np.diff(t)))) if n > 1 else 6.0
            w = rng.normal(size=(n, 2))
            k = max(1, int(round(hz / 5.0)))
            w = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, "same"), 0, w)
            d = w / (w.std(0) + 1e-9) * self.amp
        elif self.kind == "drift":
            d[:, 0] = self.amp * u; d[:, 1] = 0.5 * self.amp * u
        elif self.kind == "bump":
            for i, b in enumerate(self.bumps):
                s = (1 if i % 2 == 0 else -0.5)
                d[:, 0] += self.amp * s * (u >= b); d[:, 1] += 0.5 * self.amp * (u >= b)
        elif self.kind == "pan":
            d[:, 0] = self.amp * np.sin(2 * np.pi * (t - t[0]) / self.period_s)
        return d

    def visible_mask(self, shape):
        """Bool mask (rows, cols) of the original frame still in view after the static crop."""
        h, w = shape; m = np.ones(shape, bool)
        c = self.crop
        if c.get("left"): m[:, : int(round(w * c["left"]))] = False
        if c.get("right"): m[:, w - int(round(w * c["right"])):] = False
        if c.get("top"): m[: int(round(h * c["top"])), :] = False
        if c.get("bottom"): m[h - int(round(h * c["bottom"])):, :] = False
        return m


def suite():
    """The benchmark set: clean, 4 camera paths, 12 side crops, 1 corner crop."""
    P = [Perturbation("clean"),
         Perturbation("shake", "shake", amp=0.012),
         Perturbation("drift", "drift", amp=0.08),
         Perturbation("bumps", "bump", amp=0.06, bumps=(0.25, 0.5, 0.75)),
         Perturbation("pan", "pan", amp=0.10, period_s=60.0)]
    for side, label in (("left", "left"), ("right", "right"), ("top", "far baseline"), ("bottom", "near")):
        for f in (0.1, 0.2, 0.3):
            P.append(Perturbation(f"crop_{side}_{int(f * 100)}", "crop", crop={side: f}))
    P.append(Perturbation("corner_far_left_20", "crop", crop={"top": 0.2, "left": 0.2}))
    P.append(Perturbation("mount_right", "mount", mount={"side": "right", "w": 0.12, "h": 0.35, "y": 0.6}))
    P.append(Perturbation("mount_right_bumps", "bump", amp=0.06, bumps=(0.25, 0.5, 0.75),
                          mount={"side": "right", "w": 0.12, "h": 0.35, "y": 0.6}))
    return P


def _shift(img, dy, dx, cval=0.0):
    from scipy.ndimage import shift
    return shift(np.asarray(img, np.float32), (dy, dx), order=1, mode="constant", cval=cval)


def perturb_frames(frames, frame_t, t, pert):
    """Proxy frames at times t: the nearest stored frame, moved by the camera path, black outside the crop.
    -> float32 (len(t), h, w)."""
    frames = np.asarray(frames); h, w = frames.shape[1:]
    idx = np.clip(np.searchsorted(frame_t, t), 0, len(frame_t) - 1)
    d = pert.path(t); vm = pert.visible_mask((h, w))
    mm = pert.mount_mask((h, w)); mval = 0.35 * float(np.asarray(frames[0]).mean())     # dark, smooth (out of focus)
    out = np.empty((len(t), h, w), np.float32)
    for k, i in enumerate(idx):
        f = frames[i] if not d[k].any() else _shift(frames[i], d[k, 1] * h, d[k, 0] * w, float(frames[i].mean()))
        f = np.where(vm, f, 0.0)
        out[k] = np.where(mm, mval, f)
    return out


def perturb_energy(E, t, pert, background):
    """Motion-energy grid under the perturbation. E: (n, rows, cols) mean |frame diff| per cell (gray levels of the
    proxy frame); background: the static proxy frame (h, w) used for the ego-motion term."""
    E = np.asarray(E, np.float32); n, R, C = E.shape
    h, w = background.shape
    d = pert.path(t)
    out = E.copy()
    if d.any():
        gy, gx = np.gradient(np.asarray(background, np.float32))
        dd = np.vstack([np.zeros((1, 2)), np.diff(d, axis=0)])           # camera motion between consecutive samples
        cy, cx = h // R, w // C
        def cells(img): return img[: R * cy, : C * cx].reshape(R, cy, C, cx).mean((1, 3))
        for k in range(n):
            if d[k].any(): out[k] = _shift(E[k], d[k, 1] * R, d[k, 0] * C, 0.0)
            if dd[k].any():
                out[k] += cells(np.abs(gx * dd[k, 0] * w + gy * dd[k, 1] * h))
    vm = pert.visible_mask((R, C))
    out[:, ~vm] = 0.0
    if pert.mount:                                      # nothing moves behind the mount part: no frame difference there
        from .occluders import cells_mask
        out[:, cells_mask(pert.mount_mask(background.shape), R, C)] = 0.0
    return out


def perturb_points(px, t, pert, size, d=None):
    """Track points (n, 2) image px at times t -> moved with the camera; NaN where outside the crop. d: optional camera
    path (n, 2) already sampled at t (e.g. interpolated from the match-wide path, so tracks and frames share one path)."""
    w, h = size; px = np.asarray(px, float).copy()
    d = pert.path(t) if d is None else np.asarray(d, float)
    px[:, 0] += d[:, 0] * w; px[:, 1] += d[:, 1] * h
    c = pert.crop
    inside = ((px[:, 0] >= w * c.get("left", 0)) & (px[:, 0] < w * (1 - c.get("right", 0))) &
              (px[:, 1] >= h * c.get("top", 0)) & (px[:, 1] < h * (1 - c.get("bottom", 0))))
    if pert.mount:                                      # behind the mount part: not observable
        mh, mw = 108, 192; mm = pert.mount_mask((mh, mw))
        ok = inside & ~np.isnan(px).any(1)
        i = np.clip((px[ok, 1] * mh / h).astype(int), 0, mh - 1); j = np.clip((px[ok, 0] * mw / w).astype(int), 0, mw - 1)
        hid = np.zeros(len(px), bool); hid[np.nonzero(ok)[0][mm[i, j]]] = True
        inside &= ~hid
    px[~inside] = np.nan
    return px


def perturb_homography(H, offset, size):
    """court -> image homography after a camera offset (dx, dy fractions)."""
    w, h = size
    S = np.array([[1, 0, offset[0] * w], [0, 1, offset[1] * h], [0, 0, 1.0]])
    return S @ np.asarray(H, float)


# ----------------------------------------------------------------------------------------------------------- bad footage (ball / PW)
# Antony 19 Oct: great results from really bad recordings. Geometric suite() covers shake/drift/bump/pan/crop/mount.
# These extras degrade *tracks and candidates* (analysis-side) to stand in for blur, fence/mesh, night, bitrate, etc.


def degrade_candidates(cands, condition, seed=0, y_net=None):
    """Apply a named bad-footage condition to candidate peaks (T, K, 3) = x,y,score. Returns a copy.

    Conditions (synthetic stand-ins where real labelled clips are missing):
      clean, shake, blur, lowlight, bitrate, fence_mesh, glare, far_crop, low_angle, wind_audio_proxy
    """
    c = np.asarray(cands, float).copy()
    rng = np.random.default_rng(seed)
    T = len(c)
    if condition in ("clean", None, ""):
        return c
    if condition == "shake":
        # ~4 px RMS jitter at 1280-wide analysis (hand-held)
        jitter = rng.normal(0, 4.0, size=c[:, :, :2].shape)
        ok = np.isfinite(c[:, :, 0])
        c[:, :, :2] = np.where(ok[..., None], c[:, :, :2] + jitter, c[:, :, :2])
    elif condition == "blur":
        # motion blur: smear peak toward previous frame + drop score
        for t in range(1, T):
            ok = np.isfinite(c[t, :, 0]) & np.isfinite(c[t - 1, :, 0])
            c[t, ok, :2] = 0.55 * c[t, ok, :2] + 0.45 * c[t - 1, ok, :2]
            c[t, ok, 2] *= 0.7
    elif condition == "lowlight":
        # night / low light: kill weak peaks, add salt noise peaks
        weak = np.isfinite(c[:, :, 2]) & (c[:, :, 2] < 0.65)
        c[weak] = np.nan
        for t in range(0, T, 3):
            k = int(rng.integers(0, c.shape[1]))
            if not np.isfinite(c[t, k, 0]):
                c[t, k] = (rng.uniform(100, 1180), rng.uniform(80, 640), rng.uniform(0.35, 0.55))
    elif condition == "bitrate":
        # compression: quantise positions to 4 px grid, drop every other weak peak
        ok = np.isfinite(c[:, :, 0])
        c[:, :, :2] = np.where(ok[..., None], np.round(c[:, :, :2] / 4.0) * 4.0, c[:, :, :2])
        drop = ok & (c[:, :, 2] < 0.7) & (rng.random(ok.shape) < 0.35)
        c[drop] = np.nan
    elif condition == "fence_mesh":
        # padel glass / fence: periodic false peaks on a grid + slight score boost of clutter
        for t in range(0, T, 2):
            k = int(rng.integers(0, c.shape[1]))
            c[t, k] = (200 + 40 * ((t // 2) % 20), 150 + 30 * ((t // 5) % 12), 0.45)
    elif condition == "glare":
        # sun glare / backlight: wipe a band of the far half
        if y_net is None:
            y_net = 280.0
        far = np.isfinite(c[:, :, 1]) & (c[:, :, 1] < y_net)
        c[far & (rng.random(far.shape) < 0.55)] = np.nan
    elif condition == "far_crop":
        # low phone height / partial court: top 25% of frame gone (far baseline)
        cut = np.isfinite(c[:, :, 1]) & (c[:, :, 1] < 180)
        c[cut] = np.nan
    elif condition == "low_angle":
        # odd / low angle: squash y toward bottom (perspective exaggeration) + noise
        ok = np.isfinite(c[:, :, 1])
        c[:, :, 1] = np.where(ok, 720 - (720 - c[:, :, 1]) * 0.75 + rng.normal(0, 2, c[:, :, 1].shape), c[:, :, 1])
    elif condition == "wind_audio_proxy":
        # no visual change; harness treats audio contacts as 40% dropped (caller-side). Keep cands clean.
        pass
    else:
        raise ValueError(f"unknown bad-footage condition: {condition}")
    return c


def bad_footage_suite():
    """Named conditions for the ball/point-winner bad-footage split (eng notes §2.5.3)."""
    return [
        "clean", "shake", "blur", "lowlight", "bitrate", "fence_mesh",
        "glare", "far_crop", "low_angle", "wind_audio_proxy",
    ]


def depth_bin(y, y_net, y_near=None):
    """near / mid / far from image y (analysis px). y increases toward the near baseline on a fence cam."""
    if not np.isfinite(y) or y_net is None:
        return "unknown"
    # far = above net (smaller y), near = below a band under the net
    if y < y_net - 40:
        return "far"
    if y_near is not None and y > y_near:
        return "near"
    if y > y_net + 80:
        return "near"
    return "mid"
