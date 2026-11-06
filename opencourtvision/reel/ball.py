"""Ball detection, tracking and crop-path planning for 9:16 reframing.

Per moment: decode the source window at analysis size (default 1280x720 @ 60 fps) -> detect ball candidates
(yellow HSV & motion, small round blobs) -> temporal support -> greedy constant-velocity track ->
crop-centre path (gap fill, player-framing fallback, lead, dead zone, Savitzky-Golay, velocity limit), and QA
metrics (jerk_p95). Player-framing fallback (editorial spec §5.2): when the ball is lost for more than
BALL_LOST_FRAMES frames, or its per-frame confidence is below MIN_BALL_CONF, the target is the centre of the bbox union of
the players nearest the last known ball (the hitting pair), never a hold on empty court or a jump.
An optional TrackNet model can be fused in (TrackNet first, HSV fills gaps).
"""
import re
import subprocess
import sys

import cv2
import numpy as np
from scipy.signal import savgol_filter

from ..ffmpeg import ffmpeg_bin, is_macos

BALL_LOST_FRAMES = 6      # spec §5.2: ball lost > 6 frames -> player-fit framing
MIN_BALL_CONF = 0.5       # frames below this ball confidence count as lost (not in the spec; see frame_confidence)
FIT_PLAYERS = 2           # players framed in the fallback: the 2 blobs nearest the last ball position (hitting pair)


def decode(src, a, b, aw=1280, ah=720, fps=60, hwaccel=None):
    """Yield BGR frames of src[a:b] scaled to aw x ah at fps."""
    hw = ["-hwaccel", hwaccel] if hwaccel else []
    cmd = [ffmpeg_bin(), "-nostdin", "-loglevel", "error", *hw, "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src,
           "-vf", f"scale={aw}:{ah}", "-r", str(fps), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = aw * ah * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
    p.wait()


def detect(frames, a=0.0, fps=60, dbg_times=(), keep_small=False, with_boxes=False):
    """Ball candidates per frame [(x, y, area)], player-activity centre x per frame, debug frames, 640x360 frames (for TrackNet).
    with_boxes=True also returns, per frame, the player-sized moving blobs as [(x0, x1, area, y0, y1)] (analysis px)."""
    cands, act, frames_dbg, small, boxes = [], [], {}, [], []
    g2 = g1 = None
    for i, f in enumerate(frames):
        AH = f.shape[0]
        if keep_small:
            small.append(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA))
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        c, ax, bb = [], np.nan, []
        if g2 is not None:
            mot = cv2.absdiff(g, g2)
            mm = (mot > 18).astype(np.uint8)
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            ym = cv2.inRange(hsv, (22, 80, 130), (45, 255, 255))
            m = cv2.dilate(mm, np.ones((3, 3), np.uint8)) & (ym > 0).astype(np.uint8)
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
            for k in range(1, n):
                x, y, w, h, area = st[k]
                if 2 <= area <= 90 and w <= 14 and h <= 14 and max(w, h) <= 2.6 * min(w, h) + 1 and y > AH * 0.08:
                    c.append((float(cen[k][0]), float(cen[k][1]), int(area)))
            # activity centre: big moving blobs (players)
            big = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            n2, _, st2, cen2 = cv2.connectedComponentsWithStats(big, 8)
            keep = [k for k in range(1, n2) if 250 <= st2[k][4] <= 40000 and st2[k][1] > AH * 0.12]
            ws = [(st2[k][4], cen2[k][0]) for k in keep]
            bb = [(float(st2[k][0]), float(st2[k][0] + st2[k][2]), int(st2[k][4]), float(st2[k][1]), float(st2[k][1] + st2[k][3]))
                  for k in keep]
            if ws:
                ax = float(np.average([x for _, x in ws], weights=[w for w, _ in ws]))
        cands.append(c)
        act.append(ax)
        boxes.append(bb)
        t = a + i / fps
        for dt in dbg_times:
            if abs(t - dt) < 0.5 / fps:
                frames_dbg[dt] = (i, f.copy())
        g2, g1 = g1, g
    if with_boxes:
        return cands, np.array(act), frames_dbg, small, boxes
    return cands, np.array(act), frames_dbg, small


def frame_confidence(hsv_pos, tn_pos, agree_px=25):
    """Per-frame ball confidence from the two detectors: 1.0 both agree (< agree_px), 0.7 TrackNet only, 0.6 HSV only,
    0.3 both present but disagreeing, 0 neither. (Heuristic: the spec names the trigger, not the confidence scale.)"""
    h, t = ~np.isnan(hsv_pos[:, 0]), ~np.isnan(tn_pos[:, 0])
    c = np.zeros(len(h))
    c[t & ~h] = 0.7; c[h & ~t] = 0.6
    both = h & t
    if both.any():
        d = np.hypot(*(hsv_pos[both] - tn_pos[both]).T)
        c[np.where(both)[0]] = np.where(d < agree_px, 1.0, 0.3)
    return c


def fallback_mask(ok, conf=None, max_lost=BALL_LOST_FRAMES, min_conf=MIN_BALL_CONF):
    """True on frames framed by players instead of the ball: every frame of a run of > max_lost consecutive frames
    without a usable ball (not detected, or confidence < min_conf). The path is solved offline, so the whole gap
    switches (look-ahead), not just the frames after the 6th."""
    ok = np.asarray(ok, bool).copy()
    if conf is not None: ok &= np.asarray(conf) >= min_conf
    fb = np.zeros(len(ok), bool); i = 0
    while i < len(ok):
        if ok[i]: i += 1; continue
        j = i
        while j < len(ok) and not ok[j]: j += 1
        if j - i > max_lost: fb[i:j] = True
        i = j
    return fb


def player_fit_x(blobs, ref_x=None, k=FIT_PLAYERS):
    """Centre x of the bbox union of the k player blobs nearest ref_x (the last ball position), or of all blobs
    when ref_x is unknown. blobs: [(x0, x1, area[, y0, y1])]. NaN when there are none."""
    if not blobs: return np.nan
    if ref_x is not None and np.isfinite(ref_x):
        blobs = sorted(blobs, key=lambda b: abs((b[0] + b[1]) / 2 - ref_x))[:k]
    return (min(b[0] for b in blobs) + max(b[1] for b in blobs)) / 2


def jerk_p95(cx, frame_w=None, cy=None):
    """95th percentile of |third difference| of the crop centre, px/frame^3 (2-D norm when cy is given), and the same
    divided by frame_w. Compute per continuous shot (cuts reset the camera); returns None for < 4 frames."""
    cx = np.asarray(cx, float)
    if len(cx) < 4: return None
    j = np.abs(np.diff(cx, 3)) if cy is None else np.hypot(np.diff(cx, 3), np.diff(np.asarray(cy, float), 3))
    p = float(np.percentile(j, 95))
    return dict(px_per_frame3=float(f"{p:.6g}"), norm=float(f"{p / frame_w:.6g}") if frame_w else None, n=int(len(j)))


def smooth_mean(x, k=9):
    """Centred k-frame moving mean (edges padded with the end values): the option-(b) cross-check smoother."""
    x = np.asarray(x, float)
    if len(x) == 0: return x
    h = k // 2
    return np.convolve(np.pad(x, h, mode="edge"), np.ones(k) / k, "valid")


def crop_cmd_centres(path):
    """Crop-centre x per frame from an ffmpeg sendcmd file ('t crop w W, crop h H, crop x X, crop y Y;' lines)."""
    out = []
    for ln in open(path):
        m = re.search(r"crop w (\d+).*?crop x (\d+)", ln)
        if m: out.append(int(m.group(2)) + int(m.group(1)) / 2)
    return np.array(out, float)


def pooled_jerk(paths, frame_w, smooth_k=None):
    """jerk_p95 pooled over several continuous shots (third differences never span a cut); optional k-frame smoothing."""
    js = []
    for cx in paths:
        cx = np.asarray(cx, float)
        if smooth_k: cx = smooth_mean(cx, smooth_k)
        if len(cx) >= 4: js.append(np.abs(np.diff(cx, 3)))
    if not js: return None
    j = np.concatenate(js)
    p = float(np.percentile(j, 95))
    return dict(px_per_frame3=float(f"{p:.6g}"), norm=float(f"{p / frame_w:.6g}"), n=int(len(j)))


def track(cands, aw=1280, ah=720):
    """Greedy constant-velocity tracker over temporally supported candidates; NaN where lost."""
    n = len(cands)
    pos = np.full((n, 2), np.nan)
    # temporal support: candidate must have a neighbour within 45 px in frame t-1 or t+1
    sup = []
    for t in range(n):
        s = []
        for (x, y, _a) in cands[t]:
            ok = any(abs(x - u) < 45 and abs(y - v) < 45 and (abs(x - u) + abs(y - v)) > 0.5
                     for tt in (t - 1, t + 1) if 0 <= tt < n for (u, v, _) in cands[tt])
            if ok:
                s.append((x, y))
        sup.append(s)
    p = v = None
    miss = 99
    for t in range(n):
        if p is not None and miss <= 8:
            pred = p + v
            best = min(sup[t], key=lambda c: np.hypot(c[0] - pred[0], c[1] - pred[1]), default=None)
            if best is not None and np.hypot(best[0] - pred[0], best[1] - pred[1]) < 70 + 10 * miss:
                nb = np.array(best)
                v = 0.6 * v + 0.4 * (nb - p) / (miss + 1)
                p = nb
                miss = 0
                pos[t] = p
                continue
            miss += 1
            continue
        if sup[t]:  # (re)initialise on the supported candidate nearest the previous position / centre
            ref = p if p is not None else np.array([aw / 2, ah / 2])
            best = min(sup[t], key=lambda c: np.hypot(c[0] - ref[0], c[1] - ref[1]))
            p = np.array(best)
            v = np.zeros(2)
            miss = 0
            pos[t] = p
        else:
            miss += 1
    return pos


def path(pos, act, fallback, scale, boxes=None, conf=None, max_lost=BALL_LOST_FRAMES, min_conf=MIN_BALL_CONF, info=None):
    """Crop-centre x per frame in source pixels. `scale` = source width / analysis width.
    boxes (per-frame player blobs from detect(with_boxes=True)) and conf (per-frame ball confidence) enable the
    player-framing fallback; without boxes the fallback target is the player-activity centre, then `fallback`.
    If `info` is a dict it receives the fallback mask ("fallback") and the target ("target")."""
    n = len(pos)
    bx = pos[:, 0] * scale
    t = np.arange(n)
    ok = ~np.isnan(bx)
    actx = np.where(np.isnan(act), np.nan, act * scale)
    if np.isnan(actx).all():
        actx[:] = fallback
    good = ~np.isnan(actx)
    actx = np.interp(t, t[good], actx[good]) if good.any() else np.full(n, fallback)
    actx = savgol_filter(actx, min(n // 2 * 2 - 1, 61), 2) if n > 63 else actx
    if conf is not None: ok = ok & (np.asarray(conf) >= min_conf)
    fb = fallback_mask(ok, None, max_lost) if n else np.zeros(0, bool)
    # player-fit target: union of the players nearest the last usable ball position, else activity centre
    pfit = actx.copy()
    if boxes is not None:
        last = np.nan; px = np.full(n, np.nan)
        for i in range(n):
            if ok[i]: last = bx[i] / scale
            if fb[i]: px[i] = player_fit_x(boxes[i] if i < len(boxes) else [], last)
        g = ~np.isnan(px)
        if g.any(): pfit = np.where(fb, np.interp(t, t[g], px[g] * scale), actx)
    if ok.any():
        tgt = np.interp(t, t[ok], bx[ok])
        # short gaps (<= max_lost frames) are interpolated; longer gaps and low-confidence runs frame the players,
        # eased over ~10 frames at each edge so the switch is a move, not a jump
        w = fb.astype(float)
        if n >= 11: w = np.convolve(w, np.ones(11) / 11, mode="same")
        tgt = (1 - w) * tgt + w * pfit
    else:
        tgt = pfit
        fb = np.ones(n, bool)
    if info is not None: info["fallback"] = fb; info["target"] = tgt.copy()
    # lead in direction of travel
    vel = np.gradient(savgol_filter(tgt, 15, 2) if n > 16 else tgt)
    tgt = tgt + np.clip(vel * 8, -90, 90)
    # dead zone follower
    cam = np.empty(n)
    c = tgt[0]
    DZ = 300
    for i in range(n):
        d = tgt[i] - c
        if abs(d) > DZ:
            c += (d - np.sign(d) * DZ) * 0.07
        cam[i] = c
    if n > 91:
        cam = savgol_filter(cam, 91, 2)
    # velocity/accel limit (px per frame at source resolution)
    out = np.empty(n)
    out[0] = cam[0]
    vv = 0
    for i in range(1, n):
        dv = np.clip((cam[i] - out[i - 1]) - vv, -1.2, 1.2)
        vv = np.clip(vv + dv, -20, 20)
        out[i] = out[i - 1] + vv
    return out, bx


def zoom_at(ts, ramps, punch, pre=0.35, post=0.45):
    """Zoom factor per frame: 1.08 punch-in around each ramp (contact) time, or constant for punch moments."""
    if punch:
        return np.full(len(ts), 1.08)
    z = np.ones(len(ts))
    for h in ramps:
        a, b = h - pre - 0.25, h + post
        z = np.maximum(z, np.where((ts >= a) & (ts <= b), 1.0 + 0.08 * np.clip((ts - a) / 0.25, 0, 1), 1.0))
        z = np.maximum(z, np.where((ts > b) & (ts < b + 0.3), 1.08 - 0.08 * (ts - b) / 0.3, 1.0))
    return z


class TrackNet:
    """Optional TrackNet ball detector. Needs torch and a TrackNet checkout providing `model.BallTrackerNet`
    plus weights (not shipped). Pass the directory containing model.py and the .pt file."""

    def __init__(self, model_dir, weights="model_best.pt"):
        import os

        import torch
        sys.path.insert(0, model_dir)
        from model import BallTrackerNet  # noqa: E402  (from the TrackNet checkout)
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        net = BallTrackerNet()
        net.load_state_dict(torch.load(os.path.join(model_dir, weights), map_location="cpu"))
        self.net = net.to(self.dev).eval()

    @staticmethod
    def postprocess(fm):
        fm = (fm * 255).reshape(360, 640).astype(np.uint8)
        _, hm = cv2.threshold(fm, 127, 255, cv2.THRESH_BINARY)
        c = cv2.HoughCircles(hm, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
        return (float(c[0][0][0] * 2), float(c[0][0][1] * 2)) if c is not None and len(c) == 1 else None

    def run(self, small):
        """TrackNet (3-frame input at ~30 fps): run on every 2nd 60-fps frame using frames i, i-2, i-4."""
        n = len(small)
        out = np.full((n, 2), np.nan)
        idx = list(range(4, n, 2))
        B = 8
        with self.torch.no_grad():
            for k in range(0, len(idx), B):
                ii = idx[k:k + B]
                x = np.stack([np.rollaxis(np.concatenate((small[i], small[i - 2], small[i - 4]), axis=2).astype(np.float32) / 255.0, 2, 0) for i in ii])
                o = self.net(self.torch.from_numpy(x).to(self.dev)).argmax(dim=1).cpu().numpy()
                for j, i in enumerate(ii):
                    p = self.postprocess(o[j])
                    if p:
                        out[i] = p
        # outlier removal: drop isolated points far from both detected neighbours
        det = np.where(~np.isnan(out[:, 0]))[0]
        for a_, b_, c_ in zip(det[:-2], det[1:-1], det[2:]):
            if np.hypot(*(out[b_] - out[a_])) > 100 and np.hypot(*(out[c_] - out[b_])) > 100:
                out[b_] = np.nan
        # fill odd frames between two detections 2 frames apart
        for i in range(1, n - 1):
            if np.isnan(out[i, 0]) and not np.isnan(out[i - 1, 0]) and not np.isnan(out[i + 1, 0]):
                out[i] = (out[i - 1] + out[i + 1]) / 2
        return out


def default_hwaccel():
    return "videotoolbox" if is_macos() else None
