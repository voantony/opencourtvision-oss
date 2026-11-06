"""Point-winner v1, stage 0: per-labelled-segment ball + player tracks for the dev_match match (research measurement only).

S0 v2 (18 Oct): with --prelabels, the window is anchored on OUR last audio contact in the point (same rule as pw_stages:
last hit in [start, end-0.3]): tc-4 s .. tc+3.5 s (falls back to the old end-anchored window when there is no hit). Far-half
players are detected on a native-resolution 4K crop of the far court (per camera view, from courts_by_view) at --far_imgsz,
and merged into the 640x360 box/keypoint arrays (so pw_stages reads them unchanged); raw far detections in source px are kept
as far_boxes / far_kp. --only_points skips labelled non-points (pw_stages only reads kind=point with a winner).

Runs on Antony's Mac at low priority. Ball: TrackNet tennis weights (yastrebksv/TrackNet, 3-frame 640x360; licence to
verify before any product use). Players: Ultralytics YOLO11s-pose (AGPL-3.0: internal measurement only, never shipped;
product path is RF-DETR Nano, Apache-2.0). Output: one compact .npz per segment + a mid-frame JPEG, under --out.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tn", default=os.path.expanduser("~/models/tracknet")); ap.add_argument("--pose", default=os.path.expanduser("~/models/yolo/yolo11s-pose.pt"))
ap.add_argument("--pre", type=float, default=6.0, help="seconds before the labelled point END (winner needs the ending, not the whole rally)"); ap.add_argument("--post", type=float, default=3.0)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--fps", type=float, default=30.0); ap.add_argument("--pose_every", type=int, default=3)
ap.add_argument("--ffmpeg", default=os.path.expanduser("~/bin/ffmpeg"))
ap.add_argument("--prelabels", help="S0 v2: cc prelabels.json (audio contacts + courts_by_view, source px); enables the contact window + far crop")
ap.add_argument("--tc_pre", type=float, default=4.0); ap.add_argument("--tc_post", type=float, default=3.5)
ap.add_argument("--far_imgsz", type=int, default=1920); ap.add_argument("--only_points", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="process at most N segments (cost check)")
ap.add_argument("--court_override", help="JSON {view, H_img2court} (or a list of them): replaces that view's homography from the prelabels, "
                "e.g. the manual court-setup fit tests/fixtures/court_dev_match_view0_manual.json")
ap.add_argument("--whole_point", action="store_true", help="S0 v3: track the whole labelled point (start .. end+0.5 s) so the rally end "
                "can be found from the ball path instead of trusting the last audio contact (needs --prelabels for the far crop)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.tn)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import torch  # noqa: E402
from model import BallTrackerNet  # noqa: E402
from pw_merge import merge_far  # noqa: E402
from ultralytics import YOLO  # noqa: E402

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tn = BallTrackerNet(); tn.load_state_dict(torch.load(os.path.join(a.tn, "model_best.pt"), map_location="cpu")); tn = tn.to(dev).eval()
pose = YOLO(a.pose)
W, H = 640, 360
L = json.load(open(a.labels))
segs = [p for p in L["points"] if p.get("start") is not None]
if a.only_points:
    segs = [p for p in segs if p.get("kind") == "point" and p.get("winner") in ("A", "B")]
SW, SH, SC = 3840, 2160, 3840 / W
NET_Y, COURT_W, COURT_L = 11.885, 10.97, 23.77
hits, views = [], []
if a.prelabels:
    PL = json.load(open(a.prelabels))
    SW, SH = int(PL.get("width", SW)), int(PL.get("height", SH)); SC = SW / W
    hits = sorted(c["t"] for r in PL["rallies"] for c in r["contacts"])
    cbv = [dict(c) for c in PL["courts_by_view"]]
    if a.court_override:
        ov = json.load(open(a.court_override)); ov = ov if isinstance(ov, list) else [ov]
        for o in ov: cbv[o["view"]].update(H_img2court=o["H_img2court"], detected=True, manual=True)
    for c in cbv:
        Hi = np.linalg.inv(np.array(c["H_img2court"], float))
        views.append((c["t0"], c["t1"], Hi, bool(c.get("detected")) or c.get("line_support", 0) >= 0.5))

def c2i(Hi, x, y):
    v = Hi @ [x, y, 1.0]; return v[:2] / v[2]

def far_roi(t):
    """far-half crop (x0, y0, w, h) in source px: far court + 3.5 m sides/5.5 m back, +player height above, clipped."""
    v = [q for q in views if q[0] <= t < q[1]] or views[-1:]
    if not v: return None
    _, _, Hi, ok = v[0]
    pts = np.array([c2i(Hi, x, y) for x in (-3.5, COURT_W + 3.5) for y in (NET_Y - 1.0, COURT_L + 5.5)])
    nb = c2i(Hi, COURT_W / 2, 0.0)                          # same bad-fit test as pw_stages.View (near baseline off-frame)
    if not ok or not np.isfinite(pts).all() or not (-2000 < nb[0] < 6000): return None
    xs = np.clip(pts[:, 0], 0, SW); ys = pts[:, 1]
    ynet = max(ys[0], ys[2]); yback = min(ys[1], ys[3])
    hpx = max(ynet - yback, 150.0)                          # far half height in px; a standing player is ~0.6 of it
    y0 = int(np.clip(yback - 0.9 * hpx, 0, SH)); y1 = int(np.clip(ynet + 0.15 * hpx, 0, SH))
    x0, x1 = int(xs.min()), int(xs.max())
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2; w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    return (x0, y0, w, h) if w >= 320 and h >= 160 else None

def window(p):
    hp = [h for h in hits if p["start"] <= h <= p["end"] - 0.3]
    if a.whole_point:                                       # from 2 s before the first audio hit (pre-serve lead trimmed) to end + 0.5 s
        h0 = [h for h in hits if p["start"] <= h <= p["end"]]
        t0 = max(p["start"], (h0[0] - 2.0) if h0 else p["start"], 0.0); return t0, p["end"] + 0.5 - t0, (hp[-1] if hp else np.nan), 2
    if hp:
        tc = hp[-1]; return max(tc - a.tc_pre, 0.0), a.tc_pre + a.tc_post, tc, 1
    t0 = max(p["start"] - 1.0, p["end"] - a.pre, 0.0); return t0, p["end"] + a.post - t0, np.nan, 0
log = open(os.path.join(a.out, "progress.log"), "a")

def far_players(ff, roi, nk):
    """native-res far-half pose; returns (nk,8,5) boxes and (nk,8,17,3) kps in SOURCE px."""
    x0, y0 = roi[:2]; FB = np.full((nk, 8, 5), np.nan, np.float32); FP = np.full((nk, 8, 17, 3), np.nan, np.float32)
    for s in range(0, min(len(ff), nk), 4):
        res = pose.predict(list(ff[s:min(s + 4, nk)]), device=dev, verbose=False, imgsz=a.far_imgsz, conf=0.25, classes=[0])
        for j, r in enumerate(res):
            if r.boxes is None or not len(r.boxes): continue
            bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); kp = r.keypoints.data.cpu().numpy()
            o = np.argsort(-cf)[:8]; bx = bx[o] + [x0, y0, x0, y0]; kp = kp[o].copy(); kp[..., 0] += x0; kp[..., 1] += y0
            FB[s + j, :len(o), :4] = bx; FB[s + j, :len(o), 4] = cf[o]; FP[s + j, :len(o)] = kp
    return FB, FP

def merge(B, P, FB, FP, roi):
    return merge_far(B, P, FB, FP, roi, SC)

def ball(fr):
    n = len(fr); xy = np.full((n, 2), np.nan, np.float32); pk = np.zeros(n, np.float32)
    idx = list(range(2, n))                                 # float conversion per batch: whole points are up to ~2000 frames
    for s in range(0, len(idx), a.batch):
        b = idx[s:s + a.batch]
        inp = np.stack([np.concatenate([fr[i], fr[i - 1], fr[i - 2]], 2).transpose(2, 0, 1) for i in b]).astype(np.float32) / 255.0
        with torch.inference_mode():
            out = tn(torch.from_numpy(inp).to(dev)).argmax(1).cpu().numpy().astype(np.uint8).reshape(-1, H, W)
        for i, hm in zip(b, out):
            m = hm.max(); pk[i] = m / 255.0
            if m < 128: continue
            _, th = cv2.threshold(hm, 127, 255, cv2.THRESH_BINARY)
            c = cv2.HoughCircles(th, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
            if c is not None and len(c) == 1: xy[i] = c[0][0][:2]
            else:
                ys, xs = np.nonzero(hm == m); xy[i] = (xs.mean(), ys.mean())
    return xy, pk

def players(fr):
    k = list(range(0, len(fr), a.pose_every)); P = np.full((len(k), 8, 17, 3), np.nan, np.float32); B = np.full((len(k), 8, 5), np.nan, np.float32)
    for s in range(0, len(k), 8):
        res = pose.predict(list(fr[k[s:s + 8]]), device=dev, verbose=False, imgsz=640, conf=0.25, classes=[0])
        for j, r in enumerate(res):
            if r.boxes is None or not len(r.boxes): continue
            bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); kp = r.keypoints.data.cpu().numpy()
            o = np.argsort(-cf)[:8]
            B[s + j, :len(o), :4] = bx[o]; B[s + j, :len(o), 4] = cf[o]; P[s + j, :len(o)] = kp[o]
    return np.array(k), B, P

CHUNK = 270                                                  # frames per chunk (multiple of pose_every): bounded memory on whole points


def track_stream(t0, dur, roi):
    """Stream the window through ffmpeg in CHUNK-frame pieces (640x360 for ball + pose, 4K far crop at fps/pose_every for
    far players). TrackNet gets the 2 previous frames across chunk joins, so the result equals the one-shot version."""
    base = subprocess.Popen([a.ffmpeg, "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{t0:.3f}", "-i", a.video, "-t", f"{dur:.3f}",
                             "-vf", f"fps={a.fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    far = None
    if roi is not None:
        x0, y0, w, h = roi
        far = subprocess.Popen([a.ffmpeg, "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{t0:.3f}", "-i", a.video, "-t", f"{dur:.3f}",
                                "-vf", f"fps={a.fps / a.pose_every},crop={w}:{h}:{x0}:{y0}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               stdout=subprocess.PIPE)
    fs = W * H * 3; out = dict(xy=[], pk=[], k=[], B=[], P=[], FB=[], FP=[]); prev = None; off = 0; mid = None; t_far = 0.0
    while True:
        raw = base.stdout.read(CHUNK * fs)
        n = len(raw) // fs
        if n == 0:
            break
        fr = np.frombuffer(raw[:n * fs], np.uint8).reshape(n, H, W, 3)
        if mid is None: mid = fr[n // 2].copy()
        ext = fr if prev is None else np.concatenate([prev, fr])
        xy, pk = ball(ext)
        if prev is not None: xy, pk = xy[2:], pk[2:]
        k, B, P = players(fr)
        if far is not None:
            tf = time.time(); x0, y0, w, h = roi; ffs = w * h * 3
            fraw = far.stdout.read(len(k) * ffs); nf = len(fraw) // ffs
            ff = np.frombuffer(fraw[:nf * ffs], np.uint8).reshape(nf, h, w, 3)
            FB, FP = far_players(ff, roi, len(k)); B, P = merge(B, P, FB, FP, roi)
            out["FB"].append(FB); out["FP"].append(FP); t_far += time.time() - tf
        for key, v in (("xy", xy), ("pk", pk), ("k", k + off), ("B", B), ("P", P)): out[key].append(v)
        prev = fr[-2:].copy(); off += n
    base.wait()
    if far is not None: far.stdout.close(); far.wait()
    cat = lambda key: np.concatenate(out[key]) if out[key] else None
    return cat("xy"), cat("pk"), cat("k"), cat("B"), cat("P"), cat("FB"), cat("FP"), mid, off, t_far


T = time.time(); done = 0
for p in segs:
    dst = os.path.join(a.out, f"seg{p['idx']:03d}.npz")
    if os.path.exists(dst): continue
    if a.limit and done >= a.limit: break
    if a.prelabels: t0, dur, tc, mode = window(p)
    else: t0 = max(p["start"] - 1.0, p["end"] - a.pre, 0.0); dur = p["end"] + a.post - t0; tc, mode = np.nan, 0
    t1 = time.time()
    roi = far_roi(p["end"]) if a.prelabels else None
    xy, pk, k, B, P, FB, FP, mid, nfr, t_far = track_stream(t0, dur, roi); t2 = time.time() - t_far
    nfar = float(np.mean(np.sum(np.isfinite(FB[:, :, 0]) & (FB[:, :, 4] >= 0.4), 1))) if FB is not None else 0.0
    cv2.imwrite(os.path.join(a.out, f"seg{p['idx']:03d}.jpg"), mid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    extra = dict(tc=tc, window_mode=mode, far_roi=np.array(roi if roi else [0, 0, 0, 0]))
    if FB is not None: extra.update(far_boxes=FB, far_kp=FP.astype(np.float16))
    np.savez_compressed(dst + ".tmp.npz", t0=t0, fps=a.fps, ball=xy, ball_peak=pk, pose_k=k, boxes=B, kp=P.astype(np.float16), **extra)
    os.replace(dst + ".tmp.npz", dst); done += 1
    print(f"{time.strftime('%H:%M:%S')} seg {p['idx']} frames={nfr} det={np.mean(~np.isnan(xy[:,0])):.2f} win={['end', 'tc', 'whole'][mode]} "
          f"far_roi={roi} far_per_frame={nfar:.2f} base={t2-t1:.1f}s far={t_far:.1f}s total={time.time()-t1:.1f}s", file=log, flush=True)
print(f"{time.strftime('%H:%M:%S')} DONE {len(segs)} segs in {time.time()-T:.0f}s", file=log, flush=True)
