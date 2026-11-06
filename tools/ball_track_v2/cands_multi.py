"""Per-frame ball candidates (top-k heatmap peaks) from any of several ball detectors, in one format, for the
head-to-head benchmark and the fused multi-cue tracker (tools/ball_track_v2). Mac research tool (MPS); outputs are
private (never committed).

Backends (weights are downloaded at runtime, never committed; licences in docs/engineering-notes Legal list):
  heatnet      own 3-frame heatmap U-Net (run1 weights), 768x432
  wasb         WASB-SBDT HRNet, tennis weights (nttcom/WASB-SBDT, MIT), 3-in/3-out at 512x288, ImageNet-normalised
  tnv2         TrackNetV2 (WASB model zoo tennis weights), 3-in/3-out 512x288
  restnv2      ResTrackNetV2 (WASB zoo tennis weights)
  monotrack    MonoTrack (WASB zoo tennis weights)
  tnv3         TrackNetV3 (MIT, badminton weights; vendored model), 8 frames + median background at 512x288
Option --tile far: run the backend a second time on an upscaled crop around the far half of the court (from the clip's
stored court) and merge its peaks (test-time tiling for the few-pixel far ball).

Output OUT/<clip>.npz: cands (n, K, 3) float32 [x, y, peak] in 1280x720 px, NaN-padded, best first; t0; fps;
ms_model (per frame, model forward only); ms_total (per frame incl. decode/prep/peaks).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AW, AH = 1280, 720
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def peaks(prob, th, k, sx, sy, ox=0.0, oy=0.0, nms=7):
    """top-k local maxima of a (H, W) probability map -> (k, 3) [x, y, p] in analysis px (sub-pixel centroid)."""
    from scipy.ndimage import maximum_filter
    pk = (prob >= th) & (prob == maximum_filter(prob, size=nms))
    ys, xs = np.nonzero(pk)
    o = np.argsort(-prob[ys, xs])[:k]
    out = np.full((k, 3), np.nan, np.float32)
    for j, q in enumerate(o):
        y, x = ys[q], xs[q]
        y0, y1, x0, x1 = max(0, y - 1), min(prob.shape[0], y + 2), max(0, x - 1), min(prob.shape[1], x + 2)
        p = prob[y0:y1, x0:x1]; gy, gx = np.mgrid[y0:y1, x0:x1]; s = float(p.sum())
        out[j] = (ox + (gx * p).sum() / s * sx, oy + (gy * p).sum() / s * sy, prob[y, x])
    return out


def merge_cands(a, b, k, d_px=6.0):
    """union of two (K, 3) candidate lists (b = tile peaks), duplicates within d_px collapsed to the stronger, best first."""
    rows = [r for r in np.r_[a, b] if np.isfinite(r[0])]
    rows.sort(key=lambda r: -r[2])
    keep = []
    for r in rows:
        if all(np.hypot(r[0] - q[0], r[1] - q[1]) > d_px for q in keep):
            keep.append(r)
    out = np.full((k, 3), np.nan, np.float32)
    for j, r in enumerate(keep[:k]):
        out[j] = r
    return out


class AttrDict(dict):
    """dict with attribute access (the WASB HRNet reads cfg.MODEL...), recursive."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return AttrDict(v) if isinstance(v, dict) and not isinstance(v, AttrDict) else v

    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return AttrDict(v) if isinstance(v, dict) and not isinstance(v, AttrDict) else v


class WasbZoo:
    """WASB-SBDT model-zoo networks (3 frames in, 3 heatmaps out, 512x288). wasb_src = a checkout of nttcom/WASB-SBDT/src."""
    W, H, T = 512, 288, 3

    def __init__(self, name, weights, wasb_src, device):
        import torch
        sys.path.insert(0, wasb_src)
        from models import build_model  # noqa: E402  (WASB-SBDT src/models)
        cfgname = {"wasb": "wasb", "tnv2": "tracknetv2", "restnv2": "restracknetv2", "monotrack": "monotrack"}[name]
        cfg = json.load(open(os.path.join(wasb_src, "configs", "model", cfgname + ".json")))   # yaml configs pre-converted
        self.torch, self.device = torch, device
        self.net = build_model(AttrDict({"model": cfg}))
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        self.net.load_state_dict(sd); self.net.to(device).eval()

    def prep(self, f):
        import cv2
        x = cv2.resize(f, (self.W, self.H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.0
        return ((x - MEAN) / STD).transpose(2, 0, 1)

    def run(self, frames):
        """list of n BGR frames -> (n, H, W) probability maps (non-overlapping 3-frame windows, tail padded)."""
        n = len(frames); out = np.zeros((n, self.H, self.W), np.float32); ms = 0.0
        xs = [self.prep(f) for f in frames]
        for s in range(0, n, self.T):
            idx = [min(n - 1, s + j) for j in range(self.T)]
            x = np.concatenate([xs[i] for i in idx], 0)[None]
            t = time.perf_counter()
            with self.torch.no_grad():
                y = self.net(self.torch.from_numpy(x).to(self.device))
                y = (y[0] if isinstance(y, dict) else y).sigmoid().float().cpu().numpy()[0]
            ms += time.perf_counter() - t
            for j in range(self.T):
                if s + j < n:
                    out[s + j] = y[j]
        return out, ms


class TNV3:
    W, H, T = 512, 288, 8

    def __init__(self, weights, device):
        from opencourtvision.ball.tracknet3 import TrackNetV3
        self.m = TrackNetV3(weights=weights, device=device)

    def run(self, frames):
        n = len(frames); out = np.zeros((n, self.H, self.W), np.float32); ms = 0.0
        self.m.set_background(frames[:: max(1, n // 30)])
        for s in range(0, n, self.T):
            t = time.perf_counter()
            hm = self.m.heatmaps(frames[s:s + self.T])
            ms += time.perf_counter() - t
            out[s:s + len(hm)] = hm
        return out, ms


class HeatNet:
    def __init__(self, weights, device):
        import torch

        from opencourtvision.ball.own.heatnet import SIZE, build_model
        self.torch, self.device = torch, device
        self.W, self.H = SIZE
        self.net = build_model(16); sd = torch.load(weights, map_location="cpu"); self.net.load_state_dict(sd.get("model", sd))
        self.net.to(device).eval()

    def run(self, frames):
        import cv2
        n = len(frames); out = np.zeros((n, self.H, self.W), np.float32); ms = 0.0
        sm = [cv2.resize(f, (self.W, self.H), interpolation=cv2.INTER_AREA)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0 for f in frames]
        B = 8
        for s in range(0, n, B):
            idx = range(s, min(n, s + B))
            x = np.stack([np.concatenate([sm[max(0, i - 1)], sm[i], sm[min(n - 1, i + 1)]], 0) for i in idx])
            t = time.perf_counter()
            with self.torch.no_grad():
                y = self.torch.sigmoid(self.net(self.torch.from_numpy(x).to(self.device))).float().cpu().numpy()[:, 0]
            ms += time.perf_counter() - t
            out[s:s + len(y)] = y
        return out, ms


def far_tile(Hc2i, aspect=16 / 9, margin=1.5):
    """crop box (x0, y0, w, h) in analysis px around the far half of the court (+ margin m, + 2.5 m above the baseline
    for lobs), aspect-locked so the network sees it at a uniform upscale. None without a court."""
    if Hc2i is None or not np.isfinite(Hc2i).all():
        return None
    pts = []
    for X, Y in ((-margin, 11.885), (10.97 + margin, 11.885), (-margin, 23.77 + margin), (10.97 + margin, 23.77 + margin)):
        p = Hc2i @ np.array([X, Y, 1.0]); pts.append(p[:2] / p[2])
    pts = np.array(pts)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max(); y1 = pts[:, 1].max(); y0 = pts[:, 1].min() - 0.35 * (y1 - pts[:, 1].min())
    w = max(x1 - x0, (y1 - y0) * aspect); h = w / aspect; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(w, AW), min(h, AH)
    x0 = float(np.clip(cx - w / 2, 0, AW - w)); y0 = float(np.clip(cy - h / 2, 0, AH - h))
    if w > 0.8 * AW:          # far court already fills the frame: tiling buys nothing
        return None
    return x0, y0, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--wasb_src"); ap.add_argument("--split", default="all"); ap.add_argument("--source", default="dev_match")
    ap.add_argument("--clips", help="comma list of clip names (overrides --split)")
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--th", type=float, default=0.10)
    ap.add_argument("--tile", choices=["none", "far"], default="none"); ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    import cv2
    import torch
    torch.set_num_threads(a.threads)
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if a.backend in ("wasb", "tnv2", "restnv2", "monotrack"):
        net = WasbZoo(a.backend, a.weights, a.wasb_src, dev)
    elif a.backend == "tnv3":
        net = TNV3(a.weights, dev)
    elif a.backend == "heatnet":
        net = HeatNet(a.weights, dev)
    else:
        raise SystemExit(f"unknown backend {a.backend}")
    idx = json.load(open(os.path.join(a.data, "index.json")))["clips"]
    if a.clips:
        names = a.clips.split(",")
    else:
        names = [n for n, m in sorted(idx.items()) if m["source"] == a.source and (a.split == "all" or m["split"] == a.split)]
    si, sn = (int(v) for v in a.shard.split("/"))
    os.makedirs(a.out, exist_ok=True)
    for name in names[si::sn]:
        dst = os.path.join(a.out, name + ".npz")
        if os.path.exists(dst):
            continue
        t_all = time.perf_counter()
        z = np.load(os.path.join(a.data, "clips", name + ".npz")); n = idx[name]["n"]
        fd = os.path.join(a.data, "frames", name)
        frames = [cv2.imread(os.path.join(fd, f"{i:05d}.jpg")) for i in range(n)]
        frames = [f for f in frames if f is not None]
        n = len(frames)
        hm, ms = net.run(frames)
        sx, sy = AW / hm.shape[2], AH / hm.shape[1]
        cands = np.stack([peaks(hm[i], a.th, a.k, sx, sy) for i in range(n)])
        tile = far_tile(z["Hc2i"]) if a.tile == "far" else None
        if tile is not None:
            x0, y0, w, h = tile
            fs = frames[0].shape[1] / AW          # clip JPGs are stored at 768x432; the tile box is in 1280x720 px
            crops = [cv2.resize(f[int(y0 * fs):int((y0 + h) * fs), int(x0 * fs):int((x0 + w) * fs)], (AW, AH), interpolation=cv2.INTER_CUBIC)
                     for f in frames]
            hm2, ms2 = net.run(crops); ms += ms2
            s2x, s2y = w / hm2.shape[2], h / hm2.shape[1]
            for i in range(n):
                cands[i] = merge_cands(cands[i], peaks(hm2[i], a.th, a.k, s2x, s2y, x0, y0), a.k)
        tot = time.perf_counter() - t_all
        np.savez_compressed(dst + ".tmp.npz", cands=cands.astype(np.float32), t0=float(z["t0"]), fps=30.0,
                            ms_model=1000 * ms / max(1, n), ms_total=1000 * tot / max(1, n), tile=np.array(tile or [np.nan] * 4, float))
        os.replace(dst + ".tmp.npz", dst)
        print(f"{time.strftime('%T')} {a.backend}{'+tile' if tile else ''} {name} n={n} model {1000 * ms / max(1, n):.1f} ms/f total {1000 * tot / max(1, n):.1f} ms/f", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
