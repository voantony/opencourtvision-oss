"""Per-frame ball CANDIDATES (top-k heatmap peaks, not just the argmax) from the own heatmap net, for the physics
tracker (opencourtvision.ball.physics3d). Mac research tool: CPU only (the MPS GPU is left to other jobs), sharded,
resumable, bounded memory. Outputs are private (never committed).

  --clips DATA   run over run1/data clips (JPG frames, 1280x720)   -> OUT/<clip>.npz
  --segs DIR     run over pw_tracks segment windows (video decode)  -> OUT/segNNN.npz
Each npz: cands (n, K, 3) float32 [x, y, peak] in 1280x720 px, NaN-padded, best first; t0; fps.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def peaks(prob, th, k, nms=7):
    from scipy.ndimage import maximum_filter
    pk = (prob >= th) & (prob == maximum_filter(prob, size=nms))
    ys, xs = np.nonzero(pk)
    o = np.argsort(-prob[ys, xs])[:k]
    out = np.full((k, 3), np.nan, np.float32)
    for j, q in enumerate(o):
        y, x = ys[q], xs[q]
        y0, y1, x0, x1 = max(0, y - 1), min(prob.shape[0], y + 2), max(0, x - 1), min(prob.shape[1], x + 2)
        p = prob[y0:y1, x0:x1]; gy, gx = np.mgrid[y0:y1, x0:x1]; s = float(p.sum())
        out[j] = ((gx * p).sum() / s / C.K_NET, (gy * p).sum() / s / C.K_NET, prob[y, x])
    return out


def run_frames(model, torch, small, th, k, batch=1):
    n = len(small); out = np.full((n, k, 3), np.nan, np.float32)
    for s in range(0, n, batch):
        idx = range(s, min(n, s + batch))
        x = np.stack([np.concatenate([small[max(0, i - 1)], small[i], small[min(n - 1, i + 1)]], 0) for i in idx])
        with torch.no_grad():
            pr = torch.sigmoid(model(torch.from_numpy(x))).numpy()[:, 0]
        for j, i in enumerate(idx):
            out[i] = peaks(pr[j], th, k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clips"); ap.add_argument("--segs"); ap.add_argument("--video")
    ap.add_argument("--source", default="dev_match"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--th", type=float, default=0.10); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import cv2
    import torch
    torch.set_num_threads(a.threads)
    if os.environ.get('OCV_NO_MKLDNN'):
        torch.backends.mkldnn.enabled = False
    from opencourtvision.ball.own.heatnet import build_model, to_input
    m = build_model(16); sd = torch.load(a.weights, map_location="cpu"); m.load_state_dict(sd.get("model", sd)); m.eval()
    si, sn = (int(v) for v in a.shard.split("/"))
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    if a.clips:
        idx = json.load(open(os.path.join(a.clips, "index.json")))["clips"]
        jobs = [("clip", n) for n, mm in sorted(idx.items()) if mm["source"] == a.source]
    if a.segs:
        jobs += [("seg", f[:-4]) for f in sorted(os.listdir(a.segs)) if f.startswith("seg") and f.endswith(".npz")]
    jobs = jobs[si::sn]
    for kind, name in jobs:
        dst = os.path.join(a.out, name + ".npz")
        if os.path.exists(dst):
            continue
        st = time.perf_counter()
        if kind == "clip":
            z = np.load(os.path.join(a.clips, "clips", name + ".npz")); t0 = float(z["t0"])
            fd = os.path.join(a.clips, "frames", name); n = len([f for f in os.listdir(fd) if f.endswith(".jpg")])
            cands = np.full((n, a.k, 3), np.nan, np.float32); s = 0; pad = 1; chunk = 60
            while s < n:
                lo = max(0, s - pad); hi = min(n, s + chunk + pad)
                small = [to_input([cv2.imread(os.path.join(fd, f"{i:05d}.jpg"))]) for i in range(lo, hi)]
                c = run_frames(m, torch, small, a.th, a.k)
                k0, k1 = s - lo, min(len(c), s - lo + chunk)
                cands[s:s + (k1 - k0)] = c[k0:k1]; s += chunk
        else:
            d = np.load(os.path.join(a.segs, name + ".npz")); t0, n = float(d["t0"]), len(d["ball"])
            cands = np.full((n, a.k, 3), np.nan, np.float32); s = 0; pad = 2; chunk = 60
            while s < n:
                lo = max(0, s - pad); hi = min(n, s + chunk + pad)
                fr = C.decode(a.video, t0 + lo / C.AFPS, hi - lo)
                if not fr:
                    break
                small = [to_input([f]) for f in fr]; del fr
                c = run_frames(m, torch, small, a.th, a.k)
                k0, k1 = s - lo, min(len(c), s - lo + chunk)
                cands[s:s + (k1 - k0)] = c[k0:k1]; s += chunk
        np.savez_compressed(dst + ".tmp.npz", cands=cands, t0=t0, fps=C.AFPS); os.replace(dst + ".tmp.npz", dst)
        print(f"{time.strftime('%T')} {name} n={len(cands)} {1000 * (time.perf_counter() - st) / max(1, len(cands)):.0f} ms/f", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
