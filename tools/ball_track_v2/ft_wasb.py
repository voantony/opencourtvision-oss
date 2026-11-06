"""Fine-tune the WASB-SBDT HRNet (tennis weights) on our footage: the same physics-filtered pseudo-labels the own
heatmap net was trained on (run1 data, train split: dev_match points 1-50 + CC train videos; val = CC val videos).
3 frames in -> 3 heatmaps out at 512x288, each output supervised by its own frame's label (masked when that frame has
none). Loss: weighted BCE on a Gaussian target (sigma 2.5 px) with positive-pixel weight. Augment: scale/shift, flip,
brightness/contrast. Mac MPS, time-capped, nice 20 (run via the queue). Weights private (pseudo-labels from Antony's
footage), never committed.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 512, 288
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def gauss(xy, sigma=2.5):
    hm = np.zeros((H, W), np.float32)
    if xy is None or not np.isfinite(xy[0]):
        return hm
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2) / (2 * sigma ** 2)).astype(np.float32)


class Clips:
    def __init__(self, data):
        self.data = data; self.idx = json.load(open(os.path.join(data, "index.json")))["clips"]; self.cache = {}

    def labels(self, name):
        if name not in self.cache:
            z = np.load(os.path.join(self.data, "clips", name + ".npz"))
            self.cache[name] = (z["phys_pos"] * (W / 1280.0), z["phys_w"])
        return self.cache[name]

    def samples(self, split, source=None):
        out = []
        for name, m in self.idx.items():
            if m["split"] != split or (source and m["source"] != source):
                continue
            pos, w = self.labels(name)
            for i in range(1, min(len(w), m["n"]) - 1):
                if w[i] > 0 and np.isfinite(pos[i, 0]):
                    out.append((name, i))
        return out


def item(cl, name, i, aug, crop=None):
    pos, w = cl.labels(name)
    fr = [cv2.imread(os.path.join(cl.data, "frames", name, f"{j:05d}.jpg")) for j in (i - 1, i, i + 1)]
    fr = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) if f is not None else np.zeros((H, W, 3), np.uint8) for f in fr]
    xy = [pos[j].copy() if w[j] > 0 and np.isfinite(pos[j, 0]) else None for j in (i - 1, i, i + 1)]
    if aug:
        s = random.uniform(0.85, 1.2); c = xy[1]
        tx = random.uniform(-0.1, 0.1) * W + (1 - s) * c[0]; ty = random.uniform(-0.1, 0.1) * H + (1 - s) * c[1]
        if not (4 <= s * c[0] + tx < W - 4 and 4 <= s * c[1] + ty < H - 4):
            tx, ty = (1 - s) * c[0], (1 - s) * c[1]
        M = np.float32([[s, 0, tx], [0, s, ty]])
        fr = [cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for f in fr]
        xy = [None if p is None else np.array([s * p[0] + tx, s * p[1] + ty]) for p in xy]
        if random.random() < 0.5:
            fr = [f[:, ::-1] for f in fr]; xy = [None if p is None else np.array([W - 1 - p[0], p[1]]) for p in xy]
        a, b = random.uniform(0.8, 1.2), random.uniform(-20, 20)
        fr = [np.clip(f.astype(np.float32) * a + b, 0, 255).astype(np.uint8) for f in fr]
    x = np.concatenate([((f[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for f in fr], 0)
    y = np.stack([gauss(p) for p in xy]); mk = np.array([p is not None for p in xy], np.float32)
    if crop:              # random crop that contains the middle frame's ball (memory: the HRNet keeps full-res features)
        cw, ch = crop; c = xy[1]
        x0 = int(np.clip(random.uniform(c[0] - cw + 8, c[0] - 8), 0, W - cw)); y0 = int(np.clip(random.uniform(c[1] - ch + 8, c[1] - 8), 0, H - ch))
        x, y = x[:, y0:y0 + ch, x0:x0 + cw], y[:, y0:y0 + ch, x0:x0 + cw]
    return np.ascontiguousarray(x), np.ascontiguousarray(y), mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--wasb_src", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--minutes", type=float, default=50); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--val_n", type=int, default=300); ap.add_argument("--crop", type=int, default=224)
    a = ap.parse_args()
    import torch
    sys.path.insert(0, HERE)
    from cands_multi import WasbZoo
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = WasbZoo("wasb", a.weights, a.wasb_src, dev).net; net.train()
    cl = Clips(a.data); tr = cl.samples("train"); va = cl.samples("val")
    random.seed(0); random.shuffle(va); va = va[:a.val_n]
    print(f"train samples {len(tr)}, val {len(va)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)

    def run_val():
        net.eval(); tp = fp = fn = 0
        with torch.no_grad():
            for s in range(0, len(va), 4):
                b = [item(cl, n, i, False) for n, i in va[s:s + 4]]
                x = torch.from_numpy(np.stack([q[0] for q in b])).to(dev)
                pr = net(x)[0].sigmoid()[:, 1].float().cpu().numpy()
                for (n, i), p in zip(va[s:s + 4], pr):
                    gt = cl.labels(n)[0][i]; k = np.unravel_index(np.argmax(p), p.shape)
                    if p[k] < 0.5:
                        fn += 1
                    elif math.hypot(k[1] - gt[0], k[0] - gt[1]) <= 4:
                        tp += 1
                    else:
                        fp += 1; fn += 1
        net.train(); pr_ = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        return dict(precision=round(pr_, 4), recall=round(rc, 4), f1=round(2 * pr_ * rc / max(1e-9, pr_ + rc), 4))
    hist = [dict(epoch=-1, **run_val())]; print(hist[-1], flush=True); best = hist[-1]["f1"]
    torch.save(dict(model_state_dict=net.state_dict()), a.out)
    t0 = time.time(); ep = 0
    while (time.time() - t0) / 60 < a.minutes:
        random.shuffle(tr); tot = 0.0; nb = 0
        for s in range(0, len(tr), a.bs):
            if (time.time() - t0) / 60 >= a.minutes:
                break
            b = [item(cl, n, i, True, (a.crop, a.crop)) for n, i in tr[s:s + a.bs]]
            x = torch.from_numpy(np.stack([q[0] for q in b])).to(dev); y = torch.from_numpy(np.stack([q[1] for q in b])).to(dev)
            mk = torch.from_numpy(np.stack([q[2] for q in b])).to(dev)[:, :, None, None]
            lo = net(x)[0]
            wpix = 1 + 50 * y
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(lo, y, reduction="none") * wpix * mk).sum() / (mk.sum() * y.shape[-1] * y.shape[-2] / 100).clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
            if nb % 200 == 0:
                print(f"ep {ep} batch {nb} loss {tot / nb:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
            if nb % 500 == 0:          # checkpoint by val F1 inside long epochs
                v = run_val(); hist.append(dict(epoch=ep, batch=nb, loss=round(tot / nb, 4), minutes=round((time.time() - t0) / 60, 1), **v)); print(hist[-1], flush=True)
                if v["f1"] > best:
                    best = v["f1"]; torch.save(dict(model_state_dict=net.state_dict()), a.out)
        v = run_val(); hist.append(dict(epoch=ep, loss=round(tot / max(1, nb), 4), minutes=round((time.time() - t0) / 60, 1), **v)); print(hist[-1], flush=True)
        if v["f1"] > best:
            best = v["f1"]; torch.save(dict(model_state_dict=net.state_dict()), a.out)
        ep += 1
    json.dump(dict(history=hist, best_f1=best, train_samples=len(tr)), open(a.out + ".train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
