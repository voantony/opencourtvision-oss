"""Train the own ball heatmap net on the physics-filtered pseudo-labels (Mac MPS; weights stay private).

  python train.py --data DATA --out models/heatnet_r1.pt --minutes 70
  python train.py --data DATA --out models/heatnet_r2.pt --labels r2 --init models/heatnet_r1.pt --minutes 35

Mac-only budget (Antony, 18 Oct: no cloud GPU; M2 Pro 16 GB): crops instead of full frames (default 384x224 at the
768-wide scale, so a far-court ball keeps its pixels but each step costs ~1/4), MPS fp16 autocast with a grad scaler
(falls back to fp32 if the loss goes non-finite), a curated subset (per clip at most --per_clip frames, hard examples
first: frames filled from an arc where the teacher missed, far-court balls above the net line, fast balls), early
stopping on val F1 (--patience epochs), run under nice 20 by run_phase1.sh. The net is fully convolutional, so
inference still runs on the whole 768x432 frame.

Samples: train-split frames with a pseudo-label (observed inlier or filled-from-arc), 3-frame input (i-1, i, i+1).
Loss: CenterNet focal loss on a Gaussian at the label, weighted by the label weight (filled frames count less).
Augment (same for the 3 frames): scale 0.8-1.25 + shift, horizontal flip, brightness/contrast. Validation: the
val-split clips (CC videos held out by id), pseudo-label recall/precision at 4 px (768 wide), each epoch; the best
epoch by F1 is kept. Time-capped by --minutes.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from opencourtvision.ball.own.heatnet import SIZE, build_model, decode_heatmap, focal_loss, gaussian_target, param_count  # noqa: E402


def load_samples(data, split, labels, per_clip=0, seed=0):
    """-> [(clip, i, x, y, w)] at 768-wide px. per_clip > 0 keeps at most that many per clip, hardest first:
    score = 2 (filled: the teacher missed it) + 1.5 (ball above the net line, i.e. far court) + speed/20 + jitter."""
    rng = np.random.default_rng(seed)
    idx = json.load(open(os.path.join(data, "index.json")))["clips"]
    out = []
    for name, m in idx.items():
        if m["split"] != split:
            continue
        z = np.load(os.path.join(data, "clips", name + ".npz"))
        key_p, key_w = ("r2_pos", "r2_w") if labels == "r2" and "r2_pos" in z.files else ("phys_pos", "phys_w")
        pos, w = z[key_p] * C.K_NET, z[key_w]
        y_net = None
        if "Hc2i" in z.files and np.isfinite(z["Hc2i"]).all():
            y_net = C.project(z["Hc2i"], (C.COURT_W / 2, C.NET_Y))[1] * C.K_NET
        cand = []
        for i in range(1, min(len(w), m["n"]) - 1):
            if w[i] > 0 and np.isfinite(pos[i, 0]):
                sp = float(np.hypot(*(pos[i + 1] - pos[i - 1]))) / 2 if np.isfinite(pos[[i - 1, i + 1], 0]).all() else 0.0
                far = y_net is not None and pos[i, 1] < y_net
                score = 2.0 * (w[i] < 1.0) + 1.5 * far + min(sp, 40.0) / 20.0 + rng.random()
                cand.append((score, (name, i, float(pos[i, 0]), float(pos[i, 1]), float(w[i]))))
        cand.sort(key=lambda c: -c[0])
        out += [c[1] for c in (cand[:per_clip] if per_clip else cand)]
    return out


class DS:
    def __init__(self, data, samples, augment, crop=None):
        self.data, self.s, self.aug, self.crop = data, samples, augment, crop

    def __len__(self):
        return len(self.s)

    def __getitem__(self, k):
        name, i, x, y, w = self.s[k]
        W, H = SIZE
        fr = [cv2.imread(os.path.join(self.data, "frames", name, f"{j:05d}.jpg")) for j in (i - 1, i, i + 1)]
        fr = [f if f is not None else np.zeros((H, W, 3), np.uint8) for f in fr]
        if self.aug:
            s = random.uniform(0.8, 1.25)
            # keep the ball inside the frame after scale + shift
            tx = random.uniform(-0.15, 0.15) * W + (1 - s) * x
            ty = random.uniform(-0.15, 0.15) * H + (1 - s) * y
            nx, ny = s * x + tx, s * y + ty
            if not (4 <= nx < W - 4 and 4 <= ny < H - 4):
                tx, ty = (1 - s) * x, (1 - s) * y; nx, ny = x, y
            M = np.float32([[s, 0, tx], [0, s, ty]])
            fr = [cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for f in fr]
            x, y = nx, ny
            if random.random() < 0.5:
                fr = [f[:, ::-1] for f in fr]; x = W - 1 - x
            a, b = random.uniform(0.8, 1.2), random.uniform(-20, 20)
            fr = [np.clip(f.astype(np.float32) * a + b, 0, 255).astype(np.uint8) for f in fr]
        if self.crop:                                    # random crop that contains the ball (anywhere but the 8 px border)
            cw, ch = self.crop
            x0 = int(np.clip(random.uniform(x - cw + 8, x - 8), 0, W - cw)); y0 = int(np.clip(random.uniform(y - ch + 8, y - 8), 0, H - ch))
            fr = [f[y0:y0 + ch, x0:x0 + cw] for f in fr]; x, y, W, H = x - x0, y - y0, cw, ch
        inp = np.concatenate([f[:, :, ::-1].transpose(2, 0, 1) for f in fr], 0).astype(np.float32) / 255.0
        return np.ascontiguousarray(inp), gaussian_target(H, W, (x, y))[None], np.float32(w)


def evaluate(model, torch, dev, ds, tol=4.0, max_n=600):
    model.eval()
    tp = fp = fn = 0
    n = min(len(ds), max_n)
    step = max(1, len(ds) // n)
    with torch.no_grad():
        for k in range(0, len(ds), step):
            inp, _, _ = ds[k]
            _, _, x, y, _ = ds.s[k]
            pr = torch.sigmoid(model(torch.from_numpy(inp[None]).to(dev)))[0, 0].float().cpu().numpy()
            c = decode_heatmap(pr, 0.5, 1)
            if not c:
                fn += 1
            elif math.hypot(c[0][0] - x, c[0][1] - y) <= tol:
                tp += 1
            else:
                fp += 1; fn += 1
    model.train()
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / max(1e-9, p + r), 4), n=tp + fn)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default="phys", choices=("phys", "r2")); ap.add_argument("--init")
    ap.add_argument("--minutes", type=float, default=70); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", default="384x224", help="train crop WxH at the 768-wide scale ('' = full frame)")
    ap.add_argument("--per_clip", type=int, default=60, help="curated subset: hardest N labelled frames per clip (0 = all)")
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--amp", default="fp16", choices=("fp16", "none"))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = load_samples(a.data, "train", a.labels, a.per_clip, a.seed), load_samples(a.data, "val", a.labels, a.per_clip, a.seed)
    crop = tuple(int(v) for v in a.crop.split("x")) if a.crop else None
    print(f"train samples {len(tr)}  val samples {len(va)}  device {dev}", flush=True)
    if len(tr) < int(os.environ.get("OCV_MIN_SAMPLES", 200)):
        print("too few training samples; stopping"); sys.exit(3)
    model = build_model(a.base)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["model"])
    model.to(dev).train()
    dl = torch.utils.data.DataLoader(DS(a.data, tr, True, crop), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     drop_last=True, persistent_workers=a.workers > 0)
    vds = DS(a.data, va, False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = len(dl)
    total = a.epochs * steps_per_epoch

    def lr_at(step, elapsed_min):                     # warmup, then cosine over whichever runs out first: epochs or time
        frac = max(step / total, elapsed_min / a.minutes)
        return a.lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
    use_amp = a.amp == "fp16" and dev in ("mps", "cuda")
    scaler = torch.amp.GradScaler(dev) if use_amp else None
    t_start = time.time(); best = None; hist = []; step = 0; bad_epochs = 0
    for ep in range(a.epochs):
        run = 0.0; nb = 0
        for inp, tg, w in dl:
            inp, tg, w = inp.to(dev), tg.to(dev), w.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
                logits = model(inp)
            logits = logits.float()
            loss = torch.stack([focal_loss(logits[b:b + 1], tg[b:b + 1]) * w[b] for b in range(len(w))]).mean()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, (time.time() - t_start) / 60)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                if use_amp:                               # fp16 unstable here: continue in fp32
                    print("non-finite loss under fp16; switching to fp32", flush=True); use_amp = False; scaler = None
                continue
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            step += 1; run += loss.item(); nb += 1
            if nb % 200 == 0:
                print(f"ep {ep} it {nb}/{steps_per_epoch} loss {run / nb:.3f} {(time.time() - t_start) / 60:.1f} min", flush=True)
            if (time.time() - t_start) / 60 > a.minutes:
                break
        ev = evaluate(model, torch, dev, vds) if len(vds) else dict(f1=0.0)
        hist.append(dict(epoch=ep, loss=round(run / max(1, nb), 4), minutes=round((time.time() - t_start) / 60, 1), **ev))
        print("epoch", hist[-1], flush=True)
        if best is not None and ev["f1"] < best:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f"early stop after epoch {ep} (no val F1 gain for {a.patience} epochs)", flush=True); break
        else:
            best = ev["f1"]; bad_epochs = 0
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            torch.save(dict(model=model.state_dict(), meta=dict(base=a.base, size=SIZE, params=param_count(model), epoch=ep,
                                                                val=ev, labels=a.labels, train_samples=len(tr), crop=a.crop,
                                                                amp=use_amp, per_clip=a.per_clip)), a.out)
        if (time.time() - t_start) / 60 > a.minutes:
            break
    C.save_json(dict(history=hist, best_f1=best, minutes=round((time.time() - t_start) / 60, 1), amp_fp16=use_amp, train_samples=len(tr), val_samples=len(va), params=param_count(model)),
                os.path.splitext(a.out)[0] + ".train.json")


if __name__ == "__main__":
    main()
