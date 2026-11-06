"""Vamos ball heatmap network (own implementation; no third-party code or weights).

Input: 3 consecutive RGB frames stacked on channels (B, 9, H, W, values 0..1). Inside the model two absolute grey
frame differences are appended (motion cue for a 2-12 px ball), so an exported Core ML model takes raw frames only.
Output: one heatmap (logits) for the MIDDLE frame at full input resolution; the ball is the peak above a threshold.
This is the TrackNet idea (Huang et al. 2019, arXiv 1907.03698; TrackNetV2, Sun et al. 2020) re-written from the
papers' description with a much smaller encoder-decoder: 4 levels (base, 2b, 4b, 8b channels), additive skips with
1x1 projections (cheaper than concatenation), nearest upsampling. base=16 -> ~0.33 M parameters, ~5 GMAC at 768x432.

Training targets: a Gaussian (sigma ~2.5 px at 768 wide) at the pseudo-label, focal loss (CenterNet-style). Weights
are trained by us (tools/ball_tracker) and never committed.
"""
import numpy as np

SIZE = (768, 432)       # (W, H) network input; divisible by 8
SIGMA = 2.5


def gaussian_target(h, w, xy, sigma=SIGMA):
    """(h, w) float32 heatmap with a unit-peak Gaussian at xy (or all zeros when xy is None/NaN)."""
    hm = np.zeros((h, w), np.float32)
    if xy is None or not np.all(np.isfinite(xy)):
        return hm
    x, y = float(xy[0]), float(xy[1])
    r = int(3 * sigma) + 1
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 2)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 2)
    if x0 >= x1 or y0 >= y1:
        return hm
    gx = np.exp(-((np.arange(x0, x1) - x) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((np.arange(y0, y1) - y) ** 2) / (2 * sigma ** 2))
    hm[y0:y1, x0:x1] = np.outer(gy, gx)
    return hm


def decode_heatmap(prob, th=0.5, max_blobs=3, nms=7):
    """prob (H, W) in 0..1 -> [(x, y, peak)] local maxima >= th, best first, sub-pixel by 3x3 weighted centroid."""
    from scipy.ndimage import maximum_filter
    prob = np.asarray(prob, np.float32)
    peaks = (prob >= th) & (prob == maximum_filter(prob, size=nms))
    ys, xs = np.nonzero(peaks)
    out = []
    for y, x in sorted(zip(ys, xs), key=lambda p: -prob[p[0], p[1]])[:max_blobs]:
        y0, y1, x0, x1 = max(0, y - 1), min(prob.shape[0], y + 2), max(0, x - 1), min(prob.shape[1], x + 2)
        p = prob[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        s = float(p.sum())
        out.append((float((gx * p).sum() / s), float((gy * p).sum() / s), float(prob[y, x])))
    return out


def build_model(base=16, in_frames=3):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def cbr(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class BallHeatNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.nf = in_frames
            cin = 3 * in_frames + (in_frames - 1)
            self.e0 = nn.Sequential(cbr(cin, c[0]), cbr(c[0], c[0]))
            self.e1 = nn.Sequential(cbr(c[0], c[1]), cbr(c[1], c[1]))
            self.e2 = nn.Sequential(cbr(c[1], c[2]), cbr(c[2], c[2]))
            self.e3 = nn.Sequential(cbr(c[2], c[3]), cbr(c[3], c[3]))
            self.pool = nn.MaxPool2d(2)
            self.l2, self.d2 = nn.Conv2d(c[3], c[2], 1), cbr(c[2], c[2])
            self.l1, self.d1 = nn.Conv2d(c[2], c[1], 1), cbr(c[1], c[1])
            self.l0, self.d0 = nn.Conv2d(c[1], c[0], 1), cbr(c[0], c[0])
            self.head = nn.Conv2d(c[0], 1, 1)
            nn.init.constant_(self.head.bias, -4.0)       # start near "no ball" (rare positives)

        def forward(self, x):
            b, _, h, w = x.shape
            g = x.view(b, self.nf, 3, h, w).mean(2)
            d = (g[:, 1:] - g[:, :-1]).abs() * 4.0
            x = torch.cat([x, d], 1)
            e0 = self.e0(x); e1 = self.e1(self.pool(e0)); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            u = self.d2(F.interpolate(self.l2(e3), scale_factor=2, mode="nearest") + e2)
            u = self.d1(F.interpolate(self.l1(u), scale_factor=2, mode="nearest") + e1)
            u = self.d0(F.interpolate(self.l0(u), scale_factor=2, mode="nearest") + e0)
            return self.head(u)

    return BallHeatNet()


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss; target in 0..1 with 1 at the ball centre. Normalised by #positives (>=1)."""
    import torch
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    pl = -((1 - p) ** alpha) * torch.log(p) * pos
    nl = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    return (pl.sum() + nl.sum()) / pos.sum().clamp(min=1.0)


def param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def macs(base=16, size=SIZE, in_frames=3):
    """Multiply-accumulates per frame (convolutions only), for sizing against the Neural Engine budget."""
    w, h = size
    c = [base, base * 2, base * 4, base * 8]
    cin = 3 * in_frames + (in_frames - 1)
    px = [w * h / 4 ** k for k in range(4)]
    m = px[0] * 9 * (cin * c[0] + c[0] * c[0])
    for k in (1, 2, 3):
        m += px[k] * 9 * (c[k - 1] * c[k] + c[k] * c[k])
    for k in (2, 1, 0):
        m += px[k] * (c[k + 1] * c[k] + 9 * c[k] * c[k])
    return int(m + px[0] * c[0])


def to_input(frames_bgr, size=SIZE):
    """3 BGR uint8 frames (any size) -> float32 (9, H, W) RGB 0..1 at the network size."""
    import cv2
    w, h = size
    fs = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[1] != w or f.shape[0] != h else f for f in frames_bgr]
    return np.concatenate([f[:, :, ::-1].transpose(2, 0, 1) for f in fs], 0).astype(np.float32) / 255.0


class HeatNetTracker:
    """Per-frame ball candidates from a trained BallHeatNet. frames: list of BGR frames (one clip, any size)."""

    def __init__(self, weights, device=None, size=SIZE, base=16, th=0.5):
        import torch
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.size, self.th = device, tuple(size), th
        self.model = build_model(base)
        sd = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(sd.get("model", sd))
        self.model.to(device).eval()

    def run(self, frames, batch=8):
        """-> (cands per frame [(x, y, peak)] in the frames' px, seconds spent in the network only)."""
        import time
        torch = self.torch
        n = len(frames)
        if n == 0:
            return [], 0.0
        sx, sy = frames[0].shape[1] / self.size[0], frames[0].shape[0] / self.size[1]
        small = [to_input([f], self.size) for f in frames]
        out, spent = [], 0.0
        for s in range(0, n, batch):
            idx = range(s, min(n, s + batch))
            x = np.stack([np.concatenate([small[max(0, i - 1)], small[i], small[min(n - 1, i + 1)]], 0) for i in idx])
            t = time.perf_counter()
            with torch.no_grad():
                pr = torch.sigmoid(self.model(torch.from_numpy(x).to(self.device))).float().cpu().numpy()[:, 0]
            spent += time.perf_counter() - t
            for hm in pr:
                out.append([(x_ * sx, y_ * sy, p) for x_, y_, p in decode_heatmap(hm, self.th)])
        return out, spent
