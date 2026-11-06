"""TrackNetV3 ball detector backend (MIT, vendored model in `_vendor/tracknetv3`).

TrackNetV3 (Chen & Wang, ACM MM Asia 2023, https://github.com/qaz812345/TrackNetV3) takes a median background frame plus
8 consecutive RGB frames at 512x288 ("concat" background mode) and outputs 8 ball heatmaps. Upstream weights are trained
on badminton (Shuttlecock Trajectory Dataset); on tennis they are a starting point to fine-tune, not a finished model.

Weights are downloaded at runtime (never committed): `ocv-fetch-weights tracknetv3` puts TrackNet_best.pt (and
InpaintNet_best.pt) under $OCV_CACHE/tracknetv3 (default ~/.cache/opencourtvision/tracknetv3). SHA-256 is checked.

Pre/post-processing is our own reimplementation of the upstream behaviour (resize -> /255 -> concat median; heatmap > 0.5
-> connected components), extended to return every blob with its peak score instead of only the largest blob, so the
live-ball filter downstream can choose. Device: CUDA, else Apple MPS, else CPU (`pick_device`).
"""
import hashlib
import os
import zipfile

import numpy as np

from .. import privacy

W, H = 512, 288
SEQ_LEN = 8
GDRIVE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"     # TrackNetV3_ckpts.zip, linked from the upstream README
SHA256 = {
    "TrackNet_best.pt": "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a",
    "InpaintNet_best.pt": "5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c",
}


def cache_dir():
    return os.path.join(os.environ.get("OCV_CACHE", os.path.expanduser("~/.cache/opencourtvision")), "tracknetv3")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def fetch_weights(dest=None, zip_path=None):
    """Download (or unpack a given) TrackNetV3_ckpts.zip into dest and verify SHA-256. Returns the TrackNet .pt path."""
    dest = dest or cache_dir(); os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, "TrackNet_best.pt")
    if os.path.exists(target) and _sha256(target) == SHA256["TrackNet_best.pt"]:
        return target
    if zip_path is None:
        zip_path = os.path.join(dest, "TrackNetV3_ckpts.zip")
        if privacy.beta_mode():
            # Beta: never Google Drive. Our EU host (files are SHA-256 checked below) or the feature is disabled.
            privacy.download(privacy.model_url("tracknetv3/TrackNetV3_ckpts.zip", ""), zip_path)
        else:
            try:
                import gdown  # optional (MIT); handles Google Drive's large-file confirmation page
                gdown.download(id=GDRIVE_ID, output=zip_path, quiet=True)
            except ImportError:
                privacy.download(f"https://drive.usercontent.google.com/download?id={GDRIVE_ID}&export=download&confirm=t", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in SHA256:
                with z.open(name) as src, open(os.path.join(dest, base), "wb") as out: out.write(src.read())
    for base, digest in SHA256.items():
        p = os.path.join(dest, base)
        if not os.path.exists(p) or _sha256(p) != digest:
            raise RuntimeError(f"{base}: missing or SHA-256 mismatch in {dest}")
    return target


def pick_device(torch, want=None):
    if want: return want
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def heatmap_blobs(hm, th=0.5, scale=(1.0, 1.0), max_blobs=4):
    """Heatmap (H, W) in [0, 1] -> [(x, y, score)] in source px, strongest first. score = peak heatmap value."""
    import cv2
    m = (hm > th).astype(np.uint8)
    if not m.any(): return []
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for k in range(1, n):
        peak = float(hm[lab == k].max())
        out.append((float(cen[k][0] * scale[0]), float(cen[k][1] * scale[1]), peak))
    return sorted(out, key=lambda b: -b[2])[:max_blobs]


class TrackNetV3:
    """Usage: tn = TrackNetV3(); tn.set_background(frames_sample); blobs = tn.detect(frames8) -> 8 lists of (x, y, score)."""

    def __init__(self, weights=None, device=None, threads=None):
        import torch

        from ._vendor.tracknetv3 import TrackNet
        self.torch = torch
        if threads: torch.set_num_threads(threads)
        self.device = pick_device(torch, device)
        weights = weights or privacy.bundled("tracknetv3/TrackNet_best.pt") or os.path.join(cache_dir(), "TrackNet_best.pt")
        privacy.require_local(weights, "TrackNetV3 ball tracker")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"{weights} not found; run `ocv-fetch-weights tracknetv3` first")
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        p = ck["param_dict"]
        if p.get("bg_mode") != "concat" or p.get("seq_len") != SEQ_LEN:
            raise ValueError(f"unsupported TrackNetV3 checkpoint config {p.get('bg_mode')!r}/{p.get('seq_len')}")
        net = TrackNet(in_dim=(SEQ_LEN + 1) * 3, out_dim=SEQ_LEN)
        net.load_state_dict(ck["model"])
        self.net = net.to(self.device).eval()
        self.bg = None
        self.windows_run = 0

    @staticmethod
    def _prep(frame_bgr):
        import cv2
        return cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)

    def set_background(self, frames_bgr):
        """Median of a sample of frames (BGR, any size) = the upstream 'median image'."""
        self.bg = np.median(np.stack([self._prep(f) for f in frames_bgr]), 0).astype(np.float32)

    def heatmaps(self, frames_bgr):
        """8 BGR frames (same size) -> (8, 288, 512) heatmaps. Fewer than 8 frames are padded with the last one."""
        fr = list(frames_bgr)
        n = len(fr)
        if n == 0: return np.zeros((0, H, W), np.float32)
        fr = fr + [fr[-1]] * (SEQ_LEN - n)
        imgs = [self._prep(f) for f in fr]
        if self.bg is None: self.bg = np.median(np.stack(imgs), 0)
        x = np.concatenate([np.moveaxis(self.bg, -1, 0)] + [np.moveaxis(i, -1, 0) for i in imgs], 0) / 255.0
        with self.torch.no_grad():
            y = self.net(self.torch.from_numpy(x[None].astype(np.float32)).to(self.device)).float().cpu().numpy()[0]
        self.windows_run += 1
        return y[:n]

    def detect(self, frames_bgr, th=0.5):
        fh, fw = frames_bgr[0].shape[:2]
        return [heatmap_blobs(h, th, (fw / W, fh / H)) for h in self.heatmaps(frames_bgr)]
