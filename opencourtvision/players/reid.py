"""OSNet re-identification embeddings (body + outfit appearance, no faces). Network code is vendored from
deep-person-reid (MIT); weights (osnet_x1_0 trained on MSMT17) are downloaded at runtime, never committed.
MSMT17 is a research-only dataset: listed for Legal review (docs/player_id.md).

Beta mode (opencourtvision.privacy, Ops V7): the weights must be bundled (VAMOS_MODEL_DIR/osnet/) or come from our EU host;
Google Drive is never contacted and without local weights the OSNet cue is disabled (BetaModelUnavailable).
Embeddings are per match, in memory only (Ops V6): see players/tracker.py."""
from __future__ import annotations

import os

import numpy as np

from .. import privacy

OSNET_URL = {"osnet_x1_0_msmt17": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
             "osnet_x0_25_msmt17": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def weights_path(name="osnet_x1_0_msmt17", cache=None):
    b = privacy.bundled(f"osnet/{name}.pth")
    if b:
        return b
    cache = cache or os.environ.get("OCV_CACHE", os.path.expanduser("~/.cache/opencourtvision"))
    d = os.path.join(cache, "osnet"); os.makedirs(d, exist_ok=True); p = os.path.join(d, name + ".pth")
    if not os.path.exists(p):
        fid = OSNET_URL[name].rsplit("=", 1)[1]
        upstream = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
        privacy.download(privacy.model_url(f"osnet/{name}.pth", upstream), p)   # beta: EU host or BetaModelUnavailable
    return p


class OSNetEmbedder:
    def __init__(self, name="osnet_x1_0_msmt17", device=None):
        import torch

        from ._vendor.osnet import osnet as O
        arch = name.rsplit("_", 1)[0]
        self.torch = torch
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        m = getattr(O, arch)(num_classes=1, pretrained=False, loss="softmax")
        sd = torch.load(weights_path(name), map_location="cpu")
        sd = sd.get("state_dict", sd); sd = {k.replace("module.", ""): v for k, v in sd.items() if not k.startswith(("classifier", "module.classifier"))}
        m.load_state_dict(sd, strict=False); self.m = m.to(self.dev).eval()

    def __call__(self, crops_bgr) -> np.ndarray:
        """list of HxWx3 uint8 BGR crops -> (N, 512) L2-normalised float32."""
        import cv2
        if not crops_bgr:
            return np.zeros((0, 512), np.float32)
        x = np.stack([cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] / 255.0 for c in crops_bgr])
        x = ((x - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)
        with self.torch.inference_mode():
            f = self.m(self.torch.from_numpy(x).to(self.dev)).float().cpu().numpy()
        return (f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
