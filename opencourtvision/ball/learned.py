"""Learned parts of the multi-cue ball tracker (ball tracker v2, research; torch, CPU-sized).

1. CandNet: a small temporal net that rescores every ball candidate of every frame from detector peaks (each detector
   + agreement), motion consistency with neighbouring frames, the player-attention prior (gaze, swing/contact,
   static) and the audio envelope. Per-candidate embedding + a dilated 1D conv over time on the per-frame pooled
   embedding (context), -> one logit per candidate. Its scores replace the peaks the physics tracker sees.
2. EventNet: a small temporal 1D conv over a tracked trajectory (position, velocity, acceleration, confidence, the
   physics tracker's own hit/bounce flags), audio and the nearest player's wrist speed/distance -> per-frame P(hit),
   P(bounce).
Both are trained on the dev_match TRAIN points (1-50) only; weights are private (trained on Antony's labels/derived
pseudo-labels) and never committed.
"""
import numpy as np

CAND_F = 22          # see cand_features


def _nearest_d(c_i, c_j):
    """for each candidate in c_i (K, 2) the distance to the nearest finite candidate in c_j (K', 2) (inf if none)."""
    ok = np.isfinite(c_j[:, 0])
    if not ok.any():
        return np.full(len(c_i), np.inf)
    q = c_j[ok]
    d = np.hypot(c_i[:, None, 0] - q[None, :, 0], c_i[:, None, 1] - q[None, :, 1])
    return d.min(1)


def cand_features(fused, det_list, att_feats, audio_db, max_det=6):
    """fused (T, K, 3), det_list: list of (T, K_d, 3) per detector (<= max_det), att_feats (T, K, 4) or None,
    audio_db (T,) dB or None -> X (T, K, CAND_F) float32, valid (T, K) bool."""
    T, K, _ = fused.shape
    X = np.zeros((T, K, CAND_F), np.float32)
    valid = np.isfinite(fused[..., 0])
    uv = fused[..., :2]
    for i in range(T):
        if not valid[i].any():
            continue
        X[i, :, 0] = np.nan_to_num(fused[i, :, 2])
        for d, dc in enumerate(det_list[:max_det]):
            dd = _nearest_d(uv[i], dc[i, :, :2]) if i < len(dc) else np.full(K, np.inf)
            pk = np.zeros(K)
            ok = np.isfinite(dc[i, :, 0]) if i < len(dc) else np.zeros(0, bool)
            if ok.any():
                q = dc[i][ok]
                dist = np.hypot(uv[i, :, None, 0] - q[None, :, 0], uv[i, :, None, 1] - q[None, :, 1])
                j = np.argmin(np.nan_to_num(dist, nan=1e9), 1)
                pk = np.where(dd <= 8.0, q[j, 2], 0.0)
            X[i, :, 1 + d] = np.nan_to_num(pk)
        X[i, :, 7] = (X[i, :, 1:1 + max_det] > 0).sum(1) / max(1, min(max_det, len(det_list)))
        X[i, :, 8] = np.nan_to_num(uv[i, :, 0] / 1280.0); X[i, :, 9] = np.nan_to_num(uv[i, :, 1] / 720.0)
        if att_feats is not None:
            g, c, s, dm = (att_feats[i, :, q] for q in range(4))
            X[i, :, 10] = np.nan_to_num(g, nan=0.5); X[i, :, 11] = np.nan_to_num(c); X[i, :, 12] = np.nan_to_num(s)
            X[i, :, 13] = np.clip(np.nan_to_num(dm, nan=10.0, posinf=10.0) / 10.0, 0, 1)
        if audio_db is not None:
            X[i, :, 14] = np.clip((audio_db[i] + 60.0) / 60.0, 0, 1)
        for q, dt in enumerate((1, 2, 3)):
            for sgn, col in ((-1, 15 + 2 * q), (1, 16 + 2 * q)):
                j = i + sgn * dt
                if 0 <= j < T:
                    X[i, :, col] = np.clip(_nearest_d(uv[i], uv[j]) / (25.0 * dt), 0, 1.5)
                else:
                    X[i, :, col] = 1.5
        if 0 < i < T - 1:   # constant-velocity residual through the neighbours
            a, b = uv[i - 1][np.isfinite(uv[i - 1, :, 0])], uv[i + 1][np.isfinite(uv[i + 1, :, 0])]
            if len(a) and len(b):
                mid = (a[:, None] + b[None]) / 2
                r = np.hypot(uv[i, :, None, None, 0] - mid[None, ..., 0], uv[i, :, None, None, 1] - mid[None, ..., 1]).reshape(K, -1).min(1)
                X[i, :, 21] = np.clip(np.nan_to_num(r, nan=40.0) / 20.0, 0, 2)
            else:
                X[i, :, 21] = 2.0
        else:
            X[i, :, 21] = 2.0
    X[~valid] = 0
    return X, valid


def build_candnet(F=CAND_F, h=32):
    import torch.nn as nn

    class CandNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Sequential(nn.Linear(F, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.ctx = nn.Sequential(nn.Conv1d(h, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

        def forward(self, x, valid):            # x (B, T, K, F), valid (B, T, K)
            e = self.emb(x)                       # (B, T, K, h)
            pooled = (e * valid[..., None]).amax(2)          # (B, T, h)
            c = self.ctx(pooled.transpose(1, 2)).transpose(1, 2)   # (B, T, h)
            z = torch_cat(e, c[:, :, None].expand_as(e))
            return self.head(z)[..., 0]

    return CandNet()


def torch_cat(a, b):
    import torch
    return torch.cat([a, b], -1)


EV_F = 14


def event_features(pos, conf, phys_events, audio_db, swing, pdist):
    """pos (T, 2) px (NaN gaps), conf (T,), phys_events list of dict(kind, i), audio_db (T,), swing (T,) max wrist
    speed of the player nearest the ball, pdist (T,) that player's distance in box heights -> (T, EV_F) float32."""
    T = len(pos)
    X = np.zeros((T, EV_F), np.float32)
    ok = np.isfinite(pos[:, 0])
    p = pos.copy()
    if ok.any():
        idx = np.arange(T)
        for c in (0, 1):
            p[:, c] = np.interp(idx, idx[ok], pos[ok, c])
    v = np.gradient(p, axis=0) if T > 2 else np.zeros_like(p); a = np.gradient(v, axis=0) if T > 2 else np.zeros_like(p)
    X[:, 0] = ok; X[:, 1] = np.nan_to_num(conf)
    X[:, 2] = np.nan_to_num(p[:, 0] / 1280.0); X[:, 3] = np.nan_to_num(p[:, 1] / 720.0)
    X[:, 4] = np.clip(np.nan_to_num(v[:, 0]) / 30.0, -2, 2); X[:, 5] = np.clip(np.nan_to_num(v[:, 1]) / 30.0, -2, 2)
    X[:, 6] = np.clip(np.nan_to_num(a[:, 0]) / 10.0, -2, 2); X[:, 7] = np.clip(np.nan_to_num(a[:, 1]) / 10.0, -2, 2)
    X[:, 8] = np.clip(np.hypot(np.nan_to_num(a[:, 0]), np.nan_to_num(a[:, 1])) / 10.0, 0, 3)
    for e in phys_events:
        k = int(e["i"])
        if 0 <= k < T:
            X[k, 9 if e["kind"] == "hit" else 10] = 1.0
    if audio_db is not None:
        X[:, 11] = np.clip((audio_db + 60.0) / 60.0, 0, 1)
    X[:, 12] = np.clip(np.nan_to_num(swing) * 5, 0, 2); X[:, 13] = np.clip(np.nan_to_num(pdist, nan=10.0) / 10.0, 0, 1)
    return X


def build_eventnet(F=EV_F, h=32):
    import torch.nn as nn
    return nn.Sequential(nn.Conv1d(F, h, 7, padding=3), nn.ReLU(), nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.ReLU(),
                         nn.Conv1d(h, h, 5, padding=8, dilation=4), nn.ReLU(), nn.Conv1d(h, 2, 1))


def pick_events(prob, th, min_gap=6):
    """per-frame probabilities (T,) -> peak frames above th, at least min_gap apart (greedy, strongest first)."""
    order = np.argsort(-prob); out = []
    for k in order:
        if prob[k] < th:
            break
        if all(abs(int(k) - j) >= min_gap for j in out):
            out.append(int(k))
    return sorted(out)
