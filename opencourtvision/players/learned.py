"""Learned cues, each a small net trained per match on that match's own first set (no hand labels needed) and tested
on later sets; trained models are private match artefacts and are never committed.

- MetricHead: OSNet + mask colour -> 64-d embedding, batch-hard triplet loss. Positives = the same motion-chain
  tracklet within a point; negatives = other people in the same frame (they are by definition different people).
- In-play MLP: per-detection features (inplay.FEATS) -> in play or not.
- Pose-role MLP: normalised pose (cues.pose_norm) -> "is the server" at the serve contact (role, not identity).
- Association LR: logistic regression on per-cue cost differences -> learned fusion weights for the tracker.
"""
from __future__ import annotations

import numpy as np


def train_metric_head(X, y, groups_frame, dim=64, epochs=40, lr=3e-3, seed=0, device="cpu"):
    """X: (N, D) inputs, y: tracklet ids, groups_frame: frame ids (used to sample same-frame negatives).
    Returns a callable f(X) -> (N, dim) L2-normalised."""
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32); y = np.asarray(y); D = X.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, dim)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    ids = np.unique(y); by = {i: np.flatnonzero(y == i) for i in ids}; ids = [i for i in ids if len(by[i]) >= 2]
    if len(ids) < 4:
        raise ValueError("too few tracklets to train the metric head")
    for ep in range(epochs):
        rng.shuffle(ids)
        for b in range(0, len(ids), 32):
            sel = ids[b:b + 32]
            if len(sel) < 2:
                continue
            idx = np.concatenate([rng.choice(by[i], size=min(4, len(by[i])), replace=False) for i in sel])
            xb = torch.from_numpy(X[idx]).to(device); yb = torch.from_numpy(y[idx].astype(np.int64)).to(device)
            e = torch.nn.functional.normalize(net(xb), dim=1); dm = torch.cdist(e, e)
            same = yb[:, None] == yb[None, :]; eye = torch.eye(len(yb), dtype=torch.bool, device=device)
            pos = (dm * (same & ~eye)).max(1).values; neg = (dm + 1e3 * same).min(1).values
            loss = torch.relu(pos - neg + 0.3).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def f(Z):
        with torch.inference_mode():
            return torch.nn.functional.normalize(net(torch.from_numpy(np.asarray(Z, np.float32)).to(device)), dim=1).cpu().numpy()
    return f


def train_mlp(X, y, hidden=(32, 16), seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hidden, max_iter=2000, alpha=1e-3, random_state=seed))
    m.fit(np.asarray(X, float), np.asarray(y, int))
    return m


def train_assoc(dC, y):
    """dC: (N, n_cues) cost(swap) - cost(keep) per cue for 2x2 side decisions; y: 1 if keep is right.
    Returns per-cue weights (non-negative)."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=500).fit(np.asarray(dC, float), np.asarray(y, int))
    return np.maximum(m.coef_[0], 0.0)
