"""Unsupervised movement "syllables": split a player's keypoint stream into short repeated movements and cluster them,
with zero tennis knowledge, then map every occurrence in 2-D so similar movements sit together.

Independent, simple implementation (NOT keypoint-MoSeq, whose licence forbids commercial use):
1. egocentric features per frame: joints relative to the hip centre, scaled by torso length, plus body velocity;
2. short sliding windows -> PCA -> k-means into `k` movement states;
3. a sticky Viterbi pass (self-transition bias) smooths per-frame labels; runs shorter than min_s are merged;
4. each run is one syllable instance; instances are embedded in 2-D (PCA, or UMAP if `umap-learn` is installed).
Names are given by a human after watching example clips per syllable (see instances' `clip` times).
"""
import numpy as np

from .skeleton import hip_center, interpolate_gaps
from .strokes import body_scale


def features(kp, fps):
    """(T, F) egocentric pose + body velocity features."""
    kp = interpolate_gaps(kp, max_gap=int(fps)); sc = body_scale(kp); hc = hip_center(kp)
    rel = (kp - hc[:, None]) / sc
    vel = np.gradient(hc, axis=0) * fps / sc
    jv = np.gradient(rel, axis=0) * fps * 0.1
    f = np.concatenate([rel.reshape(len(kp), -1), vel, jv.reshape(len(kp), -1)], axis=1)
    return np.nan_to_num(f - np.nanmean(f, axis=0))


def _windows(f, w):
    pad = np.pad(f, ((w // 2, w - 1 - w // 2), (0, 0)), mode="edge")
    return np.stack([pad[i:i + len(f)] for i in range(w)], axis=1).reshape(len(f), -1)


def pca(x, d):
    x = x - x.mean(0); u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:d].T


def kmeans(x, k, iters=50, seed=0):
    rng = np.random.default_rng(seed); c = x[[rng.integers(len(x))]]
    for _ in range(1, k):                                             # k-means++ init
        d = ((x[:, None] - c[None]) ** 2).sum(-1).min(1); c = np.vstack([c, x[rng.choice(len(x), p=d / d.sum())]])
    for _ in range(iters):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        nc = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        for j in range(k):                                            # re-seed empty clusters at the worst-fit point
            if not (lab == j).any():
                far = ((x - nc[lab]) ** 2).sum(1).argmax(); nc[j] = x[far]; lab[far] = j
        if np.allclose(nc, c): break
        c = nc
    return lab, c


def sticky_viterbi(x, centers, stick=0.98):
    """Most likely label path under isotropic Gaussian emissions at the centers and a sticky transition matrix."""
    k = len(centers); d2 = ((x[:, None] - centers[None]) ** 2).sum(-1)
    var = max(float(np.median(d2.min(1))), 1e-6); ll = -d2 / (2 * var)
    lt = np.log(np.full((k, k), (1 - stick) / max(k - 1, 1))); np.fill_diagonal(lt, np.log(stick))
    T = len(x); dp = ll[0].copy(); bp = np.zeros((T, k), int)
    for t in range(1, T):
        s = dp[:, None] + lt; bp[t] = s.argmax(0); dp = s.max(0) + ll[t]
    path = np.empty(T, int); path[-1] = dp.argmax()
    for t in range(T - 1, 0, -1): path[t - 1] = bp[t, path[t]]
    return path


def runs(lab):
    """[(start, end_exclusive, label)]"""
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]: out.append((s, i, int(lab[s]))); s = i
    return out


def merge_short(lab, min_len):
    lab = lab.copy()
    for _ in range(3):
        R = runs(lab)
        for j, (a, b, _) in enumerate(R):
            if b - a < min_len and len(R) > 1:
                nb = R[j - 1][2] if j > 0 and (j == len(R) - 1 or R[j - 1][1] - R[j - 1][0] >= R[j + 1][1] - R[j + 1][0]) else R[min(j + 1, len(R) - 1)][2]
                lab[a:b] = nb
    return lab


def embed2d(x, seed=0):
    try:
        import umap  # BSD-3-Clause, optional
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
    except Exception:
        return pca(x, 2) if x.shape[0] >= 2 else np.zeros((len(x), 2))


def fit(kp, fps, k=12, win_s=0.3, dims=10, stick=0.98, min_s=0.2, seed=0):
    """-> dict(labels per frame, instances [dict(syllable, start, end, dur, speed, x, y)], k)."""
    f = features(kp, fps); w = max(1, int(win_s * fps))
    z = pca(_windows(f, w), min(dims, f.shape[1] * w))
    z = z / (z.std() + 1e-9)
    _, c = kmeans(z, k, seed=seed)
    lab = merge_short(sticky_viterbi(z, c, stick), max(1, int(min_s * fps)))
    hc = hip_center(interpolate_gaps(kp, int(fps))); sc = body_scale(kp)
    sp = np.nan_to_num(np.hypot(*np.gradient(hc, axis=0).T) * fps / sc)
    inst, feats = [], []
    for a, b, l in runs(lab):
        seg = z[a:b]
        feats.append(np.r_[seg.mean(0), seg[0], seg[-1], [np.log((b - a) / fps)]])
        inst.append(dict(syllable=l, start=round(a / fps, 3), end=round(b / fps, 3), dur=round((b - a) / fps, 3), speed=round(float(sp[a:b].mean()), 2)))
    xy = embed2d(np.asarray(feats), seed) if len(feats) > 2 else np.zeros((len(feats), 2))
    if len(xy):
        xy = (xy - xy.min(0)) / (np.ptp(xy, 0) + 1e-9)
    for d, (x, y) in zip(inst, xy): d["x"], d["y"] = round(float(x), 4), round(float(y), 4)
    return dict(labels=lab, instances=inst, k=k)


def purity(pred, truth):
    """Fraction of frames whose predicted cluster's majority truth label matches theirs (an eval helper)."""
    tot = 0
    for c in np.unique(pred):
        tot += np.bincount(truth[pred == c]).max()
    return tot / len(pred)
