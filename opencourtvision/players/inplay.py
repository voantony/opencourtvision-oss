"""In-play filter: keep the 4 people in the point (2 per side), drop spectators, umpires, ball kids and people on
other courts. Hand rule: court mask + "stay in play from the serve" (a candidate must be near a person who was in
play at the serve, moving at a human speed). Learned rule: an MLP over the same per-detection features."""
from __future__ import annotations

import numpy as np

from .geometry import CW, NET_Y, outside_dist

FEATS = ("x_off", "y_from_net", "outside", "conf", "h_ratio", "track_dist", "since_serve", "n_side")


def det_features(d, prev_xy, t_since_serve, n_side):
    x, y = d["xy"]
    td = min(4.0, float(np.min(np.linalg.norm(np.asarray(prev_xy) - [x, y], axis=1)))) if prev_xy is not None and len(prev_xy) else 4.0
    return [abs(x - CW / 2), abs(y - NET_Y), float(outside_dist([[x, y]])[0]), d["conf"], d.get("h_ratio", 1.0), td,
            min(t_since_serve, 30.0), n_side]


def hand_score(f):
    """score of one det_features row: higher = more likely in play."""
    x_off, y_net, out, conf, h_ratio, td, _, _ = f
    return conf - 0.35 * out - 0.25 * td - (2.0 if out > 3.5 else 0.0) - (1.0 if not 0.45 < h_ratio < 2.2 else 0.0)


def select(dets, score, max_per_side=2, thr=-1.0):
    """indices of the kept detections: best `max_per_side` per side with score > thr."""
    keep = []
    for s in (0, 1):
        idx = [i for i, d in enumerate(dets) if d["side"] == s and score[i] > thr]
        keep += sorted(idx, key=lambda i: -score[i])[:max_per_side]
    return sorted(keep)
