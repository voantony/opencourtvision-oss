import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "point_winner"))
from pw_merge import merge_far  # noqa: E402


def _frame():
    B = np.full((1, 8, 5), np.nan, np.float32); P = np.full((1, 8, 17, 3), np.nan, np.float32)
    B[0, 0] = [100, 200, 140, 330, 0.95]; P[0, 0] = 1.0       # near player, foot y 330 (640 px) -> 1980 src: outside the crop
    B[0, 1] = [300, 60, 310, 90, 0.50]; P[0, 1] = 2.0          # low-res far player, foot inside the crop: replaced
    FB = np.full((1, 8, 5), np.nan, np.float32); FP = np.full((1, 8, 17, 3), np.nan, np.float32)
    FB[0, 0] = [1800, 360, 1872, 540, 0.90]; FP[0, 0] = 6.0    # native-res far player (source px)
    FB[0, 1] = [2400, 380, 2460, 560, 0.80]; FP[0, 1] = 6.0
    return B, P, FB, FP


def test_near_players_survive_the_far_merge():
    B, P, FB, FP = _frame()
    B, P = merge_far(B, P, FB, FP, roi=(0, 200, 3840, 700), sc=6.0)
    rows = [tuple(np.round(b, 2)) for b in B[0] if np.isfinite(b[0])]
    assert len(rows) == 3                                       # near kept + 2 far; the low-res far one replaced
    assert (100.0, 200.0, 140.0, 330.0, 0.95) in rows          # the near box is intact, not wiped or duplicated
    assert (300.0, 60.0, 310.0, 90.0, 0.5) not in rows
    assert len(set(rows)) == len(rows)                          # no duplicates
    assert np.allclose(P[0, 0], 1.0)                            # its keypoints travel with it
    far = [b for b in B[0] if np.isfinite(b[0]) and b[4] < 0.95]
    assert np.allclose(far[0][:4], np.array([1800, 360, 1872, 540]) / 6.0)


def test_merge_without_far_detections_keeps_everything_outside_the_crop():
    B, P, FB, FP = _frame(); FB[:] = np.nan
    B, P = merge_far(B, P, FB, FP, roi=(0, 200, 3840, 700), sc=6.0)
    assert np.isfinite(B[0, 0, 0]) and np.isnan(B[0, 1, 0])
