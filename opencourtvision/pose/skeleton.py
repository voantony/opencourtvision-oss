"""Canonical 17-joint body layout used by the pose stage, plus mappings from common pose-model outputs.

Keypoint arrays are float (T, 17, 2) in image pixels with NaN for missing joints; confidences are (T, 17).
"""
import numpy as np

JOINTS = ("head",
          "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
          "l_heel", "r_heel", "l_toe", "r_toe")
J = {n: i for i, n in enumerate(JOINTS)}
BONES = (("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("l_ankle", "l_heel"), ("l_heel", "l_toe"), ("r_ankle", "r_heel"), ("r_heel", "r_toe"))

# MediaPipe Pose Landmarker (33 landmarks) -> ours
MEDIAPIPE33 = {"head": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
               "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
               "l_heel": 29, "r_heel": 30, "l_toe": 31, "r_toe": 32}
# COCO-17 (RTMPose, MoveNet, ViTPose, ...) -> ours; COCO has no heel/toe, so those stay NaN
COCO17 = {"head": 0, "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10,
          "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16}
LAYOUTS = {"mediapipe33": MEDIAPIPE33, "coco17": COCO17}


def from_layout(kp, conf=None, layout="coco17", min_conf=0.3):
    """kp: (T, K, 2) in a model's own joint order -> ((T, 17, 2), (T, 17)) in ours, low-confidence joints -> NaN."""
    kp = np.asarray(kp, float); T = kp.shape[0]
    conf = np.ones(kp.shape[:2]) if conf is None else np.asarray(conf, float)
    out = np.full((T, len(JOINTS), 2), np.nan); oc = np.zeros((T, len(JOINTS)))
    for name, k in LAYOUTS[layout].items():
        out[:, J[name]] = kp[:, k]; oc[:, J[name]] = conf[:, k]
    out[oc < min_conf] = np.nan
    return out, oc


def mid(kp, a, b):
    """Midpoint of two joints over time (T, 2)."""
    return (kp[:, J[a]] + kp[:, J[b]]) / 2


def hip_center(kp):
    return mid(kp, "l_hip", "r_hip")


def shoulder_center(kp):
    return mid(kp, "l_shoulder", "r_shoulder")


def foot_point(kp, side):
    """Best ground-contact point for one foot: heel/toe midpoint, else ankle."""
    s = side[0]
    ht = (kp[:, J[f"{s}_heel"]] + kp[:, J[f"{s}_toe"]]) / 2
    return np.where(np.isnan(ht), kp[:, J[f"{s}_ankle"]], ht)


def ground_point(kp):
    """Between the feet (T, 2), the player's court position."""
    return (foot_point(kp, "l") + foot_point(kp, "r")) / 2


def interpolate_gaps(kp, max_gap=5):
    """Linearly fill NaN runs of at most max_gap frames per joint/coordinate."""
    kp = np.array(kp, float); T = kp.shape[0]; t = np.arange(T)
    flat = kp.reshape(T, -1)
    for c in range(flat.shape[1]):
        v = flat[:, c]; ok = ~np.isnan(v)
        if ok.sum() < 2 or ok.all(): continue
        filled = np.interp(t, t[ok], v[ok])
        bad = ~ok; i = 0
        while i < T:
            if bad[i]:
                j = i
                while j < T and bad[j]: j += 1
                if i > 0 and j < T and j - i <= max_gap: v[i:j] = filled[i:j]
                i = j
            else:
                i += 1
    return flat.reshape(kp.shape)
