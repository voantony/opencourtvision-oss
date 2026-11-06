"""Pose estimators behind one small interface, plus a keypoints file format.

Backends (all optional, none shipped with weights):
- ``mediapipe``: MediaPipe Pose Landmarker (Apache-2.0 code and model card). The .task model is downloaded at
  runtime into $OCV_CACHE (default ~/.cache/opencourtvision); nothing is committed. In beta mode (privacy.py) it must be
  bundled or come from our EU host; Google is never contacted.
- ``apple-vision``: macOS Vision VNDetectHumanBodyPoseRequest through pyobjc (system framework, no weights).
- precomputed: any tool's output saved as a keypoints .npz (see save_keypoints) or COCO-17/MediaPipe-33 arrays.
Ultralytics/YOLO-pose (AGPL) and keypoint-MoSeq (non-commercial) are deliberately not supported.
"""
import os

import numpy as np

from .. import privacy
from .skeleton import JOINTS, from_layout

MEDIAPIPE_MODELS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def cache_dir():
    d = os.environ.get("OCV_CACHE", os.path.expanduser("~/.cache/opencourtvision")); os.makedirs(d, exist_ok=True)
    return d


def save_keypoints(path, kp, conf, fps, size=None, meta=None):
    np.savez_compressed(path, kp=np.asarray(kp, np.float32), conf=np.asarray(conf, np.float32), fps=float(fps),
                        size=np.asarray(size or (0, 0)), joints=np.asarray(JOINTS), meta=str(meta or {}))


def load_keypoints(path):
    """-> dict(kp (T,17,2), conf (T,17), fps, size)."""
    z = np.load(path, allow_pickle=False)
    if tuple(z["joints"]) != JOINTS: raise ValueError(f"{path}: joint layout differs from opencourtvision.pose.skeleton.JOINTS")
    return dict(kp=z["kp"].astype(float), conf=z["conf"].astype(float), fps=float(z["fps"]), size=tuple(int(v) for v in z["size"]))


def pick_player(poses, prev=None, region=None):
    """Choose one person per frame from a list of (17,2) candidate poses: the one nearest the previous pick,
    else the lowest in the image (the near-end player for a fence camera). region=(x0,y0,x1,y1) restricts by hip."""
    best, bd = None, np.inf
    for p in poses:
        h = np.nanmean(p[[7, 8]], axis=0)
        if np.isnan(h).any(): continue
        if region and not (region[0] <= h[0] <= region[2] and region[1] <= h[1] <= region[3]): continue
        d = np.hypot(*(h - np.nanmean(prev[[7, 8]], axis=0))) if prev is not None and not np.isnan(prev[[7, 8]]).all() else -h[1]
        if d < bd: best, bd = p, d
    return best


class PoseEstimator:
    """Interface: __call__(frame_bgr) -> list of ((17,2) px, (17,) conf) for each person found."""
    name = "base"

    def __call__(self, frame):
        raise NotImplementedError


class MediaPipeEstimator(PoseEstimator):
    name = "mediapipe"

    def __init__(self, variant="full", num_poses=4, model_path=None):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError("pip install mediapipe (Apache-2.0) to use the mediapipe pose backend") from e
        name = f"pose_landmarker_{variant}.task"
        path = model_path or privacy.bundled(f"mediapipe/{name}") or os.path.join(cache_dir(), name)
        if not os.path.exists(path):
            # Beta mode (privacy.py): never storage.googleapis.com; our EU host or the feature is disabled.
            url = privacy.model_url(f"mediapipe/{name}", MEDIAPIPE_MODELS[variant])
            print(f"downloading MediaPipe pose model ({variant}) from {privacy.host_of(url)} -> {path}")
            privacy.download(url, path)
        opts = mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=path),
                                                     running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=num_poses)
        self._mp = mp; self._lm = mp.tasks.vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, frame):
        h, w = frame.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[:, :, ::-1]))
        out = []
        for lms in self._lm.detect(img).pose_landmarks:
            kp = np.array([[p.x * w, p.y * h] for p in lms])[None]; c = np.array([p.visibility for p in lms])[None]
            k, cc = from_layout(kp, c, "mediapipe33", 0.0); out.append((k[0], cc[0]))
        return out


class AppleVisionEstimator(PoseEstimator):
    """macOS 11+ Vision body pose (19 joints, no heel/toe). Requires pyobjc-framework-Vision; untested off-Mac."""
    name = "apple-vision"
    NAMES = {"head": "head_joint", "l_shoulder": "left_shoulder_1_joint", "r_shoulder": "right_shoulder_1_joint",
             "l_elbow": "left_forearm_joint", "r_elbow": "right_forearm_joint", "l_wrist": "left_hand_joint", "r_wrist": "right_hand_joint",
             "l_hip": "left_upLeg_joint", "r_hip": "right_upLeg_joint", "l_knee": "left_leg_joint", "r_knee": "right_leg_joint",
             "l_ankle": "left_foot_joint", "r_ankle": "right_foot_joint"}

    def __init__(self):
        try:
            import Quartz  # noqa: F401
            import Vision
        except ImportError as e:
            raise RuntimeError("apple-vision backend needs macOS with pyobjc-framework-Vision and pyobjc-framework-Quartz") from e
        self._V = Vision

    def __call__(self, frame):  # pragma: no cover - macOS only
        import Quartz
        h, w = frame.shape[:2]; rgba = np.ascontiguousarray(np.dstack([frame[:, :, ::-1], np.full((h, w), 255, np.uint8)]))
        cs = Quartz.CGColorSpaceCreateDeviceRGB()
        prov = Quartz.CGDataProviderCreateWithData(None, rgba.tobytes(), rgba.nbytes, None)
        img = Quartz.CGImageCreate(w, h, 8, 32, 4 * w, cs, Quartz.kCGImageAlphaNoneSkipLast, prov, None, False, 0)
        req = self._V.VNDetectHumanBodyPoseRequest.alloc().init()
        self._V.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None).performRequests_error_([req], None)
        out = []
        for obs in req.results() or []:
            pts, _ = obs.recognizedPointsForJointsGroupName_error_(self._V.VNHumanBodyPoseObservationJointsGroupNameAll, None)
            kp = np.full((len(JOINTS), 2), np.nan); c = np.zeros(len(JOINTS))
            for i, n in enumerate(JOINTS):
                p = pts.get(self.NAMES.get(n, ""))
                if p is not None and p.confidence() > 0:
                    kp[i] = (p.location().x * w, (1 - p.location().y) * h); c[i] = p.confidence()
            out.append((kp, c))
        return out


BACKENDS = {"mediapipe": MediaPipeEstimator, "apple-vision": AppleVisionEstimator}


def extract(frames, estimator, fps, stride=1, region=None):
    """Run an estimator over an iterable of BGR frames (every `stride`th), tracking one player.
    Returns (kp (T,17,2), conf (T,17), fps_out). Frames between strides are skipped, not interpolated."""
    K, C, prev = [], [], None
    for i, f in enumerate(frames):
        if i % stride: continue
        people = estimator(f)
        p = pick_player([k for k, _ in people], prev, region)
        if p is None:
            K.append(np.full((len(JOINTS), 2), np.nan)); C.append(np.zeros(len(JOINTS)))
        else:
            K.append(p); C.append(next(c for k, c in people if k is p)); prev = p
    return np.asarray(K, float).reshape(-1, len(JOINTS), 2), np.asarray(C, float).reshape(-1, len(JOINTS)), fps / stride
