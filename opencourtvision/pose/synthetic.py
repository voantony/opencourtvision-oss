"""Synthetic stick-figure player for tests and the pose benchmark (no real footage or real pose data).

Camera model: a top-down-like linear view, S px per cm, so cm_per_px is constant and known. Court corners for
that view come from synthetic_court(). The player has their back to the camera (near end), right-handed by default.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator as CubicSpline  # monotone: no overshoot past keyframes

from .skeleton import JOINTS, J

S = 0.5                       # px per cm
W_PX, H_PX = 1000, 1400
OX, OY = 220.0, 1300.0        # image px of the near-left doubles corner


def synthetic_court():
    w, l = 1097 * S, 2377 * S
    return np.array([(OX, OY), (OX + w, OY), (OX + w, OY - l), (OX, OY - l)])


def court_to_px(x_cm, y_cm, z_cm=0.0):
    """court cm (+ height above ground, drawn upward in the image) -> image px."""
    return np.array([OX + x_cm * S, OY - y_cm * S - z_cm * S])


# body template: (lateral cm from ground point, height cm), back to camera so +lateral = player's right = image right
BODY = {"head": (0, 170), "l_shoulder": (-20, 145), "r_shoulder": (20, 145), "l_elbow": (-26, 118), "r_elbow": (26, 118),
        "l_wrist": (-28, 92), "r_wrist": (28, 92), "l_hip": (-12, 95), "r_hip": (12, 95), "l_knee": (-13, 50), "r_knee": (13, 50)}


def _pose(ground, stance_deg=0.0, stance_w=60.0, body_dx=0.0, wrist=None, lean=0.0, hand="right", bob=0.0):
    """One frame: ground (x_cm, y_cm), feet on a line at stance_deg to the baseline, optional hitting-wrist
    (lateral, height) override, upper body shifted laterally by `lean` cm (positive = to player's left)."""
    gx, gy = ground; p = np.full((len(JOINTS), 2), np.nan)
    for n, (dx, z) in BODY.items():
        sh = -lean * (z - 95) / 50 if z > 95 else 0.0       # lean grows with height above the hips
        p[J[n]] = court_to_px(gx + dx + body_dx + sh, gy, z + bob)
    a = np.radians(stance_deg); u = np.array([np.cos(a), np.sin(a)])   # left foot -> right foot direction on court
    for s, sign in (("l", -1), ("r", 1)):
        f = np.array([gx, gy]) + sign * u * stance_w / 2
        p[J[f"{s}_ankle"]] = court_to_px(f[0], f[1] + 4, 8)
        p[J[f"{s}_heel"]] = court_to_px(f[0], f[1] - 4, 2)
        p[J[f"{s}_toe"]] = court_to_px(f[0], f[1] + 18, 0)
    if wrist is not None:
        w = J["r_wrist" if hand == "right" else "l_wrist"]; e = J["r_elbow" if hand == "right" else "l_elbow"]
        sx = 1 if hand == "right" else -1
        p[w] = court_to_px(gx + body_dx + sx * wrist[0], gy, wrist[1])
        sh = p[J["r_shoulder" if hand == "right" else "l_shoulder"]]
        p[e] = (p[w] + sh) / 2
    return p


def stroke_keys(kind, contact_cm=55.0, finish_cm=-15.0, shoulder_z=145.0):
    """Hitting-wrist keyframes (dt, lateral cm on the hitting-hand side, height cm). Contact at dt=0.
    Backhand contact is on the other side (negative lateral)."""
    if kind == "forehand":
        return [(-0.9, 28, 92), (-0.4, 75, 105), (0.0, contact_cm, 100), (0.35, -35, shoulder_z + finish_cm), (0.9, 28, 92)]
    if kind == "backhand":
        return [(-0.9, 28, 92), (-0.4, -70, 100), (0.0, -contact_cm, 100), (0.35, 45, shoulder_z + finish_cm), (0.9, 28, 92)]
    if kind == "serve":
        return [(-1.2, 28, 92), (-0.6, 40, 130), (-0.25, 30, 150), (0.0, 15, 235), (0.4, -30, 80), (1.0, 28, 92)]
    raise ValueError(kind)


def rally(strokes, fps=30.0, dur=None, hand="right", seed=0, noise_px=0.0, start=(500.0, -50.0)):
    """strokes: list of dict(t, kind, stance=deg, contact=cm, finish=cm, lean=cm). Between strokes the player
    shuffles sideways. Returns dict(kp (T,17,2), conf, fps, hits [t], kinds [..], court corners px)."""
    rng = np.random.default_rng(seed)
    dur = dur or (max(s["t"] for s in strokes) + 1.5)
    T = int(round(dur * fps)); ts = np.arange(T) / fps
    # ground path: sidestep to each stroke's x (alternating), smooth
    gx_keys = [(0.0, start[0])] + [(s["t"] - 0.3, start[0] + (150 if i % 2 else -150) * (s["kind"] != "serve")) for i, s in enumerate(strokes)] + [(dur, start[0])]
    kt, kx = zip(*sorted(gx_keys)); gx = np.interp(ts, kt, kx)
    kp = np.zeros((T, len(JOINTS), 2))
    for i, t in enumerate(ts):
        s = min(strokes, key=lambda s: abs(s["t"] - t)); dt = t - s["t"]
        keys = stroke_keys(s["kind"], s.get("contact", 55.0), s.get("finish", -15.0))
        if keys[0][0] <= dt <= keys[-1][0]:
            k = np.array(keys); w = (CubicSpline(k[:, 0], k[:, 1])(dt), CubicSpline(k[:, 0], k[:, 2])(dt))
            win = np.exp(-(dt / 0.3) ** 2)
            stance = s.get("stance", 0.0); lean = s.get("lean", 0.0) * win
            kp[i] = _pose((gx[i], start[1]), stance, 60.0, 0.0, w, lean, hand)
        else:
            kp[i] = _pose((gx[i], start[1]), 0.0, 55.0, hand=hand, bob=3 * np.sin(2 * np.pi * 2.5 * t))
    kp += rng.normal(0, noise_px, kp.shape) if noise_px else 0
    return dict(kp=kp, conf=np.ones(kp.shape[:2]), fps=fps, hits=[s["t"] for s in strokes], kinds=[s["kind"] for s in strokes],
                court=synthetic_court())


def movement_sequence(fps=30.0, seed=0, reps=6):
    """Concatenated movement types for syllable tests: ready bounce, shuffle right, shuffle left, run forward,
    walk back, forehand, backhand. Returns (kp, truth_labels per frame, names)."""
    rng = np.random.default_rng(seed); names = ["bounce", "shuffle_r", "shuffle_l", "run_fwd", "walk_back", "forehand", "backhand"]
    K, L = [], []; g = np.array([500.0, -50.0])
    for _ in range(reps):
        for li in rng.permutation(len(names)):
            n = int(fps * rng.uniform(0.8, 1.2)); kind = names[li]
            for f in range(n):
                t = f / fps
                if kind == "bounce": p = _pose(g, bob=6 * np.sin(2 * np.pi * 3 * t))
                elif kind in ("shuffle_r", "shuffle_l"):
                    g = g + [(250 if kind == "shuffle_r" else -250) / fps, 0]; p = _pose(g, stance_w=55 + 20 * np.sin(2 * np.pi * 3 * t))
                elif kind in ("run_fwd", "walk_back"):
                    v = 450 if kind == "run_fwd" else -120; g = g + [0, v / fps]
                    p = _pose(g, stance_deg=90, stance_w=50 * np.sin(2 * np.pi * (3 if v > 0 else 1.2) * t), bob=4 * abs(np.sin(2 * np.pi * 3 * t)))
                else:
                    keys = np.array(stroke_keys(kind)); dt = -0.4 + 0.8 * t / (n / fps)
                    p = _pose(g, 30, 60, wrist=(CubicSpline(keys[:, 0], keys[:, 1])(dt), CubicSpline(keys[:, 0], keys[:, 2])(dt)))
                g[0] = np.clip(g[0], 150, 950); g[1] = np.clip(g[1], -300, 600)
                K.append(p); L.append(li)
    return np.asarray(K), np.asarray(L), names
