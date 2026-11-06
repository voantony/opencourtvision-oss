"""Shared helpers for the own ball tracker tools (Mac research pipeline; nothing here is shipped)."""
import json
import os
import subprocess

import numpy as np

AW, AH, AFPS = 1280, 720, 30.0          # analysis frames (teacher, baselines, physics)
NET = (768, 432)                        # network input (heatnet.SIZE)
K_NET = NET[0] / AW                     # 1280 px -> 768 px
COURT_W, NET_Y = 10.97, 11.885


def ffmpeg_bin():
    for p in (os.environ.get("OCV_FFMPEG"), os.path.expanduser("~/bin/ffmpeg"), "ffmpeg"):
        if p and (os.path.sep not in p or os.path.exists(p)):
            return p
    return "ffmpeg"


_NO_HW = set()


def decode(path, a, n, w=AW, h=AH, fps=AFPS, hw=True):
    """n BGR frames of path from time a at fps, scaled to w x h. Falls back to software decode (and remembers it per
    file) when VideoToolbox can't take the codec (AV1 on M2)."""
    hw = hw and path not in _NO_HW
    cmd = [ffmpeg_bin(), "-nostdin", "-loglevel", "error"] + (["-hwaccel", "videotoolbox"] if hw else []) + [
        "-ss", f"{a:.3f}", "-i", path, "-frames:v", str(n), "-vf", f"fps={fps},scale={w}:{h}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL if hw else None, check=False)
    buf, sz = p.stdout, w * h * 3
    if not buf and hw:
        _NO_HW.add(path)
        return decode(path, a, n, w, h, fps, hw=False)
    return [np.frombuffer(buf[k * sz:(k + 1) * sz], np.uint8).reshape(h, w, 3) for k in range(len(buf) // sz)]


def h_court2img(H_img2court, src_w, dst_w=AW):
    """court metres -> analysis px, from a stored image->court homography in source px."""
    s = dst_w / float(src_w)
    return np.diag([s, s, 1.0]) @ np.linalg.inv(np.asarray(H_img2court, float))


def override_courts(courts_by_view, path):
    """courts_by_view with the manual court-setup fits in JSON `path` ({view, H_img2court} or a list) swapped in."""
    cbv = [dict(c) for c in courts_by_view]
    if path:
        ov = json.load(open(path))
        for o in ov if isinstance(ov, list) else [ov]:
            cbv[o["view"]].update(H_img2court=o["H_img2court"], detected=True, manual=True)
    return cbv


def view_at(courts_by_view, t):
    v = [c for c in courts_by_view if c["t0"] <= t < c["t1"]] or courts_by_view[-1:]
    return v[0] if v else None


def project(Hc2i, xy):
    p = Hc2i @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


def boxes_fn(npz, scale):
    """player boxes at analysis frame i from a pw_tracks npz (640x360 boxes at pose_k frames)."""
    k, B = npz["pose_k"], npz["boxes"]

    def at(i):
        if not len(k):
            return []
        j = int(np.argmin(np.abs(k - i)))
        if abs(int(k[j]) - i) > 3:
            return []
        return [bb[:4] * scale for bb in B[j] if np.isfinite(bb[0]) and bb[4] >= 0.4]
    return at


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def live_arc_check(gate, fps=AFPS):
    """arc -> True unless the court gate calls it static, slow (ground speed below the live-ball floor) or off court
    (same rules as ball.proxy.false_track_stats). None when there is no court."""
    if gate is None:
        return None
    from opencourtvision.ball.proxy import false_track_stats

    def ok(arc):
        span = np.full((arc.i1 - arc.i0 + 1, 2), np.nan)
        span[arc.idx - arc.i0] = arc.xy
        st = false_track_stats(span, fps, gate, min_len=2)
        return st["false"] == 0
    return ok
