"""Synthetic A/B for the physics ball tracker (known 3D truth): physics3d.track vs the plain 2D arc filter
(ball.own.physics.filter_track on the best candidate per frame, i.e. the current tracker's post-processing).
Camera = the dev_match view-0 calibration (or a generic fence camera). Prints and writes a JSON summary.
  python tools/ball_tracker/physball_synth_ab.py --n 30 --out synth_ab.json [--court tests/fixtures/court_dev_match_view0_manual.json]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from opencourtvision.ball import physics3d as P3  # noqa: E402
from opencourtvision.ball import physics3d_synth as S  # noqa: E402
from opencourtvision.ball.own import physics as P2  # noqa: E402


def camera(court):
    if court and os.path.exists(court):
        d = json.load(open(court))
        H = np.diag([1 / 3, 1 / 3, 1]) @ np.linalg.inv(np.array(d["H_img2court"]))
        return P3.Camera.from_homography(H, (1280, 720))
    return P3.Camera.look_at([5.5, -6.0, 2.8], [5.5, 12.0, 0.0], 830.0, (1280, 720))


def score_track(pos, uv_true, tol=5.0):
    vis = np.isfinite(uv_true[:, 0]); rep = np.isfinite(pos[:, 0])
    d = np.hypot(*(pos - uv_true).T)
    good = rep & vis & (d <= tol)
    return int(good.sum()), int(vis.sum()), int(rep.sum())


def match_bounces(found, truth, fps, tol_k=3.0, tol_m=1.0):
    """found [(k, x, y)], truth [(t, x, y)] -> (matched, pos errors m, timing errors frames, false)"""
    used, errs, terr = set(), [], []
    for k, x, y in found:
        best = None
        for j, (t, tx, ty) in enumerate(truth):
            if j in used:
                continue
            dk = abs(k - t * fps); dm = float(np.hypot(x - tx, y - ty))
            if dk <= tol_k and dm <= tol_m and (best is None or dm < best[1]):
                best = (j, dm, dk)
        if best:
            used.add(best[0]); errs.append(best[1]); terr.append(best[2])
    return len(used), errs, terr, len(found) - len(used)


def run(n, cam, false_share, gap, seed0=100):
    A = {k: dict(good=0, vis=0, rep=0, b_true=0, b_found=0, b_false=0, b_err=[], b_terr=[], ms=0.0, frames=0) for k in ("plain2d", "physics3d")}
    Hi = np.linalg.inv(cam.P[:, [0, 1, 3]])
    for s in range(n):
        rng = np.random.default_rng(seed0 + s)
        R = S.rally(rng)
        c, uv = S.detections(rng, cam, R["X"], false_share=false_share, gap=gap)
        T = len(c)
        best = np.full((T, 2), np.nan)
        for i in range(T):
            if np.isfinite(c[i, 0, 0]):
                best[i] = c[i, 0, :2]
        t0 = time.perf_counter(); ph = P2.filter_track(best, R["fps"], frame_w=1280); t2 = time.perf_counter() - t0
        b2 = []
        for e in ph["events"]:
            if e["kind"] == "bounce":
                g = Hi @ np.array([e["x"], e["y"], 1.0]); b2.append((e["i"], g[0] / g[2], g[1] / g[2]))
        t0 = time.perf_counter(); out = P3.track(cam, c); t3 = time.perf_counter() - t0
        b3 = [(e["k"], e["xy"][0], e["xy"][1]) for e in out["events"] if e["kind"] == "bounce"]
        for k, pos, bb, dt in (("plain2d", ph["pos"], b2, t2), ("physics3d", out["pos"], b3, t3)):
            g, v, r = score_track(pos, uv)
            m, errs, terr, fa = match_bounces(bb, R["bounces"], R["fps"])
            a = A[k]; a["good"] += g; a["vis"] += v; a["rep"] += r; a["b_true"] += len(R["bounces"]); a["b_found"] += m
            a["b_false"] += fa; a["b_err"] += errs; a["b_terr"] += terr; a["ms"] += 1000 * dt; a["frames"] += T
    res = {}
    for k, a in A.items():
        res[k] = dict(recall_5px=round(a["good"] / max(1, a["vis"]), 3), precision_5px=round(a["good"] / max(1, a["rep"]), 3),
                      bounces_found=f'{a["b_found"]}/{a["b_true"]}', false_bounces=a["b_false"],
                      bounce_err_m_median=round(float(np.median(a["b_err"])), 3) if a["b_err"] else None,
                      bounce_err_m_p90=round(float(np.percentile(a["b_err"], 90)), 3) if a["b_err"] else None,
                      bounce_timing_err_frames_median=round(float(np.median(a["b_terr"])), 2) if a["b_terr"] else None,
                      ms_per_frame=round(a["ms"] / max(1, a["frames"]), 2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--out"); ap.add_argument("--court")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    cam = camera(a.court)
    R = {}
    for name, fs, gap in (("clean", 0.0, None), ("false50_gap8", 0.5, (0.45, 8)), ("false70_gap8", 0.7, (0.45, 8))):
        R[name] = run(a.n, cam, fs, gap)
        print(name, json.dumps(R[name]), flush=True)
    if a.out:
        json.dump(dict(n_rallies=a.n, camera=dict(f=cam.f, centre=cam.centre.tolist()), results=R), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
