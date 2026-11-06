"""Run the own ball tracker (best weights, plain pass + physics filter) over the windows of an existing pw_tracks
segment folder (e.g. the dev_match S0 v3 whole-point tracks) and write a copy of each segment npz with the ball replaced
by ours (+ our physics bounces), so pw_stages (S1-S6, --anchor rally, --hard_rules) can run on v3 players + own ball.
Mac research tool: nice 20, chunked decode (bounded memory), resumable (existing outputs are skipped). Outputs are
private (never committed)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from opencourtvision.ball.geometry import CourtGate  # noqa: E402
from opencourtvision.ball.own import physics as P  # noqa: E402
from opencourtvision.ball.own.fastpath import FastRunner  # noqa: E402
from opencourtvision.ball.own.heatnet import HeatNetTracker, to_input  # noqa: E402


def track_window(runner, video, t0, n, chunk=240, pad=2):
    """own-net positions (n, 2) in analysis px over [t0, t0 + n/fps), decoded in overlapping chunks."""
    pos = np.full((n, 2), np.nan)
    s = 0
    while s < n:
        a = max(0, s - pad); b = min(n, s + chunk + pad)
        fr = C.decode(video, t0 + a / C.AFPS, b - a)
        if not fr:
            break
        small = [to_input([f]) for f in fr]
        p, _ = runner.run(small, "plain")
        p = p / C.K_NET
        k0, k1 = s - a, min(len(p), s - a + chunk)
        pos[s:s + (k1 - k0)] = p[k0:k1]
        s += chunk
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--tracks", required=True)
    ap.add_argument("--prelabels", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--th", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    PL = json.load(open(a.prelabels)); sw = int(PL.get("width", 3840)); S_SRC = sw / C.AW
    runner = FastRunner(HeatNetTracker(a.weights, th=a.th))
    segs = sorted(f for f in os.listdir(a.tracks) if f.startswith("seg") and f.endswith(".npz"))
    tot_f, tot_s = 0, 0.0
    for f in segs:
        dst = os.path.join(a.out, f)
        if os.path.exists(dst):
            continue
        d = np.load(os.path.join(a.tracks, f))
        t0, n = float(d["t0"]), len(d["ball"])
        v = C.view_at(PL["courts_by_view"], t0 + n / C.AFPS / 2)
        Hc2i = C.h_court2img(v["H_img2court"], sw) if v else None
        gate = CourtGate(Hc2i, image_size=(C.AW, C.AH)) if Hc2i is not None else None
        y_net = C.project(Hc2i, (C.COURT_W / 2, C.NET_Y))[1] if Hc2i is not None else None
        st = time.perf_counter()
        raw = track_window(runner, a.video, t0, n)
        ph = P.filter_track(raw, C.AFPS, boxes_at=C.boxes_fn(d, C.AW / 640.0), y_net=y_net, frame_w=C.AW,
                            arc_ok=C.live_arc_check(gate))
        out = {k: d[k] for k in d.files}
        out["ball"] = (ph["pos"] / 2.0).astype(np.float32); out["ball_peak"] = ph["conf"].astype(np.float32)
        out["bounces"] = np.array([[t0 + e["t"], e["x"] * S_SRC, e["y"] * S_SRC] for e in ph["events"] if e["kind"] == "bounce"],
                                  float).reshape(-1, 3)
        np.savez_compressed(dst + ".tmp.npz", **out); os.replace(dst + ".tmp.npz", dst)
        dt = time.perf_counter() - st; tot_f += n; tot_s += dt
        print(f"{time.strftime('%T')} {f} n={n} det={np.isfinite(ph['pos'][:, 0]).mean():.2f} "
              f"bounces={len(out['bounces'])} {1000 * dt / max(1, n):.1f} ms/f", flush=True)
    print(f"DONE {len(segs)} segs, {tot_f} frames, {tot_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
