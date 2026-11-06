"""Self-training round: run a trained net over the train/val clips, merge with the teacher, re-run the physics filter.

New labels (r2_pos, r2_w) are written into each clip npz. Per frame the candidate is the teacher's detection where the
teacher's own arcs accepted it, else the net's best blob, else the teacher's raw one; only arc-accepted frames become
labels, so a net error has to be physically consistent over several frames to survive (adds labels, never trades
teacher-arc labels away). Held-out clips are never touched.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from opencourtvision.ball.geometry import CourtGate  # noqa: E402
from opencourtvision.ball.own import physics as P  # noqa: E402
from opencourtvision.ball.own.heatnet import HeatNetTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--th", type=float, default=0.5)
    a = ap.parse_args()
    ip = os.path.join(a.data, "index.json")
    index = json.load(open(ip))
    trk = HeatNetTracker(a.weights, th=a.th)
    tot = dict(clips=0, frames=0, r1_labels=0, r2_labels=0)
    for name, m in index["clips"].items():
        if m["split"] == "heldout":
            continue
        zp = os.path.join(a.data, "clips", name + ".npz")
        z = dict(np.load(zp))
        d = os.path.join(a.data, "frames", name)
        frames = [cv2.imread(os.path.join(d, f"{i:05d}.jpg")) for i in range(m["n"])]
        if any(f is None for f in frames):
            continue
        cands, _ = trk.run(frames)
        net = np.full((m["n"], 2), np.nan)
        for i, c in enumerate(cands):
            if c:
                net[i] = np.array(c[0][:2]) / C.K_NET          # 768 -> 1280 analysis px
        teach = z["teacher_raw"][:m["n"]]
        keep = (z["phys_w"][:m["n"]] >= 1.0)[:, None]
        both = np.where(keep, teach, np.where(np.isfinite(net), net, teach))
        H = z.get("Hc2i")
        gate = CourtGate(H, image_size=(C.AW, C.AH)) if H is not None and np.isfinite(H).all() else None
        ph = P.filter_track(both, C.AFPS, frame_w=C.AW, arc_ok=C.live_arc_check(gate))
        z["r2_pos"], z["r2_w"] = ph["pos"], ph["conf"]
        np.savez_compressed(zp, **z)
        m["r2_labels"] = int((ph["conf"] > 0).sum())
        tot["clips"] += 1; tot["frames"] += m["n"]; tot["r1_labels"] += m["labels"]; tot["r2_labels"] += m["r2_labels"]
    index["r2_counts"] = tot
    C.save_json(index, ip)
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
