"""Build the candidate sets for one benchmark arm: a single detector, a rule fusion of several, and/or the player-
attention prior on top. Writes OUT/<clip>.npz in the extract_cands format (cands (T, K, 3), t0, fps) plus agree and
ms_* timings, so tools/ball_tracker/physball_eval.py runs the physics tracker on it unchanged. CPU; private outputs.

  --src wasb=DIR --src tnv3=DIR ...   detector candidate dirs (cands_multi.py / extract_cands.py output)
  --use wasb,tnv3                     which sources (one = single detector, several = rule fusion)
  --att none|all|gaze|contact|static|gaze,contact,...  attention cues;  --hide  gaze pseudo-candidate when none
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from opencourtvision.ball import attention as AT  # noqa: E402
from opencourtvision.ball.fuse import fuse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True); ap.add_argument("--use", required=True)
    ap.add_argument("--weights", default=""); ap.add_argument("--data", required=True); ap.add_argument("--tracks17", required=True)
    ap.add_argument("--att", default="none"); ap.add_argument("--hide", action="store_true")
    ap.add_argument("--split", default="heldout"); ap.add_argument("--out", required=True); ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    src = dict(s.split("=", 1) for s in a.src)
    use = a.use.split(","); w = [float(x) for x in a.weights.split(",")] if a.weights else None
    cues = () if a.att == "none" else (("gaze", "contact", "static") if a.att == "all" else tuple(a.att.split(",")))
    cfg = AT.Config(**json.loads(a.cfg))
    idx = json.load(open(os.path.join(a.data, "index.json")))["clips"]
    names = [n for n, m in sorted(idx.items()) if m["source"] == "dev_match" and (a.split == "all" or m["split"] == a.split)]
    os.makedirs(a.out, exist_ok=True)
    tot = dict(frames=0, det_ms=0.0, fuse_ms=0.0, att_ms=0.0, clips=0)
    for name in names:
        paths = [os.path.join(src[u], name + ".npz") for u in use]
        if not all(os.path.exists(p) for p in paths):
            continue
        zs = [np.load(p) for p in paths]
        cl = [z["cands"].astype(float) for z in zs]
        n = idx[name]["n"]; cl = [c[:n] if len(c) >= n else np.concatenate([c, np.full((n - len(c),) + c.shape[1:], np.nan)]) for c in cl]
        det_ms = sum(float(z["ms_model"]) if "ms_model" in z.files else np.nan for z in zs)
        t = time.perf_counter()
        if len(cl) == 1:
            c = cl[0]; agree = np.isfinite(c[..., 0]).astype(int)
        else:
            c, agree = fuse(cl, w)
        f_ms = 1000 * (time.perf_counter() - t) / n
        a_ms = 0.0
        if cues or a.hide:
            t = time.perf_counter()
            z = np.load(os.path.join(a.data, "clips", name + ".npz"))
            d = np.load(os.path.join(a.tracks17, f"seg{idx[name]['idx']:03d}.npz"))
            off = int(round((float(z["t0"]) - float(d["t0"])) * 30.0))
            ps = AT.players_from_npz(d, n, off)
            H = z["Hc2i"]; y_net = None
            if np.isfinite(H).all():
                p = H @ np.array([10.97 / 2, 11.885, 1.0]); y_net = p[1] / p[2]
            F = AT.features(c, ps, cfg, y_net)
            c = AT.reweight(c, F, ps, cfg, use=cues, hide=a.hide, y_net=y_net)
            a_ms = 1000 * (time.perf_counter() - t) / n
        t0 = float(zs[0]["t0"])
        np.savez_compressed(os.path.join(a.out, name + ".npz"), cands=c.astype(np.float32), t0=t0, fps=30.0, agree=agree,
                            ms_det=det_ms, ms_fuse=f_ms, ms_att=a_ms)
        tot["frames"] += n; tot["clips"] += 1; tot["det_ms"] += det_ms * n; tot["fuse_ms"] += f_ms * n; tot["att_ms"] += a_ms * n
    s = {k: (round(v / max(1, tot["frames"]), 2) if k.endswith("_ms") else v) for k, v in tot.items()}
    json.dump(dict(use=use, att=list(cues), hide=a.hide, **s), open(os.path.join(a.out, "_arm.json"), "w"), indent=1)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
