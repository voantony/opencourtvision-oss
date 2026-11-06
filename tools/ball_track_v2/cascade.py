"""Cascade: run the cheap detector everywhere, and the expensive one only on frames where the cheap track is unsure,
then fuse there (rule fusion) and let the physics tracker run on the mix. Writes a candidate dir for physball_eval
plus the cost model: det_ms = cheap_ms + share_escalated x expensive_ms (expensive nets run on 3/8-frame windows, so the
share is counted on whole windows). Policies (no ground truth used):
  gap:D        frames with no physics inlier of the cheap track within +-D frames
  tail:S       frames from the last audio contact - S s to the clip end
  lcw:S        last-contact window: last audio contact - S s to + 2.5 s (where the deciding shot and bounce are)
  gap:D+tail:S union of both
CPU (box); private outputs.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from opencourtvision.ball.fuse import fuse  # noqa: E402

_VF = os.environ.get("OCV_VF", "/workspace/vf")
DATA = os.environ.get("OCV_DATA", f"{_VF}/ballbench/run1/data"); PL = f"{_VF}/dev_match_tracks/prelabel/prelabels.json"


def escalate_mask(n, conf, hits_k, policy, win=3):
    m = np.zeros(n, bool)
    for part in policy.split("+"):
        kind, v = part.split(":"); v = float(v)
        if kind == "gap":
            inl = np.convolve((conf >= 1.0).astype(float), np.ones(int(2 * v + 1)), "same") > 0
            m |= ~inl
        elif kind == "tail":
            k0 = int(max(hits_k)) - int(v * 30) if hits_k else n - int(v * 30)
            m[max(0, k0):] = True
        elif kind == "lcw":         # last-contact window: last audio contact - S s .. + 2.5 s (the shot and bounce that decide the point)
            kl = int(max(hits_k)) if hits_k else n - 75
            m[max(0, kl - int(v * 30)):min(n, kl + 75)] = True
        elif kind == "all":
            m[:] = True
    # whole windows of the expensive net
    w = np.zeros(n, bool)
    for s in range(0, n, win):
        if m[s:s + win].any():
            w[s:s + win] = True
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", required=True); ap.add_argument("--cheap_dump", required=True); ap.add_argument("--exp", required=True, action="append")
    ap.add_argument("--policy", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cheap_ms", type=float, required=True); ap.add_argument("--exp_ms", type=float, required=True)
    ap.add_argument("--win", type=int, default=3)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(DATA, "index.json")))["clips"]
    hits_all = sorted(c["t"] for r in json.load(open(PL))["rallies"] for c in r["contacts"])
    os.makedirs(a.out, exist_ok=True); fr = esc = 0
    for f in sorted(os.listdir(a.cheap_dump)):
        nm = f[:-4]
        if not all(os.path.exists(os.path.join(d, nm + ".npz")) for d in [a.cheap] + a.exp):
            continue
        n = idx[nm]["n"]
        ch = np.load(os.path.join(a.cheap, nm + ".npz"))["cands"].astype(float)[:n]
        ex = [np.load(os.path.join(d, nm + ".npz"))["cands"].astype(float)[:n] for d in a.exp]
        t = np.load(os.path.join(a.cheap_dump, f)); conf = t["conf"][:n]; t0 = float(t["t0"])
        hk = [(h - t0) * 30 for h in hits_all if t0 <= h <= t0 + n / 30.0]
        m = escalate_mask(n, conf, hk, a.policy, a.win)
        fu, _ = fuse([ch] + ex)
        K = max(ch.shape[1], fu.shape[1]); out = np.full((n, K, 3), np.nan)
        out[:, :ch.shape[1]] = ch; out[m] = np.nan; out[m, :fu.shape[1]] = fu[m]
        np.savez_compressed(os.path.join(a.out, f), cands=out.astype(np.float32), t0=t0, fps=30.0, escalated=m)
        fr += n; esc += int(m.sum())
    share = esc / max(1, fr)
    s = dict(policy=a.policy, share_escalated=round(share, 3), det_ms=round(a.cheap_ms + share * a.exp_ms, 2), fuse_ms=0.0, att_ms=0.0, frames=fr)
    json.dump(s, open(os.path.join(a.out, "_arm.json"), "w"), indent=1); print(json.dumps(s))


if __name__ == "__main__":
    main()
