"""Learned candidate scorer (opencourtvision.ball.learned.CandNet): build features, train on the dev_match TRAIN points
(teacher = physics-verified inlier picks of the best rule-fused arm, dumped by physball_eval --dump_pos), apply to the
held-out points -> a candidate dir for physball_eval (peaks = learned scores). CPU (box). Weights/outputs private.

  candnet.py train --src wasb=DIR ... --use wasb,tnv3,heatnet --teacher DUMPDIR --model OUT.pt [--no_att] [--no_audio]
  candnet.py apply --src ... --use ... --model OUT.pt --out CANDDIR
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
from opencourtvision.ball import learned as L  # noqa: E402
from opencourtvision.ball.fuse import fuse  # noqa: E402

_VF = os.environ.get("OCV_VF", "/workspace/vf")
DATA = os.environ.get("OCV_DATA", f"{_VF}/ballbench/run1/data"); TR = f"{_VF}/dev_match_tracks_v2/seg_r"; AUD = f"{_VF}/ballbench/audio_env_dev_match.npy"


def clip_inputs(name, m, src, use, att=True, audio=True, env=None):
    n = m["n"]
    cl = []
    for u in use:
        c = np.load(os.path.join(src[u], name + ".npz"))["cands"].astype(float)
        cl.append(c[:n] if len(c) >= n else np.concatenate([c, np.full((n - len(c),) + c.shape[1:], np.nan)]))
    t = time.perf_counter()
    fused, _ = fuse(cl) if len(cl) > 1 else (cl[0], None)
    F = None
    if att:
        z = np.load(os.path.join(DATA, "clips", name + ".npz")); d = np.load(os.path.join(TR, f"seg{m['idx']:03d}.npz"))
        ps = AT.players_from_npz(d, n, int(round((float(z["t0"]) - float(d["t0"])) * 30)))
        H = z["Hc2i"]; y_net = None
        if np.isfinite(H).all():
            p = H @ np.array([5.485, 11.885, 1.0]); y_net = p[1] / p[2]
        F = AT.features(fused, ps, None, y_net)
    adb = None
    if audio and env is not None:
        t0 = float(np.load(os.path.join(DATA, "clips", name + ".npz"))["t0"])
        ii = np.clip(((t0 + np.arange(n) / 30.0) * 100).astype(int), 0, env.shape[1] - 1)
        adb = env[0][ii]
    X, valid = L.cand_features(fused, cl, F, adb)
    return fused, X, valid, 1000 * (time.perf_counter() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "apply"]); ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--use", required=True); ap.add_argument("--teacher"); ap.add_argument("--model", required=True)
    ap.add_argument("--out"); ap.add_argument("--no_att", action="store_true"); ap.add_argument("--no_audio", action="store_true")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--split", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4); torch.manual_seed(0)
    src = dict(s.split("=", 1) for s in a.src); use = a.use.split(",")
    idx = json.load(open(os.path.join(DATA, "index.json")))["clips"]
    split = a.split or ("train" if a.cmd == "train" else "heldout")
    names = [nm for nm, m in sorted(idx.items()) if m["source"] == "dev_match" and m["split"] == split
             and all(os.path.exists(os.path.join(src[u], nm + ".npz")) for u in use)]
    env = np.load(AUD) if not a.no_audio else None
    if a.cmd == "train":
        Xs, Vs, Ys, Ms = [], [], [], []
        for nm in names:
            tp = os.path.join(a.teacher, nm + ".npz")
            if not os.path.exists(tp):
                continue
            fused, X, valid, _ = clip_inputs(nm, idx[nm], src, use, not a.no_att, not a.no_audio, env)
            t = np.load(tp); pos, conf = t["pos"], t["conf"]
            T = min(len(pos), len(X)); Y = np.zeros(valid.shape, np.float32); M = np.zeros(len(X), np.float32)
            for i in range(T):
                if conf[i] >= 1.0 and np.isfinite(pos[i, 0]):
                    d = np.hypot(fused[i, :, 0] - pos[i, 0], fused[i, :, 1] - pos[i, 1])
                    if np.isfinite(d).any() and np.nanmin(d) <= 8.0:
                        Y[i, int(np.nanargmin(d))] = 1.0
                    M[i] = 1.0
                elif not np.isfinite(pos[i, 0]) and valid[i].any():
                    M[i] = 0.25          # frames the physics tracker left empty: weak negatives
            Xs.append(X); Vs.append(valid); Ys.append(Y); Ms.append(M)
        print(f"train clips {len(Xs)}, positives {int(sum(y.sum() for y in Ys))}", flush=True)
        net = L.build_candnet(); opt = torch.optim.Adam(net.parameters(), 2e-3)
        for ep in range(a.epochs):
            tot = 0.0
            for j in np.random.default_rng(ep).permutation(len(Xs)):
                x = torch.from_numpy(Xs[j])[None]; v = torch.from_numpy(Vs[j].astype(np.float32))[None]
                y = torch.from_numpy(Ys[j])[None]; mm = torch.from_numpy(Ms[j])[None, :, None] * v
                lo = net(x, v)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(lo, y, reduction="none", pos_weight=torch.tensor(3.0)) * mm).sum() / mm.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            if ep % 10 == 9:
                print(f"epoch {ep + 1} loss {tot / len(Xs):.4f}", flush=True)
        torch.save(dict(model=net.state_dict(), use=use, att=not a.no_att, audio=not a.no_audio), a.model)
    else:
        ck = torch.load(a.model, weights_only=False); net = L.build_candnet(); net.load_state_dict(ck["model"]); net.eval()
        os.makedirs(a.out, exist_ok=True); ms_f = ms_n = 0.0; nfr = 0
        for nm in names:
            fused, X, valid, f_ms = clip_inputs(nm, idx[nm], src, use, ck["att"], ck["audio"], env)
            t = time.perf_counter()
            with torch.no_grad():
                p = torch.sigmoid(net(torch.from_numpy(X)[None], torch.from_numpy(valid.astype(np.float32))[None]))[0].numpy()
            n_ms = 1000 * (time.perf_counter() - t) / len(X)
            c = fused.copy(); c[..., 2] = np.where(valid, p, np.nan)
            o = np.argsort(-np.nan_to_num(c[..., 2], nan=-1), 1)
            c = np.take_along_axis(c, o[..., None], 1)
            np.savez_compressed(os.path.join(a.out, nm + ".npz"), cands=c.astype(np.float32), t0=float(np.load(os.path.join(DATA, "clips", nm + ".npz"))["t0"]),
                                fps=30.0, ms_feat=f_ms, ms_net=n_ms)
            ms_f += f_ms * len(X); ms_n += n_ms * len(X); nfr += len(X)
        json.dump(dict(use=use, model=os.path.basename(a.model), ms_feat=round(ms_f / max(1, nfr), 2), ms_net=round(ms_n / max(1, nfr), 3)),
                  open(os.path.join(a.out, "_arm.json"), "w"))
        print(f"applied {len(names)} clips, feat {ms_f / max(1, nfr):.2f} ms/f, net {ms_n / max(1, nfr):.3f} ms/f")


if __name__ == "__main__":
    main()
