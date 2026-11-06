"""Held-out evaluation of the own ball tracker vs the current trackers on dev_match points 51-101, then the point-winner
stages re-run with the new track. Research measurement; outputs are private (never committed).

Trackers (all compared in 1280x720 analysis px, 30 fps, same per-point windows as the S0 v2 tracks):
  ours_raw        own heatmap net, best blob >= th per frame
  ours            ours_raw + physics filter (arcs, outliers out, in-arc gaps filled)
  engine_v2       engine-v2 BallEngine (HSV re-acquire) live positions  [the current reel/engine tracker]
  engine_v2_phys  engine-v2 pre-live positions + the same physics filter
  reel_ball       reel/ball.py detect + track (the original reel tracker)
  tracknet_s0     the point-winner S0 v2 ball (research TrackNet, 640x360, peak >= 0.5)  [current pw input]
  tracknet_s0_phys  tracknet_s0 + physics filter
A/B (Antony, 18 Oct: every efficiency trick is benchmarked against the plain pass and kept only if it wins):
  weights arms  r1_all (all labelled frames) vs r1_cur (curated hard-example subset, same minutes) vs r2 (self-trained)
  inference     plain / court / motion / track (opencourtvision.ball.own.fastpath) x fp32 / fp16, on the best weights
A trick is "kept" when it is >= 10% faster (ms/frame, so timing noise can't pick it) and loses <= 1 pt of sv_recall and of precision_proxy and <= 1 bounce.
The point-winner re-run uses the best weights with the plain pass, and the kept trick when one wins.
There is no hand-labelled ball ground truth (no new labelling from Antony), so recall/precision are PROXIES:
  sv_recall      share of SwingVision shot times with a detection within +-150 ms (SwingVision = independent system)
  audio_recall   same for our audio contacts;  coverage_600ms  frames carrying the ball in the 600 ms after each contact
  precision_proxy  1 - share of tracked frames in false tracklets (static / slow / off-court; court gate)
  arc_share      share of raw detections the physics filter accepts (raw trackers only)
plus bounce found per point (a bounce 0.1-2.5 s after our last audio contact, the window pw_stages uses), bounce
position agreement with SwingVision's last-shot bounce (<= 1.5 m on court), hitter cues at the last contact vs
SwingVision's last-hit side (near/far), and ms/frame.
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from opencourtvision.ball.geometry import CourtGate  # noqa: E402
from opencourtvision.ball.own import physics as P  # noqa: E402
from opencourtvision.ball.own.fastpath import FastRunner, court_roi  # noqa: E402
from opencourtvision.ball.own.heatnet import HeatNetTracker, param_count, to_input  # noqa: E402
from opencourtvision.ball.proxy import event_detection_rate, false_track_stats, post_hit_coverage  # noqa: E402

BASELINES = ("engine_v2", "engine_v2_phys", "reel_ball", "tracknet_s0", "tracknet_s0_phys")


def load_pw(path):
    spec = importlib.util.spec_from_file_location("pw_stages", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def time_net(trk, frames, n=120):
    """ms/frame of the network alone at batch 1 (after warm-up) and batch 8."""
    import torch
    sz = trk.size
    x1 = torch.rand(1, 9, sz[1], sz[0]).to(trk.device); x8 = torch.rand(8, 9, sz[1], sz[0]).to(trk.device)

    def sync():
        if trk.device == "mps":
            torch.mps.synchronize()
    with torch.no_grad():
        for _ in range(10):
            trk.model(x1)
        sync(); t = time.perf_counter()
        for _ in range(n):
            trk.model(x1)
        sync(); b1 = (time.perf_counter() - t) / n * 1000
        for _ in range(3):
            trk.model(x8)
        sync(); t = time.perf_counter()
        for _ in range(n // 8):
            trk.model(x8)
        sync(); b8 = (time.perf_counter() - t) / (n // 8 * 8) * 1000
    return round(b1, 2), round(b8, 2)


def new_acc():
    return dict(frames=0, det=0, sv=[0, 0], audio=[0, 0], cov=[0, 0], in_tracks=0, false=0, raw_det=0, raw_in=0,
                bounce_pts=0, bounce_sv=[0, 0], pts=0, cue={}, cue_all=[0, 0], cue_pair=[0, 0])


def r_(x):
    return round(x[0] / x[1], 3) if x[1] else None


def summarise(A):
    return dict(points=A["pts"], det_rate=round(A["det"] / max(1, A["frames"]), 3), sv_recall=r_(A["sv"]), sv_n=A["sv"][1],
                audio_recall=r_(A["audio"]), coverage_600ms=r_(A["cov"]),
                precision_proxy=round(1 - A["false"] / A["in_tracks"], 3) if A["in_tracks"] else None,
                arc_share=round(A["raw_in"] / A["raw_det"], 3) if A["raw_det"] else None,
                bounce_found=A["bounce_pts"], bounce_sv_agree=r_(A["bounce_sv"]), bounce_sv_n=A["bounce_sv"][1],
                hitter_cue_acc_vs_sv={c: r_(v) for c, v in A["cue"].items()}, hitter_cue_n={c: v[1] for c, v in A["cue"].items()},
                ball_cues_all_agree_sv=r_(A["cue_all"]), ball_cue_pairwise_agree=r_(A["cue_pair"]))


class Ctx:
    """Everything about one dev_match clip that the metrics need."""

    def __init__(self, a, name, m, p, pw, views, hits_all, SV, S_SRC):
        self.name, self.m, self.p = name, m, p
        self.z = np.load(os.path.join(a.data, "clips", name + ".npz"))
        self.d = np.load(os.path.join(a.tracks17, f"seg{m['idx']:03d}.npz"))
        self.n, self.t0 = m["n"], float(self.z["t0"])
        H = self.z["Hc2i"]
        self.Hc2i = H if np.isfinite(H).all() else None
        self.gate = CourtGate(self.Hc2i, image_size=(C.AW, C.AH)) if self.Hc2i is not None else None
        self.y_net = C.project(self.Hc2i, (C.COURT_W / 2, C.NET_Y))[1] if self.Hc2i is not None else None
        self.bx = C.boxes_fn(self.d, C.AW / 640.0)
        self.hits = [h for h in hits_all if self.t0 - 0.1 <= h <= self.t0 + self.n / C.AFPS + 0.1]
        self.hits_idx = [(h - self.t0) * C.AFPS * 1.0 for h in self.hits]
        self.tc = float(self.d["tc"]) if "tc" in self.d.files else np.nan
        sv = SV.get(str(m["idx"])) or {}
        self.sv_t = [s["t"] for s in sv.get("shots", [])]
        last = sv["shots"][-1] if sv.get("shots") else None
        self.sv_bounce = last["bounce_xy"] if last and last["result"] in ("In", "Out") else None
        self.side_truth = {"near": 1, "far": -1}.get(sv.get("last_hit_side"))
        self.sv_side = sv.get("last_hit_side")
        self.view = pw.view_at(views, p["end"])
        self.S_SRC = S_SRC
        Hn = None if self.Hc2i is None else np.diag([C.K_NET, C.K_NET, 1.0]) @ self.Hc2i
        self.roi = court_roi(Hn, C.NET)

    def small(self, a):
        fr = [cv2.imread(os.path.join(a.data, "frames", self.name, f"{i:05d}.jpg")) for i in range(self.n)]
        return [to_input([f]) for f in fr]

    def phys(self, pos):
        return P.filter_track(pos, C.AFPS, boxes_at=self.bx, y_net=self.y_net, frame_w=C.AW, arc_ok=C.live_arc_check(self.gate))

    def add(self, A, pos, phr, pw, hits_all, env, raw_phys=None):
        """accumulate one tracker's metrics for this clip. phr: physics result when pos IS a physics output."""
        n, t0 = self.n, self.t0
        A["pts"] += 1
        ok = np.isfinite(pos[:, 0]); A["frames"] += n; A["det"] += int(ok.sum())
        for key, ev in (("sv", self.sv_t), ("audio", self.hits)):
            h, tot = event_detection_rate(pos, t0, C.AFPS, ev); A[key][0] += h; A[key][1] += tot
        h, tot = post_hit_coverage(pos, t0, C.AFPS, self.hits); A["cov"][0] += h; A["cov"][1] += tot
        st = false_track_stats(pos, C.AFPS, self.gate); A["in_tracks"] += st["frames_in_tracks"]; A["false"] += st["frames_false"]
        if phr is None:
            q = raw_phys or self.phys(pos)
            A["raw_det"] += q["stats"]["detections"]; A["raw_in"] += q["stats"]["inliers"]; evs = q["events"]
        else:
            evs = phr["events"]
        tc = self.tc
        bo = [e for e in evs if e["kind"] == "bounce" and tc + 0.1 < t0 + e["t"] <= tc + 2.5]
        A["bounce_pts"] += bool(bo)
        if bo and self.sv_bounce is not None and self.gate is not None:
            g = self.gate.to_court(np.array([[bo[0]["x"], bo[0]["y"]]]))[0]
            A["bounce_sv"][1] += 1; A["bounce_sv"][0] += bool(math.hypot(g[0] - self.sv_bounce[0], g[1] - self.sv_bounce[1]) <= 1.5)
        dd = {kk: self.d[kk] for kk in self.d.files}
        dd["ball"] = (pos / 2.0).astype(np.float32)
        dd["ball_peak"] = (phr["conf"] if phr is not None else ok.astype(float)).astype(np.float32)
        bl = [dict(t=t0 + e["t"], x=e["x"] * self.S_SRC, y=e["y"] * self.S_SRC) for e in evs if e["kind"] == "bounce"]
        f, _ = pw.point_features(self.p, dd, self.view, hits_all, bl, env) if self.view is not None else ({}, None)
        cues = {c: f[c] for c in ("l2_ball_side", "l2_dir_after", "l2_dir_before", "l2_box") if c in f and f[c] != 0}
        if self.side_truth is not None:
            for c, v in cues.items():
                cc = A["cue"].setdefault(c, [0, 0]); cc[1] += 1; cc[0] += int(np.sign(v) == self.side_truth)
            ball_c = [np.sign(v) for c, v in cues.items() if c != "l2_box"]
            if ball_c:
                A["cue_all"][1] += 1; A["cue_all"][0] += int(all(s == self.side_truth for s in ball_c))
            if len(ball_c) >= 2:
                pairs = [(i, j) for i in range(len(ball_c)) for j in range(i + 1, len(ball_c))]
                A["cue_pair"][1] += len(pairs); A["cue_pair"][0] += sum(ball_c[i] == ball_c[j] for i, j in pairs)
        return dict(det=int(ok.sum()), bounce=bool(bo), cues={c: round(float(v), 2) for c, v in cues.items()})

    def write_npz(self, path, ph):
        out = {k: self.d[k] for k in self.d.files}
        out["ball"] = (ph["pos"] / 2.0).astype(np.float32); out["ball_peak"] = ph["conf"].astype(np.float32)
        out["bounces"] = np.array([[self.t0 + e["t"], e["x"] * self.S_SRC, e["y"] * self.S_SRC] for e in ph["events"]
                                   if e["kind"] == "bounce"], float).reshape(-1, 3)
        np.savez_compressed(path, **out)


def run_variant(runner, ctx, small, mode):
    pos, st = runner.run(small, "plain" if mode == "plain" else mode, ctx.roi, ctx.hits_idx)
    return pos / C.K_NET, st


def keep_trick(t, base):
    return (t["ms_per_frame"] <= 0.9 * base["ms_per_frame"] and (t["sv_recall"] or 0) >= (base["sv_recall"] or 0) - 0.01
            and (t["precision_proxy"] or 0) >= (base["precision_proxy"] or 0) - 0.01 and t["bounce_found"] >= base["bounce_found"] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True, nargs="+", help="NAME=path ... (weights arms); the first is the default")
    ap.add_argument("--tracks17", required=True); ap.add_argument("--labels17", required=True)
    ap.add_argument("--prelabels17", required=True); ap.add_argument("--sv17", required=True)
    ap.add_argument("--audio"); ap.add_argument("--pw_stages", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--modes", default="court,motion,track,plain_fp16,court_fp16,track_fp16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pw = load_pw(a.pw_stages)
    index = json.load(open(os.path.join(a.data, "index.json")))["clips"]
    PL = json.load(open(a.prelabels17)); S_SRC = int(PL.get("width", 3840)) / C.AW
    views = [pw.View(c) for c in PL["courts_by_view"]]
    hits_all = sorted(c["t"] for r in PL["rallies"] for c in r["contacts"])
    pre_bounces = sorted((b for r in PL["rallies"] for b in r.get("bounces", [])), key=lambda b: b["t"])
    env = np.load(a.audio)[0] if a.audio else None
    SV = json.load(open(a.sv17))["points"]
    labels = {p["idx"]: p for p in json.load(open(a.labels17))["points"]}
    arms = dict(w.split("=", 1) if "=" in w else (os.path.splitext(os.path.basename(w))[0], w) for w in a.weights)
    trk = {k: HeatNetTracker(v, th=a.th) for k, v in arms.items() if os.path.exists(v)}
    clips = sorted((m["idx"], n) for n, m in index.items() if m["source"] == "dev_match")
    held = [(i, n) for i, n in clips if index[n]["split"] == "heldout"]
    acc, timing, per_point = {}, {}, []

    def tally(key, st, n):
        T = timing.setdefault(key, dict(frames=0, net_s=0.0, wall_s=0.0, px=0.0, phys_s=0.0))
        T["frames"] += n; T["net_s"] += st["net_s"]; T["wall_s"] += st["wall_s"]; T["px"] += st["px_share"] * n

    # ---- pass 1: held-out, every arm (plain) and every trick on every arm's plain run kept for the arm choice
    for idx, name in held:
        ctx = Ctx(a, name, index[name], labels[idx], pw, views, hits_all, SV, S_SRC)
        small = ctx.small(a); row = dict(point_no=ctx.p["point_no"], idx=idx, sv_last_side=ctx.sv_side)
        for arm, t in trk.items():
            for mode in ["plain"] + ([m for m in a.modes.split(",") if m] if arm == list(trk)[0] or len(trk) == 1 else []):
                base_mode, fp16 = mode.replace("_fp16", ""), mode.endswith("_fp16")
                runner = FastRunner(t, fp16=fp16)
                raw, st = run_variant(runner, ctx, small, base_mode)
                tp = time.perf_counter(); ph = ctx.phys(raw); st["wall_s"] += time.perf_counter() - tp
                key = f"{arm}:{mode}"
                tally(key, st, ctx.n)
                row[key + ":raw"] = ctx.add(acc.setdefault(key + ":raw", new_acc()), raw, None, pw, hits_all, env, raw_phys=ph)
                row[key] = ctx.add(acc.setdefault(key, new_acc()), ph["pos"], ph, pw, hits_all, env)
        n = ctx.n
        s0 = ctx.d["ball"].astype(float) * 2.0; s0[ctx.d["ball_peak"] < 0.5] = np.nan
        s0 = s0[:n] if len(s0) >= n else np.r_[s0, np.full((n - len(s0), 2), np.nan)]
        ev2 = ctx.phys(ctx.z["teacher_raw"][:n]); s0p = ctx.phys(s0)
        for k, pos, phr in (("engine_v2", ctx.z["teacher_live"][:n], None), ("engine_v2_phys", ev2["pos"], ev2),
                            ("reel_ball", ctx.z["reel"][:n] if "reel" in ctx.z.files else np.full((n, 2), np.nan), None),
                            ("tracknet_s0", s0, None), ("tracknet_s0_phys", s0p["pos"], s0p)):
            row[k] = ctx.add(acc.setdefault(k, new_acc()), pos, phr, pw, hits_all, env)
        pb = [b for b in pre_bounces if ctx.tc + 0.1 < b["t"] <= ctx.tc + 2.5]
        A = acc.setdefault("prelabel_bounces", new_acc()); A["pts"] += 1; A["bounce_pts"] += bool(pb)
        if pb and ctx.sv_bounce is not None and ctx.gate is not None:
            g = ctx.gate.to_court(np.array([[pb[0]["x"] / S_SRC, pb[0]["y"] / S_SRC]]))[0]
            A["bounce_sv"][1] += 1; A["bounce_sv"][0] += bool(math.hypot(g[0] - ctx.sv_bounce[0], g[1] - ctx.sv_bounce[1]) <= 1.5)
        per_point.append(row)

    table = {k: summarise(v) for k, v in acc.items()}
    for k, T in timing.items():
        for kk in (k, k + ":raw"):
            if kk in table:
                table[kk]["ms_per_frame"] = round(1000 * (T["net_s"] if kk.endswith(":raw") else T["wall_s"]) / max(1, T["frames"]), 2)
                table[kk]["net_ms_per_frame"] = round(1000 * T["net_s"] / max(1, T["frames"]), 2)
                table[kk]["px_share"] = round(T["px"] / max(1, T["frames"]), 3)
    # best arm by held-out sv_recall x precision_proxy (plain pass, physics output)
    def score(k):
        t = table.get(k) or {}
        return (t.get("sv_recall") or 0) * (t.get("precision_proxy") or 0) + 0.002 * t.get("bounce_found", 0)
    best_arm = max(trk, key=lambda arm: score(f"{arm}:plain"))
    base = table[f"{list(trk)[0]}:plain"]
    tricks = {}
    for k, t in table.items():
        if k.startswith(list(trk)[0] + ":") and not k.endswith(":raw") and not k.endswith(":plain"):
            tricks[k.split(":", 1)[1]] = dict(kept=bool(keep_trick(t, base)), ms_per_frame=t["ms_per_frame"],
                                              speedup=round(base["ms_per_frame"] / max(1e-6, t["ms_per_frame"]), 2),
                                              d_sv_recall=round((t["sv_recall"] or 0) - (base["sv_recall"] or 0), 3),
                                              d_precision=round((t["precision_proxy"] or 0) - (base["precision_proxy"] or 0), 3),
                                              d_bounce=t["bounce_found"] - base["bounce_found"], px_share=t["px_share"])
    kept = [m for m, v in tricks.items() if v["kept"]]
    best_mode = min(kept, key=lambda m: tricks[m]["ms_per_frame"]) if kept else "plain"
    b1, b8 = time_net(trk[best_arm], None)
    import torch
    meta = {k: torch.load(v, map_location="cpu").get("meta", {}) for k, v in arms.items() if os.path.exists(v)}
    res = dict(schema="ocv.ball_tracker_phase1.v2", heldout="dev_match point_no >= 51", arms=list(trk), best_arm=best_arm,
               weights_meta=meta, tricks_vs_plain=tricks, chosen_mode=best_mode,
               speed=dict(net_ms_per_frame_batch1=b1, net_ms_per_frame_batch8=b8, device=trk[best_arm].device,
                          params=param_count(trk[best_arm].model), input=list(C.NET),
                          teacher_engine_v2_ms_per_frame=round(1000 * sum(m.get("teacher_s", 0) for m in index.values() if m["source"] == "dev_match")
                                                              / max(1, sum(m["n"] for m in index.values() if m["source"] == "dev_match")), 2),
                          reel_ball_ms_per_frame=round(1000 * sum(m.get("reel_s", 0) for m in index.values() if "reel_s" in m)
                                                       / max(1, sum(m["n"] for m in index.values() if "reel_s" in m)), 2)),
               table=table, notes="recall/precision are proxies (no hand ball labels); see module docstring")
    C.save_json(res, os.path.join(a.out, "ball_metrics.json"))
    C.save_json(per_point, os.path.join(a.out, "ball_per_point.json"))
    print(json.dumps(dict(best_arm=best_arm, chosen_mode=best_mode, tricks=tricks, speed=res["speed"]), indent=1), flush=True)

    # ---- pass 2: all 101 points with the best arm (plain, and the kept trick) -> pw_stages
    variants = ["plain"] + ([best_mode] if best_mode != "plain" else [])
    dirs = {v: os.path.join(a.out, f"tracks_{best_arm}_{v}") for v in variants}
    for v in dirs.values():
        os.makedirs(v, exist_ok=True)
    for idx, name in clips:
        ctx = Ctx(a, name, index[name], labels[idx], pw, views, hits_all, SV, S_SRC)
        small = ctx.small(a)
        for v in variants:
            raw, _ = run_variant(FastRunner(trk[best_arm], fp16=v.endswith("_fp16")), ctx, small, v.replace("_fp16", ""))
            ctx.write_npz(os.path.join(dirs[v], f"seg{idx:03d}.npz"), ctx.phys(raw))
    py = sys.executable
    runs = {"current_s0": (a.tracks17, False)}
    for v, dd in dirs.items():
        runs[f"own_{v}"] = (dd, False); runs[f"own_{v}_own_bounces"] = (dd, True)
    pwres = {}
    for nm, (td, ob) in runs.items():
        o = os.path.join(a.out, f"pw_{nm}.json")
        cmd = [py, a.pw_stages, "--labels", a.labels17, "--tracks", td, "--prelabels", a.prelabels17, "--out", o]
        if a.audio:
            cmd += ["--audio", a.audio]
        if ob:
            cmd += ["--bounces-from-tracks"]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        print(nm, cp.stdout[-2500:], cp.stderr[-1500:], flush=True)
        if cp.returncode == 0:
            R = json.load(open(o))
            pwres[nm] = dict(s0=R["s0"], majority=R["majority_baseline"],
                             stages={k: dict(acc=v["first_call_acc"], acc_51_101=v["acc_test_51_101"], acc_1_50=v["acc_test_1_50"],
                                             taps99=round(v["taps_per_100_at_99"], 1), taps999=round(v["taps_per_100_at_999"], 1))
                                     for k, v in R["stages"].items()})
    res["point_winner"] = pwres
    C.save_json(res, os.path.join(a.out, "ball_metrics.json"))
    print(json.dumps(pwres, indent=1))


if __name__ == "__main__":
    main()
