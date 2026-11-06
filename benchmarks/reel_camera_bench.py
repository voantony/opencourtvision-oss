#!/usr/bin/env python3
"""Reel camera bench (v4.6): the full camera check set on cached inputs, no decode, no render.

  package   python -m benchmarks.reel_camera_bench package <reel_cfg.json> [--camera REF] [--out stats.json]
            reel.build(plan_only) on a config with "det_cache" (detections from a previous run): the Media Director's
            check path (stats identical to a render run; the render is skipped).
  harness   python -m benchmarks.reel_camera_bench harness <dir of in_*.pkl> [--camera REF] [--out stats.json]
            camera inputs dumped by the Mac reel harness (Apple Vision player boxes with court side, fused ball track,
            verified contacts with sides / kinds): plan_clip -> tilt -> the same checks as build().

--camera REF runs an older camera.py (git ref, e.g. 37302d7 or 0c59fa6) for before/after; the v4.6 checks
(pre-serve, source edge, far partner) are computed on its path with the current check code. Old refs get the
contacts without sides / kinds, as their build() passed them. Inputs and outputs are local (no private labels).
"""
import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import tempfile

import numpy as np

from opencourtvision.reel import camera as NEW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_FUNCS = ("serve_role", "server_track", "source_edge_check", "preserve_qa", "framing_checks", "partner_stats", "_ymid",
               "far_keep_rows", "crop_height_qa", "crop_height_target", "jerk_qa", "role_clip_qa", "infeasible_decision")


def load_camera(ref):
    """Current camera, or camera.py at a git ref wrapped so build() / the checks can use it."""
    if not ref: return NEW
    src = subprocess.run(["git", "show", f"{ref}:opencourtvision/reel/camera.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    f = tempfile.NamedTemporaryFile("w", suffix=f"_camera_{ref}.py", delete=False); f.write(src); f.close()
    sp = importlib.util.spec_from_file_location(f"camera_{ref}", f.name); old = importlib.util.module_from_spec(sp); sp.loader.exec_module(old)
    plan0 = old.plan_clip
    def plan_clip(boxes, ball_xy, ball_x, contacts, t0, fps, SW=3840, SH=2160, cfg=None, scale=3.0, fallback=None, need_v=None,
                  big_shots=(), punch_in=False, sides=None, kinds=None, **kw):
        cfg = dict(old.CAMERA, **{k: v for k, v in (cfg or {}).items() if k in old.CAMERA or k in ("max_wide", "infeasible_action", "jerk_baseline")})
        p = plan0(boxes, ball_xy, ball_x, contacts, t0, fps, SW, SH, cfg, scale, fallback, need_v, big_shots, punch_in)
        k_ = None if not kinds else kinds
        sr = NEW.serve_role(p["roles"], t0, NEW.CAMERA, k_)
        p.update(serve_i=sr["i"] if sr else None, server=NEW.server_track(boxes, sr) if sr else {},
                 source_edge=NEW.source_edge_check(p["roles"], scale, SW, fps, t0, NEW.CAMERA))
        return p
    old.plan_clip = plan_clip
    for k in CHECK_FUNCS:
        if k in ("crop_height_qa", "crop_height_target", "jerk_qa", "role_clip_qa") and hasattr(old, k): continue
        setattr(old, k, getattr(NEW, k))
    for k, v in NEW.CAMERA.items(): old.CAMERA.setdefault(k, v)
    old.REF = ref
    return old


def run_package(cfg_path, cam_mod, out):
    from opencourtvision.reel import reel as R
    cfg = json.load(open(cfg_path)); base = os.path.dirname(os.path.abspath(cfg_path))
    cfg["det_cache"] = os.path.join(base, cfg.get("det_cache", "det")); cfg["plan_only"] = True
    if not os.path.isabs(cfg["events"]) or not os.path.exists(cfg["events"]): cfg["events"] = os.path.join(base, os.path.basename(cfg["events"]))
    if cfg.get("player_boxes") and not os.path.isdir(cfg["player_boxes"]):   # same key as the Mac config, local copy
        cfg["player_boxes"] = os.path.join(base, os.path.basename(cfg["player_boxes"].rstrip("/")))
    keep = R.camera; R.camera = cam_mod
    try:
        with tempfile.TemporaryDirectory() as work:
            st = R.build(cfg, work, None, None)
    finally:
        R.camera = keep
    return st


def run_harness(d, cam, out, SW=3840, SH=2160, scale=3.0):
    """Mirror of build()'s camera section on Vision-box inputs; reel-level stats in build()'s keys."""
    files = sorted(f for f in os.listdir(d) if f.startswith("in_") and f.endswith(".pkl"))
    order = json.load(open(os.path.join(d, "order.json"))) if os.path.exists(os.path.join(d, "order.json")) else None
    if order: files = [f"in_{n}.pkl" for n in order if f"in_{n}.pkl" in files]
    C = cam.CAMERA if cam is NEW else dict(cam.CAMERA)
    cfg = dict(C, max_wide=1); stats = dict(frames=0, far_out_frames=0, any_out_frames=0, sliced_frames=0, ball_frames=0, ball_in_crop=0)
    h_all, jerk, clips, rq_tot = [], [], {}, dict(contacts_hitter_clipped=0, contacts=0, receptions_receiver_clipped=0, receptions=0)
    for f in files:
        m = pickle.load(open(os.path.join(d, f), "rb"))
        name, a, fps, boxes, pos, bx = m["name"], m["a"], m["fps"], m["boxes"], m["pos"], m["bx"]
        n = len(boxes); ct = sorted(t for t in m["contacts"] if m["a"] <= t <= m["b"])
        top, bot = cam.vertical_box(boxes, scale, n)
        need = cam.need_height(top, bot, fps, SH, cfg) if not np.isnan(top).all() else None
        fbm = np.zeros(n, bool) if m.get("end_card") else m["fbm"]
        if not ct or m.get("end_card"):                             # end card / no contacts: steady default crop
            h = np.full(n, cfg["h_default"]); cw, ch = cam.crop_size(h, SH); cx = np.full(n, SW / 2.0); gp = None
        else:
            sd = dict(zip(m["contacts"], m.get("sides") or [None] * len(m["contacts"])))
            kd = dict(zip(m["contacts"], m.get("kinds") or [None] * len(m["contacts"])))
            sides, kinds = [sd.get(t) for t in ct], [kd.get(t) for t in ct]
            gp = cam.plan_clip(boxes, pos, bx, ct, a, fps, SW, SH, cfg, scale, fbm, need, [t for t in m["big"] if a <= t <= m["b"]], False,
                               sides=sides, kinds=kinds)
            h, cw, ch, cx = gp["h"], gp["cw"], gp["ch"], gp["cx"]
        x = np.clip(np.round(cx - cw / 2), 0, SW - cw).astype(int)
        y = cam.vertical_follow(top, bot, ch, fps, SH, cfg, None)
        wide = m.get("mode") == "wide"
        ps = NEW.partner_stats(boxes, x, cw, scale, wide, SW)
        for k in ("far_out", "any_out", "sliced"): stats[k + "_frames"] += int(ps[k].sum())
        det = ~np.isnan(bx); inn = det & (bx >= x + 20) & (bx <= x + cw - 20)
        stats["frames"] += n; stats["ball_frames"] += int(det.sum()); stats["ball_in_crop"] += int(inn.sum())
        h_all.append(np.ones(n) if wide else h)
        if n > 3: jerk.append(np.abs(np.diff(cx, 3)))
        c = dict(frames=n, far_out=round(float(ps["far_out"].mean()), 3), sliced=round(float(ps["sliced"].mean()), 3),
                 mean_crop_height=round(float(np.mean(h_all[-1])), 3), wide=wide)
        if gp is not None:
            rq = NEW.role_clip_qa(gp["roles"], x, y, cw, ch, scale, SW, fps, a, cfg, wide)
            for k in rq_tot: rq_tot[k] += rq[k]
            pq = NEW.preserve_qa(gp, x, y, cw, ch, scale, fps, a, wide, cfg)
            v_, a_, q_ = gp["caps"]
            c.update(roles_qa=rq, preserve=pq, source_edge=gp["source_edge"], caps=dict(v_over_cap=round(v_, 3), a_over_cap=round(a_, 3), quick_reversals=len(q_)),
                     far_keep_dropped=bool(gp.get("far_keep_dropped")))
            clips[name] = dict(preserve=pq, source_edge=gp["source_edge"], wide=wide)
        stats[name] = c
    H = np.concatenate(h_all); F = max(1, stats["frames"])
    hq = NEW.crop_height_qa(H, NEW.CAMERA)
    stats.update(rq_tot, roles_check={"pass": rq_tot["contacts_hitter_clipped"] == 0 and rq_tot["receptions_receiver_clipped"] == 0},
                 mean_crop_height=hq["mean_crop_height"], pct_frames_ge95=hq["pct_frames_ge95"],
                 crop_height_check=dict(target=NEW.crop_height_target(NEW.CAMERA), ok=not hq["warnings"], warnings=hq["warnings"]),
                 jerk_p95={"px_per_frame3": float(f"{np.percentile(np.concatenate(jerk), 95):.6g}")},
                 pct_ball_frames_in_crop=round(100 * stats["ball_in_crop"] / max(1, stats["ball_frames"]), 1))
    stats["jerk_qa"] = NEW.jerk_qa(stats["jerk_p95"]["px_per_frame3"], dict(NEW.CAMERA, jerk_baseline=0.333333))
    for k, lab in (("far_out", "far_player_out_of_frame"), ("any_out", "any_player_out_of_frame"), ("sliced", "player_sliced_by_crop_edge")):
        stats["pct_frames_" + lab] = round(100 * stats[k + "_frames"] / F, 1)
    stats.update(NEW.framing_checks(clips, stats["pct_frames_far_player_out_of_frame"], NEW.CAMERA))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["package", "harness"]); ap.add_argument("input")
    ap.add_argument("--camera", default=None, help="git ref of an older camera.py (before/after)")
    ap.add_argument("--out", default=None)
    A = ap.parse_args(argv)
    cam = load_camera(A.camera)
    st = run_package(A.input, cam, A.out) if A.mode == "package" else run_harness(A.input, cam, A.out)
    st["camera_ref"] = A.camera or "working tree"
    if A.out: json.dump(st, open(A.out, "w"), indent=1, default=str)
    return st


if __name__ == "__main__":
    main()
