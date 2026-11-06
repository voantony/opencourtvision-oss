"""4-player ID tracker: detect -> vision -> eval on a local doubles match (research measurement tool).

  detect  RF-DETR Nano (384, pretrained COCO) on the full frame + 4K far-court tiles at 5 fps from 1 s before each
          labelled point's serve contact to its end; court mask; crops of in-zone people; OSNet embeddings; colour.
  vision  Apple Vision person instance masks + body pose on every crop (tools/player_id/vision_crops.swift).
  eval    rules truth + motion-chain truth, hand + learned cues, tracker variants, per-cue ablation, cascade sweep,
          2-tap vs 4-tap labelling, deduction failures, check-set contact sheets, report.{json,md}.

Everything it writes stays under --out (frames, crops, labels-derived truth and models trained on this
match are never committed). Run with nice 20 (tools/player_id/run_pid.sh).

Research only, never shipped: unlike the app engine (opencourtvision.players, in memory per match, Ops V6) this tool caches
OSNet / colour features under --out so the eval can be re-run. It refuses to run in beta mode.
Pass --video, --prelabels and --court (or PID_VIDEO, PID_PRELABELS, PID_COURT) for the local match files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

if os.environ.get("VAMOS_BETA", "").strip().lower() in ("1", "true", "yes", "on"):
    sys.exit("pid_run.py is a research tool that caches appearance features; it never runs in a beta build (Ops V6)")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from opencourtvision.players import geometry as G  # noqa: E402
from opencourtvision.players.cues import COLOUR_DIM  # noqa: E402

HOME = os.path.expanduser("~")
FPS, PRE_S = 2.5, 1.2                                   # serve contact = frame round(PRE_S * FPS) = 3
SERVE_I = int(round(PRE_S * FPS))

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["detect", "vision", "eval", "all"])
ap.add_argument("--video", default=os.environ.get("PID_VIDEO", ""), help="path to the match video")
ap.add_argument("--labels", default=os.environ.get("PID_LABELS", ""), help="path to the point labels JSON")
ap.add_argument("--prelabels", default=os.environ.get("PID_PRELABELS", ""), help="path to prelabels.json")
ap.add_argument("--court", default=os.environ.get("PID_COURT", ""), help="path to a manual court/homography JSON")
ap.add_argument("--out", default=os.environ.get("PID_OUT", os.path.join(os.getcwd(), "pid_out")))
ap.add_argument("--ffmpeg", default=os.environ.get("PID_FFMPEG", "ffmpeg"))
ap.add_argument("--rfw", default=os.environ.get("PID_RFDETR", os.path.join(HOME, "models", "rfdetr")))
ap.add_argument("--limit", type=int, default=0, help="first N points only (smoke test)")
a = ap.parse_args()
VID = a.video
LAB = a.labels or (VID[:-4] + ".labels.json" if VID else "")
PRE = a.prelabels
OVR = a.court
if a.cmd in ("detect", "eval", "all") and not (VID and PRE and OVR):
    sys.exit("pid_run.py needs --video, --prelabels and --court (or PID_VIDEO, PID_PRELABELS, PID_COURT)")
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "progress.log"), "a")


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, file=LOG, flush=True); print(time.strftime("%H:%M:%S"), *x, flush=True)


def views():
    PL = json.load(open(PRE)); cbv = [dict(c) for c in PL["courts_by_view"]]; ov = json.load(open(OVR))
    for o in (ov if isinstance(ov, list) else [ov]):
        cbv[o["view"]].update(H_img2court=o["H_img2court"], detected=True, manual=True)
    return PL, cbv


def view_at(cbv, t):
    v = [c for c in cbv if c["t0"] <= t < c["t1"]] or cbv[-1:]
    return v[0]


def points():
    L = json.load(open(LAB)); PL, _ = views()
    hits = sorted(c["t"] for r in PL["rallies"] for c in r["contacts"])
    P = [p for p in L["points"] if p.get("kind") == "point" and p.get("start") is not None]
    for p in P:
        h = [x for x in hits if p["start"] <= x <= p["end"]]
        p["t_serve"] = h[0] if h else None
    P = [p for p in P if p["t_serve"] is not None]
    return P[: a.limit] if a.limit else P, L


# ------------------------------------------------------------------------------------------------------------ detect
def cmd_detect():
    import cv2
    import rfdetr
    import torch
    from PIL import Image
    from rfdetr.util.coco_classes import COCO_CLASSES

    from opencourtvision.players.cues import colour_hist
    from opencourtvision.players.reid import OSNetEmbedder
    inv = {v: k for k, v in COCO_CLASSES.items()}; PERSON = inv["person"]
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    cwd = os.getcwd(); os.chdir(a.rfw)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = rfdetr.RFDETRNano(resolution=384, device=dev)
    try:
        m.predict(Image.new("RGB", (640, 360)), threshold=0.5)
    except Exception as e:
        log("MPS failed, CPU:", repr(e)[:160]); dev = "cpu"; m = rfdetr.RFDETRNano(resolution=384, device="cpu")
    os.chdir(cwd)
    emb = OSNetEmbedder(device=dev); P, _ = points(); _, cbv = views(); SW, SH = 3840, 2160; fs = SW * SH * 3

    def run(pils, offs, sc):
        out = m.predict(pils if len(pils) > 1 else pils[0], threshold=0.25); out = out if isinstance(out, list) else [out]; dets = []
        for det, (ox, oy), s in zip(out, offs, sc):
            for b, c, k in zip(det.xyxy, det.confidence, det.class_id):
                if int(k) == PERSON:
                    dets.append([float(b[0] * s + ox), float(b[1] * s + oy), float(b[2] * s + ox), float(b[3] * s + oy), float(c)])
        return dets
    T = time.time()
    for p in P:
        dst = os.path.join(a.out, "det", f"p{p['idx']:03d}.npz")
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True); cdir = os.path.join(a.out, "crops", f"p{p['idx']:03d}"); os.makedirs(cdir, exist_ok=True)
        t0 = max(0.0, p["t_serve"] - PRE_S); dur = p["end"] - t0 + 0.2
        c = view_at(cbv, p["t_serve"]); H = np.asarray(c["H_img2court"], float); Hi = np.linalg.inv(H); roi = G.far_roi(H, SW, SH)
        ff = subprocess.Popen([a.ffmpeg, "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{t0:.3f}", "-i", VID, "-t", f"{dur:.3f}",
                               "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        rows = []; ms_det = []; ms_os = []; ms_cb = []; nfr = 0; nout = []
        while True:
            raw = ff.stdout.read(fs)
            if len(raw) < fs:
                break
            img = np.frombuffer(raw, np.uint8).reshape(SH, SW, 3); t = t0 + nfr / FPS
            t1 = time.perf_counter()
            small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            dets = run([Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))], [(0, 0)], [SW / 640])
            if roi is not None:
                TL = G.tiles(roi)
                far = G.nms(run([Image.fromarray(cv2.cvtColor(img[y:y + s, x:x + s], cv2.COLOR_BGR2RGB)) for x, y, s, _ in TL], [(x, y) for x, y, _, _ in TL], [1.0] * len(TL)))
                dets = G.merge_far(dets, far, roi)
            dets = G.nms(dets, 0.6, contain=0.8); ms_det.append(1000 * (time.perf_counter() - t1))
            if dets:
                xy = G.to_court(H, G.feet([d[:4] for d in dets])); zin = G.in_zone(xy)
            else:
                xy = np.zeros((0, 2)); zin = np.zeros(0, bool)
            nout.append(int((~zin).sum())); crops = []; keep = []
            for j, (d, q, z) in enumerate(zip(dets, xy, zin)):
                if not z:
                    continue
                bw, bh = d[2] - d[0], d[3] - d[1]; px, py = 0.08 * bw, 0.04 * bh
                X0, Y0, X1, Y1 = int(max(0, d[0] - px)), int(max(0, d[1] - py)), int(min(SW, d[2] + px)), int(min(SH, d[3] + py))
                cr = img[Y0:Y1, X0:X1]
                if cr.shape[0] < 16 or cr.shape[1] < 6:
                    continue
                if cr.shape[0] > 256:
                    cr = cv2.resize(cr, (max(6, int(cr.shape[1] * 256 / cr.shape[0])), 256), interpolation=cv2.INTER_AREA)
                a1 = Hi @ [q[0], q[1], 1.0]; a2 = Hi @ [q[0] + 1.0, q[1], 1.0]; ppm = float(np.linalg.norm(a1[:2] / a1[2] - a2[:2] / a2[2]))
                cp = os.path.join(cdir, f"f{nfr:03d}_{j}.jpg"); cv2.imwrite(cp, cr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                tc = time.perf_counter(); cb = colour_hist(cr); ms_cb.append(1000 * (time.perf_counter() - tc))
                crops.append(cr); keep.append((d, q, ppm, cp, cb))
            if crops:
                to = time.perf_counter(); E = emb(crops); ms_os.append(1000 * (time.perf_counter() - to) / len(crops))
                for (d, q, ppm, cp, cb), e in zip(keep, E):
                    rows.append(dict(f=nfr, t=t, box=d[:4], conf=d[4], xy=[float(q[0]), float(q[1])], h_ratio=(d[3] - d[1]) / max(1.0, 1.75 * ppm),
                                     crop=os.path.relpath(cp, a.out), osnet=e.astype(np.float16), colour_box=cb.astype(np.float16)))
            nfr += 1
        ff.wait()
        np.savez_compressed(dst + ".tmp.npz", meta=json.dumps(dict(idx=p["idx"], t0=t0, n=nfr, view=c["view"], roi=roi,
                                                                    rows=[{k: v for k, v in r.items() if k not in ("osnet", "colour_box")} for r in rows])),
                            osnet=np.array([r["osnet"] for r in rows], np.float16).reshape(-1, 512), colour_box=np.array([r["colour_box"] for r in rows], np.float16).reshape(len(rows), COLOUR_DIM),
                            ms_det=np.array(ms_det), ms_osnet=np.array(ms_os), ms_colour_box=np.array(ms_cb), n_out_zone=np.array(nout))
        os.replace(dst + ".tmp.npz", dst)
        log(f"p{p['idx']} frames={nfr} dets/frame={len(rows) / max(1, nfr):.1f} det={np.mean(ms_det):.0f}ms osnet={np.mean(ms_os) if ms_os else 0:.1f}ms/crop total={time.time() - T:.0f}s")
    log("detect done")


# ------------------------------------------------------------------------------------------------------------ vision
def cmd_vision():
    exe = os.path.join(a.out, "vision_crops")
    if not os.path.exists(exe):
        subprocess.run(["xcrun", "swiftc", "-O", os.path.join(HERE, "vision_crops.swift"), "-o", exe], check=True)
    for d in sorted(glob.glob(os.path.join(a.out, "crops", "p*"))):
        dst = d + ".vision.jsonl"
        if os.path.exists(dst):
            continue
        L = sorted(glob.glob(os.path.join(d, "*.jpg"))); lf = d + ".list.txt"; open(lf, "w").write("\n".join(L) + "\n")
        t = time.time(); subprocess.run([exe, lf, dst + ".tmp"], check=True); os.replace(dst + ".tmp", dst)
        log(f"vision {os.path.basename(d)} {len(L)} crops {1000 * (time.time() - t) / max(1, len(L)):.1f} ms/crop")
    log("vision done")


if a.cmd in ("detect", "all"):
    cmd_detect()
if a.cmd in ("vision", "all"):
    cmd_vision()
if a.cmd in ("eval", "all"):
    from pid_eval import run_eval
    run_eval(a.out, LAB, points, log)
