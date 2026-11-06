"""Phase-1 data for the own ball tracker: frames + physics-filtered pseudo-labels (no new hand labelling).

Sources
  dev_match match (Antony's, private): the same per-point windows as the point-winner S0 v2 tracks (one 7.5 s window per
      point around our last audio contact, 30 fps). Points 1-50 give training pseudo-labels; points 51-101 are the
      held-out set: frames + baselines only, their pseudo-labels are never used for training.
  CC BY footage (tennis, padel, pickleball; footage_sources.md): K evenly spaced windows inside each usable,
      fixed-camera video's play range. 1 video in 10 (by id hash) is validation.
Teacher: engine-v2 BallEngine with HSV re-acquire (our own code; no TrackNet, no third-party weights), its pre-live
  filter positions, then our physics filter (arcs, outliers out, gaps inside arcs filled) and, where a court
  homography exists, the live-ball check per arc (not static, not slow on the ground plane, on court). Only frames on
  accepted arcs become labels.
Output (under --out, private, never committed): frames/<clip>/<i>.jpg at 768x432, clips/<clip>.npz, index.json.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from opencourtvision.ball.geometry import CourtGate  # noqa: E402
from opencourtvision.ball.own import physics as P  # noqa: E402
from opencourtvision.ball.tracker import BallEngine  # noqa: E402


def save_frames(frames, d):
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), cv2.resize(f, C.NET, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


def teacher(frames, t0, hits, Hc2i):
    gate = CourtGate(Hc2i, image_size=(C.AW, C.AH)) if Hc2i is not None else None
    t = time.perf_counter()
    r = BallEngine(gate, None, (), C.AFPS, t0, hits, reacquire="hsv_full").run(frames)
    return np.asarray(r["raw"], float), np.asarray(r["pos"], float), time.perf_counter() - t


def reel_baseline(frames):
    from opencourtvision.reel.ball import detect, track
    t = time.perf_counter()
    c, *_ = detect(frames, 0.0, C.AFPS)
    return np.asarray(track(c, C.AW, C.AH), float), time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video17", required=True); ap.add_argument("--labels17", required=True)
    ap.add_argument("--tracks17", required=True, help="pw_tracks S0 v2 npz dir (windows, player boxes, TrackNet S0 ball)")
    ap.add_argument("--prelabels17", required=True)
    ap.add_argument("--cc_root", default=os.path.expanduser("~/VamosFootage"))
    ap.add_argument("--cc_windows", type=int, default=8); ap.add_argument("--cc_win_s", type=float, default=6.0)
    ap.add_argument("--cc_max_videos", type=int, default=0)
    ap.add_argument("--heldout_from", type=int, default=51, help="dev_match point_no >= this is held out")
    ap.add_argument("--limit17", type=int, default=0, help="smoke test: at most N dev_match points per split")
    ap.add_argument("--court_override", help="manual court-setup JSON (e.g. tests/fixtures/court_dev_match_view0_manual.json); "
                    "the stored dev_match view-0 court is oversized, so pass this when building the dev_match clips")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    idx_path = os.path.join(a.out, "index.json")
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else {"clips": {}}

    def done(name):
        return name in index["clips"] and os.path.exists(os.path.join(a.out, "clips", name + ".npz"))

    def record(name, meta, arrays):
        np.savez_compressed(os.path.join(a.out, "clips", name + ".npz"), **arrays)
        index["clips"][name] = meta
        C.save_json(index, idx_path)

    # ---------------- dev_match
    L = json.load(open(a.labels17))["points"]
    PL = json.load(open(a.prelabels17))
    PL["courts_by_view"] = C.override_courts(PL["courts_by_view"], a.court_override)
    SW = int(PL.get("width", 3840))
    hits_all = sorted(c["t"] for r in PL["rallies"] for c in r["contacts"])
    taken = {"train": 0, "heldout": 0}
    for p in L:
        if p.get("kind") != "point" or p.get("winner") not in ("A", "B"):
            continue
        f = os.path.join(a.tracks17, f"seg{p['idx']:03d}.npz")
        name = f"dev_match_seg{p['idx']:03d}"
        sp = "heldout" if p["point_no"] >= a.heldout_from else "train"
        if a.limit17 and taken[sp] >= a.limit17:
            continue
        taken[sp] += 1
        if not os.path.exists(f) or done(name):
            continue
        d = np.load(f)
        t0, n = float(d["t0"]), len(d["ball"])
        frames = C.decode(a.video17, t0, n)
        if len(frames) < n * 0.9:
            print("short decode", name, len(frames), n, flush=True); continue
        n = len(frames)
        v = C.view_at(PL["courts_by_view"], t0 + n / C.AFPS / 2)
        Hc2i = C.h_court2img(v["H_img2court"], SW) if v else None
        hits = [h for h in hits_all if t0 - 0.1 <= h <= t0 + n / C.AFPS + 0.1]
        raw, live, dt_teach = teacher(frames, t0, hits, Hc2i)
        y_net = C.project(Hc2i, (C.COURT_W / 2, C.NET_Y))[1] if Hc2i is not None else None
        gate = CourtGate(Hc2i, image_size=(C.AW, C.AH)) if Hc2i is not None else None
        ph = P.filter_track(raw, C.AFPS, boxes_at=C.boxes_fn(d, C.AW / 640.0), y_net=y_net, frame_w=C.AW,
                            arc_ok=C.live_arc_check(gate))
        held = p["point_no"] >= a.heldout_from
        arrays = dict(t0=t0, fps=C.AFPS, teacher_raw=raw, teacher_live=live, phys_pos=ph["pos"], phys_w=ph["conf"],
                      Hc2i=Hc2i if Hc2i is not None else np.full((3, 3), np.nan), hits=np.asarray(hits, float))
        meta = dict(source="dev_match", sport="tennis", split="heldout" if held else "train", point_no=p["point_no"], idx=p["idx"],
                    n=n, labels=int((ph["conf"] > 0).sum()) if not held else 0, teacher_s=round(dt_teach, 2), **ph["stats"])
        if held:
            rb, dt_reel = reel_baseline(frames)
            arrays["reel"] = rb; meta["reel_s"] = round(dt_reel, 2)
        save_frames(frames, os.path.join(a.out, "frames", name))
        record(name, meta, arrays)
        print(name, meta["split"], "labels", meta["labels"], "det", ph["stats"]["detections"], flush=True)

    # ---------------- CC BY footage
    qa = list(csv.DictReader(open(os.path.join(a.cc_root, "prelabels", "qa_summary.csv"))))
    vids = [r for r in qa if r["usable"] == "True" and r["fixed_camera"] == "True" and r["overlay_suspect"] != "True"
            and r["vertical"] != "True"]
    if a.cc_max_videos:
        vids = vids[:a.cc_max_videos]
    for r in vids:
        sport, vid = r["sport"], r["video_id"]
        path = (glob.glob(os.path.join(a.cc_root, sport, vid + ".*")) or [None])[0]
        pre = os.path.join(a.cc_root, "prelabels", sport, vid)
        if path is None or not os.path.isdir(pre):
            continue
        trim = json.load(open(os.path.join(pre, "trim.json"))) if os.path.exists(os.path.join(pre, "trim.json")) else {}
        pj = json.load(open(os.path.join(pre, "prelabels.json")))
        dur = float(r["duration_s"] or 0)
        s0, s1 = float(trim.get("play_start_s", 0.0)), float(trim.get("play_end_s", dur) or dur)
        if s1 - s0 < a.cc_win_s * 2:
            continue
        W = int(r["resolution"].split("x")[0])
        court = pj.get("court") or {}
        Hc2i = None
        if court.get("H_img2court") and (court.get("detected") or court.get("line_support", 0) >= 0.5) and sport == "tennis":
            Hc2i = C.h_court2img(court["H_img2court"], W)        # the court gate's dimensions are tennis
        split = "val" if zlib.crc32(vid.encode()) % 10 == 0 else "train"
        n = int(a.cc_win_s * C.AFPS)
        for k in range(a.cc_windows):
            name = f"cc_{sport}_{vid}_w{k}"
            if done(name):
                continue
            t0 = s0 + (k + 0.5) * (s1 - s0) / a.cc_windows
            frames = C.decode(path, t0, n)
            if len(frames) < n * 0.9:
                continue
            raw, live, dt_teach = teacher(frames, t0, (), Hc2i)
            gate = CourtGate(Hc2i, image_size=(C.AW, C.AH)) if Hc2i is not None else None
            ph = P.filter_track(raw, C.AFPS, frame_w=C.AW, arc_ok=C.live_arc_check(gate))
            meta = dict(source="cc", sport=sport, split=split, video_id=vid, n=len(frames), labels=int((ph["conf"] > 0).sum()),
                        teacher_s=round(dt_teach, 2), **ph["stats"])
            save_frames(frames, os.path.join(a.out, "frames", name))
            record(name, meta, dict(t0=t0, fps=C.AFPS, teacher_raw=raw, teacher_live=live, phys_pos=ph["pos"],
                                    phys_w=ph["conf"], Hc2i=Hc2i if Hc2i is not None else np.full((3, 3), np.nan)))
            print(name, split, "labels", meta["labels"], flush=True)

    # ---------------- counts
    cnt = {}
    for name, m in index["clips"].items():
        key = f"{m['source']}:{m['sport']}:{m['split']}"
        c = cnt.setdefault(key, dict(clips=0, frames=0, labels=0, detections=0, inliers=0, filled=0))
        c["clips"] += 1; c["frames"] += m["n"]; c["labels"] += m["labels"]
        for k in ("detections", "inliers", "filled"):
            c[k] += m.get(k, 0)
    index["counts"] = cnt
    C.save_json(index, idx_path)
    print(json.dumps(cnt, indent=1))


if __name__ == "__main__":
    main()
