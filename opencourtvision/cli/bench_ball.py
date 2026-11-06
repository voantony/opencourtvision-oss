"""ocv-bench-ball: benchmark ball trackers on real footage. Every number it prints is a PROXY unless --labels is given.

Proxy metrics (no ball ground truth needed):
  sv_rate     fraction of SwingVision shot times (Video Time + 2.00 s lag, mapped onto the clip clock, see
              ball/proxy.align_clips) with a ball detection within +-150 ms,
  audio_rate  the same for our audio hits,
  false_rate  fraction of a variant's tracklets (>= 5 frames) that are static, slow/rolling, or off our court,
  coverage    fraction of frames in the 600 ms after each audio hit (ball just struck, in flight) with a detection,
  fps         frames per second of tracking on this machine (decode excluded), plus the ROI pixel fraction.
With --labels labels/ball/*.csv (ocv-label-ball; rows with status 'ok' only) it also reports precision / recall / F1
at --tol-px against those labels: those are real numbers, not proxies.

Variants: engine (v2: ROI + TrackNetV3 re-acquire + size gate + live filter), engine_raw (v2 before the live filter),
engine_hsv (v2 re-acquiring with full-frame HSV, no TrackNet), reel_hsv (the old reel.ball detect+track),
tracknet3_full (TrackNetV3 on every frame, top blob). External baselines come in as CSVs (t,x,y on the proxy clock,
in the bench frame size) via --baseline NAME=DIR, DIR holding clip_<k>.csv; detections inside the overlay masks are
dropped for every variant alike.

Example:
  ocv-bench-ball --video sv_proxy.mp4 --sv shots.xlsx --clips clips.json --calib court_calib.json --out work/bench \\
      --max-clips 6 --baseline yastrebksv=work/base/yastrebksv
calib JSON: {"court_m": [[x, y], ...], "image_px": [[u, v], ...], "ref_time": s} (>= 4 pairs, OpenCourtVision court
metres, see pose/court.py, for the frame size used here). With ref_time (the frame the points were picked on) each clip
gets the court carried over by `ball.geometry.transfer_homography` (exports that re-frame per point, bumped camera);
a clip where that fails runs without court/size gates (reported as court: null).
"""
import argparse
import json
import os
import time


def _variant_runs(names, frames, a, fps, gate, hits, overlays, tn):
    import numpy as np

    from ..ball.hsv import overlay_mask
    from ..ball.tracker import BallEngine
    from ..reel.ball import detect, track
    out = {}
    h, w = frames[0].shape[:2]
    if {"engine", "engine_raw"} & set(names):
        t = time.perf_counter(); r = BallEngine(gate, tn, overlays, fps, a, hits).run(frames); dt = time.perf_counter() - t
        out["engine"] = (r["pos"], dt, r["stats"]); out["engine_raw"] = (r["raw"], dt, r["stats"])
    if "engine_hsv" in names:
        t = time.perf_counter(); r = BallEngine(gate, None, overlays, fps, a, hits, reacquire="hsv_full").run(frames)
        out["engine_hsv"] = (r["pos"], time.perf_counter() - t, r["stats"])
    if "reel_hsv" in names:
        t = time.perf_counter(); c, *_ = detect(frames, a, fps); p = track(c, w, h)
        out["reel_hsv"] = (p, time.perf_counter() - t, {})
    if "tracknet3_full" in names and tn is not None:
        t = time.perf_counter(); p = np.full((len(frames), 2), np.nan)
        for i in range(0, len(frames), 8):
            for k, bl in enumerate(tn.detect(frames[i:i + 8])):
                if bl: p[i + k] = bl[0][:2]
        out["tracknet3_full"] = (p, time.perf_counter() - t, {})
    m = overlay_mask(h, w, overlays) if overlays else None
    for k, (p, dt, st) in out.items():
        if m is not None:
            ok = ~np.isnan(p[:, 0]); xi = np.clip(np.nan_to_num(p[:, 0]).astype(int), 0, w - 1); yi = np.clip(np.nan_to_num(p[:, 1]).astype(int), 0, h - 1)
            p = p.copy(); p[ok & (m[yi, xi] == 0)] = np.nan
        out[k] = (p, dt, st)
    return {k: v for k, v in out.items() if k in names}


def label_metrics(pos, labels, tol_px):
    """labels: dict frame -> (x, y) or None (ball not visible). -> dict(tp, fp, fn, precision, recall, f1)."""
    import numpy as np
    tp = fp = fn = 0
    for i, g in labels.items():
        if not 0 <= i < len(pos): continue
        p = pos[i]; has = not np.isnan(p[0])
        if g is None:
            fp += has
        elif not has:
            fn += 1
        elif np.hypot(p[0] - g[0], p[1] - g[1]) <= tol_px:
            tp += 1
        else:
            fp += 1; fn += 1
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=round(pr, 3), recall=round(rc, 3), f1=round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0)


def format_table(summary):
    rows = ["| variant (PROXY) | SV shots detected +-150 ms | audio hits detected +-150 ms | ball in 0-600 ms after hits | false-track rate | fps (box CPU) | ROI px frac |",
            "|---|---|---|---|---|---|---|"]
    for k, s in summary.items():
        rows.append(f"| {k} | {s['sv_hit']}/{s['sv_tot']} = {s['sv_rate']:.0%} | {s['au_hit']}/{s['au_tot']} = {s['audio_rate']:.0%} | "
                    f"{s['coverage']:.0%} | {s['false']}/{s['tracks']} = {s['false_rate']:.0%} | {s['fps']:.1f} | {s.get('roi_pixel_frac', '')} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-bench-ball", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--sv", help="SwingVision export .xlsx (shot times)")
    ap.add_argument("--clips", help="clips.json: [[start, end], ...] proxy seconds of the export's point clips")
    ap.add_argument("--segments", nargs="*", type=float, help="explicit start end pairs (s) instead of --clips")
    ap.add_argument("--calib", help="court calibration JSON for the court gate / size gate")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--overlays", default="swingvision960", help="'swingvision960', 'none' or x0,y0,x1,y1;...")
    ap.add_argument("--variants", default="engine,engine_raw,engine_hsv,reel_hsv,tracknet3_full")
    ap.add_argument("--baseline", action="append", default=[], help="NAME=DIR with clip_<k>.csv (t,x,y)")
    ap.add_argument("--max-clips", type=int, default=6)
    ap.add_argument("--clip-ids", help="comma list of clip indices to use (overrides the automatic pick)")
    ap.add_argument("--min-matches", type=int, default=3, help="min SV-contact/audio matches for a clip's alignment to count")
    ap.add_argument("--labels", nargs="*", default=[], help="labels/ball/*.csv for real (non-proxy) metrics")
    ap.add_argument("--tol-px", type=float, default=6.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    A = ap.parse_args(argv)
    import numpy as np

    from ..ball import proxy
    from ..ball.geometry import CourtGate
    from ..ball.hsv import SWINGVISION_OVERLAYS_960
    from ..detection.audio import detect_hits, load_audio, loudness
    from ..reel.ball import decode
    os.makedirs(A.out, exist_ok=True)
    W, H = (int(v) for v in A.size.lower().split("x"))
    overlays = SWINGVISION_OVERLAYS_960 if A.overlays == "swingvision960" else [] if A.overlays == "none" else \
        [tuple(int(v) for v in b.split(",")) for b in A.overlays.split(";")]
    gate0, ref = None, None
    if A.calib:
        c = json.load(open(A.calib)); gate0 = CourtGate.from_points(c["court_m"], c["image_px"], image_size=(W, H))
        if c.get("ref_time") is not None:   # the calibration frame: other clips get the court via transfer_homography
            ref = list(decode(A.video, float(c["ref_time"]), float(c["ref_time"]) + 1.5 / A.fps, W, H, A.fps))[0]
    sr, x = load_audio(A.video); le, bg, _ = loudness(sr, x); hits, _ = detect_hits(le, bg); hits = np.asarray(hits, float)
    if A.clips:
        clips = json.load(open(A.clips))
    else:
        s = A.segments or []; clips = list(zip(s[0::2], s[1::2]))
    al = [dict(clip=k, start=float(a), end=float(b), point=None, offset=None, matches=0, n_sv=0) for k, (a, b) in enumerate(clips)]
    pts = None
    if A.sv and A.clips:
        from ..importers import swingvision as sv
        from ..importers.swingvision import SV_SHOT_LAG_S
        m = sv.load(A.sv)
        pts = [[s["video_time"] + SV_SHOT_LAG_S for s in m.shots if s["point"] == p["point"] and s.get("video_time") is not None] for p in m.points]
        al = proxy.align_clips(clips, pts, hits)
        json.dump(al, open(os.path.join(A.out, "alignment.json"), "w"), indent=1)
    if A.clip_ids:
        use = [al[int(k)] for k in A.clip_ids.split(",")]
    else:
        good = [a for a in al if a["offset"] is not None and a["matches"] >= A.min_matches and a["matches"] >= 0.6 * max(a["n_sv"], 1)]
        use = good[:: max(1, len(good) // A.max_clips)][:A.max_clips] if good else al[:A.max_clips]
    names = [v for v in A.variants.split(",") if v]
    tn = None
    if any(v in names for v in ("engine", "engine_raw", "tracknet3_full")):
        from ..ball.tracknet3 import TrackNetV3
        tn = TrackNetV3(device=A.device, threads=A.threads)
    labels = {}
    if A.labels:
        from .label_ball import ground_truth, read
        for f in A.labels:
            for (clip, t), g in ground_truth(read(f)).items(): labels.setdefault(str(clip), {})[t] = g
    agg, per_clip = {}, []
    for a in use:
        k, t0, t1 = a["clip"], a["start"], a["end"]
        frames = list(decode(A.video, t0, t1, W, H, A.fps))
        if len(frames) < 10: continue
        gate, gate_info = gate0, None
        if ref is not None:
            from ..ball.geometry import transfer_homography
            from ..ball.hsv import overlay_mask
            Hk, gate_info = transfer_homography(ref, frames[len(frames) // 2], gate0.H, overlay_mask(H, W, overlays) if overlays else None)
            gate = CourtGate(Hk, image_size=(W, H)) if Hk is not None else None
        if tn is not None: tn.set_background(frames[:: max(1, len(frames) // 24)])
        svt = proxy.sv_times_in_clip(a, [t for p in pts for t in p]) if pts is not None and a["offset"] is not None else np.zeros(0)
        aut = hits[(hits >= t0) & (hits <= t1)]
        runs = _variant_runs(names, frames, t0, A.fps, gate, aut, overlays, tn)
        for spec in A.baseline:
            nm, d = spec.split("=", 1); f = os.path.join(d, f"clip_{k}.csv")
            if os.path.exists(f):
                tm = json.load(open(os.path.join(d, "timing.json"))).get(str(k), 0.0) if os.path.exists(os.path.join(d, "timing.json")) else 0.0
                runs[nm] = (proxy.read_track_csv(f, t0, A.fps, len(frames)), tm, {})
        rec = dict(clip=k, start=t0, end=t1, frames=len(frames), n_sv=len(svt), n_audio=len(aut), court=gate_info, variants={})
        for nm, (p, dt, st) in runs.items():
            np.savetxt(os.path.join(A.out, f"{nm}_clip{k}.csv"), np.c_[t0 + np.arange(len(p)) / A.fps, p], delimiter=",", header="t,x,y", comments="", fmt="%.4f")
            sh, stot = proxy.event_detection_rate(p, t0, A.fps, svt); ah, atot = proxy.event_detection_rate(p, t0, A.fps, aut)
            fs = proxy.false_track_stats(p, A.fps, gate)
            ch, ctot = proxy.post_hit_coverage(p, t0, A.fps, aut)
            v = dict(sv_hit=sh, sv_tot=stot, au_hit=ah, au_tot=atot, cov_hit=ch, cov_tot=ctot, seconds=round(dt, 2), **fs)
            if st.get("roi_pixel_frac") is not None: v["roi_pixel_frac"] = st["roi_pixel_frac"]; v["tn_frame_frac"] = st.get("tn_frame_frac")
            if nm == "engine" and "rejected" in st: v["live_filter"] = dict(live=st["tracklets_live"], rejected=st["rejected"])
            lab = labels.get(str(k))
            if lab:
                v["labels"] = label_metrics(p, {int(round((t - t0) * A.fps)): g for t, g in lab.items()}, A.tol_px)
            rec["variants"][nm] = v
            g = agg.setdefault(nm, dict(sv_hit=0, sv_tot=0, au_hit=0, au_tot=0, cov_hit=0, cov_tot=0, false=0, tracks=0, frames=0, seconds=0.0, roi=[]))
            for key in ("sv_hit", "sv_tot", "au_hit", "au_tot", "cov_hit", "cov_tot", "false", "tracks"): g[key] += v[key]
            g["frames"] += len(frames); g["seconds"] += dt
            if "roi_pixel_frac" in v: g["roi"].append(v["roi_pixel_frac"])
        per_clip.append(rec)
        print(f"clip {k} [{t0:.1f}, {t1:.1f}] s: " + ", ".join(f"{n} sv {v['sv_hit']}/{v['sv_tot']} au {v['au_hit']}/{v['au_tot']} false {v['false']}/{v['tracks']}"
                                                         for n, v in rec["variants"].items()), flush=True)
    summary = {}
    for nm, g in agg.items():
        summary[nm] = dict(sv_hit=g["sv_hit"], sv_tot=g["sv_tot"], sv_rate=g["sv_hit"] / max(g["sv_tot"], 1), au_hit=g["au_hit"], au_tot=g["au_tot"],
                           audio_rate=g["au_hit"] / max(g["au_tot"], 1), coverage=g["cov_hit"] / max(g["cov_tot"], 1), false=g["false"], tracks=g["tracks"], false_rate=g["false"] / max(g["tracks"], 1),
                           fps=g["frames"] / g["seconds"] if g["seconds"] else float("nan"), frames=g["frames"])
        if g["roi"]: summary[nm]["roi_pixel_frac"] = round(float(np.mean(g["roi"])), 4)
    rep = dict(proxy=True, video=os.path.basename(A.video), size=[W, H], fps=A.fps, clips=[a["clip"] for a in use], summary=summary, per_clip=per_clip,
               note="PROXY metrics: event-time agreement and track shape, not labelled ball positions.")
    json.dump(rep, open(os.path.join(A.out, "bench.json"), "w"), indent=1, default=float)
    md = "# ocv-bench-ball (PROXY metrics)\n\n" + format_table(summary) + "\n"
    open(os.path.join(A.out, "bench.md"), "w").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
