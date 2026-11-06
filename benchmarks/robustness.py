"""Perturbation benchmark: is the edit robust where SwingVision breaks (moving / bumped camera, court or players partly
out of frame)? Eng notes §2.8.4b (robustness block) and §2.9.x.

Perturbations are geometric and applied on the analysis side (robust/perturb.py): the cached proxy frames, the regional
motion-energy grid, and (when present) tracks and the court fit are moved / cropped. Nothing is re-encoded and audio is
untouched, so US boundaries (audio only) must not move; the harness checks that.

Per perturbation and build (US: audio boundaries + motion keep/drop gate; EU: motion admission + audio boundaries), with
camera-motion handling off ("naive") and on ("handled": robust/camera_motion.py + robust/framing.py):
recall, non-points, boundary error vs labels (median |start|, |end| error, s), boundary change vs the clean run
(max |dA|, |dB| over clips kept in both, s; must be 0 in the US), and edit taps per 100 points
(= missed points + non-point clips to delete, per 100 labelled points; score taps are unchanged: no vision winner yet).

  python benchmarks/robustness.py --synthetic --out /tmp/rob                              # CI / no private data
  python benchmarks/robustness.py --labels L.json --features features.npz --motion motion.npz --hits rallies.json \\
      --sv-xlsx shots.xlsx --start 879.45 --final "6-1 6-2" --out /tmp/rob                  # a real match (private)
  python benchmarks/robustness.py --tracks dev_match_tracks/seg --motion motion.npz --start 879.45 --out /tmp/rob_tracks
      # EU track checks only (no labels needed): ball / player-feet tracks (pw_tracks seg*.npz) under each perturbation,
      # naive vs handled (offset undone with the tracked camera fit): observable share, re-alignment error, reliable share.
      # With --labels etc. as well, both blocks run.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from benchmarks import scoreboard as SB  # noqa: E402
from opencourtvision.editloop import segment as SG  # noqa: E402
from opencourtvision.editloop.motion import MotionCues, admit_candidates, live_gate  # noqa: E402
from opencourtvision.editloop.smart import smart_points  # noqa: E402
from opencourtvision.robust import camera_motion as CM  # noqa: E402
from opencourtvision.robust import framing as FR  # noqa: E402
from opencourtvision.robust import occluders as OC  # noqa: E402
from opencourtvision.robust.perturb import perturb_energy, perturb_frames, suite  # noqa: E402

# Court in the clean proxy frame (fractions of width / height), hand-placed on the dev_match median thumbnail: far baseline
# ~0.40 of the height, near baseline below the frame (already partly out of view in the clean video). Used only for the
# visible-court share under crops; the product computes it from the court fit (framing.court_visible_fraction).
COURT_POLY_DEV = [(0.30, 0.40), (0.86, 0.40), (1.35, 1.25), (-0.35, 1.25)]


def poly_visible_share(poly, pert, shape=(108, 192), occ=None):
    from opencourtvision.ball.geometry import point_in_poly
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h], 1)
    inside = point_in_poly(pts, np.asarray(poly, float)).reshape(h, w)
    vm = pert.visible_mask(shape)
    if occ is not None: vm = vm & ~occ
    return float((inside & vm).sum() / max(1, inside.sum()))


def _iou(a, b):
    u = (a | b).sum()
    return round(float((a & b).sum() / u), 3) if u else None


def build_cues(E, t, thumbs, thumb_t, pert, handled, poly, start=0.0):
    bgimg = np.median(thumbs[:: max(1, len(thumbs) // 200)], 0).astype(np.float32)
    Ep = perturb_energy(E, t, pert, bgimg)
    if not handled:
        return MotionCues(Ep, t), dict(mode="naive")
    rel = np.ones(len(t), bool); info = dict(events=0, refits=0, refits_ok=0)
    occ = None
    if pert.kind != "crop":                             # clean too: real phones get bumped (dev_match was)
        fr = perturb_frames(thumbs, thumb_t, t, pert)
        i0 = int(np.searchsorted(t, start))               # reference = the framing at match start, not the warm-up
        A = CM.analyze(fr, t, ref=fr[min(i0, len(t) - 1)])
        sub = slice(None, None, 6)
        occ, oi = OC.detect_occluders(fr[sub], A["offset"][sub])
        info.update(occluder_area=oi["area"], occluder_conf=oi["confidence"], occluder_iou=_iou(occ, pert.mount_mask(occ.shape)))
        if occ.any():                                   # re-run without the mount part (it would pin the offset to 0)
            fr[:, occ] = fr[:, ~occ].mean()
            A = CM.analyze(fr, t, ref=fr[min(i0, len(t) - 1)])
        rel = A["reliable"]; info["_A"] = dict(fit_offset=A["fit_offset"], reliable=A["reliable"], t=t)
        Ep = CM.realign_energy(Ep, A["fit_offset"], (thumbs.shape[2], thumbs.shape[1]), order=int(os.environ.get("ROB_REALIGN_ORDER", "1")))
        info.update(events=len(A["events"]), refits=len(A["refits"]), refits_ok=sum(r[2] for r in A["refits"]))
        if occ.any(): Ep[:, OC.cells_mask(occ, *E.shape[1:])] = 0.0
    vis = poly_visible_share(poly, pert, thumbs.shape[1:], occ)
    info["visible"] = round(vis, 3)
    mode = FR.vision_mode(vis, float(rel.mean()))
    info.update(mode=mode, reliable=round(float(rel.mean()), 3))
    if mode == "audio":
        return None, info
    return MotionCues(Ep, t, reliable=rel, visible=vis if mode == "partial" else 1.0), info


def run_builds(le, bg, lb, start, sets, cues):
    P, R = SG.audio_points(le, bg, start)
    P3, _ = SG.gap_fill(P, R, start, sets)
    us_audio, _ = smart_points(P3, le, bg, lb, sets)
    us = live_gate(us_audio, cues) if cues is not None else us_audio
    eu = smart_points(admit_candidates(P3, cues, R, start, sets), le, bg, lb, sets)[0] if cues is not None else us_audio
    return dict(US=us, EU=eu, US_audio=us_audio)


def metrics(L, clips, hits, ct, clean=None):
    preds = [(c["A"], c["B"]) for c in clips]; parts = [c.get("parts") or [(c["A"], c["B"])] for c in clips]
    m = SB.stage_metrics(L, preds, hits, ct, parts)
    out = dict(recall=round(m["recall"], 4), found=m["found"], n_points=m["n_points"], non_points=m["non_points"],
               start_err_med=None if m["start_err_med"] is None else round(m["start_err_med"], 2),
               end_err_med=None if m["end_err_med"] is None else round(m["end_err_med"], 2),
               taps_per_100=round(100 * ((m["n_points"] - m["found"]) + m["non_points"]) / m["n_points"], 1))
    if clean is not None:
        cm = SB.match(clean, preds)
        dA = [abs(preds[j][0] - clean[i][0]) for i, j in cm.items()]; dB = [abs(preds[j][1] - clean[i][1]) for i, j in cm.items()]
        out.update(boundary_change_max=round(max(dA + dB, default=0.0), 3), kept_vs_clean=f"{len(cm)}/{len(clean)}")
    return out


def synthetic(seed=0, n_pts=24):
    """A synthetic match: audio envelopes with rallies (5-9 hits) + faults, a textured static court frame, motion energy
    from moving players during rallies only, labels and contacts. Deterministic."""
    rng = np.random.default_rng(seed)
    FRm = 100; dur = 60 + n_pts * 40.0
    n = int(dur * FRm); le = np.zeros(n); bg = np.zeros(n); lb = np.zeros(n)
    segs, ct, hits, t0 = [], {}, [], 30.0
    for k in range(n_pts):
        h = [t0 + i * 1.2 for i in range(int(rng.integers(5, 10)))]
        for v in h: le[int(v * FRm):int(v * FRm) + 3] = 25.0
        segs.append(NS(kind="point", start=h[0] - 3.0, end=h[-1] + 4.0)); ct[len(segs) - 1] = (h[0], h[-1]); hits += h
        t0 = h[-1] + float(rng.uniform(22, 34))
    L = NS(segments=segs)
    from scipy.ndimage import gaussian_filter
    img = np.clip(gaussian_filter(rng.normal(size=(108, 192)), 1.5) * 60 + 110, 0, 255)
    img[40:42, 50:165] = 230; img[60:62, 30:185] = 230; img[90:92, 5:190] = 230               # court lines
    t = np.arange(0, dur, 1 / 6.0)
    E = np.abs(rng.normal(0.3, 0.1, size=(len(t), 12, 24))).astype(np.float32)
    for s in segs:
        m = (t >= s.start + 2.0) & (t <= s.end - 3.0)
        E[m, 3:11, 4:20] += rng.uniform(3, 6, size=(m.sum(), 8, 16)) * (rng.random((m.sum(), 8, 16)) < 0.11)
    thumb_t = np.arange(0, dur, 1.0)
    thumbs = np.repeat(img[None].astype(np.uint8), len(thumb_t), 0)
    return dict(le=le, bg=bg, lb=lb, start=0.0, sets=None, L=L, ct=ct, hits=hits, E=E, t=t, thumbs=thumbs,
                thumb_t=thumb_t, poly=[(0.25, 0.37), (0.86, 0.37), (1.0, 0.85), (0.0, 0.85)])


def load_tracks(d, size=(640, 360)):
    """pw_tracks output (seg*.npz: t0, fps, ball (n,2) px, pose_k, boxes (k,8,5) px at `size`) -> ball and player-feet
    points with times (video s). Feet = bottom-centre of each person box with conf >= 0.5."""
    import glob
    tb, b, tf, f = [], [], [], []
    for fn in sorted(glob.glob(os.path.join(d, "seg*.npz"))):
        if fn.endswith(".tmp.npz"): continue
        z = np.load(fn); n = len(z["ball"]); ts = float(z["t0"]) + np.arange(n) / float(z["fps"])
        ok = ~np.isnan(z["ball"][:, 0]); tb.append(ts[ok]); b.append(z["ball"][ok])
        k = z["pose_k"]; B = z["boxes"]
        for j, i in enumerate(k):
            for bb in B[j]:
                if np.isfinite(bb[4]) and bb[4] >= 0.5: tf.append(ts[min(n - 1, int(i))]); f.append(((bb[0] + bb[2]) / 2, bb[3]))
    cat = lambda x: np.concatenate(x) if x else np.zeros((0, 2))
    return dict(size=size, t_ball=np.concatenate(tb) if tb else np.zeros(0), ball=cat(b),
                t_feet=np.array(tf), feet=np.array(f, float).reshape(-1, 2))


def track_metrics(TR, pert, t_path, A=None, proxy=(192, 108)):
    """Ball / feet tracks under the perturbation. naive (A None): observed position vs the true one; handled: the tracked
    camera fit (A["fit_offset"], proxy px) undone first. -> per kind: observable share, median / p95 error (px at the
    track size), share of samples on reliable vision."""
    from opencourtvision.robust.perturb import perturb_points
    W, H = TR["size"]; out = {}
    dpath = pert.path(t_path)
    for kind in ("ball", "feet"):
        px, t = TR[kind], TR["t_" + kind]
        if not len(t): out[kind] = None; continue
        d = np.stack([np.interp(t, t_path, dpath[:, 0]), np.interp(t, t_path, dpath[:, 1])], 1)
        obs = perturb_points(px, t, pert, (W, H), d=d)
        if A is not None:
            fo = A["fit_offset"]
            obs = obs - np.stack([np.interp(t, A["t"], fo[:, 0]) * W / proxy[0], np.interp(t, A["t"], fo[:, 1]) * H / proxy[1]], 1)
        seen = ~np.isnan(obs[:, 0]); err = np.hypot(*(obs[seen] - px[seen]).T)
        rel = float(np.mean(np.interp(t, A["t"], A["reliable"].astype(float)) >= 0.5)) if A is not None else None
        out[kind] = dict(n=int(len(t)), observable=round(float(seen.mean()), 3),
                         err_med_px=round(float(np.median(err)), 2) if len(err) else None,
                         err_p95_px=round(float(np.percentile(err, 95)), 2) if len(err) else None, reliable=None if rel is None else round(rel, 3))
    return out


def run_tracks(D, TR, perts=None, log=print):
    perts = perts or suite(); res = {}
    for pert in perts:
        for handled in (False, True):
            _, info = build_cues(D["E"], D["t"], D["thumbs"], D["thumb_t"], pert, handled, D["poly"], D["start"])
            A = info.pop("_A", None) if handled else None
            if handled and A is None:                         # crop-only: no camera motion analysis, nothing to undo
                A = dict(fit_offset=np.zeros((len(D["t"]), 2)), reliable=np.ones(len(D["t"]), bool), t=D["t"])
            key = f"{pert.name}/{'handled' if handled else 'naive'}"
            res[key] = dict(tracks=track_metrics(TR, pert, D["t"], A), camera=info)
            log(f"{key:28s} " + "  ".join(f"{k}: obs {m['observable']} err {m['err_med_px']}/{m['err_p95_px']} rel {m['reliable']}"
                                          for k, m in res[key]["tracks"].items() if m) + f"  mode {info.get('mode')}")
    return res


def run(D, perts=None, log=print):
    perts = perts or suite()
    res, clean = {}, {}
    for pert in perts:
        for handled in (False, True):
            cues, info = build_cues(D["E"], D["t"], D["thumbs"], D["thumb_t"], pert, handled, D["poly"], D["start"])
            B = run_builds(D["le"], D["bg"], D["lb"], D["start"], D["sets"], cues)
            key = f"{pert.name}/{'handled' if handled else 'naive'}"
            for b, clips in B.items():
                iv = [(c["A"], c["B"]) for c in clips]
                if pert.kind == "clean" and not handled: clean[b] = iv
                res.setdefault(key, {})[b] = metrics(D["L"], clips, D["hits"], D["ct"], clean.get(b))
            info.pop("_A", None)
            res[key]["camera"] = info
            log(f"{key:28s} " + "  ".join(f"{b}: rec {m['recall']:.3f} non {m['non_points']} taps {m['taps_per_100']} "
                                          f"dB {m.get('boundary_change_max', 0)}" for b, m in res[key].items() if b != "camera")
                + f"  {info}")
    return res


def load_real(a):
    from opencourtvision.labels.points import load_point_labels
    from opencourtvision.scoring.games import parse_final
    L = load_point_labels(a.labels)
    z = np.load(a.features); mz = np.load(a.motion)
    hits = sorted(t for r in json.load(open(a.hits)) for t in r.get("times", []))
    ct = SB.contacts(L, hits, SB.read_sv_shots(a.sv_xlsx) if a.sv_xlsx else None)
    return dict(le=z["le"], bg=z["bg"], lb=z["lb"], start=a.start, sets=parse_final(a.final), L=L, ct=ct, hits=hits,
                E=mz["E"], t=mz["t"], thumbs=mz["thumbs"], thumb_t=mz["thumb_t"], poly=COURT_POLY_DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    for k in ("labels", "features", "motion", "hits", "sv_xlsx", "final", "tracks"): ap.add_argument("--" + k.replace("_", "-"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--only", help="comma-separated perturbation names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    perts = suite()
    if a.only: perts = [p for p in perts if p.name in a.only.split(",") or p.name == "clean"]
    if a.tracks:                                         # EU track checks (motion npz for the camera analysis)
        mz = np.load(a.motion)
        DM = dict(E=mz["E"], t=mz["t"], thumbs=mz["thumbs"], thumb_t=mz["thumb_t"], start=a.start, poly=COURT_POLY_DEV)
        rt = run_tracks(DM, load_tracks(a.tracks), perts)
        json.dump(rt, open(os.path.join(a.out, "robustness_tracks.json"), "w"), indent=1)
        if not a.labels: return
    D = synthetic() if a.synthetic else load_real(a)
    res = run(D, perts)
    json.dump(res, open(os.path.join(a.out, "robustness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
