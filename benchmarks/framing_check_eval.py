#!/usr/bin/env python3
"""Framing-check evaluation on the dev_match match (PRIVATE frames, read-only; only aggregate numbers go in the repo).

    python benchmarks/framing_check_eval.py --refit /workspace/btv2/refit --out benchmarks/results/framing_check_eval.json \
        [--examples /workspace/framing_check] [--det rfdetr|none]

--refit holds refit/p{idx:03d}{a,b}.jpg + ref{view}.jpg (1280x720 proxies, two per point), refit_{a,b}.json (per-point
court fit re-estimated by ORB at 1280 px: H_refit; the view's stale fit: H_img2court) and grab.txt (frame times).
Everything is checked at 640x360 gray, the size the phone would use. Person boxes come from RF-DETR Nano (384, conf 0.5)
run on the exact pixels being checked, including every edited frame. Boxes stay in memory; nothing is written but the
aggregate JSON and (optionally) a few example PNGs outside the repo.

Cases (truth from the court fits, never from the checker):
  same_point   lock on a point's first frame, check its second (seconds apart): steady truth from the two ORB fits
  view_drift   lock on the view's setup frame, check every point of that view: real camera creep, incl. points 47-53
  synthetic    one real frame edited with a known change: shift, rotate, zoom, tilt, squash, zoom_out (court fits:
               truth = A @ H), blur / haze (lines hard to see), erase_far / erase_near (that team's pixels replaced
               by the empty-court background of the same view, then the detector re-run)
  live         every frame of a view in time order through the live (debounced) checker, locked at the view's setup
Per app-facing check (steady, baselines, lines, level, players): precision / recall, plus false-alarm rates on frames
whose truth is ok.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from opencourtvision.framing import CHECK, FramingChecker, drift_px, geometry_warnings, normalise_H  # noqa: E402
from opencourtvision.framing.check import NET_Y, _proj, far_roi, line_report  # noqa: E402
from opencourtvision.framing.status import CHECKS  # noqa: E402
from opencourtvision.framing.synth import warp_matrix  # noqa: E402

SIZE = (640, 360)
SRC_W = 3840
S = np.diag([SIZE[0] / SRC_W, SIZE[0] / SRC_W, 1.0])
DIAG = float(np.hypot(*SIZE))
THR = CHECK["DRIFT_FRAC"] * DIAG
GEO_KINDS = ("shift", "rotate", "zoom", "tilt", "squash", "zoom_out")


class Detector:
    def __init__(self, kind, roi=True):
        self.kind = kind; self.ms = []; self.roi = roi
        if kind == "rfdetr":
            import torch
            from rfdetr import RFDETRNano
            torch.set_num_threads(int(os.environ.get("DET_THREADS", "4")))
            self.m = RFDETRNano(resolution=384)

    def _run(self, bgr):
        d = self.m.predict(bgr[:, :, ::-1].copy(), threshold=0.5)
        keep = d.class_id == 1                                       # COCO person; boxes only
        return np.c_[d.xyxy[keep], d.confidence[keep]].astype(float).reshape(-1, 5)

    def __call__(self, bgr, H=None):
        """Full frame, plus a far-court crop (upscaled to 640 wide, at most 3x) when H is given; boxes merged by NMS."""
        if self.kind == "none": return np.zeros((0, 5))
        import cv2
        t0 = time.perf_counter()
        b = self._run(bgr)
        roi = None if H is None or not self.roi else far_roi(H, SIZE)
        if roi is not None:
            x0, y0, x1, y1 = roi; k = min(3.0, SIZE[0] / (x1 - x0))
            crop = cv2.resize(bgr[y0:y1, x0:x1], (int((x1 - x0) * k), int((y1 - y0) * k)), interpolation=cv2.INTER_LINEAR)
            r = self._run(crop); r[:, [0, 2]] = r[:, [0, 2]] / k + x0; r[:, [1, 3]] = r[:, [1, 3]] / k + y0
            b = np.vstack([b, r])
            if len(b):
                keep = cv2.dnn.NMSBoxes([[float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])] for x in b], b[:, 4].tolist(), 0.5, 0.5)
                b = b[np.asarray(keep).reshape(-1)]
        self.ms.append(1000 * (time.perf_counter() - t0))
        return b


def H640(H_i2c):
    return normalise_H(S @ np.linalg.inv(np.asarray(H_i2c, float)))


def load(root):
    import cv2
    times = dict(line.split() for line in open(os.path.join(root, "grab.txt")) if line.strip())
    fr = []
    for ab in "ab":
        js = json.load(open(os.path.join(root, f"refit_{ab}.json")))
        for p in js["points"]:
            img = cv2.imread(os.path.join(root, "refit", f"p{p['idx']:03d}{ab}.jpg"))
            if img is None: continue
            fr.append(dict(idx=p["idx"], ab=ab, view=p["view"], t=float(times[f"p{p['idx']:03d}{ab}"]),
                           bgr=cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA), H=H640(p["H_refit"]), H_view=H640(p["H_img2court"])))
        views = js["views"]
    refs = {}
    for v in views:
        img = cv2.imread(os.path.join(root, "refit", f"ref{v}.jpg"))
        Hv = next(f["H_view"] for f in fr if str(f["view"]) == v)
        refs[int(v)] = dict(bgr=cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA), H=Hv, t=float(times[f"ref{v}"]))
    return sorted(fr, key=lambda f: f["t"]), refs


def fit_whole_court(H, margin=0.04):
    """Zoom-out about the frame centre just enough that all four corners sit inside the frame with a margin (the
    'moved the phone back / 0.5x' version of this exact view)."""
    from opencourtvision.framing import court_corners_px
    W, Hh = SIZE; cx, cy = W / 2, Hh / 2
    k = 1.0
    for (u, v), _ in court_corners_px(H).values():
        for d, half in ((abs(u - cx), W / 2), (abs(v - cy), Hh / 2)):
            if d > 0: k = min(k, (1 - margin) * half / d)
    T = lambda dx, dy: np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return T(cx, cy) @ np.diag([k, k, 1.0]) @ T(-cx, -cy)


def gray(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def side_of(H, boxes):
    if not len(boxes): return np.zeros(0, "<U4")
    c, _ = _proj(np.linalg.inv(H), np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    return np.where(c[:, 1] >= NET_Y, "far", "near")


def backgrounds(fr, refs):
    """Empty-court picture per view: median of the view's frames warped onto its setup frame through the court fits."""
    import cv2
    bg = {}
    for v, r in refs.items():
        stack = [cv2.warpPerspective(f["bgr"], r["H"] @ np.linalg.inv(f["H"]), SIZE, borderMode=cv2.BORDER_REPLICATE) for f in fr if f["view"] == v]
        if len(stack) >= 5: bg[v] = np.median(np.stack(stack), axis=0).astype(np.uint8)
    return bg


def erase(f, boxes, side, bg, refs):
    import cv2
    v = f["view"]
    back = cv2.warpPerspective(bg[v], f["H"] @ np.linalg.inv(refs[v]["H"]), SIZE, borderMode=cv2.BORDER_REPLICATE)
    out = f["bgr"].copy()
    for b, s in zip(boxes, side_of(f["H"], boxes)):
        if s != side: continue
        x0, y0, x1, y1 = (int(round(x)) for x in b[:4]); pw, ph = 4 + (x1 - x0) // 6, 4 + (y1 - y0) // 10
        x0, y0, x1, y1 = max(0, x0 - pw), max(0, y0 - ph), min(SIZE[0], x1 + pw), min(SIZE[1], y1 + ph)
        out[y0:y1, x0:x1] = back[y0:y1, x0:x1]
    return out


def degrade(bgr, kind, mag):
    import cv2
    if kind == "blur": return cv2.GaussianBlur(bgr, (0, 0), mag)
    if kind == "haze":                                   # glare / fog / dirty lens: contrast squashed toward light gray
        return np.clip(bgr.astype(np.float32) * mag + 200 * (1 - mag), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def truth(H_lock, H_true, team_missing=(), unclear=None, steady_override=None):
    """Truth per app check: True = should warn, False = should be ok, None = not scored."""
    g, cut, info = geometry_warnings(H_true, SIZE)
    d = drift_px(H_lock, H_true, SIZE) if steady_override is None else steady_override
    steady = None if 0.5 * THR < d < 2.0 * THR else d >= 2.0 * THR
    out = bool(cut["corners"])
    lr = line_report(H_true, SIZE)["lines"]
    both_missing = all(lr[k]["in_frame"] < CHECK["BASELINE_MISSING"] for k in ("near_baseline", "far_baseline"))
    # firm truth: only what breaks scoring. Single frames can't show a *sustained* missing team: players firm unscored.
    firm = dict(steady=steady, baselines=both_missing, lines=False, level=False, players=None if team_missing else False)
    # v3: a cut corner/sideline is the baselines check's hint; "lines" only flags lines that are hard to see
    return dict(steady=steady, baselines=out, lines=bool(unclear), unclear=unclear,
                level="too_low_or_tilted" in g, players=bool(team_missing), firm=firm), d


class Score:
    def __init__(self): self.c = {}

    def add(self, case, check, t, pred):
        if t is None: return
        k = ("tp" if pred else "fn") if t else ("fp" if pred else "tn")
        for key in (case, "all"):
            self.c.setdefault(key, {}).setdefault(check, dict(tp=0, fp=0, fn=0, tn=0))[k] += 1

    @staticmethod
    def prf(d):
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else None
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else None
        fa = d["fp"] / (d["fp"] + d["tn"]) if d["fp"] + d["tn"] else None
        rnd = lambda x: None if x is None else round(x, 3)
        return dict(precision=rnd(p), recall=rnd(r), false_alarm_rate=rnd(fa), **d)

    def table(self, key="all"):
        return {c: self.prf(d) for c, d in self.c.get(key, {}).items()}


REPLAY_P = dict(WINDOW=3, DEBOUNCE=2, TEAM_WARN_CHECKS=4)


def replay(fr, det):
    """What a user would have seen over the whole dev_match match. Every frame in time order through one recording
    session (set up on the first frame with its tapped court fit). The user dismisses each hint the first time it is
    shown, and taps Re-lock on each firm "phone moved" (re-taps the corners when the court can't be re-found).
    Sampling is the per-point frames (~2 per point, tens of seconds apart), so live params are in checks, not seconds:
    a message needs 2 checks in a row, and a team must be missing for 4 checks (~2 points) to be firm."""
    def new_fc(f):
        return FramingChecker(SIZE, p=REPLAY_P).lock(gray(f["bgr"]), f["H"])
    f0 = fr[0]; fc = new_fc(f0)
    setup = []
    for _ in range(len(CHECKS)):                          # setup screen: one fix at a time, each dismissed
        v = fc.check(gray(f0["bgr"]), det(f0["bgr"], f0["H"]), live=False, track=False)
        if not v.fix_check: break
        setup.append(dict(check=v.fix_check, level=v.level, fix=v.fix))
        if v.level == "hint": fc.dismiss(v.fix_check)
        else: break
    firm, hints, hint_checks_undismissed = [], [], 0
    prev = {k: "ok" for k in CHECKS}
    for f in fr[1:]:
        v = fc.check(gray(f["bgr"]), det(f["bgr"], fc.H_lock), live=True, recording=True)
        hint_checks_undismissed += sum(1 for k in CHECKS if v.checks[k]["level"] == "hint")
        for k in CHECKS:
            lv = v.checks[k]["level"]
            if lv == "warn" and prev[k] != "warn":
                firm.append(dict(check=k, t=round(f["t"]), point=f["idx"], view=f["view"], fix=v.checks[k]["fix"]))
            if lv == "hint" and not v.checks[k]["dismissed"] and prev[k] == "ok":
                hints.append(dict(check=k, t=round(f["t"]), point=f["idx"], fix=v.checks[k]["fix"])); fc.dismiss(k)
            prev[k] = lv
        if v.checks["steady"]["level"] == "warn":         # the user taps Re-lock (or re-taps the corners)
            Hn, _ = fc.estimate_H(gray(f["bgr"]))
            dis = fc.dismissed
            fc = FramingChecker(SIZE, p=REPLAY_P).lock(gray(f["bgr"]), Hn if Hn is not None else f["H"]); fc.dismissed = dis
            prev["steady"] = "ok"
    by = lambda xs: {k: sum(1 for x in xs if x["check"] == k) for k in CHECKS if any(x["check"] == k for x in xs)}
    return dict(checks=len(fr) - 1, span_min=round((fr[-1]["t"] - fr[0]["t"]) / 60, 1), params=REPLAY_P,
                setup_messages=setup, firm_warnings=len(firm), firm_by_check=by(firm), firm_detail=[{k: x[k] for k in ("check", "point", "view", "t")} for x in firm],
                hints_shown=len(hints), hints_by_check=by(hints), hint_detail=[{k: x[k] for k in ("check", "point", "fix")} for x in hints],
                hint_check_count_if_never_dismissed=hint_checks_undismissed)


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", required=True); ap.add_argument("--out"); ap.add_argument("--examples")
    ap.add_argument("--det", default="rfdetr"); ap.add_argument("--seed", type=int, default=21); ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--skip-views", default="2", help="views whose stored court fit is wrong (dev_match view 2: the auto-fit "
                    "landed on the fence, so it cannot serve as truth); comma list, '' for none")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    rng = np.random.default_rng(a.seed)
    det = Detector(a.det)
    fr, refs = load(a.refit)
    skip = {int(x) for x in a.skip_views.split(",") if x.strip()}
    fr = [f for f in fr if f["view"] not in skip]; refs = {v: r for v, r in refs.items() if v not in skip}
    bg = backgrounds(fr, refs)
    sc = Score(); lat = []; good_any = dict(real=[0, 0], fullframe=[0, 0]); n_cases = {}; ex = {}
    team_ok_det = []

    real_firm = [0, 0]

    def run(case, fc, bgr, boxes, tr, live=False, tag=None):
        t0 = time.perf_counter(); v = fc.check(gray(bgr), boxes, live=live, recording=True); lat.append(1000 * (time.perf_counter() - t0))
        n_cases[case] = n_cases.get(case, 0) + 1
        warn = {k: v.checks[k]["level"] != "ok" for k in CHECKS}          # flagged: hint or warn
        for k in CHECKS:
            sc.add(case, k, tr[k], warn[k])
            sc.add(case, f"firm_{k}", tr["firm"][k], v.checks[k]["level"] == "warn")
        if case in ("same_point", "view_drift"):
            real_firm[1] += 1; real_firm[0] += any(v.checks[k]["level"] == "warn" and tr["firm"][k] is False for k in CHECKS)
        sc.add(case, "lines_unclear", tr.get("unclear"), "lines_unclear" in (v.shown if live else v.warnings))
        if tag and tag not in ex and v.fix: ex[tag] = (bgr.copy(), v, boxes)
        return v, warn

    def scored_ok(tr, warn):
        """(any scored check that is truly ok warned?, at least one truly-ok check scored)"""
        oks = [k for k in CHECKS if tr[k] is False]
        return any(warn[k] for k in oks), bool(oks)

    lock = lambda bgr, H, p=None: FramingChecker(SIZE, p=p).lock(gray(bgr), H)

    # same_point + synthetic
    by_pt = {}
    for f in fr: by_pt.setdefault(f["idx"], {})[f["ab"]] = f
    for idx, d in sorted(by_pt.items()):
        if "a" not in d or "b" not in d or d["a"]["view"] != d["b"]["view"]: continue
        fa, fb = d["a"], d["b"]
        bb = det(fb["bgr"], fa["H"]); s = side_of(fb["H"], bb); team_ok_det.append(bool((s == "far").any() and (s == "near").any()))
        tr, _ = truth(fa["H"], fb["H"])
        v, warn = run("same_point", lock(fa["bgr"], fa["H"]), fb["bgr"], bb, tr)
        x, y = scored_ok(tr, warn); good_any["real"][0] += x; good_any["real"][1] += y
        f = fa; base_boxes = det(f["bgr"], f["H"])
        fc0 = lock(f["bgr"], f["H"])
        for _ in range(a.reps):
            kind = GEO_KINDS[int(rng.integers(len(GEO_KINDS)))]; mag = float(rng.uniform(0.0, 1.0))
            A = warp_matrix(kind, SIZE, mag, rng)
            img2 = cv2.warpPerspective(f["bgr"], A, SIZE, borderMode=cv2.BORDER_REPLICATE)
            tr, _ = truth(f["H"], normalise_H(A @ f["H"]))
            run(f"synthetic_{kind}", lock(f["bgr"], f["H"]), img2, det(img2, f["H"]), tr, tag=f"synthetic_{kind}")
        # whole court in frame: the "good framing" control (phone moved back / 0.5x), locked on itself
        A = fit_whole_court(f["H"]); img2 = cv2.warpPerspective(f["bgr"], A, SIZE, borderMode=cv2.BORDER_REPLICATE); H2 = normalise_H(A @ f["H"])
        tr, _ = truth(H2, H2)
        v, warn = run("fullframe", lock(img2, H2), img2, det(img2, H2), tr, tag="fullframe_ok" if all(tr[k] is False for k in CHECKS) else None)
        if all(tr[k] is False for k in CHECKS):
            good_any["fullframe"][0] += any(warn.values()); good_any["fullframe"][1] += 1
        # both baselines missing: zoomed far in on mid-court (the one framing that stops scoring)
        c, _ = _proj(f["H"], [(5.485, 0.0), (5.485, 23.77)]); cx, cy = c.mean(0)      # midway between the baselines
        A = np.array([[3.0, 0, SIZE[0] / 2 - 3.0 * cx], [0, 3.0, SIZE[1] / 2 - 3.0 * cy], [0, 0, 1.0]])
        img2 = cv2.warpPerspective(f["bgr"], A, SIZE, borderMode=cv2.BORDER_REPLICATE); H2 = normalise_H(A @ f["H"])
        tr, _ = truth(H2, H2); tr["lines"] = None             # a 3x upscale really is soft: an unclear-lines hint is fair
        tr["players"] = tr["firm"]["players"] = None                 # players are cut by the zoom itself; no truth for them
        run("both_baselines_missing", lock(img2, H2), img2, det(img2, H2), tr, tag="both_baselines_missing")
        for kind, mag, unclear in (("blur", 1.0, False), ("blur", 4.0, True), ("haze", 0.6, False), ("haze", 0.2, True)):
            img2 = degrade(f["bgr"], kind, mag)
            tr, _ = truth(f["H"], f["H"], unclear=unclear)
            run(f"{kind}_{'bad' if unclear else 'mild'}", fc0, img2, det(img2, f["H"]), tr, tag=f"{kind}_bad" if unclear else None)
        if f["view"] in bg:
            for side in ("far", "near"):
                img2 = erase(f, base_boxes, side, bg, refs)
                tr, _ = truth(f["H"], f["H"], team_missing=(side,))
                run(f"erase_{side}", fc0, img2, det(img2, f["H"]), tr, tag=f"erase_{side}")

    # view_drift (single check) and live (debounced, time order), locked at each view's setup frame
    for v, r in refs.items():
        fl = lock(r["bgr"], r["H"], dict(WINDOW=5, DEBOUNCE=2))
        for f in [f for f in fr if f["view"] == v]:
            bb = det(f["bgr"], r["H"])
            tr, dd = truth(r["H"], f["H"])
            vv, warn = run("view_drift", lock(r["bgr"], r["H"]), f["bgr"], bb, tr, tag="view_drift_47_53" if 47 <= f["idx"] <= 53 else None)
            if 47 <= f["idx"] <= 53:
                sc.add("drift_47_53", "steady", tr["steady"], warn["steady"])
            run("live", fl, f["bgr"], bb, tr, live=True)
        # a whole team out of the picture for a sustained stretch: live checker over the view with that team erased
        if v in bg:
            for side in ("far", "near"):
                fe = lock(r["bgr"], r["H"], dict(WINDOW=5, DEBOUNCE=2))
                for i, f in enumerate([f for f in fr if f["view"] == v][:40]):
                    img2 = erase(f, det(f["bgr"], r["H"]), side, bg, refs)
                    tr, _ = truth(r["H"], f["H"], team_missing=(side,))
                    tr["firm"]["players"] = None if i < CHECK["TEAM_WARN_CHECKS"] + 2 else True     # after ~20 checks
                    run(f"live_erase_{side}", fe, img2, det(img2, r["H"]), tr, live=True)

    rp = replay(fr, det)
    res = dict(schema="vamos.bench.framing_check_eval.v2", data="dev_match doubles, per-point refit frames (PRIVATE; numbers only)",
               frames=len(fr), skipped_views=sorted(skip), views={str(v): sum(1 for f in fr if f["view"] == v) for v in refs}, size=list(SIZE), detector=a.det,
               drift_threshold_px=round(THR, 2), seed=a.seed, cases=n_cases, params={k: v for k, v in CHECK.items() if k != "ZONE_X"},
               detector_sees_both_teams=round(float(np.mean(team_ok_det)), 3) if team_ok_det else None,
               overall=sc.table("all"),
               by_case={k: sc.table(k) for k in sorted(sc.c) if k != "all"},
               false_alarm_any_check=dict(real_unedited=dict(frames=good_any["real"][1], with_false_warning=good_any["real"][0]),
                                          fullframe_all_ok=dict(frames=good_any["fullframe"][1], with_any_warning=good_any["fullframe"][0]),
                                          real_unedited_firm=dict(frames=real_firm[1], with_false_firm_warning=real_firm[0])),
               match_replay=rp,
               latency_ms=dict(check_median=round(statistics.median(lat), 2), check_p95=round(float(np.percentile(lat, 95)), 2),
                               detector_median=round(statistics.median(det.ms), 1) if det.ms else None,
                               detector_threads=int(os.environ.get("DET_THREADS", "4")), check_threads=1, cpu=os.uname().machine,
                               frame=f"{SIZE[0]}x{SIZE[1]}"))
    pub = {**res, "match_replay": {k: v for k, v in rp.items() if not k.endswith("_detail")}}   # per-point detail: print only
    if a.out: open(a.out, "w").write(json.dumps(pub, indent=1) + "\n")
    print(json.dumps(dict(replay=rp, overall=res["overall"], fa=res["false_alarm_any_check"], lat=res["latency_ms"], cases=n_cases,
                          d4753=res["by_case"].get("drift_47_53"), det2=res["detector_sees_both_teams"]), indent=1))
    if a.examples:
        from opencourtvision.framing.overlay import draw
        os.makedirs(a.examples, exist_ok=True)
        for tag, (img, v, boxes) in ex.items():
            cv2.imwrite(os.path.join(a.examples, f"example_{tag}.png"), draw(img, v, boxes))
        print("examples:", sorted(ex))


if __name__ == "__main__":
    main()
