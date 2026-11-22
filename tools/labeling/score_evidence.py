#!/usr/bin/env python3
"""Score existing points.json with TrackNetV3 ball tracks + pw_rules soft/hard cues.

No pose / player-ID weights (excluded pending Legal). Court is image-relative for
fence cams: bottom of frame = near baseline, top = far. Player sides via OpenCV
HOG only when available (classical, no neural weight download).

  python tools/labeling/score_evidence.py --video V.mp4 --candidates points.json \
      --out scored.json [--work DIR] [--device cuda] [--max-points N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PW = ROOT / "tools" / "point_winner"
if str(PW) not in sys.path:
    sys.path.insert(0, str(PW))

from opencourtvision.ball.tracknet3 import TrackNetV3  # noqa: E402
from opencourtvision.ball.tracker import BallEngine  # noqa: E402
import pw_rules as RU  # noqa: E402


class ImageView:
    """Simple fence-cam view: maps image y to court y with a 3-point linear model.

    near baseline ~ 0.88*H, net ~ 0.48*H, far baseline ~ 0.12*H (tunable).
    """

    def __init__(self, w: int, h: int, near_frac=0.88, net_frac=0.48, far_frac=0.12):
        self.w, self.h = w, h
        self.ok = True
        self.t0, self.t1 = 0.0, 1e9
        self.near_y = near_frac * h
        self.net_y = net_frac * h
        self.far_y = far_frac * h

    def c2i(self, x, y=None):
        if y is None:
            x, y = x
        # court x 0..10.97, y 0..23.77
        ix = (float(x) / 10.97) * self.w
        if float(y) <= 11.885:
            t = float(y) / 11.885
            iy = self.near_y + t * (self.net_y - self.near_y)
        else:
            t = (float(y) - 11.885) / 11.885
            iy = self.net_y + t * (self.far_y - self.net_y)
        return np.array([ix, iy], float)

    def i2c(self, xy):
        xy = np.atleast_2d(xy).astype(float)
        out = np.zeros_like(xy)
        out[:, 0] = xy[:, 0] / max(self.w, 1) * 10.97
        for i, iy in enumerate(xy[:, 1]):
            if iy >= self.net_y:
                t = (iy - self.net_y) / max(self.near_y - self.net_y, 1e-6)
                t = float(np.clip(t, 0, 1))
                out[i, 1] = 11.885 * (1.0 - t)  # between net and near
            else:
                t = (self.net_y - iy) / max(self.net_y - self.far_y, 1e-6)
                t = float(np.clip(t, 0, 1))
                out[i, 1] = 11.885 + 11.885 * t
        return out


def read_window(cap, start_s: float, end_s: float, fps: float, max_w: int = 960):
    """Yield BGR frames for [start_s, end_s], optionally downscaled."""
    f0 = max(0, int(start_s * fps))
    f1 = max(f0 + 1, int(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(f1 - f0):
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if w > max_w:
            scale = max_w / w
            fr = cv2.resize(fr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    return frames, f0


def soft_ball_from_track(pos: np.ndarray, fps: float, t0: float, near_is_A: bool = True):
    """Soft (p_a, weight) from where the live ball ends and last crossing of mid-frame."""
    if pos is None or len(pos) < 8:
        return None
    good = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
    if good.sum() < 8:
        return None
    ys = pos[good, 1]
    xs = pos[good, 0]
    # ending: last 0.4 s median y
    n_end = max(3, int(0.4 * fps))
    y_end = float(np.nanmedian(ys[-n_end:]))
    y_med = float(np.nanmedian(ys))
    # In fence cam, larger y = nearer camera = near side.
    # If ball ends clearly on far side (small y), near team likely won (bounce in on far / far missed).
    # If ball ends on near side, far team likely won.
    spread = float(np.nanpercentile(ys, 90) - np.nanpercentile(ys, 10))
    if spread < 20:
        return None
    mid = y_med
    # signed: +1 means end on near half
    end_near = y_end > mid
    # p that A won: if near_is_A and ball ends far -> A more likely
    if near_is_A:
        p = 0.72 if not end_near else 0.28
    else:
        p = 0.72 if end_near else 0.28
    # weight grows with how far from mid the ending is
    conf = min(1.0, abs(y_end - mid) / max(spread * 0.5, 1.0))
    w = 0.35 + 0.45 * conf
    return (float(p), float(w))


def hog_side_cue(frames, near_is_A: bool = True):
    """Optional classical HOG person detector: compare near vs far occupancy at end of point."""
    if not frames:
        return None
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception:
        return None
    # sample last ~1s
    sample = frames[-max(1, len(frames) // 10):][::3]
    near_n = far_n = 0
    for fr in sample:
        h, w = fr.shape[:2]
        # HOG prefers larger people; downscale if huge
        scale = 1.0
        img = fr
        if w > 640:
            scale = 640 / w
            img = cv2.resize(fr, (640, int(h * scale)))
        rects, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        for (x, y, bw, bh) in rects:
            cy = (y + bh) / scale if scale != 1 else (y + bh)
            if cy > h * 0.5:
                near_n += 1
            else:
                far_n += 1
    if near_n + far_n < 2:
        return None
    # weak behaviour cue: more motion/people finishing celebration on winner side — too weak; skip
    return None


def score_point(tn, frames, fps, t0, hits, near_is_A=True):
    if len(frames) < 8:
        return RU.p_a_from_cues()
    eng = BallEngine(tracknet=tn, fps=fps, t0=t0, hits=hits or (), use_live_filter=True, reacquire="tracknet")
    out = eng.run(frames)
    pos = out["pos"]
    bounces = out.get("bounces") or []
    ball_soft = soft_ball_from_track(pos, fps, t0, near_is_A=near_is_A)
    # Image-relative net/bounce approximations via FakeView + last shot from y extrema
    h, w = frames[0].shape[:2]
    view = ImageView(w, h)
    serve = net = bounce = None
    # build minimal rally from y turning points for serve/net rules
    good = np.isfinite(pos[:, 1])
    if good.sum() >= 8:
        t = t0 + np.arange(len(pos)) / fps
        ys = pos[:, 1].copy()
        ys[~good] = np.nan
        # fill small gaps
        idx = np.where(good)[0]
        y_s = ys[idx]
        t_s = t[idx]
        # turning points: local extrema
        kind_times = []
        for i in range(2, len(y_s) - 2):
            if y_s[i] >= y_s[i - 1] and y_s[i] >= y_s[i + 1] and y_s[i] - min(y_s[i - 2:i + 3]) > 15:
                kind_times.append((float(t_s[i]), +1))  # near contact
            if y_s[i] <= y_s[i - 1] and y_s[i] <= y_s[i + 1] and max(y_s[i - 2:i + 3]) - y_s[i] > 15:
                kind_times.append((float(t_s[i]), -1))  # far contact
        # snap to audio hits when close
        if hits:
            snapped = []
            for tt, k in kind_times:
                close = [h for h in hits if abs(h - tt) <= 0.25]
                snapped.append((min(close, key=lambda h: abs(h - tt)) if close else tt, k))
            kind_times = snapped
        if kind_times:
            shots = [(tt, 0.0, k) for tt, k in kind_times]
            r = {
                "shots": shots,
                "n_shots": len(shots),
                "first_kind": shots[0][2],
                "last_kind": shots[-1][2],
                "t_first": shots[0][0],
                "t_last": shots[-1][0],
            }
            # serve rule needs server; assume A serves unknown -> skip unless n_shots tiny
            # net rule: ball stays on hitter side after last shot
            def rows_fn(d, view_):
                return t[good], (pos[good, 1] - view_.net_y) / max(view_.near_y - view_.net_y, 50.0), pos[good, 0], pos[good, 1]

            # fabricate d for rule_net
            d = {"t0": t0, "fps": fps, "ball": pos / 1.0, "ball_peak": np.where(good, 1.0, 0.0)}
            # scale: our pos is already in frame px; rule_net uses rows_fn
            try:
                side, name = RU.rule_net(r, d, view, rows_fn)
                if side is not None:
                    net = (side, name)
            except Exception:
                net = None
            # bounce rule with image-space bounce list
            bo = []
            for b in bounces:
                if isinstance(b, dict):
                    bo.append(b)
                else:
                    # BallEngine bounce objects may be dataclasses
                    bo.append({"t": getattr(b, "t", None), "x": getattr(b, "x", np.nan), "y": getattr(b, "y", np.nan)})
            bo = [b for b in bo if b.get("t") is not None and np.isfinite(b.get("x", np.nan))]
            try:
                side, name = RU.rule_bounce(r, bo, view, ok_views=True)
                if side is not None:
                    bounce = (side, name)
            except Exception:
                bounce = None

    result = RU.p_a_from_cues(serve=serve, net=net, bounce=bounce, near_is_A=near_is_A, ball=ball_soft)
    result["track_stats"] = out.get("stats")
    result["n_bounces"] = len(bounces)
    result["ball_soft"] = ball_soft
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--max-w", type=int, default=960)
    a = ap.parse_args(argv)

    spec = json.loads(a.candidates.read_text(encoding="utf-8"))
    points = list(spec.get("points") or [])
    if a.max_points:
        points = points[: a.max_points]
    near_is_A = True
    match = spec.get("match") or {}
    # mine/first_server unused for near mapping; fence cam assumes near = A (mine default)

    tn = TrackNetV3(device=a.device)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    t0_wall = time.perf_counter()
    scored = []
    rule_counts: dict[str, int] = {}
    n_non = 0
    for i, p in enumerate(points):
        start = max(0.0, float(p["start"]) - a.pad)
        end = float(p["end"]) + a.pad
        frames, _ = read_window(cap, start, end, fps, max_w=a.max_w)
        hits = []  # audio hits inside window unknown here; BallEngine still works
        r = score_point(tn, frames, fps, start, hits, near_is_A=near_is_A)
        q = dict(p)
        q["p_a"] = float(r["p_a"])
        q["tier"] = r.get("tier")
        q["reason"] = (r.get("rules") or [r.get("tier") or "abstain"])[0] if not r.get("abstain") else (r.get("tier") or "abstain")
        q["rules"] = list(r.get("rules") or [])
        q["abstain"] = bool(r.get("abstain"))
        if r.get("ball_soft"):
            q["ball_soft"] = list(r["ball_soft"])
        scored.append(q)
        if not q["abstain"]:
            n_non += 1
        key = q["reason"] if not q["abstain"] else "abstain"
        if q.get("rules"):
            for name in q["rules"]:
                rule_counts[name] = rule_counts.get(name, 0) + 1
        else:
            rule_counts[key] = rule_counts.get(key, 0) + 1
        if (i + 1) % 5 == 0 or i == 0:
            print(json.dumps({"progress": i + 1, "of": len(points), "non_abstain": n_non, "last": q["id"], "p_a": q["p_a"], "tier": q["tier"]}))
    cap.release()
    elapsed = time.perf_counter() - t0_wall
    video_span = sum(max(0.0, float(p["end"]) - float(p["start"]) + 2 * a.pad) for p in points)
    hours = max(1e-9, video_span / 3600.0)
    out = dict(spec)
    out["points"] = scored
    out["meta"] = dict(spec.get("meta") or {})
    out["meta"].update({
        "evidence_s": round(elapsed, 2),
        "evidence_seconds_per_video_hour": round(elapsed / hours, 2),
        "evidence_method": "TrackNetV3+BallEngine+pw_rules (no pose weights; image-relative court)",
        "n_non_abstain": n_non,
        "n_points": len(scored),
        "rule_counts": rule_counts,
        "device": tn.device,
    })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(a.out), **{k: out["meta"][k] for k in ("evidence_s", "evidence_seconds_per_video_hour", "n_non_abstain", "n_points", "rule_counts", "device")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
