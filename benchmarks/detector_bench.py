#!/usr/bin/env python3
"""Synthetic detector baseline. No image or video file is written or read.

Frames are drawn in memory: a green court, a yellow ball, one near player and one far player.
Fast frames draw the ball as a dim, stretched streak. The ball baseline is the shipped HSV
candidate function (opencourtvision.ball.hsv.candidates). Players are a coarse colour split
that stands in until a contributed detector replaces YOLO11 (see docs/issue_drafts).

Primary metrics, each with a Wilson 95% interval and the sample count:
  far-half ball recall, fast/blurry ball recall, far-player recall,
  downstream bounce recall and point-end recall on the synthetic rallies.
Secondary: ball average precision and CPU frames per second.
A contribution regresses a primary metric when its new point estimate is below
baseline - that metric's CI half-width.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from opencourtvision.ball.hsv import candidates

H, W = 360, 640
FAR_Y = H // 2


def _wilson(k, n, z=1.96):
    if n <= 0:
        return {"p": 0.0, "lo": 0.0, "hi": 0.0, "half": 0.0, "k": 0, "n": 0}
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return {"p": float(p), "lo": float(lo), "hi": float(hi), "half": float((hi - lo) / 2), "k": int(k), "n": int(n)}


def draw(spec, prev_ball):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (45, 130, 48)
    cv2.line(img, (40, FAR_Y), (W - 40, FAR_Y), (230, 230, 230), 2)
    players = []
    for pl in spec["players"]:
        x, y, bw, bh, color = pl
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), color, -1)
        players.append((x + bw / 2, y + bh / 2, bw, bh, y < FAR_Y))
    ball = spec["ball"]
    if ball is not None:
        x, y, fast = ball
        if fast:
            cv2.ellipse(img, (int(x), int(y)), (14, 3), 20, 0, 360, (180, 210, 140), -1)
        else:
            cv2.circle(img, (int(x), int(y)), 4 if y < FAR_Y else 7, (40, 230, 240), -1)
    return img, players, ball


def detect_ball(frame, prev_gray):
    cands = candidates(frame, prev_gray)
    if not cands:
        return None
    x, y = max(cands, key=lambda c: c[4])[:2]
    return float(x), float(y)


def detect_players(frame):
    """Coarse stand-in: saturated non-green blobs. Far players in this bench are dull on purpose."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 80, 40), (140, 255, 255))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 30:
            continue
        out.append((float(cen[k][0]), float(cen[k][1]), st[k, cv2.CC_STAT_AREA]))
    return out


def _hit(pred, gt, tol):
    if pred is None or gt is None:
        return False
    return abs(pred[0] - gt[0]) <= tol and abs(pred[1] - gt[1]) <= tol


def make_specs(n=80, seed=3):
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        far_ball = i % 2 == 0
        fast = i % 5 == 0
        y = float(rng.uniform(30, FAR_Y - 12) if far_ball else rng.uniform(FAR_Y + 20, H - 20))
        x = float(rng.uniform(50, W - 50))
        near = (float(rng.uniform(80, W - 120)), float(rng.uniform(FAR_Y + 30, H - 50)), 36, 70, (200, 80, 40))
        far = (float(rng.uniform(80, W - 80)), float(rng.uniform(24, FAR_Y - 28)), 16, 28, (70, 90, 60))
        specs.append({"ball": (x, y, fast), "players": [near, far], "far_ball": far_ball, "fast": fast, "i": i})
    return specs


def bounce_rally(x0=70.0):
    """One synthetic rally: the ball falls, bounces once, then the next frame is the point end (ball gone)."""
    frames = []
    y, vy = 40.0, 18.0
    bounce_at = None
    for i in range(40):
        y += vy
        if y > 240 and vy > 0 and bounce_at is None:
            vy = -14.0
            bounce_at = i
            y = 240.0
        x = x0 + i * 6
        end = bounce_at is not None and i == bounce_at + 6
        frames.append({
            "ball": None if end else (float(x), float(y), False),
            "bounce": i == bounce_at,
            "point_end": end,
        })
        if end:
            break
    return frames


def run(n=80, seed=3):
    specs = make_specs(n, seed)
    prev = np.full((H, W), 40, np.uint8)
    far_hit = far_n = fast_hit = fast_n = player_hit = player_n = 0
    ball_tp = ball_fp = 0
    t0 = time.perf_counter()
    ball_preds = []
    for spec in specs:
        frame, players, ball = draw(spec, None)
        pred = detect_ball(frame, prev)
        ball_preds.append(pred)
        prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gt = None if ball is None else (ball[0], ball[1])
        ok = _hit(pred, gt, 12)
        if spec["far_ball"]:
            far_n += 1
            far_hit += int(ok)
        if spec["fast"]:
            fast_n += 1
            fast_hit += int(ok)
        if ok:
            ball_tp += 1
        elif pred is not None:
            ball_fp += 1
        found_far = False
        for px, py, _area in detect_players(frame):
            for pl in players:
                if pl[4] and abs(px - pl[0]) < 20 and abs(py - pl[1]) < 24:
                    found_far = True
        player_n += 1
        player_hit += int(found_far)
    elapsed = time.perf_counter() - t0
    # downstream: did a sign-change of detected vertical motion land on the true bounce, and did the track die at point end?
    bounce_hit = end_hit = bounce_n = end_n = 0
    for _rally_i in range(20):
        rally = bounce_rally(x0=60.0 + 8 * _rally_i)
        prev = np.full((H, W), 40, np.uint8)
        ys = []
        for fr in rally:
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = (45, 130, 48)
            if fr["ball"]:
                cv2.circle(img, (int(fr["ball"][0]), int(fr["ball"][1])), 6, (40, 230, 240), -1)
            pred = detect_ball(img, prev)
            prev = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ys.append(None if pred is None else pred[1])
            if fr["bounce"]:
                bounce_n += 1
                if len(ys) >= 3 and None not in ys[-3:]:
                    if ys[-3] < ys[-2] and ys[-1] < ys[-2]:
                        bounce_hit += 1
            if fr["point_end"]:
                end_n += 1
                end_hit += int(pred is None)
    ap = ball_tp / max(1, ball_tp + ball_fp)
    return {
        "dataset": "synthetic-in-memory",
        "seed": seed,
        "n_frames": n,
        "primary": {
            "ball_recall_far_half": _wilson(far_hit, far_n),
            "ball_recall_fast_blurry": _wilson(fast_hit, fast_n),
            "far_player_recall": _wilson(player_hit, player_n),
            "bounce_recall": _wilson(bounce_hit, bounce_n),
            "point_end_recall": _wilson(end_hit, end_n),
        },
        "secondary": {
            "ball_average_precision": ap,
            "ball_average_precision_n": ball_tp + ball_fp,
            "fps_cpu": n / elapsed if elapsed else None,
        },
        "regression_rule": "A change regresses a primary metric when new_p < baseline_p - baseline_half.",
        "swap_point": "tools/point_winner/pw_tracks.py YOLO() call; ball candidates: opencourtvision.ball.hsv.candidates",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    report = run(a.frames, a.seed)
    text = json.dumps(report, indent=2)
    print(text)
    if a.json_out:
        open(a.json_out, "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
