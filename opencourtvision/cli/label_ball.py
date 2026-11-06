"""ocv-label-ball: make ball ground truth (labels/ball/*.csv) for ocv-bench-ball.

CSV columns: clip, frame, t, x, y, visibility, status
  t           seconds on the video's clock (frame = round((t - start) * fps) within the clip),
  x, y        ball centre in px of the labelling frame size (--size), empty when not visible,
  visibility  1 visible, 0 not visible / occluded, (empty = not labelled yet),
  status      'pred' (pre-filled from a tracker, unchecked), 'ok' (checked by a person), 'skip'.
Only rows with status 'ok' count as ground truth in ocv-bench-ball.

Sub-commands:
  template  write the CSV for a segment, every --every frames, pre-filled from a tracker CSV (t,x,y; e.g. an
            ocv-bench-ball output) when --prefill is given, plus numbered contact sheets (JPEG, 4 x 3 frames with a
            50 px grid and the prefilled point) so labels can be read off and typed in without a GUI.
  click     OpenCV window: left-click = ball here (visibility 1), 0 = not visible, space/n = next, b = back,
            s = save, q = save and quit. Needs a display (on the Mac: run it there).
  check     validate a labels CSV and print counts.

Example:
  ocv-label-ball template --video match.mp4 --start 139.2 --end 149.0 --clip 16 --every 3 \\
      --prefill work/bench/engine_clip16.csv --out labels/ball/2026-10-17_clip16.csv
  ocv-label-ball click --video match.mp4 --labels labels/ball/2026-10-17_clip16.csv
"""
import argparse
import csv
import os

COLS = ["clip", "frame", "t", "x", "y", "visibility", "status"]


def read(path):
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def write(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in COLS})


def ground_truth(rows):
    """{(clip, t): (x, y) or None} from rows with status 'ok' (None = labelled not visible)."""
    out = {}
    for r in rows:
        if r.get("status") != "ok": continue
        vis = r.get("visibility") == "1" and r.get("x")
        out[(r.get("clip", ""), round(float(r["t"]), 4))] = (float(r["x"]), float(r["y"])) if vis else None
    return out


def _frames(video, start, end, size, fps):
    from ..reel.ball import decode
    w, h = size
    return list(decode(video, start, end, w, h, fps))


def _prefill(path):
    import numpy as np
    d = {}
    for r in read(path):
        if r.get("x") not in (None, "", "nan"):
            try:
                x, y = float(r["x"]), float(r["y"])
            except ValueError:
                continue
            if np.isfinite(x): d[round(float(r["t"]), 2)] = (x, y)
    return d


def template(A):
    import cv2
    import numpy as np
    size = tuple(int(v) for v in A.size.lower().split("x"))
    fr = _frames(A.video, A.start, A.end, size, A.fps)
    pre = _prefill(A.prefill) if A.prefill else {}
    rows, picks = [], list(range(0, len(fr), A.every))
    for i in picks:
        t = A.start + i / A.fps
        p = pre.get(round(t, 2))
        rows.append(dict(clip=A.clip, frame=i, t=f"{t:.4f}", x=f"{p[0]:.1f}" if p else "", y=f"{p[1]:.1f}" if p else "",
                         visibility="1" if p else "", status="pred" if p else ""))
    write(A.out, rows)
    sheets = os.path.splitext(A.out)[0] + "_sheets"; os.makedirs(sheets, exist_ok=True)
    tw, th = size[0] // 2, size[1] // 2
    for s in range(0, len(picks), 12):
        canvas = np.zeros((3 * th, 4 * tw, 3), np.uint8)
        for j, i in enumerate(picks[s:s + 12]):
            f = fr[i].copy()
            for gx in range(0, size[0], 50): cv2.line(f, (gx, 0), (gx, size[1]), (80, 80, 80), 1)
            for gy in range(0, size[1], 50): cv2.line(f, (0, gy), (size[0], gy), (80, 80, 80), 1)
            r = rows[s + j]
            if r["x"]: cv2.circle(f, (int(float(r["x"])), int(float(r["y"]))), 9, (255, 0, 255), 1)
            cv2.putText(f, f"frame {i}  t {r['t']}", (8, size[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            canvas[(j // 4) * th:(j // 4 + 1) * th, (j % 4) * tw:(j % 4 + 1) * tw] = cv2.resize(f, (tw, th))
        cv2.imwrite(os.path.join(sheets, f"sheet_{s // 12:03d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {A.out} ({len(rows)} rows, {sum(bool(r['x']) for r in rows)} prefilled) and contact sheets in {sheets}")


def click(A):
    import cv2
    rows = read(A.labels)
    if not rows: raise SystemExit("empty labels file; make one with `ocv-label-ball template`")
    size = tuple(int(v) for v in A.size.lower().split("x"))
    t0 = float(rows[0]["t"]) - int(rows[0]["frame"]) / A.fps
    fr = _frames(A.video, t0, float(rows[-1]["t"]) + 1.0 / A.fps, size, A.fps)
    k, cur = 0, {}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN: cur["xy"] = (x, y)
    try:
        cv2.namedWindow("ocv-label-ball"); cv2.setMouseCallback("ocv-label-ball", on_mouse)
    except cv2.error as e:   # pragma: no cover - headless
        raise SystemExit(f"no display for the click UI ({e}); use `ocv-label-ball template` and its contact sheets") from e
    while 0 <= k < len(rows):
        r = rows[k]; f = fr[min(int(r["frame"]), len(fr) - 1)].copy()
        if "xy" in cur:
            r.update(x=str(cur["xy"][0]), y=str(cur["xy"][1]), visibility="1", status="ok"); cur.clear()
        if r.get("x"): cv2.circle(f, (int(float(r["x"])), int(float(r["y"]))), 8, (0, 255, 0) if r.get("status") == "ok" else (255, 0, 255), 1)
        cv2.putText(f, f"{k + 1}/{len(rows)} t {r['t']} {r.get('status', '')}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ocv-label-ball", f)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord(" "), ord("n")):
            if r.get("status") == "pred": r["status"] = "ok"         # accepting a prediction = checked
            k += 1
        elif key == ord("b"): k = max(0, k - 1)
        elif key == ord("0"): r.update(x="", y="", visibility="0", status="ok"); k += 1
        elif key == ord("s"): write(A.labels, rows)
        elif key == ord("q"): break
    write(A.labels, rows); cv2.destroyAllWindows()
    print(f"saved {A.labels}")


def check(A):
    rows = read(A.labels)
    bad = [r for r in rows if set(COLS) - set(r)]
    ok = [r for r in rows if r.get("status") == "ok"]
    vis = [r for r in ok if r.get("visibility") == "1"]
    print(f"{A.labels}: {len(rows)} rows, {len(ok)} checked ({len(vis)} visible, {len(ok) - len(vis)} not visible), "
          f"{sum(r.get('status') == 'pred' for r in rows)} unchecked predictions, {len(bad)} malformed")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-label-ball", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template"); t.add_argument("--video", required=True); t.add_argument("--start", type=float, required=True)
    t.add_argument("--end", type=float, required=True); t.add_argument("--clip", default="0"); t.add_argument("--every", type=int, default=3)
    t.add_argument("--prefill"); t.add_argument("--out", required=True)
    c = sub.add_parser("click"); c.add_argument("--video", required=True); c.add_argument("--labels", required=True)
    k = sub.add_parser("check"); k.add_argument("--labels", required=True)
    for p in (t, c):
        p.add_argument("--size", default="960x540"); p.add_argument("--fps", type=float, default=30.0)
    A = ap.parse_args(argv)
    return {"template": template, "click": click, "check": check}[A.cmd](A) or 0


if __name__ == "__main__":
    raise SystemExit(main())
