"""ocv-pose: player pose -> strokes, stroke mechanics, footwork, movement syllables, and group comparisons.

Subcommands:
  extract  VIDEO --out kp.npz      run a pose backend (mediapipe | apple-vision) over the video, optionally only near hits
  analyze  KP.npz --out pose.json  strokes (serve/forehand/backhand) + mechanics + footwork + syllables + verified events
  compare  POSE.json ...           era/group comparison of stroke mechanics (Welch t-test, Holm, effect-size gate)
  demo     --out-dir DIR           synthetic player (no footage) through analyze, to try it without a video

Examples:
  ocv-pose extract match.mp4 --rallies work/rallies.json --out work/kp_near.npz
  ocv-pose analyze work/kp_near.npz --rallies work/rallies.json --court 412,1820,3420,1830,2610,640,1240,636 --out work/pose.json
  ocv-pose compare spring/*.json:spring autumn/*.json:autumn --metrics contact_cm finish_cm --kind forehand
"""
import argparse
import glob
import json
import os

import numpy as np

from ..stages import StageCache


def _court(s):
    from ..pose.court import Court
    return Court.parse(s) if s else None


def _hits(path):
    from ..windows import hit_times
    return hit_times(json.load(open(path))) if path else None


def cmd_extract(A):
    import cv2

    from ..pose.estimate import BACKENDS, extract, save_keypoints
    from ..windows import frame_mask, windows
    cap = cv2.VideoCapture(A.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    mask = frame_mask(n, fps, windows(_hits(A.rallies))) if A.rallies else None
    est = BACKENDS[A.backend]() if A.backend != "mediapipe" else BACKENDS["mediapipe"](A.variant)
    region = tuple(float(v) for v in A.region.split(",")) if A.region else None

    def frames():
        i = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            yield f if mask is None or (i < len(mask) and mask[i]) else np.zeros((1, 1, 3), np.uint8)
            i += 1

    class Gate:                                       # skip inference on frames outside the hit windows
        def __call__(self, f): return [] if f.shape[0] == 1 else est(f)

    kp, conf, fps_out = extract(frames(), Gate(), fps, A.stride, region)
    save_keypoints(A.out, kp, conf, fps_out, size, dict(backend=A.backend, stride=A.stride))
    print(json.dumps(dict(out=A.out, frames=len(kp), fps=fps_out, detected=float(np.isfinite(kp[:, 7, 0]).mean()))))


def cmd_analyze(A):
    from ..pose.analyze import analyze
    from ..pose.estimate import load_keypoints
    d = load_keypoints(A.keypoints)
    params = dict(rallies=A.rallies, court=A.court, hand=A.hand, k=A.k, syllables=not A.no_syllables)
    fn = lambda: analyze(d["kp"], d["fps"], _hits(A.rallies), _court(A.court), A.hand, A.k, do_syllables=not A.no_syllables)
    inputs = [A.keypoints] + ([A.rallies] if A.rallies else [])
    res, key, hit = StageCache(A.work).run("pose", fn, inputs, params) if A.work else (fn(), None, False)
    if A.out: json.dump(res, open(A.out, "w"), indent=1, default=float)
    ev = res["events"]
    print(json.dumps(dict(strokes=len(res["strokes"]), verified_events=sum(e["verified"] for e in ev), events=len(ev),
                          footwork=res["footwork"], syllable_instances=len(res.get("syllables", {}).get("instances", [])),
                          cache=dict(key=key, hit=hit) if key else None, out=A.out), indent=1))


def cmd_compare(A):
    from ..pose.compare import compare
    from ..pose.metrics import STROKE_METRICS
    rows = []
    for spec in A.inputs:
        pat, _, group = spec.rpartition(":") if ":" in spec else (spec, "", None)
        for p in sorted(glob.glob(pat)):
            r = json.load(open(p)); g = group or os.path.basename(os.path.dirname(p)) or "all"
            rows += [dict(s, session=p, group=g) for s in r["strokes"] if not A.kind or s["kind"] == A.kind]
    res = compare(rows, A.metrics or STROKE_METRICS, a=A.a, b=A.b, min_effect=A.min_effect)
    for r in res:
        f = lambda v: f"{v:8.2f}" if v is not None else "       -"
        print(f"{r['metric']:<12} {r['a']}: {f(r['mean_a'])} (n={r['n_sessions_a']})  {r['b']}: {f(r['mean_b'])} (n={r['n_sessions_b']})  "
              f"p_holm={r['p_holm']:.4f} {r['stars']:<3} {'CHANGED' if r['changed'] else ''}")
    if A.out: json.dump(res, open(A.out, "w"), indent=1, default=float)


def cmd_demo(A):
    from ..pose import synthetic as syn
    from ..pose.analyze import analyze
    from ..pose.court import Court
    os.makedirs(A.out_dir, exist_ok=True)
    r = syn.rally([dict(t=1.5, kind="serve", lean=25), dict(t=4.0, kind="forehand", stance=50), dict(t=6.5, kind="backhand", stance=20),
                   dict(t=9.0, kind="forehand", stance=50)])
    res = analyze(r["kp"], r["fps"], r["hits"], Court(r["court"]))
    p = os.path.join(A.out_dir, "pose_demo.json"); json.dump(res, open(p, "w"), indent=1, default=float)
    for s in res["strokes"]:
        print(f"{s['t']:5.2f}s {s['kind']:<9} conf {s['confidence']:.2f} stance {s['stance_deg']}° contact {s['contact_cm']} cm "
              f"finish {s['finish_cm']} cm lean {s['lean_cm']}")
    print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-pose", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="video -> keypoints .npz")
    e.add_argument("video"); e.add_argument("--out", required=True)
    e.add_argument("--backend", choices=("mediapipe", "apple-vision"), default="mediapipe")
    e.add_argument("--variant", choices=("lite", "full", "heavy"), default="full", help="MediaPipe model size (downloaded at runtime)")
    e.add_argument("--rallies", help="rallies.json / hits JSON: run pose only in windows around hits")
    e.add_argument("--stride", type=int, default=1, help="run every Nth frame")
    e.add_argument("--region", help="x0,y0,x1,y1 px: only track a player whose hips are inside (e.g. the near half)")
    e.set_defaults(fn=cmd_extract)
    a = sub.add_parser("analyze", help="keypoints -> strokes, mechanics, footwork, syllables, events")
    a.add_argument("keypoints"); a.add_argument("--out")
    a.add_argument("--rallies", help="rallies.json / hits JSON from ocv-swingcut (contact times; enables verification)")
    a.add_argument("--court", help="court corners px: near-left,near-right,far-right,far-left as x1,y1,...,x4,y4")
    a.add_argument("--hand", choices=("right", "left"), default="right")
    a.add_argument("--k", type=int, default=12, help="number of movement syllables")
    a.add_argument("--no-syllables", action="store_true")
    a.add_argument("--work", help="work dir for the content-hash stage cache")
    a.set_defaults(fn=cmd_analyze)
    c = sub.add_parser("compare", help="compare stroke mechanics between two groups of sessions")
    c.add_argument("inputs", nargs="+", help="pose JSON globs, each optionally suffixed :GROUP")
    c.add_argument("--metrics", nargs="*"); c.add_argument("--kind", choices=("forehand", "backhand", "serve"))
    c.add_argument("--a"); c.add_argument("--b"); c.add_argument("--min-effect", type=float, default=0.4); c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)
    d = sub.add_parser("demo", help="run analyze on a synthetic player")
    d.add_argument("--out-dir", default="work/pose_demo"); d.set_defaults(fn=cmd_demo)
    A = ap.parse_args(argv)
    A.fn(A)


if __name__ == "__main__":
    main()
