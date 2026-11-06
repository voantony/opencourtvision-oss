"""Extend tracker clip windows past the last contact (same rule as the bounce-marker fix).

The dev_match tracker clips are the last 7.5 s of each labelled point, so when the point label ends early the clip
stops before the final bounce (point 24: last contact at frame ~209 of 225). New clip end:
    min(last contact + AFTER s, next point start - GAP s)      (defaults 3.0 s / 0.3 s: the next-serve cue)
Last contact = the point's last SwingVision shot when given (--sv), else its last audio contact (--prelabels).
Only appends frames; frame k keeps its time t0 + k / fps, so existing labels and candidates stay valid.

  plan    (any machine)  -> PLAN.json: clips to extend, old/new frame counts
  decode  (Mac, video)   -> OUT data dir: index.json (planned clips, new n), clips/ + frames/ (old frames linked,
                            new ones decoded from the source video at 768x432 like the clip frames)
  serve_plan (any)       -> PLAN.json of NEW clips around each point's first audio hit (the serve): first hit - PRE s
                            .. + POST s, for serve-end cues from the ball (decode them with `decode`)
  merge   (box)          -> OUT data dir over SRC data (all clips, new n for planned ones; files linked), and
                            optionally a merged candidates dir (--cands SRC --cands_ext EXT --cands_out DIR)
Outputs reference private footage/labels; never commit them.
"""
import argparse
import json
import os
import subprocess

import numpy as np


def window_end(t_last, starts, after=3.0, gap=0.3):
    """End time of a clip / marker window: last contact + after s, but stop gap s before the next point starts."""
    nxt = [s for s in starts if s > t_last + 0.5]
    return min(t_last + after, min(nxt) - gap) if nxt else t_last + after


def last_contact(p, sv, contacts):
    shots = sorted((sv.get(str(p["idx"])) or {}).get("shots", []), key=lambda s: s["t"]) if sv else []
    if shots:
        return float(shots[-1]["t"]), "sv"
    hp = [c for c in contacts if p["start"] <= c <= p["end"] + 1.0]
    return (float(hp[-1]), "audio") if hp else (None, None)


def plan(a):
    idx = json.load(open(os.path.join(a.data, "index.json")))["clips"]
    pts = sorted(json.load(open(a.labels))["points"], key=lambda p: p["start"]); by = {p["idx"]: p for p in pts}
    starts = [p["start"] for p in pts]
    sv = json.load(open(a.sv))["points"] if a.sv else None
    contacts = sorted(c["t"] for r in json.load(open(a.prelabels))["rallies"] for c in r["contacts"]) if a.prelabels else []
    out = []
    for name, m in sorted(idx.items()):
        if m.get("source") != a.source or m["idx"] not in by:
            continue
        z = np.load(os.path.join(a.data, "clips", name + ".npz")); t0, fps = float(z["t0"]), float(z["fps"])
        tl, src = last_contact(by[m["idx"]], sv, contacts)
        if tl is None:
            continue
        n_new = int(np.floor((window_end(tl, starts, a.after, a.gap) - t0) * fps)) + 1
        if n_new > m["n"]:
            out.append(dict(name=name, idx=m["idx"], split=m.get("split"), t0=t0, fps=fps, n_old=m["n"], n_new=n_new,
                            t_last=round(tl, 3), last_src=src))
    json.dump(dict(after_s=a.after, gap_s=a.gap, clips=out), open(a.out, "w"), indent=1)
    print(f"{len(out)} clips to extend, +{sum(c['n_new'] - c['n_old'] for c in out)} frames")


def serve_plan(a):
    pts = sorted(json.load(open(a.labels))["points"], key=lambda p: p["start"])
    contacts = sorted(c["t"] for r in json.load(open(a.prelabels))["rallies"] for c in r["contacts"])
    out = []
    for p in pts:
        if p.get("kind", "point") != "point":
            continue
        h = [c for c in contacts if p["start"] <= c <= p["end"]]
        if not h:
            continue
        ta = h[0] if a.anchor == "first_hit" else p["start"]   # audio often misses the serve itself: anchor at the segment start
        t0 = round(ta - a.pre, 3); n = int(round((a.pre + a.post) * a.fps))
        out.append(dict(name=f"{a.source}_{'srv' if a.anchor == 'first_hit' else 'srs'}{p['idx']:03d}", idx=p["idx"], split=None, t0=t0, fps=a.fps, n_old=0, n_new=n,
                        t_first_hit=round(h[0], 3), t_anchor=round(ta, 3), anchor=a.anchor))
    json.dump(dict(pre_s=a.pre, post_s=a.post, clips=out), open(a.out, "w"), indent=1)
    print(f"{len(out)} serve windows, {sum(c['n_new'] for c in out)} frames")


def _link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def decode(a):
    P = json.load(open(a.plan))["clips"]
    src_idx = json.load(open(os.path.join(a.data, "index.json")))["clips"] if a.data else {}
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True); idx = {}
    for c in P:
        nm = c["name"]; fd = os.path.join(a.out, "frames", nm); os.makedirs(fd, exist_ok=True)
        if nm in src_idx:
            _link(os.path.join(a.data, "clips", nm + ".npz"), os.path.join(a.out, "clips", nm + ".npz"))
        else:                                                   # a new window (serve_plan): its own clip header
            np.savez(os.path.join(a.out, "clips", nm + ".npz"), t0=c["t0"], fps=c["fps"], Hc2i=np.full((3, 3), np.nan))
        for k in range(c["n_old"]):
            _link(os.path.join(a.data, "frames", nm, f"{k:05d}.jpg"), os.path.join(fd, f"{k:05d}.jpg"))
        k0, want = c["n_old"], c["n_new"] - c["n_old"]
        if not os.path.exists(os.path.join(fd, f"{c['n_new'] - 1:05d}.jpg")):
            cmd = [a.ffmpeg, "-nostdin", "-loglevel", "error", "-ss", f"{c['t0'] + k0 / c['fps']:.3f}", "-i", a.video,
                   "-frames:v", str(want), "-vf", f"fps={c['fps']},scale=768:432:flags=area", "-q:v", "3",
                   "-start_number", str(k0), os.path.join(fd, "%05d.jpg")]
            subprocess.run(cmd, check=True)
        n = len([f for f in os.listdir(fd) if f.endswith(".jpg")])
        base = src_idx.get(nm, dict(source=nm.split("_")[0], idx=c["idx"], split=c.get("split")))
        idx[nm] = dict(base, n=n, n_orig=c["n_old"]); print(nm, c["n_old"], "->", n, flush=True)
    json.dump(dict(clips=idx), open(os.path.join(a.out, "index.json"), "w"), indent=1)


def merge(a):
    P = {c["name"]: c for c in json.load(open(a.plan))["clips"]}
    d = json.load(open(os.path.join(a.data, "index.json")))
    for nm, c in P.items():
        if nm in d["clips"]:
            d["clips"][nm] = dict(d["clips"][nm], n=c["n_new"], n_orig=c["n_old"])
    os.makedirs(a.out, exist_ok=True)
    for sub in ("clips", "frames"):
        if os.path.isdir(os.path.join(a.data, sub)):
            _link(os.path.join(a.data, sub), os.path.join(a.out, sub))
    json.dump(d, open(os.path.join(a.out, "index.json"), "w"))
    if a.cands:
        os.makedirs(a.cands_out, exist_ok=True)
        for f in os.listdir(a.cands):
            if f.endswith(".npz"):
                e = os.path.join(a.cands_ext, f)
                _link(e if (f[:-4] in P and os.path.exists(e)) else os.path.join(a.cands, f), os.path.join(a.cands_out, f))
    print("merged", len(P), "extended clips")


def main():
    ap = argparse.ArgumentParser(); s = ap.add_subparsers(dest="cmd", required=True)
    p = s.add_parser("plan"); p.add_argument("--data", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--sv"); p.add_argument("--prelabels"); p.add_argument("--source", default="dev_match")
    p.add_argument("--after", type=float, default=3.0); p.add_argument("--gap", type=float, default=0.3); p.add_argument("--out", required=True)
    v = s.add_parser("serve_plan"); v.add_argument("--labels", required=True); v.add_argument("--prelabels", required=True)
    v.add_argument("--source", default="dev_match"); v.add_argument("--pre", type=float, default=0.5); v.add_argument("--post", type=float, default=2.0)
    v.add_argument("--fps", type=float, default=30.0); v.add_argument("--out", required=True)
    v.add_argument("--anchor", choices=("first_hit", "start"), default="first_hit")
    q = s.add_parser("decode"); q.add_argument("--plan", required=True); q.add_argument("--data")
    q.add_argument("--video", required=True); q.add_argument("--ffmpeg", default="ffmpeg"); q.add_argument("--out", required=True)
    r = s.add_parser("merge"); r.add_argument("--plan", required=True); r.add_argument("--data", required=True); r.add_argument("--out", required=True)
    r.add_argument("--cands"); r.add_argument("--cands_ext"); r.add_argument("--cands_out")
    a = ap.parse_args(); dict(plan=plan, serve_plan=serve_plan, decode=decode, merge=merge)[a.cmd](a)


if __name__ == "__main__":
    main()
