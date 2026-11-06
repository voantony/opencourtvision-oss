"""Pick ~30 train-split points (1-50 only) for the last-bounce marker and dump the physics tracker's guess.
Outputs (pick.json) are PRIVATE: never commit. Analysis px 1280x720; k = frame in clip (t = t0 + k/fps). Prefers points
where the tracker missed or was unsure about the final bounce; fills the rest with a spread of no-guess points.

Window (fix, 19 Oct 2026): last contact -0.1 s to min(last contact + 3.0 s, next point start - 0.3 s), the same rule as
tools/ball_track_v2/extend_clips.py. Run it on the extended clip data (extend_clips.py merge) so the window is not cut
at the old 7.5 s clip end (point 24 used to stop before its bounce)."""
import argparse
import json
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ball_tracker")); sys.path.insert(0, os.path.join(HERE, "..", "ball_track_v2"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common as C  # noqa: E402
import physball_eval as PE  # noqa: E402
from extend_clips import window_end  # noqa: E402

from opencourtvision.ball import physics3d as P3  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("data", "cands", "tracks17", "labels17", "prelabels17", "sv17", "audio", "pw_stages", "court_override"):
    ap.add_argument("--" + k)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=30); ap.add_argument("--idx")
ap.add_argument("--after", type=float, default=3.0); ap.add_argument("--gap", type=float, default=0.3)
a = ap.parse_args(); warnings.filterwarnings("ignore")
EH = PE.load(os.path.join(os.path.dirname(PE.__file__), "eval_heldout.py"), "eval_heldout")
pw = EH.load_pw(a.pw_stages)
index = json.load(open(os.path.join(a.data, "index.json")))["clips"]
PL = json.load(open(a.prelabels17)); S_SRC = int(PL.get("width", 3840)) / C.AW
cbv = pw.apply_court_override(PL["courts_by_view"], a.court_override); views = [pw.View(c) for c in cbv]
hits_all = sorted(c["t"] for r in PL["rallies"] for c in r["contacts"])
SV = json.load(open(a.sv17))["points"]; labels = {p["idx"]: p for p in json.load(open(a.labels17))["points"]}
starts = sorted(p["start"] for p in labels.values())
manual = json.load(open(a.court_override)); cfg = P3.Config(sigma=2.5)
want = set(json.load(open(a.idx))) if a.idx else None
rows = []
for idx, name in sorted((m["idx"], n) for n, m in index.items() if m["source"] == "dev_match" and m["split"] == "train"):
    if labels[idx].get("point_no", idx) > 50 or not os.path.exists(os.path.join(a.cands, name + ".npz")):
        continue
    if want is not None and idx not in want:
        continue
    ctx = EH.Ctx(argparse.Namespace(data=a.data, tracks17=a.tracks17), name, index[name], labels[idx], pw, views, hits_all, SV, S_SRC)
    PE.fix_court(ctx, manual, EH); cam = PE.camera_for(ctx, manual)
    cz = np.load(os.path.join(a.cands, name + ".npz"))["cands"].astype(float)
    cz = cz[:ctx.n] if len(cz) >= ctx.n else np.concatenate([cz, np.full((ctx.n - len(cz),) + cz.shape[1:], np.nan)])
    r = PE.run_p3(cam, cz, cfg, ctx.bx) if cam is not None else dict(pos=np.full((ctx.n, 2), np.nan), events=[], conf=np.zeros(ctx.n))
    shots = (SV.get(str(idx)) or {}).get("shots", [])
    last = shots[-1] if shots else None
    hits = [h for h in ctx.hits if h <= ctx.t0 + ctx.n / C.AFPS]
    t_last = last["t"] if last and ctx.t0 <= last["t"] <= ctx.t0 + ctx.n / C.AFPS else (hits[-1] if hits else None)
    if t_last is None:
        continue
    bo = [e for e in r["events"] if e["kind"] == "bounce" and np.isfinite(e.get("x", np.nan))]
    fb = [e for e in bo if t_last + 0.1 < ctx.t0 + e["t"] <= t_last + 2.5]
    # SV bounce is only trustworthy when we used the SV shot's time (not an audio substitute)
    sv_uv = None
    if last and abs(t_last - last["t"]) < 0.05 and last.get("bounce_xy") and cam is not None and last.get("result") in ("In", "Out"):
        sv_uv = [float(v) for v in cam.project(np.r_[last["bounce_xy"], 0.0])]
    guess = None
    if fb:
        e = fb[0]; guess = dict(k=int(e["i"]), u=float(e["x"]), v=float(e["y"]), p_in=None if e.get("p_in") is None else float(e["p_in"]))
    status, pri = "no_sv", 1.0
    if sv_uv is not None:
        if not guess:
            status, pri = "missed", 3.0
        else:
            dpx = math.hypot(guess["u"] - sv_uv[0], guess["v"] - sv_uv[1])
            status, pri = ("disagree", 2.5) if dpx > 25 else ("ok", 0.5)
    elif not guess:
        status, pri = "no_guess", 2.0
    if guess and guess["p_in"] is not None and 0.2 < guess["p_in"] < 0.8:
        pri += 0.5
    # window: last contact -0.1 s to min(+3.0 s, next point start -0.3 s), clipped to the (extended) clip
    t_end = window_end(float(t_last), starts, a.after, a.gap)
    k0 = max(0, int(round((t_last - 0.1 - ctx.t0) * C.AFPS))); k1 = min(ctx.n, int(np.floor((t_end - ctx.t0) * C.AFPS)) + 1)
    if k1 - k0 < 15:                                                     # very short: take the clip's last 2.2 s
        k0 = max(0, ctx.n - int(2.2 * C.AFPS)); k1 = ctx.n
    pos = r["pos"]
    rows.append(dict(idx=idx, point_no=labels[idx].get("point_no", idx), name=name, t0=ctx.t0, n=ctx.n, fps=C.AFPS,
                     t_last=float(t_last), last_src="sv" if last and abs(t_last - last["t"]) < 0.05 else "audio",
                     sv_result=last.get("result") if last else None, sv_uv=sv_uv, guess=guess, status=status, priority=pri,
                     k0=k0, k1=k1, ball=[[None, None] if not np.isfinite(p[0]) else [round(float(p[0]), 1), round(float(p[1]), 1)]
                                        for p in pos[k0:k1]]))
    print(idx, status, flush=True)
if want is None:
    rows.sort(key=lambda r: (-r["priority"], r["idx"]))
    # prefer missed/disagree/no_sv; fill with a spread of no_guess
    hard = [r for r in rows if r["status"] in ("missed", "disagree", "no_sv")]
    rest = [r for r in rows if r["status"] == "no_guess"]
    need = max(0, a.n - len(hard))
    fill = [rest[int(round(i))] for i in np.linspace(0, len(rest) - 1, need)] if need and rest else []
    rows = sorted(hard + fill, key=lambda r: r["idx"])[:a.n]
json.dump(dict(note="PRIVATE. Analysis px 1280x720. k = frame in the full clip (t = t0 + k/fps). Train points 1-50 only.",
               items=rows), open(a.out, "w"))
print("picked", len(rows), {s: sum(r["status"] == s for r in rows) for s in ("missed", "disagree", "no_guess", "no_sv", "ok")})
