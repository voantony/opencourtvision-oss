"""Eval scoreboard: SwingVision vs our pipeline, stage by stage, against a human-labelled match (tools/labeler).

  python benchmarks/scoreboard.py LABELS.json --sv-fixture points.json \
      --sv-clips sv_clips_raw.json --ours segments.csv --hits rallies.json [--pred NAME=segs.csv ...] [--md out.md]

Truth: the labelled points (start/end = the labeller's segment times; NOTE they were seeded from our shipped segmenter and
only edited where wrong, so boundary errors vs truth favour that segmenter; the audio-hit metrics below do not).
Predictions are interval lists on the raw video timeline:
  sv     SwingVision's exported clips, located in the raw video by audio cross-correlation (align_sv_clips()).
  ours   our segmenter's segments.csv (any CSV with start_s,end_s); --pred adds more (e.g. the PR #12 segment log).
Matching: greedy 1:1 by overlap seconds. recall = matched points / points; precision = predictions matched to a point /
all predictions; non-points = predictions whose best overlap is a labelled X (fault, let) or nothing.
Boundaries (matched points): signed start/end error vs the labelled segment. Serve contact and last contact per point come
from contacts(): SwingVision shot TIMES snapped to our audio hits (fallback: played_rally() on audio hits): cut-off serve = clip starts later than serve contact - 0.5 s;
cut-off celebration = clip ends before last contact + 1.5 s; dead time = clip seconds outside [first hit - 2.0, last hit + 3.0].
Winner / overlay score: SwingVision's per-point winner and its running score (fixture) vs the labels.
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencourtvision.labels.points import load_point_labels, match_sv_points  # noqa: E402

LEAD, TAIL, SERVE_PRE, CELEB, SERVE_GAP = 2.0, 3.0, 0.5, 1.5, 2.5


def align_sv_clips(sv_wav, env_npy, clips_csv):
    """SwingVision export clips (export timeline) -> raw-video intervals by normalised cross-correlation of the 1.5-6 kHz
    10 ms loudness envelope (same features as the hit detector; env_npy row 0)."""
    import numpy as np
    import scipy.io.wavfile as w
    import scipy.signal as s
    sr, x = w.read(sv_wav); x = x.astype(np.float32) / 32768
    b, a = s.butter(4, [1500, 6000], btype="band", fs=sr); y = s.filtfilt(b, a, x)
    hop = 160; n = len(y) // hop
    le = 20 * np.log10(np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-6)
    R = np.load(env_npy)[0]; Rz = R - np.convolve(R, np.ones(300) / 300, "same")
    out = []
    for i, r in enumerate(csv.DictReader(open(clips_csv))):
        seg = le[max(0, int(float(r["start_s"]) * 100)):int(float(r["end_s"]) * 100)]
        if len(seg) < 100:
            continue
        q = (seg - seg.mean()) / (seg.std() + 1e-9)
        c = s.fftconvolve(Rz, q[::-1], "valid") / (np.sqrt(np.convolve(Rz ** 2, np.ones(len(q)), "valid")) + 1e-9) / np.sqrt(len(q))
        k = int(np.argmax(c))
        out.append(dict(clip=i + 1, raw_start=k / 100, raw_end=k / 100 + len(q) / 100, score=float(c[k])))
    return out


def read_sv_shots(xlsx):
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True)["Shots"]
    rows = list(ws.iter_rows(values_only=True))
    return [dict(zip(rows[0], r)) for r in rows[1:] if r and r[0] is not None]


def read_intervals(path):
    if path.endswith(".json"):
        return [(float(o["raw_start"]), float(o["raw_end"])) for o in json.load(open(path)) if o]
    return [(float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(open(path))]


def read_parts(path):
    """Optional `parts` column ("a-b;c-d"): a clip shown as several spans (e.g. first-serve fault + point, jump cut).
    Returns one list of spans per row (the row's [start_s, end_s] when the column is absent or empty)."""
    if path.endswith(".json"):
        return None
    out = []
    for r in csv.DictReader(open(path)):
        ps = [tuple(float(v) for v in x.split("_")) for x in (r.get("parts") or "").split(";") if x]
        out.append(ps or [(float(r["start_s"]), float(r["end_s"]))])
    return out


def played_rally(h):
    """Hits inside a labelled point -> (serve contact, last contact) of the played rally: split at pauses > SERVE_GAP and
    take the run with the most hits (ties: the later run, e.g. the second serve of a double fault). This skips a first-serve
    fault + its return before, and ball-collection hits after."""
    runs, cur = [], [h[0]]
    for v in h[1:]:
        if v - cur[-1] > SERVE_GAP:
            runs.append(cur); cur = [v]
        else:
            cur.append(v)
    runs.append(cur)
    r = max(runs, key=lambda r: (len(r), r[0]))
    return r[0], r[-1]


def ov(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match(truth, preds):
    pairs = sorted(((ov(t, p), i, j) for i, t in enumerate(truth) for j, p in enumerate(preds) if ov(t, p) > 0), reverse=True)
    ti, pj, m = set(), set(), {}
    for o, i, j in pairs:
        if i not in ti and j not in pj:
            ti.add(i); pj.add(j); m[i] = j
    return m


def contacts(L, hits, sv_shots=None, offset=2.0, snap=0.4):
    """Per labelled point: (serve contact, last contact) of the played rally. With SwingVision shot rows: the last serve row
    (first_serve / second_serve) and the last shot of the matched SV point (+offset), snapped to an audio hit within `snap`
    s (SV's timing, not its outcome, is trusted). Points without an SV match: played_rally() over the audio hits."""
    segs = L.segments
    out = {}
    by_pt = {}
    for r in sv_shots or []:
        by_pt.setdefault(int(r["Point"]), []).append(r)
    svm = match_sv_points(segs, [dict(point=k, t_last=max(float(x["Video Time"]) for x in v)) for k, v in by_pt.items()], offset) if by_pt else {}

    def sn(t):
        c = min(hits, key=lambda v: abs(v - t)) if hits else None
        return c if c is not None and abs(c - t) <= snap else t
    for i, s in enumerate(segs):
        if s.kind != "point":
            continue
        if i in svm:
            rows = sorted(by_pt[svm[i]["point"]], key=lambda x: float(x["Video Time"]))
            serves = [x for x in rows if x["Type"] in ("first_serve", "second_serve")]
            fs = float((serves[-1] if serves else rows[0])["Video Time"]) + offset
            out[i] = (sn(fs), sn(float(rows[-1]["Video Time"]) + offset))
        else:
            h = [v for v in hits if s.start <= v <= s.end]
            if h:
                out[i] = played_rally(h)
    return out


WINDOW_PAD = 0.5


def match_windows(pts, pidx, ct, preds, pad=WINDOW_PAD):
    """Match clips to points by the played window [serve - pad, last contact + pad] (falls back to the labelled span when
    a point has no contacts). Label spans often wrap an earlier fault or 40 s of walking, so span-overlap matching can
    pair the fault clip with the point and then count the real rally clip as a non-point: one miss counted twice."""
    win = [(ct[i][0] - pad, ct[i][1] + pad) if i in ct else t for i, t in zip(pidx, pts)]
    return match(win, preds)


def stage_metrics(L, preds, hits, ct=None, parts=None, matcher="window"):
    segs = L.segments
    pidx = [i for i, s in enumerate(segs) if s.kind == "point"]
    pts = [(segs[i].start, segs[i].end) for i in pidx]
    ct = ct if ct is not None else contacts(L, hits)
    xs = [(s.start, s.end) for s in segs if s.kind == "non_point"]
    m = match_windows(pts, pidx, ct, preds) if matcher == "window" else match(pts, preds)
    used = set(m.values())
    non = 0
    for j, p in enumerate(preds):
        if j in used:
            continue
        non += 1          # unmatched prediction: a fault/let/X, a fragment of a point, or junk; all are non-points in a reel
    on_x = sum(1 for j, p in enumerate(preds) if j not in used and any(ov(p, x) > 0 for x in xs))
    se, ee, cs, cc, dead, lens = [], [], 0, 0, [], []
    for i, j in m.items():
        t, p = pts[i], preds[j]
        se.append(p[0] - t[0]); ee.append(p[1] - t[1])
        if pidx[i] in ct:
            fh, lh = ct[pidx[i]]
            cs += p[0] > fh - SERVE_PRE
            cc += p[1] < lh + CELEB
            w = (fh - LEAD, lh + TAIL)
            spans = parts[j] if parts else [p]
            dead.append(sum((q[1] - q[0]) - ov(q, w) for q in spans))
        lens.append(sum(q[1] - q[0] for q in (parts[j] if parts else [p])))
    n = len(m)
    return dict(n_pred=len(preds), recall=n / len(pts), precision=n / len(preds) if preds else 0.0, found=n, n_points=len(pts),
                start_err_med=statistics.median(abs(v) for v in se) if se else None, start_err_mean=statistics.mean(se) if se else None,
                end_err_med=statistics.median(abs(v) for v in ee) if ee else None, end_err_mean=statistics.mean(ee) if ee else None,
                cut_serve=cs, cut_celebration=cc, n_with_hits=len(dead), dead_s=statistics.mean(dead) if dead else None,
                non_points=non, non_points_on_x=on_x, clip_len=statistics.mean(lens) if lens else None)


def sv_winner_and_overlay(L, sv):
    segs, states = L.segments, L.states()
    mp = match_sv_points(segs, sv)
    on_pts = {i: q for i, q in mp.items() if segs[i].kind == "point"}
    agree = sum(("A" if q["winner"] == "host" else "B") == segs[i].winner for i, q in on_pts.items())
    ok = 0
    for i, q in on_pts.items():
        st = states[i]
        sv_state = (tuple(q["sets_before"]), tuple(q["games_before"]), (q["host_pts"], q["guest_pts"]))
        ok += sv_state == (st.sets, st.games, st.pts)
    n = sum(s.kind == "point" for s in segs)
    return dict(found=len(on_pts), n_points=n, winner_agree=agree, overlay_ok=ok)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def table(L, stages, svw, milestone):
    sv, ours = stages["sv"], stages["ours"]
    extra = [k for k in stages if k not in ("sv", "ours")]

    def b(m):
        return (f"start {m['start_err_mean']:+.1f} s (med \\|{m['start_err_med']:.1f}\\|), end {m['end_err_mean']:+.1f} s "
                f"(med \\|{m['end_err_med']:.1f}\\|)")

    def row(name, f_sv, f_ours, target, ms, fx=None):
        cells = [name, f_sv, f_ours, f_ours] + [fx(stages[k]) if fx else "n/a" for k in extra] + [target, ms]
        return "| " + " | ".join(cells) + " |"
    hdr = ["Stage", "SwingVision", "Ours now, US build", "Ours now, EU build"] + [f"{k} (branch)" for k in extra] + ["Target", "Milestone"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    rp = lambda m: f"{m['found']}/{m['n_points']} = {pct(m['found'], m['n_points'])} / {pct(m['found'], m['n_pred'])}"  # noqa: E731
    out.append(row("Point detection: recall / precision", rp(sv), rp(ours), "≥ 98% / ≥ 97%", milestone["E"], rp))
    out.append(row("Boundaries vs labelled segment (signed mean; median abs)", b(sv), b(ours), "\\|err\\| ≤ 1 s", milestone["E"], b))
    cut = lambda m: f"{m['cut_serve']} / {m['cut_celebration']} of {m['n_with_hits']}"  # noqa: E731
    out.append(row("Cut-off serves / cut-off celebrations", cut(sv), cut(ours), "0 / ≤ 1%", milestone["E"], cut))
    dd = lambda m: f"{m['dead_s']:.1f} s (clip {m['clip_len']:.1f} s)"  # noqa: E731
    out.append(row("Dead time per point (outside first hit − 2 s … last hit + 3 s)", dd(sv), dd(ours), "≤ 1 s", milestone["E"], dd))
    nn = lambda m: f"{m['non_points']} of {m['n_pred']} ({m['non_points_on_x']} on faults/lets)"  # noqa: E731
    out.append(row("Non-points in the points reel", nn(sv), nn(ours), "≤ 2% (MD)", milestone["E"], nn))
    out.append(row("Winner accuracy, unaided (first call)", f"{svw['winner_agree']}/{svw['found']} = {pct(svw['winner_agree'], svw['found'])} "
                   f"({pct(svw['winner_agree'], svw['n_points'])} of all {svw['n_points']})", "n/a (no vision winner yet)",
                   "≥ 98% (EU, first call)", milestone["S"]))
    out.append(row("Score shown correct on overlays (before each point)", f"{svw['overlay_ok']}/{svw['found']} = {pct(svw['overlay_ok'], svw['found'])}",
                   "n/a (editable score; no auto winner)", "≥ 98% (EU, first call)", milestone["S"]))
    out.append(row("Taps per 100 points to reach 99.9%", f"no confirm flow: review all ~100, fix ~{svw['found'] - svw['winner_agree'] + svw['n_points'] - svw['found']}",
                   "100 (every point tapped)", "≤ 5 (US, forward-only)", milestone["S"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--sv-fixture", required=True)
    ap.add_argument("--sv-clips", required=True, help="sv_clips_raw.json from align_sv_clips (or a CSV of raw intervals)")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--sv-xlsx", help="SwingVision export workbook (Shots sheet) for serve / last-contact times")
    ap.add_argument("--hits", required=True, help="rallies.json with per-rally hit 'times' (raw timeline)")
    ap.add_argument("--pred", action="append", default=[], help="NAME=path.csv extra segmenter outputs")
    ap.add_argument("--milestone-e", default="Gate E, wk 5")
    ap.add_argument("--milestone-s", default="Gate S, wk 12")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    L = load_point_labels(a.labels)
    hits = sorted(t for r in json.load(open(a.hits)) for t in r.get("times", []))
    shots = read_sv_shots(a.sv_xlsx) if a.sv_xlsx else None
    ct = contacts(L, hits, shots)
    stages = {"sv": stage_metrics(L, read_intervals(a.sv_clips), hits, ct), "ours": stage_metrics(L, read_intervals(a.ours), hits, ct, read_parts(a.ours))}
    for p in a.pred:
        k, v = p.split("=", 1)
        stages[k] = stage_metrics(L, read_intervals(v), hits, ct, read_parts(v))
    svw = sv_winner_and_overlay(L, json.load(open(a.sv_fixture))["points"])
    md = table(L, stages, svw, {"E": a.milestone_e, "S": a.milestone_s})
    print(md)
    if a.md:
        open(a.md, "w").write(md + "\n")
    if a.json:
        json.dump(dict(stages=stages, sv=svw), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
