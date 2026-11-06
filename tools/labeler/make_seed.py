#!/usr/bin/env python3
"""Pre-seed a labeler session from what we already know about a match (stdlib only).

  python3 make_seed.py --segments segments.csv [--sv sv_points.json] --video "match.MP4" --date 2026-10-17 --out seed.json

segments.csv: our rally segmenter output (idx,start_s,end_s,...; times on the video timeline).
sv_points.json: story.highlight.points_from_swingvision() output ({"points": [{winner, server, t_last, ...}]}), whose
t_last is SwingVision "Video Time" (video time = t_last + --sv-offset, 2.00 s on dev_match). Matched SwingVision points give
SUGGESTED winners/servers only: the labeler shows them greyed and they are not ground truth until confirmed.
Teams: A = host (the recording user's side, as in scoring/decoder.py), B = guest.
"""
import argparse
import csv
import json
import os


def build_seed(segments, sv_points=None, sv_offset=2.0, video=None, date=None, final=None, tol=5.0):
    segs = [dict(start=round(float(r["start_s"]), 3), end=round(float(r["end_s"]), 3), src_idx=int(r["idx"])) for r in segments]
    side = {"host": "A", "guest": "B"}
    used = set()
    for p in sv_points or []:
        t = float(p["t_last"]) + sv_offset
        def gap(s):
            return 0.0 if s["start"] <= t <= s["end"] else min(abs(t - s["start"]), abs(t - s["end"]))
        cands = [i for i, s in enumerate(segs) if gap(s) <= tol and i not in used]
        if cands:
            i = min(cands, key=lambda j: gap(segs[j]))
        else:                                  # the segmenter missed this point: add a segment around SwingVision's
            span = float(p.get("rally_span_s") or 0.0)
            segs.append(dict(start=round(max(0.0, t - span - 6.0), 3), end=round(t + 2.0, 3), src_idx=None, sv_only=True))
            i = len(segs) - 1
        used.add(i)
        segs[i]["sugg_winner"] = side.get(p.get("winner"))
        segs[i]["sugg_server"] = side.get(p.get("server"))
        segs[i]["sv_point"] = p.get("point")
        if p.get("detail"):
            segs[i]["sv_detail"] = p["detail"]
    segs.sort(key=lambda s: s["start"])
    for a, b in zip(segs, segs[1:]):          # one segment held two SwingVision points: split it where the 2nd starts
        if a["end"] > b["start"]:
            a["end"] = round(max(a["start"] + 1.0, b["start"] - 0.05), 3)
            a["split_by_seed"] = True
    for s in segs:
        if "sv_point" not in s:
            s["sugg_kind"] = "non_point?"     # no SwingVision point: often a fault, let, or a cut-short rally
    return {"schema": "ocv.labeler_seed.v1", "video": video, "date": date, "sport": "tennis", "preset": "advantage",
            "final_hint": final, "teams": {"A": "Host (my side)", "B": "Guest"},
            "first_server": next((s["sugg_server"] for s in segs if s.get("sugg_server")), None), "points": segs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--sv")
    ap.add_argument("--sv-offset", type=float, default=2.0)
    ap.add_argument("--video")
    ap.add_argument("--date")
    ap.add_argument("--final")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    segs = list(csv.DictReader(open(a.segments)))
    sv = json.load(open(a.sv))["points"] if a.sv else None
    seed = build_seed(segs, sv, a.sv_offset, a.video and os.path.basename(a.video), a.date, a.final)
    json.dump(seed, open(a.out, "w"), indent=1)
    n = sum("sv_point" in s for s in seed["points"])
    print(f"{a.out}: {len(seed['points'])} segments, {n} with SwingVision suggestions")


if __name__ == "__main__":
    main()
