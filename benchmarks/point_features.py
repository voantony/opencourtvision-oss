"""Per-point features for reel work (one row per labelled point), from a labelled match + our audio + SwingVision.

  python benchmarks/point_features.py LABELS.json --sv-fixture sv_points.json --sv-xlsx export.xlsx \
      --sv-clips sv_clips_raw.json --ours segments.csv --hits rallies.json --env env.npy --out features.json

Row: start/end (labelled), winner, server, score before / after (sets, games, points, call), game / break / set / match point
flags, run of points (streak per team before the point), rally length (audio hits in the played rally, all audio hits in
the segment, SwingVision shots if matched), serve / last contact times, audio reaction (broadband level peak in
[last contact + 0.6, + 5 s] minus the median of the 3 s before the serve, and its time after the last contact), favourite,
and our segment's / SwingVision's clip boundaries vs the labelled ones. Output is private when built from real labels:
keep it out of git.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoreboard import contacts, match, played_rally, read_intervals, read_sv_shots  # noqa: E402

from opencourtvision.labels.points import load_point_labels  # noqa: E402
from opencourtvision.scoring.decoder import score_call, serve_side  # noqa: E402


def reaction(lb, t_serve, t_last, hz=100):
    import numpy as np
    a, b = int((t_last + 0.6) * hz), int((t_last + 5.0) * hz)
    pre = lb[max(0, int((t_serve - 3.0) * hz)):max(1, int(t_serve * hz))]
    if b <= a or len(pre) == 0 or b > len(lb):
        return None, None
    w = lb[a:b]
    k = int(np.argmax(w))
    return round(float(w[k] - np.median(pre)), 1), round((a + k) / hz - t_last, 2)


def build(L, sv_points, shots, sv_clips, ours, hits, lb=None):
    segs = L.segments
    states = L.states()
    fmt = L.fmt
    hr = {r["idx"]: r for r in L.highlight_rows(sv_points)}
    ct = contacts(L, hits, shots)
    pidx = [i for i, s in enumerate(segs) if s.kind == "point"]
    pts = [(segs[i].start, segs[i].end) for i in pidx]
    m_ours, m_sv = match(pts, ours), match(pts, sv_clips) if sv_clips else {}

    def st(x):
        return dict(sets=list(x.sets), games=list(x.games), pts=list(x.pts), server=x.server, call=None if x.done else score_call(x, fmt),
                    done=x.done)
    streak = {"A": 0, "B": 0}
    rows = []
    for k, i in enumerate(pidx):
        s = segs[i]; r = hr.get(s.idx, {})
        nxt = next((states[j] for j in range(i + 1, len(states)) if j == len(segs) or segs[j].kind == "point" or j == len(states) - 1), states[-1])
        h = [v for v in hits if s.start <= v <= s.end]
        pr = played_rally(h) if h else None
        fh, lh = ct.get(i, (None, None))
        row = dict(point=k + 1, idx=s.idx, start=s.start, end=s.end, winner=s.winner, winner_is_mine=s.winner == L.match.get("mine", "A"),
                   server=states[i].server, serve_side=serve_side(states[i], fmt), game_state=r.get("game_state"), score_before=st(states[i]), score_after=st(nxt),
                   game_point_for=r.get("game_point_for"), break_point=r.get("break_point"), set_point=r.get("set_point"),
                   match_point=r.get("match_point"), streak_before=dict(streak),
                   audio_hits_rally=len([v for v in h if pr and pr[0] <= v <= pr[1]]) if pr else 0, audio_hits_segment=len(h),
                   sv_shots=r.get("rally_shots") if r.get("has_sv") else None, sv_detail=r.get("detail") or None,
                   serve_contact=fh, last_contact=lh, favorite=s.favorite, fav_seen=s.fav_seen, notes=s.notes,
                   ranker_pq=r.get("pq"), great_lost=r.get("great_lost"))
        if lb is not None and fh is not None:
            row["reaction_db"], row["reaction_t_after_last"] = reaction(lb, fh, lh)
        for name, mm, pred in (("ours", m_ours, ours), ("sv", m_sv, sv_clips)):
            if k in mm:
                p = pred[mm[k]]
                row[f"{name}_segment"] = dict(start=round(p[0], 3), end=round(p[1], 3), start_err=round(p[0] - s.start, 2),
                                             end_err=round(p[1] - s.end, 2),
                                             lead_before_serve=round(fh - p[0], 2) if fh else None,
                                             tail_after_last=round(p[1] - lh, 2) if lh else None)
            else:
                row[f"{name}_segment"] = None
        rows.append(row)
        w, o = s.winner, ("B" if s.winner == "A" else "A")
        streak = {w: streak[w] + 1, o: 0}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--sv-fixture", required=True)
    ap.add_argument("--sv-xlsx")
    ap.add_argument("--sv-clips")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--hits", required=True)
    ap.add_argument("--env", help="env.npy from the hit detector (rows: band dB, background, broadband dB)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = load_point_labels(a.labels)
    hits = sorted(t for r in json.load(open(a.hits)) for t in r.get("times", []))
    lb = None
    if a.env:
        import numpy as np
        lb = np.load(a.env)[2]
    rows = build(L, json.load(open(a.sv_fixture))["points"], read_sv_shots(a.sv_xlsx) if a.sv_xlsx else None,
                 read_intervals(a.sv_clips) if a.sv_clips else [], read_intervals(a.ours), hits, lb)
    meta = dict(schema="ocv.point_features.v1", video=L.video.get("name"), date=L.match.get("date"), teams=L.match.get("teams"),
                mine=L.match.get("mine"), final=L.final_score(), n_points=len(rows),
                notes="times in seconds on the raw video; streak_before = consecutive points won by each team before this point; "
                      "reaction_db relative to the 3 s before the serve")
    json.dump(dict(meta=meta, points=rows), open(a.out, "w"), indent=1)
    print(f"{a.out}: {len(rows)} points")


if __name__ == "__main__":
    main()
