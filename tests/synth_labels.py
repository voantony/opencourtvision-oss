"""Synthetic ocv.point_labels.v1 labels for tests, built in memory.

Segments and suggestions come from tests/synthetic_match.py (a seeded rules match, not a recording).
Every seeded point is labelled a point with the suggested winner; the warm-up segment is a non_point.
Server, side and score_before are then derived by the rules, so verify() is clean.
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthetic_match import MATCH  # noqa: E402

_cache = {}


def synthetic_point_labels():
    if "raw" not in _cache:
        sys.path.insert(0, os.path.join(HERE, "..", "tools", "labeler"))
        from make_seed import build_seed

        from opencourtvision.labels.points import SCHEMA, parse_point_labels
        from opencourtvision.scoring.decoder import score_call, serve_side

        seed = build_seed(MATCH["segments"], MATCH["points"])
        pts = []
        for i, s in enumerate(seed["points"], 1):
            is_pt = "sv_point" in s
            pts.append(dict(idx=i, start=s["start"], end=s["end"], kind="point" if is_pt else "non_point",
                            winner=s.get("sugg_winner") if is_pt else None,
                            winner_src="user_accepted_suggestion" if is_pt else "user", server_src="rules",
                            notes="", server_override=None, sv_point=s.get("sv_point"), sugg_winner=s.get("sugg_winner"),
                            sugg_server=s.get("sugg_server"), sugg_kind=s.get("sugg_kind")))
        raw = {"schema": SCHEMA, "video": {"name": "synthetic.mp4", "duration": MATCH["duration_s"]},
               "match": {"date": "2026-01-01", "sport": "tennis", "preset": "advantage",
                         "teams": {"A": "Host (my side)", "B": "Guest"}, "mine": "A",
                         "first_server": seed["first_server"], "final_hint": MATCH["final_score"]},
               "labeler": {"version": 1, "synthetic": True}, "points": pts, "decoded": {}}
        L = parse_point_labels(raw)
        for p, st in zip(raw["points"], L.states()):
            p["server"], p["side"] = st.server, serve_side(st, L.fmt)
            p["score_before"] = dict(sets=list(st.sets), games=list(st.games), pts=list(st.pts), tiebreak=bool(st.tb),
                                     call=score_call(st, L.fmt))
        raw["decoded"] = dict(score=L.final_score(), done=L.states()[-1].done, provisional=False,
                              n_segments=len(pts), n_points=sum(p["kind"] == "point" for p in pts))
        _cache["raw"] = raw
    return copy.deepcopy(_cache["raw"])


def write_synthetic_point_labels(path):
    import json
    with open(path, "w") as f:
        json.dump(synthetic_point_labels(), f)
    return str(path)
