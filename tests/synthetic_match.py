"""Synthetic doubles match for tests. No real match, SwingVision export, or person is involved.

A finished tennis match (advantage, best of 3) is played with a fixed seed. Score columns come from
the same rules state machine the decoder uses, so label verify() agrees. Rally features are invented.
"""
from __future__ import annotations

import random

from opencourtvision.scoring.decoder import advance, initial_state
from opencourtvision.scoring.formats import get_format

FMT = get_format("tennis", "advantage")
FIRST_SERVER = "B"          # guest serves the first game
SEED = 1


def _play(seed=SEED):
    rng = random.Random(seed)
    st = initial_state(FMT, FIRST_SERVER)
    winners, before = [], []
    while not st.done and len(winners) < 400:
        before.append(st)
        w = "A" if rng.random() < 0.57 else "B"
        winners.append(w)
        st = advance(st, w, FMT)
    if not st.done:
        raise RuntimeError("synthetic match did not finish")
    return winners, before, st


def _final_score(before, winners):
    sets, prev = [], None
    states = list(before) + [advance(before[-1], winners[-1], FMT)]
    for cur in states:
        if prev is not None and cur.sets != prev.sets:
            g = list(prev.games)
            g[0 if cur.sets[0] > prev.sets[0] else 1] += 1
            sets.append(f"{g[0]}-{g[1]}")
        prev = cur
    return " ".join(sets)


def build(seed=SEED):
    winners, before, _final = _play(seed)
    rng = random.Random(seed + 99)
    side = {"A": "host", "B": "guest"}
    points, segments = [], []
    # A warm-up segment with no scored point, so labeler tests have a non-point.
    segments.append({"idx": "0", "start_s": "5.000", "end_s": "12.000"})
    for i, (w, st) in enumerate(zip(winners, before), 1):
        start = 20.0 + (i - 1) * 4.0
        end = start + 3.2
        shots = rng.choice([0, 1, 2, 3, 4, 6, 8, 12, 16])
        if shots == 0:
            ending, detail = "double_fault", "Double Fault"
        elif shots <= 2 and rng.random() < 0.4:
            ending, detail = "service_winner", "Service Winner"
        elif shots >= 8 and rng.random() < 0.5:
            ending, detail = "winner", "Forehand Winner"
        else:
            ending, detail = "unforced_error", "Backhand Unforced Error"
        react = rng.choice([None, 1.2, 4.0, 6.3, 9.5, 14.0])
        points.append({
            "point": i,
            "server": side[st.server],
            "winner": side[w],
            "host_pts": int(st.pts[0]),
            "guest_pts": int(st.pts[1]),
            "games_before": [int(st.games[0]), int(st.games[1])],
            "sets_before": [int(st.sets[0]), int(st.sets[1])],
            "rally_shots": shots,
            "rally_span_s": round(shots * 0.85, 2),
            "fast_exchanges": 1 if shots >= 8 else 0,
            "net_exchange": 2 if shots >= 12 else 0,
            "ending": ending,
            "detail": detail,
            "react_db": react,
            "react_dur": None,
            "high_difficulty": shots >= 12,
            "game": int(st.games[0] + st.games[1] + 1),
            "set": int(st.sets[0] + st.sets[1] + 1),
            "t_last": round(start + 1.0, 3),   # + sv offset 2.0 lands inside [start, end]
        })
        segments.append({"idx": str(i), "start_s": f"{start:.3f}", "end_s": f"{end:.3f}"})
    return {
        "about": "Synthetic tennis doubles. Not a recorded match.",
        "n_points": len(points),
        "first_server": FIRST_SERVER,
        "final_score": _final_score(before, winners),
        "final_sets": [int(_final.sets[0]), int(_final.sets[1])],
        "points": points,
        "segments": segments,
        "duration_s": float(segments[-1]["end_s"]) + 2.0,
    }


MATCH = build()
