"""Apple Watch score taps -> weak labels.

Each tap is {t_s, team: 'us'|'them', kind: 'point'|'undo'|'highlight'|'record_start'|'record_stop', source: 'watch'}.
Taps are stored under session/taps/ and converted to weak point-winner labels (confidence 0.6) that the score engine and
the personal retrain can use. Undo removes the previous point tap. Highlights become favorite flags for the ranker /
slow-mo eligibility. Never blocks the in-order score check-in UI.
"""
import json
import os
import time
import uuid


def ingest(session, taps):
    """Append taps (list or path to JSON) and refresh weak labels on session points. Returns summary."""
    if isinstance(taps, str):
        taps = json.load(open(taps))
    if isinstance(taps, dict):
        taps = taps.get("taps") or taps.get("events") or []
    saved = []
    for raw in taps:
        ev = dict(raw)
        ev.setdefault("id", str(uuid.uuid4()))
        ev.setdefault("source", "watch")
        ev.setdefault("received", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        path = os.path.join(session["paths"]["taps"], f"{ev['id']}.json")
        json.dump(ev, open(path, "w"), indent=2)
        saved.append(ev)
    weak = apply_weak_labels(session)
    return dict(n=len(saved), weak_labels=weak)


def apply_weak_labels(session):
    """Walk stored taps in time order; map us/them point taps onto the nearest session point as weak winners."""
    files = sorted(os.listdir(session["paths"]["taps"]))
    events = [json.load(open(os.path.join(session["paths"]["taps"], f))) for f in files if f.endswith(".json")]
    events.sort(key=lambda e: float(e.get("t_s", 0)))
    points = sorted(session.get("points") or [], key=lambda p: p.get("first_hit", 0))
    team_map = {"us": session.get("user_team", "A"), "them": ("B" if session.get("user_team", "A") == "A" else "A"),
                "A": "A", "B": "B"}
    stack = []          # point taps for undo
    n = 0
    for ev in events:
        kind = ev.get("kind") or ev.get("type")
        if kind == "undo":
            if stack:
                prev = stack.pop()
                p = prev["point"]
                if p.get("winner_src") == "watch":
                    p["winner"] = None; p["winner_src"] = None; p["winner_confidence"] = 0.5
            continue
        if kind == "highlight":
            # mark nearest point as favorite (slow-mo eligibility)
            t = float(ev.get("t_s", 0))
            if points:
                p = min(points, key=lambda q: abs(q.get("first_hit", 0) - t))
                p["favorite"] = True
            continue
        if kind not in ("point", "score"):
            continue
        t = float(ev.get("t_s", 0))
        team = team_map.get(ev.get("team") or ev.get("side"))
        if team is None or not points:
            continue
        # nearest point whose window covers t, else nearest by start
        cand = [p for p in points if p.get("A", p.get("first_hit", 0)) - 1 <= t <= p.get("B", p.get("last_hit", 0)) + 2]
        p = min(cand or points, key=lambda q: abs(q.get("first_hit", 0) - t))
        if p.get("winner_src") == "user":
            continue          # never overwrite a confirmed user fix
        p["winner"] = team; p["winner_src"] = "watch"; p["winner_confidence"] = 0.6
        stack.append(dict(point=p, ev=ev)); n += 1
    return n
