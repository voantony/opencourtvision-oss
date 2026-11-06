"""Fix screen: surface the least-certain calls (bounce, point winner, player ID) for one-tap confirm / click to fix.

Priority: model struggle tags from capture-quality first (hard conditions), then lowest confidence. Every answer is stored
as a labelled example under session/fixes/ (no faces, no names). Watch taps arrive via taps.py as weak labels.
"""
import json
import os
import time
import uuid

KINDS = ("bounce", "point_winner", "player_id", "score_check", "cut_off")


def prioritise(items, struggle_tags=()):
    """Sort review items: struggle-tagged first, then ascending confidence, then time order."""
    struggle = set(struggle_tags or ())

    def key(it):
        hard = 0 if (set(it.get("struggle_tags") or ()) & struggle or it.get("hard_condition")) else 1
        return (hard, float(it.get("confidence", 1.0)), float(it.get("t", 0.0)))

    return sorted(items, key=key)


def build_queue(session, n=12):
    """Build / refresh the fix-screen queue from session points + any engine uncertainty flags."""
    items = []
    struggle = (session.get("quality") or {}).get("struggle_tags") or []
    for p in session.get("points") or []:
        if p.get("winner_confidence", 1.0) < 0.85 or p.get("winner_src") != "user":
            items.append(dict(id=f"pw-{p['id']}", kind="point_winner", point=p["id"], t=p.get("first_hit", 0),
                              confidence=float(p.get("winner_confidence", 0.5)),
                              prompt="Who won this point?", options=["A", "B", "skip"],
                              current=p.get("winner"), struggle_tags=struggle,
                              hard_condition=bool(struggle)))
        if p.get("bounce_confidence", 1.0) < 0.8:
            items.append(dict(id=f"bc-{p['id']}", kind="bounce", point=p["id"], t=p.get("last_hit", 0),
                              confidence=float(p.get("bounce_confidence", 0.5)),
                              prompt="Click where the final bounce landed (or N if none / cut off).",
                              current=p.get("bounce"), struggle_tags=struggle, hard_condition=bool(struggle)))
        if p.get("player_id_confidence", 1.0) < 0.75:
            items.append(dict(id=f"pid-{p['id']}", kind="player_id", point=p["id"], t=p.get("first_hit", 0),
                              confidence=float(p.get("player_id_confidence", 0.5)),
                              prompt="Confirm who is who (court side / role, never a face).",
                              current=p.get("player_id"), struggle_tags=struggle, hard_condition=bool(struggle)))
    queue = prioritise(items, struggle)[:n]
    session["review"] = queue
    return queue


def apply_fix(session, item_id, value, source="user"):
    """Record one fix as a labelled example. source: user | watch | confirm.
    Returns the example dict. Never stores faces or names."""
    item = next((x for x in session.get("review") or [] if x["id"] == item_id), None)
    if item is None:
        raise KeyError(item_id)
    ex = dict(schema="vamos.fix.v1", id=str(uuid.uuid4()), item_id=item_id, kind=item["kind"],
              point=item.get("point"), t=item.get("t"), value=value, source=source,
              struggle_tags=item.get("struggle_tags") or [],
              created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              # privacy: no face embeddings, no biometric templates, no real names
              privacy=dict(faces=False, biometrics=False, names=False))
    path = os.path.join(session["paths"]["fixes"], f"{ex['id']}.json")
    json.dump(ex, open(path, "w"), indent=2)
    # reflect into session points
    for p in session.get("points") or []:
        if p["id"] != item.get("point"): continue
        if item["kind"] == "point_winner":
            p["winner"] = value; p["winner_src"] = source; p["winner_confidence"] = 1.0
        elif item["kind"] == "bounce":
            p["bounce"] = value; p["bounce_confidence"] = 1.0
        elif item["kind"] == "player_id":
            p["player_id"] = value; p["player_id_confidence"] = 1.0
    session["review"] = [x for x in session["review"] if x["id"] != item_id]
    session.setdefault("n_fixes", 0)
    session["n_fixes"] += 1
    return ex


def list_fixes(session):
    d = session["paths"]["fixes"]
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            out.append(json.load(open(os.path.join(d, name))))
    return out
