"""Two-tap labelling: the user names two players (ideally one per team, e.g. himself and one opponent); the other
two are deduced. His partner is the other in-play person on his side, the remaining person is the other opponent.

taps: list of (det_index, slot) with slot in A1/A2/B1/B2 (team A = the user's team). 0, 1 or 2 taps (4 also work).
Returns (mapping det_index -> slot, status) where status is one of
  ok            all four named
  anon_opp      opponents unnamed (fewer than one tap per team): the side is known, which opponent is B1 is arbitrary
  anon_all      no taps: teams by side only (near pair = A), names inside each team arbitrary
  defer:<why>   cannot deduce on this frame (e.g. a side does not have exactly two in-play people); try a later frame
"""
from __future__ import annotations

TEAM = {"A1": "A", "A2": "A", "B1": "B", "B2": "B"}
OTHER = {"A1": "A2", "A2": "A1", "B1": "B2", "B2": "B1"}


def deduce_ids(dets, taps, near_default: str = "A"):
    sides = {0: [i for i, d in enumerate(dets) if d["side"] == 0], 1: [i for i, d in enumerate(dets) if d["side"] == 1]}
    for s, v in sides.items():
        if len(v) != 2:
            return {}, f"defer:side{s}_has_{len(v)}"
    m = {}; team_side = {}
    for i, slot in taps:
        if i not in sides[0] + sides[1]:
            return {}, "defer:tap_not_in_play"
        s = dets[i]["side"]; t = TEAM[slot]
        if team_side.get(t, s) != s or any(team_side.get(o) == s for o in team_side if o != t):
            return {}, "defer:taps_conflict"
        team_side[t] = s; m[i] = slot
    for i, slot in list(m.items()):                     # partner = the other in-play person on the same side
        j = [k for k in sides[dets[i]["side"]] if k != i][0]
        if j in m and m[j] != OTHER[slot]:
            return {}, "defer:taps_conflict"
        m[j] = OTHER[slot]
    if len(team_side) == 2:
        return m, "ok"
    if len(team_side) == 1:
        t = next(iter(team_side)); o = "B" if t == "A" else "A"; s = 1 - team_side[t]
        for k, slot in zip(sides[s], (o + "1", o + "2")):
            m[k] = slot
        return m, "anon_opp"
    far = "B" if near_default == "A" else "A"
    for s, t in ((0, near_default), (1, far)):
        for k, slot in zip(sides[s], (t + "1", t + "2")):
            m[k] = slot
    return m, "anon_all"
