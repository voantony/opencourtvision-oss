"""Grades and the opt-in accuracy leaderboard for Call the Point.

Grades follow points called. The leaderboard ranks only opt-in accuracy on
hidden checks, and only after a minimum number of those checks. A high volume
of calls does not earn a place.
"""
from __future__ import annotations

GRADES = (
    ("Iniciación", 1),
    ("Intermedio", 10),
    ("Avanzado", 50),
    ("Pro", 200),
)
MIN_CHECKS = 10
XP_PER_CALL = 10


def to_team_call(call: str, a_side_team: str | None) -> str | None:
    """Map a near/far call onto the match's teams. A is the near side of that clip."""
    if call == "F":
        return "F"
    if a_side_team not in ("A", "B") or call not in ("A", "B"):
        return None
    if a_side_team == "A":
        return call
    return "B" if call == "A" else "A"


def playable_clips(clips: list[dict], warmup_ids) -> list[dict]:
    warm = {str(i) for i in warmup_ids}
    return [c for c in clips if c.get("id") not in warm and c.get("a_side_team") in ("A", "B")]


def training_rows(calls: list[dict], clips: list[dict], warmup_ids) -> list[dict]:
    """Scoring and human-baseline rows. Warm-up ids are never counted."""
    by_id = {c.get("id"): c for c in clips}
    rows = []
    for call in calls:
        pid = call.get("point_id")
        if pid in set(warmup_ids):
            continue
        clip = by_id.get(pid)
        if not clip or clip.get("a_side_team") not in ("A", "B"):
            continue
        team = to_team_call(call.get("call"), clip.get("a_side_team"))
        if team is None:
            continue
        rows.append({"point_id": pid, "call": call.get("call"), "team_call": team})
    return rows


def grade_for(points_called: int) -> str | None:
    current = None
    for name, need in GRADES:
        if points_called >= need:
            current = name
    return current


def leaderboard(rows: list[dict], min_checks: int = MIN_CHECKS) -> list[dict]:
    eligible = []
    for row in rows:
        if row.get("opt_in") is not True:
            continue
        if not isinstance(row.get("check_count"), int) or row["check_count"] < min_checks:
            continue
        if not isinstance(row.get("accuracy"), (int, float)):
            continue
        eligible.append(row)
    eligible.sort(key=lambda row: (-float(row["accuracy"]), str(row.get("nickname") or "")))
    return [
        {"nickname": str(row["nickname"]), "accuracy_band": row.get("accuracy_band") or "—"}
        for row in eligible
    ]
