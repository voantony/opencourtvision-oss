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
