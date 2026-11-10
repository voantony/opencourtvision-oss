"""Keep practice clips off the same match as check, gold, or private-test points."""
from __future__ import annotations

import json
from pathlib import Path

HOLD_KEYS = ("check", "gold", "private_test", "private-test")
ROLES = {"check", "gold", "private-test", "private_test"}
SKIP_DIRS = {".git", "public", "node_modules", ".venv", "fonts", "__pycache__", ".pytest_cache"}


class LeakError(Exception):
    pass


def video_id(point: dict) -> str:
    return str(point.get("video_id") or point.get("video") or "")


def _as_points(rows, role: str) -> list[dict]:
    out = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("id"):
            item = dict(row)
            item["_role"] = role
            out.append(item)
    return out


def points_in(data) -> list[dict]:
    found: list[dict] = []
    if not isinstance(data, dict):
        return found
    role = data.get("set") or data.get("role")
    if role in ROLES:
        found.extend(_as_points(data.get("points") or data.get("clips") or [], str(role)))
    for key in HOLD_KEYS:
        rows = data.get(key)
        if isinstance(rows, list):
            found.extend(_as_points(rows, key))
    return found


def assert_warmup_not_counted(warmup_ids, round_ids, scored_ids) -> None:
    """Warm-up clip ids never enter a round and are never scored or baselined."""
    warm = {str(i) for i in warmup_ids}
    in_round = sorted(warm.intersection(str(i) for i in round_ids))
    if in_round:
        raise LeakError("warm-up ids in a round: " + ", ".join(in_round))
    counted = sorted(warm.intersection(str(i) for i in scored_ids))
    if counted:
        raise LeakError("warm-up ids counted: " + ", ".join(counted))


def assert_whole_match_disjoint(practice: list[dict], holdouts: list[dict]) -> None:
    """Fail when a warm-up point shares an id or a source video with a held-out point."""
    hold_ids = {p["id"] for p in holdouts}
    hold_videos = {video_id(p) for p in holdouts if video_id(p)}
    for point in practice:
        pid = point.get("id")
        vid = video_id(point)
        if pid and pid in hold_ids:
            raise LeakError(f"practice point {pid} is also a check, gold, or private-test point")
        if vid and vid in hold_videos:
            raise LeakError(f"practice video {vid} is the same match as a check, gold, or private-test point")


def discover_holdouts(root: Path) -> list[dict]:
    found: list[dict] = []
    for path in root.rglob("*.json"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        found.extend(points_in(data))
    return found


def answer_tokens(holdouts: list[dict]) -> list[str]:
    return [str(p["answer_token"]) for p in holdouts if p.get("answer_token")]


def assert_no_answer_tokens(tree: Path, tokens: list[str]) -> None:
    if not tokens:
        return
    for path in tree.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".html", ".js", ".json", ".css", ".svg", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in tokens:
            if token in text:
                raise LeakError(f"{token} shipped in {path}")
