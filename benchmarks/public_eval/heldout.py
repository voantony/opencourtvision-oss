"""Frozen tune / report split. Roles live in split.json next to this file."""
from __future__ import annotations

import json
from pathlib import Path

SPLIT_PATH = Path(__file__).with_name("split.json")


def roles(tune: str | None = None) -> dict:
    spec = json.loads(SPLIT_PATH.read_text())
    tune = (tune or spec["tune"]).strip()
    report = spec["report"] if tune == spec["tune"] else spec["tune"]
    if tune not in (spec["tune"], spec["report"]):
        raise ValueError(f"unknown tune bundle {tune!r}; file names {spec['tune']!r} and {spec['report']!r}")
    return {"tune": tune, "report": report, "split_file": str(SPLIT_PATH)}


def claim_label(bundle_id: str, tune: str | None = None) -> str:
    r = roles(tune)
    if bundle_id == r["tune"]:
        return "IN-SAMPLE (tuning bundle)"
    if bundle_id == r["report"]:
        return "HELD-OUT (report bundle)"
    raise ValueError(f"unknown bundle {bundle_id!r}")
