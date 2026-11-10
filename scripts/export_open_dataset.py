#!/usr/bin/env python3
"""Write an open point list only when it shares no match with held-out sets.

Practice / warm-up points must not come from the same source video as any
check, gold, or private-test point. The same whole-match rule applies to the
rows being exported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opencourtvision.labels.leak_guard import LeakError, assert_whole_match_disjoint, points_in


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--practice", required=True, type=Path)
    parser.add_argument("--holdout", action="append", default=[], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    practice = points_in(_read(args.practice)) or _read(args.practice).get("points") or []
    holdouts: list[dict] = []
    for path in args.holdout:
        holdouts.extend(points_in(_read(path)))
    exporting = points_in(_read(args.points)) or _read(args.points).get("points") or _read(args.points).get("clips") or []
    try:
        assert_whole_match_disjoint(practice, holdouts)
        assert_whole_match_disjoint(exporting, holdouts)
    except LeakError as exc:
        print(exc, file=sys.stderr)
        return 2
    args.out.write_text(json.dumps({"points": exporting}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
