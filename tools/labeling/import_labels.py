#!/usr/bin/env python3
"""Check human labels and, once enough exist, freeze the public split by video.

  python tools/labeling/import_labels.py --labels ~/label-session/labels.json

The existing validator has to pass. Skip/unclear points are not ground truth.
A video goes entirely to cc_tune or entirely to cc_report. Points inside one video are never split.
The lists in split.json change only when the human-verified total reaches the freeze count (400).
Synthetic licences are reported and are not added to the public split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.public_eval.validate_labels import validate  # noqa: E402

FREEZE_AT = 400
PUBLIC = {"CC0", "CC BY", "CC0 1.0", "CC BY 4.0", "CC BY 3.0", "CC BY 2.0"}


def verified_points(lab: dict) -> list:
    if lab.get("labeler") != "human":
        return []
    out = []
    for p in lab.get("points") or []:
        if p.get("kind") != "point":
            continue
        if p.get("winner") in ("A", "B") and p.get("verified") is True:
            out.append(p)
    return out


def video_id_of(lab: dict, fallback: str) -> str:
    video = lab.get("video") or {}
    return str(video.get("name") or fallback)


def public_licence(lab: dict) -> bool:
    lic = ((lab.get("match") or {}).get("source_licence") or (lab.get("video") or {}).get("licence") or "")
    lic = lic.upper().replace("-", " ")
    return lic in PUBLIC


def assign_videos(video_ids: list[str]) -> tuple[list[str], list[str]]:
    """Whole videos only. One video cannot be both tune and report, so it is held out."""
    ids = sorted(set(video_ids))
    if len(ids) <= 1:
        return [], ids
    n_tune = max(1, len(ids) // 5)
    return ids[:n_tune], ids[n_tune:]


def load_many(paths: list[Path]) -> list[tuple[str, dict]]:
    found = []
    for path in paths:
        if path.is_dir():
            file = path / "labels.json"
            if file.is_file():
                found.append((file.stem if file.stem != "labels" else path.name, json.loads(file.read_text())))
            continue
        found.append((path.stem, json.loads(path.read_text())))
    return found


def run(paths: list[Path], split_path: Path, dest: Path | None, freeze_at: int = FREEZE_AT, write: bool = True) -> dict:
    groups: dict[str, dict] = {}
    problems = []
    for fallback, lab in load_many(paths):
        errors, warnings, _pts = validate(lab, name=fallback)
        if errors:
            problems.append({"file": fallback, "errors": errors})
            continue
        vid = video_id_of(lab, fallback)
        if not public_licence(lab):
            problems.append({"file": vid, "errors": ["not CC0 or CC BY; left out of the public split"]})
            continue
        groups[vid] = lab
        if warnings:
            problems.append({"file": vid, "warnings": warnings})
    counted = {vid: verified_points(lab) for vid, lab in groups.items()}
    total = sum(len(v) for v in counted.values())
    frozen = False
    tune: list[str] = []
    report: list[str] = []
    spec = json.loads(split_path.read_text()) if split_path.is_file() else {}
    already = list(spec.get("cc_tune") or []) + list(spec.get("cc_report") or [])
    if already:
        frozen = True
        tune, report = list(spec.get("cc_tune") or []), list(spec.get("cc_report") or [])
    elif total >= freeze_at and groups:
        tune, report = assign_videos(list(groups))
        frozen = True
        if write:
            spec["cc_tune"] = tune
            spec["cc_report"] = report
            note = spec.get("note") or ""
            spec["note"] = (note + " " if note else "") + f"CC split frozen by video at {total} human-verified points."
            split_path.write_text(json.dumps(spec, indent=2) + "\n")
            if dest is not None:
                dest.mkdir(parents=True, exist_ok=True)
                for vid, lab in groups.items():
                    (dest / f"{vid}.json").write_text(json.dumps(lab, indent=2) + "\n")
    return {
        "videos": sorted(groups),
        "verified": total,
        "per_video": {vid: len(v) for vid, v in counted.items()},
        "frozen": frozen and total >= freeze_at or bool(already),
        "wrote_split": frozen and total >= freeze_at and write and not already,
        "cc_tune": tune,
        "cc_report": report,
        "problems": [p for p in problems if p.get("errors")],
        "freeze_at": freeze_at,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True, type=Path, help="labels.json files or session directories")
    ap.add_argument("--split", type=Path, default=ROOT / "benchmarks" / "public_eval" / "split.json")
    ap.add_argument("--dest", type=Path, default=ROOT / "benchmarks" / "public_data" / "labels")
    ap.add_argument("--freeze-at", type=int, default=FREEZE_AT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    result = run(a.labels, a.split, a.dest, freeze_at=a.freeze_at, write=not a.dry_run)
    print(json.dumps({k: result[k] for k in ("videos", "verified", "per_video", "frozen", "wrote_split", "cc_tune", "cc_report", "problems")}, indent=2))
    if result["problems"]:
        print("IMPORT FAIL", file=sys.stderr)
        return 2
    if result["verified"] < a.freeze_at:
        print(f"not frozen: {result['verified']} human-verified points, freeze at {a.freeze_at}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
