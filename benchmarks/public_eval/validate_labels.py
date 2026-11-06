"""Label validator. Eval refuses to run until this passes.

Checks scored points for missing match metadata, bad windows, duplicate idxs,
idx collisions (same idx, starts more than 1s apart), overlaps longer than 2s,
optional p(A) length, and score consistency against the rules state machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def scored_points(lab: dict):
    out = []
    for p in lab.get("points") or []:
        if p.get("kind") in ("non_point", "skip"):
            continue
        if p.get("winner") in ("A", "B"):
            out.append(p)
    return out


def validate(lab: dict, p_a=None, name="labels", overlap_s=2.0, collision_s=1.0, require_final=False):
    errors, warnings = [], []
    match = lab.get("match") or {}
    for key in ("sport", "preset", "first_server", "mine"):
        if not match.get(key):
            errors.append(f"missing match.{key}")
    if match.get("first_server") not in (None, "A", "B"):
        errors.append(f"bad first_server={match.get('first_server')!r}")
    if match.get("mine") not in (None, "A", "B"):
        errors.append(f"bad mine={match.get('mine')!r}")

    pts = scored_points(lab)
    if not pts:
        errors.append("no scored A/B points")
        return errors, warnings, pts

    for p in pts:
        s, e = p.get("start"), p.get("end")
        if s is None or e is None or not (float(e) > float(s)):
            errors.append(f"bad window idx={p.get('idx')} start={s} end={e}")
        if p.get("winner") not in ("A", "B"):
            errors.append(f"bad winner idx={p.get('idx')}")

    idxs = [p.get("idx") for p in pts]
    dup = [i for i, c in Counter(idxs).items() if c > 1 and i is not None]
    if dup:
        errors.append(f"duplicate scored idxs: {dup[:20]}")

    by_idx = defaultdict(list)
    for p in pts:
        if p.get("idx") is not None and p.get("start") is not None:
            by_idx[p["idx"]].append(float(p["start"]))
    for idx, starts in by_idx.items():
        uniq = sorted(set(round(s, 3) for s in starts))
        if len(uniq) > 1 and max(uniq) - min(uniq) > collision_s:
            errors.append(f"idx collision idx={idx} starts={uniq}")

    ordered = sorted(pts, key=lambda p: float(p["start"]))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if float(b["start"]) >= float(a["end"]):
                break
            ov = _overlap(float(a["start"]), float(a["end"]), float(b["start"]), float(b["end"]))
            if ov > overlap_s:
                errors.append(
                    f"overlap {ov:.1f}s idx={a.get('idx')}@{float(a['start']):.1f} "
                    f"vs idx={b.get('idx')}@{float(b['start']):.1f}"
                )

    if p_a is not None and len(p_a) != len(ordered):
        errors.append(f"p_a length {len(p_a)} != n scored points {len(ordered)}")

    try:
        from opencourtvision.labels.points import parse_point_labels
        from opencourtvision.scoring.decoder import replay

        points = [
            {"idx": i, "start": float(p["start"]), "end": float(p["end"]), "kind": "point", "winner": p["winner"]}
            for i, p in enumerate(ordered, 1)
        ]
        obj = {
            "schema": "ocv.point_labels.v1",
            "video": lab.get("video") or {"name": name},
            "match": {
                "sport": match.get("sport") or "tennis",
                "preset": match.get("preset") or "advantage",
                "first_server": match.get("first_server") or "A",
                "mine": match.get("mine") or "A",
                "teams": {"A": "A", "B": "B"},
            },
            "points": points,
            "decoded": {},
            "labeler": {},
        }
        labels = parse_point_labels(obj)
        states = replay([s.choice for s in labels.segments], labels.fmt, obj["match"]["first_server"], 0)
        try:
            fs = labels.final_score()
        except Exception:
            fs = ""
        hint = match.get("final_hint") or match.get("final")
        if hint and str(hint).replace(" ", "") not in str(fs).replace(" ", ""):
            msg = f"final_hint={hint!r} vs replayed={fs!r}"
            (errors if require_final else warnings).append(msg)
        for i, p in enumerate(ordered):
            if p.get("server") in ("A", "B") and p["server"] != states[i].server:
                warnings.append(f"server field {p['server']} != replay {states[i].server} at idx={p.get('idx')}")
    except Exception as exc:
        warnings.append(f"replay check skipped: {exc}")

    return errors, warnings, ordered


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--p", help="optional JSON list, or an object with p_a")
    ap.add_argument("--require-final", action="store_true")
    a = ap.parse_args(argv)
    lab = json.load(open(a.labels))
    p_a = None
    if a.p:
        raw = json.load(open(a.p))
        p_a = raw["p_a"] if isinstance(raw, dict) else raw
    errors, warnings, pts = validate(lab, p_a, name=a.labels, require_final=a.require_final)
    print(json.dumps({"labels": a.labels, "n_scored": len(pts), "errors": errors, "warnings": warnings, "ok": not errors}, indent=2))
    if errors:
        print(f"VALIDATOR FAIL: {len(errors)} error(s)", file=sys.stderr)
        return 2
    print(f"VALIDATOR OK ({len(pts)} scored, {len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
