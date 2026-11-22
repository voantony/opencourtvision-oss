"""Human point labels in the docs/label_schema.json shape. Stdlib only.

The travel labeler writes these. Skip/unclear is a non-point, so it is not eval ground truth.
A and B are points with verified true once the labeler confirms them.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "label_schema.json"


def build_labels(queue: dict, answers: dict) -> dict:
    """answers: {point id: {winner: 'A'|'B'|'skip', server: 'A'|'B'|None, confirmed?: bool}}.

    Points are stored in source-time order so the score replay is well defined.
    Unanswered points are left out; the file is still valid once one A/B exists.
    Unconfirmed answers keep their winner values but verified=false.
    """
    video = queue.get("video") or {}
    match = queue.get("match") or {}
    points = []
    ordered = sorted(queue.get("points") or [], key=lambda p: (float(p["start"]), int(p["idx"])))
    for p in ordered:
        ans = answers.get(str(p["id"]))
        if not ans:
            continue
        winner = ans.get("winner")
        row = {
            "idx": int(p["idx"]),
            "start": float(p["start"]),
            "end": float(p["end"]),
        }
        if winner in ("A", "B"):
            row["kind"] = "point"
            row["winner"] = winner
            # Default True for legacy answers that lack the flag; False when explicitly unconfirmed.
            row["verified"] = bool(ans.get("confirmed", True))
            if ans.get("server") in ("A", "B"):
                row["server"] = ans["server"]
        else:
            row["kind"] = "non_point"
        points.append(row)
    return {
        "schema": "ocv.public_labels.v1",
        "labeler": "human",
        "video": {
            "name": video.get("name") or queue.get("video_id") or "clip",
            "licence": video.get("licence") or match.get("source_licence") or "CC BY",
            **({"url": video["url"]} if video.get("url") else {}),
            **({"attribution": video["attribution"]} if video.get("attribution") else {}),
            **({"sha256": video["sha256"]} if video.get("sha256") else {}),
        },
        "match": {
            "sport": match["sport"],
            "preset": match["preset"],
            "first_server": match["first_server"],
            "mine": match["mine"],
            "source_licence": match["source_licence"],
            **({"final_hint": match["final_hint"]} if match.get("final_hint") else {}),
        },
        "points": points,
    }


def _type_ok(value, spec) -> bool:
    kinds = spec if isinstance(spec, list) else [spec]
    for kind in kinds:
        if kind == "null" and value is None:
            return True
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
        if kind == "array" and isinstance(value, list):
            return True
    return False


def schema_errors(doc, schema=None, path="$") -> list[str]:
    """Check doc against docs/label_schema.json. No third-party jsonschema dependency."""
    if schema is None:
        schema = json.loads(SCHEMA_PATH.read_text())
    errors = []
    if "const" in schema and doc != schema["const"]:
        errors.append(f"{path} must be {schema['const']!r}")
    if "enum" in schema and doc not in schema["enum"]:
        errors.append(f"{path} not in {schema['enum']!r}")
    if "type" in schema and not _type_ok(doc, schema["type"]):
        errors.append(f"{path} type {schema['type']!r}")
        return errors
    if schema.get("type") == "object" and isinstance(doc, dict):
        allowed = set((schema.get("properties") or {}))
        if schema.get("additionalProperties") is False:
            for key in doc:
                if key not in allowed:
                    errors.append(f"{path}.{key} is not in the schema")
        for key in schema.get("required") or []:
            if key not in doc:
                errors.append(f"{path}.{key} missing")
        props = schema.get("properties") or {}
        for key, value in doc.items():
            if key in props:
                errors.extend(schema_errors(value, props[key], f"{path}.{key}"))
    if schema.get("type") == "array" and isinstance(doc, list) and "items" in schema:
        for i, value in enumerate(doc):
            errors.extend(schema_errors(value, schema["items"], f"{path}[{i}]"))
    if "pattern" in schema and isinstance(doc, str):
        import re
        if re.fullmatch(schema["pattern"], doc) is None:
            errors.append(f"{path} does not match {schema['pattern']}")
    return errors
