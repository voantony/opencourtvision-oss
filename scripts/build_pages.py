#!/usr/bin/env python3
"""Assemble the GitLab Pages tree: homepage at / and the labeler at /label/."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from opencourtvision.labels.leak_guard import (
    answer_tokens,
    assert_no_answer_tokens,
    assert_whole_match_disjoint,
    discover_holdouts,
)

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
LABEL = ROOT / "labeler-pages"
DROP = {"answer", "reason_en", "reason_es", "role", "check", "gold", "is_check", "answer_token"}


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _clips_for_client(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    clean = []
    for clip in data.get("clips") or []:
        row = {k: v for k, v in clip.items() if k not in DROP}
        if "answer_token" in clip:
            raise SystemExit(f"refusing to ship answer token on {clip.get('id')}")
        clean.append(row)
    data["clips"] = clean
    return data


def build(public: Path | None = None) -> Path:
    public = Path(public or ROOT / "public")
    if public.exists():
        shutil.rmtree(public)
    public.mkdir(parents=True)
    for name in ("index.html", "es.html", "home.css"):
        shutil.copy(SITE / name, public / name)
    _copy_tree(SITE / "fonts", public / "fonts")
    _copy_tree(SITE / "assets", public / "assets")
    label = public / "label"
    label.mkdir()
    for name in ("index.html", "style.css", "app.js", "round.js", "submit.js", "submit_config.js",
                 "game_config.js", "i18n.js", "share.js", "practice.json"):
        shutil.copy(LABEL / name, label / name)
    shutil.copy(LABEL / "clips.sha256", label / "clips.sha256")
    shutil.copy(ROOT / "docs" / "CC_ATTRIBUTION.md", public / "CC_ATTRIBUTION.txt")
    clips = _clips_for_client(LABEL / "clips.json")
    (label / "clips.json").write_text(json.dumps(clips, indent=1) + "\n", encoding="utf-8")
    practice = json.loads((LABEL / "practice.json").read_text(encoding="utf-8"))["points"]
    holdouts = discover_holdouts(ROOT)
    assert_whole_match_disjoint(practice, holdouts)
    assert_no_answer_tokens(public, answer_tokens(holdouts))
    return public


def enable_submit(public: Path) -> None:
    """Turn Send on in the deployed copy only. git keeps submitEnabled false."""
    path = public / "label" / "submit_config.js"
    text = path.read_text(encoding="utf-8")
    if text.count("submitEnabled: false,") != 1:
        raise SystemExit("submit_config.js: expected exactly one 'submitEnabled: false,'")
    path.write_text(text.replace("submitEnabled: false,", "submitEnabled: true,", 1), encoding="utf-8")


def fetch_clips(public: Path, clips_url: str) -> None:
    dest = public / "label" / "clips"
    dest.mkdir(parents=True, exist_ok=True)
    sha_path = LABEL / "clips.sha256"
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        url = clips_url.rstrip("/") + "/" + name
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            raise SystemExit(f"checksum failed for {name}")
        (dest / name).write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fetch-clips", action="store_true")
    parser.add_argument("--clips-url", default="")
    parser.add_argument("--submit-enabled", action="store_true", help="turn Send on in the built copy")
    args = parser.parse_args(argv)
    public = build(args.out)
    if args.submit_enabled:
        enable_submit(public)
    if args.fetch_clips:
        if not args.clips_url:
            raise SystemExit("--clips-url is required with --fetch-clips")
        fetch_clips(public, args.clips_url)
    print(public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
