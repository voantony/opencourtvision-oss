#!/usr/bin/env python3
"""Download public CC0 / CC BY clips into a cache outside the repo, checking sha256.

The manifest ships with no clips pinned. Add an entry only after you have the file's digest.
There is no default URL, so a fresh tree downloads nothing and exits 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "benchmarks", "public_data", "clips_manifest.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=os.path.expanduser("~/.cache/opencourtvision/public_clips"))
    a = ap.parse_args(argv)
    spec = json.load(open(MANIFEST))
    clips = spec.get("clips") or []
    if not clips:
        print("no public clips pinned; synthetic benches need no download")
        return 0
    os.makedirs(a.dest, exist_ok=True)
    for clip in clips:
        lic = (clip.get("licence") or "").upper().replace("-", " ")
        if lic not in ("CC0", "CC BY", "CC0 1.0", "CC BY 4.0"):
            print(f"{clip.get('id')}: licence must be CC0 or CC BY", file=sys.stderr)
            return 2
        digest = (clip.get("sha256") or "").lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            print(f"{clip.get('id')}: refusing download without a sha256", file=sys.stderr)
            return 2
        if not clip.get("attribution"):
            print(f"{clip.get('id')}: CC BY/CC0 entry needs an attribution string", file=sys.stderr)
            return 2
        dest = os.path.join(a.dest, clip["id"] + os.path.splitext(clip["url"])[1])
        if not (os.path.exists(dest) and _sha256(dest) == digest):
            urllib.request.urlretrieve(clip["url"], dest)
        got = _sha256(dest)
        if got != digest:
            print(f"{clip['id']}: sha256 mismatch", file=sys.stderr)
            return 2
        print(f"{dest} {got} {clip['licence']} {clip['attribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
