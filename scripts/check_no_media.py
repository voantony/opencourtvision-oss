#!/usr/bin/env python3
"""Fail if the tree contains an image or video, including Git LFS pointers to one.

Extensions: png jpg jpeg gif webp heic bmp tiff tif svg ico mp4 mov webm avi mkv m4v.
Content is sniffed from magic bytes (and from the LFS object a pointer names, when the
object is present). A clean tree exits 0.
"""
from __future__ import annotations

import os
import subprocess
import sys

EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".bmp", ".tiff", ".tif",
    ".svg", ".ico", ".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v",
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".build"}
LFS_MARK = b"version https://git-lfs.github.com/spec/v1"


def _kind(head: bytes, path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in EXTS:
        return f"extension {ext}"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "magic png"
    if head.startswith(b"\xff\xd8\xff"):
        return "magic jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "magic gif"
    if head.startswith(b"BM"):
        return "magic bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "magic tiff"
    if head.startswith(b"\x00\x00\x01\x00"):
        return "magic ico"
    if head.startswith(b"RIFF") and b"WEBP" in head[:16]:
        return "magic webp"
    if head.startswith(b"RIFF") and b"AVI " in head[:16]:
        return "magic avi"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "magic webm/mkv"
    if len(head) > 12 and head[4:8] == b"ftyp":
        return "magic iso-media (mp4/mov/heic)"
    stripped = head.lstrip()
    if stripped.lower().startswith(b"<svg") or (stripped.lower().startswith(b"<?xml") and b"<svg" in head[:800].lower()):
        return "magic svg"
    return None


def _lfs_oid(text: bytes):
    for line in text.splitlines():
        if line.startswith(b"oid sha256:"):
            return line.split(b":", 1)[1].decode().strip()
    return None


def _lfs_object(root: str, oid: str) -> str | None:
    path = os.path.join(root, ".git", "lfs", "objects", oid[:2], oid[2:4], oid)
    return path if os.path.isfile(path) else None


def scan(root: str):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            try:
                with open(path, "rb") as f:
                    head = f.read(512)
            except OSError as exc:
                hits.append(f"{rel}: unreadable ({exc})")
                continue
            kind = _kind(head, name)
            if not kind:
                try:
                    mime = subprocess.run(["file", "-b", "--mime-type", path], capture_output=True, text=True, timeout=5)
                    mt = (mime.stdout or "").strip()
                except (OSError, subprocess.TimeoutExpired):
                    mt = ""
                if mt.startswith("image/") or mt.startswith("video/"):
                    kind = f"file(1) {mt}"
            if kind:
                hits.append(f"{rel}: {kind}")
                continue
            if head.startswith(LFS_MARK):
                oid = _lfs_oid(head)
                obj = _lfs_object(root, oid) if oid else None
                if obj:
                    with open(obj, "rb") as f:
                        inner = f.read(64)
                    inner_kind = _kind(inner, name)
                    if inner_kind:
                        hits.append(f"{rel}: git-lfs pointer -> {inner_kind}")
                elif os.path.splitext(name)[1].lower() in EXTS:
                    hits.append(f"{rel}: git-lfs pointer for a media path")
    return hits


def main(argv=None):
    root = os.path.abspath(argv[0] if argv else os.path.join(os.path.dirname(__file__), ".."))
    # `file(1)` is a second opinion on the same bytes. A disagreement still fails via magic above.
    hits = scan(root)
    if hits:
        print("MEDIA CHECK FAIL", file=sys.stderr)
        for h in hits:
            print(" ", h, file=sys.stderr)
        return 1
    print(f"MEDIA CHECK OK ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
