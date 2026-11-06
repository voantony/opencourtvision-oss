#!/usr/bin/env python3
"""Download v1 release weights (ball / court / scoring) and check checksums.

Player-ID and pose assets are refused. Nothing is written into the git tree.
Assets with hosted=false are listed and skipped: this script does not invent a URL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "weights", "MANIFEST.json")
EXCLUDED = {"osnet_x1_0_msmt17", "osnet", "yolo11s-pose", "yolo", "ultralytics", "mediapipe-pose-landmarker", "pose", "rfdetr-coco", "rfdetr"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", default="tracknetv3", help="asset id, or 'list'")
    a = ap.parse_args(argv)
    spec = json.load(open(MANIFEST))
    if a.id == "list":
        for row in spec["assets"]:
            print(f"{row['id']:20s} role={row['role']:8s} hosted={row['hosted']}")
        print("excluded pending Legal:")
        for row in spec["excluded_v1_pending_legal"]:
            print(f"  {row['id']} ({row['role']})")
        return 0
    if a.id in EXCLUDED or any(a.id == row["id"] for row in spec["excluded_v1_pending_legal"]):
        print(f"{a.id} is not a v1 release asset (player-ID or pose, pending Legal)", file=sys.stderr)
        return 2
    row = next((r for r in spec["assets"] if r["id"] == a.id), None)
    if row is None:
        print(f"unknown asset {a.id}", file=sys.stderr)
        return 2
    if not row["hosted"]:
        print(f"{a.id}: not hosted yet ({row.get('training_data', '')})")
        return 0
    if a.id == "tracknetv3":
        from opencourtvision.ball.tracknet3 import SHA256, fetch_weights
        path = fetch_weights()
        print(path)
        for name, digest in SHA256.items():
            print(f"  {name} sha256={digest}")
        return 0
    print(f"no downloader for {a.id}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
