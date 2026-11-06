#!/usr/bin/env python3
"""Generate the synthetic benchmark clips (benchmarks/data/, git-ignored) and ground truth (benchmarks/ground_truth/)."""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from opencourtvision.synthetic import SCENARIOS, make_clip, scenario  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true", help="WAV only (faster)")
    A = ap.parse_args(argv)
    gtd = os.path.join(HERE, "ground_truth")
    os.makedirs(gtd, exist_ok=True)
    for name in SCENARIOS:
        p = make_clip(scenario(name, A.seed), A.out, video=not A.no_video)
        shutil.copy(p["gt"], os.path.join(gtd, f"{name}.json"))
        gt = json.load(open(p["gt"]))
        print(f"{name}: {len(gt['hits'])} hits, {len(gt['rallies'])} rallies, {gt['total_s']} s -> {', '.join(sorted(p))}")


if __name__ == "__main__":
    main()
