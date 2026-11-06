#!/usr/bin/env python3
"""Assemble a variant artefact of the engine (eng notes §2.8.8).

  python scripts/build_variant.py --variant us   --out dist/us      # "Vamos US": best-variant code not packaged
  python scripts/build_variant.py --variant full --out dist/full    # non-US "Vamos" (build on EU CI only)

The US artefact is the package tree minus every path in EXCLUDE_US; then run scripts/scan_us_build.py on it.
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXCLUDE_US = (
    "opencourtvision/variants/full",
    "opencourtvision/scoring/decoder.py",
    "opencourtvision/pose/syllables.py",
    # Unconstrained research track (eng notes §2.5.4 / §2.10): 3D ODE, speed, multi-phone 3D, etc.
    "opencourtvision/research",
    "opencourtvision/ball/physics3d.py",
    "opencourtvision/ball/physics3d_synth.py",
)


def build(variant, out):
    if os.path.exists(out): shutil.rmtree(out)
    src = os.path.join(ROOT, "opencourtvision")
    def ignore(d, names):
        rel = os.path.relpath(d, ROOT).replace(os.sep, "/")
        skip = {n for n in names if n in ("__pycache__",) or n.endswith(".pyc")}
        if variant == "us":
            skip |= {n for n in names if f"{rel}/{n}" in EXCLUDE_US}
        return skip
    shutil.copytree(src, os.path.join(out, "opencourtvision"), ignore=ignore)
    with open(os.path.join(out, "VARIANT"), "w") as f: f.write(variant + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("us", "full"), required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.variant == "full" and os.environ.get("OCV_CI_REGION", "EU") == "US":
        sys.exit("full-variant builds run on EU infrastructure only (OCV_CI_REGION=US refused)")
    print(build(a.variant, a.out))


if __name__ == "__main__":
    main()
