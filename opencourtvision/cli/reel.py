"""ocv-reel: ball-follow 9:16 reel (1080x1920) from a moments config JSON (work in progress).

Example:
  ocv-reel examples/reel_moments.example.json --work work/reel --out exports/reel.mp4
"""
import argparse
import json
import os

from ..reel.reel import build


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-reel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="moments config JSON (see examples/reel_moments.example.json)")
    ap.add_argument("--work", default="work/reel", help="scratch dir for crop commands, segments, debug sheet, stats")
    ap.add_argument("--out", help="output mp4 (overrides config 'output')")
    ap.add_argument("--tracknet", default=os.environ.get("OCV_TRACKNET_DIR"),
                    help="TrackNet checkout dir with model.py + model_best.pt (optional; env OCV_TRACKNET_DIR)")
    A = ap.parse_args(argv)
    cfg = json.load(open(A.config))
    build(cfg, A.work, A.out, A.tracknet)


if __name__ == "__main__":
    main()
