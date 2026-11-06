"""ocv-swingcut: tennis match video -> warmup cutoff, point segments, highlights, score timeline (+ optional lossless 4K render).

Example:
  ocv-swingcut match.mp4 --final "6-4 6-3" --work work/ --story --render exports/ --name "Club final 6-4 6-3"
"""
import argparse

from ..pipeline import add_arguments, run


def main(argv=None):
    ap = add_arguments(argparse.ArgumentParser(prog="ocv-swingcut", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
