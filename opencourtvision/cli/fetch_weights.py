"""ocv-fetch-weights: download model weights into the local cache (never into the repo).

  ocv-fetch-weights tracknetv3            # TrackNetV3 checkpoints (MIT) -> $OCV_CACHE/tracknetv3 (default
                                          #   ~/.cache/opencourtvision/tracknetv3), SHA-256 verified
  ocv-fetch-weights tracknetv3 --zip TrackNetV3_ckpts.zip   # install from an already downloaded upstream zip
"""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-fetch-weights", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["tracknetv3"])
    ap.add_argument("--zip", help="upstream TrackNetV3_ckpts.zip already on disk")
    ap.add_argument("--dest", help="target directory (default: the cache directory)")
    A = ap.parse_args(argv)
    from ..ball.tracknet3 import fetch_weights
    paths = fetch_weights(A.dest, A.zip)
    print("\n".join(str(p) for p in (paths if isinstance(paths, (list, tuple)) else [paths])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
