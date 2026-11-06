"""Full-app whole-match score decoder (Viterbi + forward-backward over a lattice of score states, anchors re-inferring
points on both sides). Kept as a SWITCHABLE module for a possible Wingfield EP 3834171 B1 opposition (deadline ~15 Jan
2027): not packaged into Vamos US (scripts/build_variant.py excludes it and scoring/decoder.py), and OFF by default in
the full app (Profile.scorer = "forward"). Only `make_scorer(profile)` in scoring.engine may select it, when a full-app
profile sets scorer="global".
"""
from ...scoring.decoder import Decoded, ScoreConflict, confirm_loop, decode  # noqa: F401

ENABLED_BY_DEFAULT = False
