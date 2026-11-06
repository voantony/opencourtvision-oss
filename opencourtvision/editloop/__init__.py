"""Edit loop (eng notes v3.7 §2.8.7/§2.8.8): which parts of the match become clips, and how clips are picked.

Design intent (engineering, not a legal conclusion): clip events are never identified from *tracked ball* motion.

- E1  `segment.audio_points`: point segments from audio hit onsets + rhythm rules only.
- E3  `segment.gap_fill`: structure-guided audio re-scan (final score -> changeovers; in-game gaps; after-fault serves).
- Vision fallback `motion.rally_cues` (flag `OCV_VISION_RALLY_CUES`, off by default): regional frame-motion energy
  (no object tracking) used only to merge/split audio segments.
- E2  `select.select_clips`: selection reads only whitelisted audio/segment/user features; decoded score only when the
  build allows it (US build: off).

Structural guard: nothing in this package may import `opencourtvision.ball`, the SwingVision importer or the score
decoder (tests/test_editloop_guard.py), and every segment carries `sources` checked by `provenance.check`.
"""
from .provenance import ALLOWED_SOURCES, FORBIDDEN_SOURCES, ProvenanceError, check  # noqa: F401
