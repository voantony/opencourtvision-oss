"""4-player identity tracking for doubles (no faces, no biometrics).

Pipeline: person detections (RF-DETR Nano, run by the Mac tool) -> court mask + in-play filter -> per-side 2x2
assignment with a cheap-first cue cascade (court side/team schedule, motion, colour, OSNet, learned head, serve
role + rotation) -> 4 stable ids. The user taps two players (ideally one per team); `deduce` fills in the rest.
See docs/player_id.md.

Appearance data (OSNet embeddings, CIELAB colour histograms) stays in memory, per match: nothing in this package writes
it to disk, puts it in a session or export, or loads it across matches (Ops V6; tests/test_players_privacy.py).
"""
from .deduce import deduce_ids  # noqa: F401
from .tracker import APPEARANCE_KEYS, IDTracker  # noqa: F401


def strip_appearance(obj):
    """Copy of `obj` (frames / dets, nested dicts and lists) without appearance keys: use before anything is saved."""
    if isinstance(obj, dict):
        return {k: strip_appearance(v) for k, v in obj.items() if k not in APPEARANCE_KEYS}
    if isinstance(obj, list):
        return [strip_appearance(v) for v in obj]
    return obj
