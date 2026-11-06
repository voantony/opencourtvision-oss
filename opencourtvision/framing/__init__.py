"""Record-screen framing check: before and during recording, are all court lines and both teams in frame, did the
camera move, is the phone too low or tilted? -> per-frame verdict + one guidance hint. Design: docs/framing_check.md.

    from opencourtvision.framing import FramingChecker
    fc = FramingChecker(size=(640, 360), players_per_side=2)
    fc.lock(ref_gray, H_court2img)                # at setup: the court fit (tap 4 corners / auto) + its frame
    v = fc.check(gray, person_boxes)              # every frame (or every Nth): Verdict
    v.checks, v.fix                               # {steady, baselines, lines, level, players: ok|warn}, one plain fix
    v.verdict, v.warnings, v.hint                 # v1: "ok" | "court_cut_off" | ..., all warnings, short hint

App engine call (JSON in/out, ~1 Hz): opencourtvision.framing.api.FramingSession.

Inputs are a small gray frame (any size; 640x360 is plenty) and person boxes from any detector (Apple Vision human
rectangles on iPhone; YOLO / RF-DETR on the Mac). The checker itself is numpy + OpenCV ORB: no model, no faces.
"""
from .check import (
                    CHECK,
                    VERDICTS,
                    FramingChecker,
                    Verdict,
                    court_corners_px,
                    drift_px,
                    far_roi,
                    geometry_warnings,
                    horizon_roll_deg,
                    line_report,
                    normalise_H,
                    players_by_side,
)
from .status import CHECKS, summarise

__all__ = ["FramingChecker", "Verdict", "VERDICTS", "CHECK", "CHECKS", "summarise", "geometry_warnings", "players_by_side", "drift_px",
           "horizon_roll_deg", "court_corners_px", "normalise_H", "line_report", "far_roi"]
