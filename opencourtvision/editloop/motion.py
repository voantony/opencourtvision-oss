"""Vision fallback for segmentation, behind a flag (off by default): regional frame-motion energy, no object tracking.

Input: an npz with E[t, rows, cols] = mean |frame difference| per grid cell (gray 192x108, 24x12 cells, 6 Hz) and t,
as written by scripts/motion_energy.py (one cheap decode pass on the Mac: ~25 min for an 86-min 4K60 match, ~17% CPU).
Signal: `moving` = number of cells whose energy is > 3x that cell's 3-minute median (players moving; static texture,
wind in trees and lighting drift are normalised away). Players are never detected, identified or tracked here.

Measured on dev_match (eng notes §2.8.7, in-sample, one match): it cannot tell a fault pause from a between-points pause,
and it finds no missed points by itself; it helps precision by dropping idle-court audio segments and by gating
lower-threshold E3 candidates. Pose cues (serve stance) are a hook only (`pose_cues`): not built in this pass (cost).
"""
import numpy as np

from .segment import attach_faults, gap_fill

CUES = dict(ratio=3.0, base_s=180.0, idle_min=9.0, fill_min=15.0, fill_single_amp=14.0)


class MotionCues:
    def __init__(self, E, t, p=None, reliable=None, visible=1.0):
        """reliable: bool per sample (camera stable and court fit accepted, robust/camera_motion.py); visible: share of
        the court in frame (robust/framing.py). Cell-count thresholds scale with `visible`; windows with too few
        reliable samples give no decision (None), and no decision never drops a point."""
        from scipy.ndimage import median_filter
        self.p = {**CUES, **(p or {})}
        E = np.asarray(E, np.float32); self.t = np.asarray(t, float)
        F = E.reshape(len(self.t), -1)
        hz = 1.0 / max(1e-6, float(np.median(np.diff(self.t)))) if len(self.t) > 1 else 6.0
        step = max(1, int(round(hz)))                                  # baseline on a 1 Hz subsample, then held
        b = median_filter(F[::step], size=(max(3, int(self.p["base_s"])), 1)) + 0.05
        base = np.repeat(b, step, axis=0)[: len(F)]
        self.moving = (F / base > self.p["ratio"]).sum(1).astype(float)
        self.reliable = np.ones(len(self.t), bool) if reliable is None else np.asarray(reliable, bool)
        self.visible = float(visible)
        if self.visible < 1.0:
            self.p = {**self.p, **{k: self.p[k] * max(self.visible, 1e-3) for k in ("idle_min", "fill_min")}}

    @classmethod
    def load(cls, path, p=None):
        z = np.load(path)
        return cls(z["E"], z["t"], p)

    def activity(self, t0, t1, q=75):
        m = (self.t >= t0) & (self.t <= t1)
        return float(np.percentile(self.moving[m], q)) if m.any() else 0.0

    def decide(self, t0, t1, q=75, min_reliable=0.5):
        """activity over the reliable samples, or None (no decision) when fewer than min_reliable of them are."""
        m = (self.t >= t0) & (self.t <= t1)
        if not m.any(): return 0.0
        r = m & self.reliable
        if r.sum() < min_reliable * m.sum(): return None
        return float(np.percentile(self.moving[r], q))


def apply_cues(points, cues, rallies=None, start=0.0, sets=None):
    """Motion-gated pipeline after E1: (1) E3 at a lower onset threshold, keeping a structure candidate only if the court
    is active around it; (2) audio fault attach; (3) drop points during which the court is idle. Sources gain
    "motion_energy" where the cue decided."""
    p = cues.p
    pts = list(points)
    if rallies is not None:
        base = [q for q in pts if "structure" not in q["sources"]]
        filled, add = gap_fill(base, rallies, start, sets, {"single_amp": p["fill_single_amp"]})
        keep = [q for q in add if (_d(cues, q["t0"] - 1.0, q["t1"] + 3.0) or 0.0) >= p["fill_min"]]
        for q in keep: q["sources"] = set(q["sources"]) | {"motion_energy"}
        pts = sorted(base + keep, key=lambda x: x["t0"])
    pts = attach_faults(pts)
    return [q for q in pts if _keep(cues, q, p["idle_min"])]


def admit_candidates(points, cues, rallies=None, start=0.0, sets=None):
    """EU only (flags.motion_boundaries): steps (1) and (3) of apply_cues without the fault attach, for the smart
    segmenter, which merges faults itself."""
    p = cues.p
    pts = list(points)
    if rallies is not None:
        base = [q for q in pts if "structure" not in q["sources"]]
        _, add = gap_fill(base, rallies, start, sets, {"single_amp": p["fill_single_amp"]})
        keep = [q for q in add if (_d(cues, q["t0"] - 1.0, q["t1"] + 3.0) or 0.0) >= p["fill_min"]]   # no decision: not added
        for q in keep: q["sources"] = set(q["sources"]) | {"motion_energy"}
        pts = sorted(base + keep, key=lambda x: x["t0"])
    return [q for q in pts if _keep(cues, q, p["idle_min"])]


def _d(cues, a, b):
    return cues.decide(a, b) if hasattr(cues, "decide") else cues.activity(a, b)


def _keep(cues, q, th):
    """Drop only on a confident idle court; no decision (unstable camera, fit not accepted) keeps the audio point."""
    a = _d(cues, q["t0"], q["t1"] + 1.0)
    return a is None or a >= th


def live_gate(points, cues):
    """US-safe use of motion ('807/'160 design intent): motion only says whether play is live. Whole clips during which
    the court is idle are dropped; no clip is added, split, merged or trimmed, and every boundary stays audio-derived."""
    p = cues.p
    return [q for q in points if _keep(cues, q, p["idle_min"])]


def pose_cues(*_a, **_k):
    raise NotImplementedError("pose rally cues are not built in this pass (cost); see eng notes §2.8.7")
