"""Engine call for the app's record screen: one session per recording, about one check per second.

    from opencourtvision.framing.api import FramingSession
    s = FramingSession()                                         # 640x360 check frame, doubles
    s.setup(frame, court_corners=[(x, y)] * 4)                   # the setup tap (near-left, near-right, far-right, far-left)
    r = s.check(frame, person_boxes)                             # ~1 Hz before recording
    s.start_recording()
    r = s.check(frame, person_boxes, t=12.0)                     # ~1 Hz while recording (t = seconds into the video)
    r["status"], r["fix"], r["fix_check"], r["checks"]           # "ok" | "hint" | "warn", one fix, the check it belongs to
    s.dismiss(r["fix_check"])                                    # user dismissed a hint: not shown again this session
    s.relock(frame)                                              # user tapped Re-lock after a firm "phone moved"
    s.quality()                                                  # vamos.framing_quality.v1 -> app.analyze(framing=...)

Levels (framing/status.py): "hint" is quiet, dismissable and never blocks recording; "warn" is firm and only for what
breaks scoring (phone moved mid-recording, both baselines missing, a team out of frame ~20 s). blocks_recording is
always False: imperfect framing is recorded and analysed, and the engine widens its uncertainty from quality().

Before recording the session follows the court when the phone is adjusted (it re-locks itself on the tracked fit), so
moving the phone during setup is never a warning. frame: any size, gray or BGR (downscaled here). court_corners and
person boxes are in that frame's pixels; boxes from any detector (Apple Vision on iPhone, RF-DETR Nano on the Mac),
x0, y0, x1, y1[, conf]. The session keeps the setup frame and fit in memory only and writes nothing; quality() is
numbers only (no pixels, boxes, faces or identities). When #28 (opencourtvision.api) lands it re-exports this.
"""
import time

import numpy as np

from .check import FramingChecker, normalise_H
from .quality import QualityTrack

SCHEMA = "vamos.framing.v2"
SIZE = (640, 360)
LIVE = dict(WINDOW=5, DEBOUNCE=2)      # at ~1 Hz: team presence over 5 s; a message must hold 2 checks (~2 s)


def _small(frame, size):
    import cv2
    g = np.asarray(frame)
    if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    s = size[0] / g.shape[1]
    out = g if g.shape[1] == size[0] and g.shape[0] == size[1] else cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return out, s


def court_H_from_corners(corners):
    """Four tapped doubles-court corners (near-left, near-right, far-right, far-left; image px) -> H court m -> image px."""
    from ..pose.court import COURT_L, COURT_W, homography
    return normalise_H(homography([(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)], [tuple(map(float, c)) for c in corners]))


class FramingSession:
    def __init__(self, size=SIZE, players_per_side=2, p=None):
        self.size = tuple(size); self.pps = players_per_side
        self.p = {**LIVE, **(p or {})}
        self.fc = None; self.scale = 1.0; self.recording = False; self.track = QualityTrack(); self._t0 = None

    def setup(self, frame, court_corners=None, H=None):
        """Lock the setup frame and its court fit (from the 4-corner tap, or a given H court -> frame px)."""
        g, s = _small(frame, self.size)
        Hf = court_H_from_corners(court_corners) if H is None else normalise_H(H)
        self.scale = s
        dismissed = self.fc.dismissed if self.fc is not None else set()
        self.fc = FramingChecker(self.size, self.pps, self.p).lock(g, np.diag([s, s, 1.0]) @ Hf)
        self.fc.dismissed = dismissed
        return self._check(g, None, live=False, track=False)

    def start_recording(self):
        self.recording = True; self._t0 = time.monotonic()

    def stop_recording(self):
        self.recording = False

    def dismiss(self, check):
        if self.fc is not None and check: self.fc.dismiss(check)

    def relock(self, frame):
        """Accept the camera where it is now: the tracked fit becomes the new setup fit."""
        if self.fc is None: raise RuntimeError("call setup() first")
        g, _ = _small(frame, self.size)
        self._relock(g)
        return self._check(g, None, live=False, track=False)

    def _relock(self, g):
        """Lock on the current frame with the tracked fit. False (nothing changed) when the court can't be re-found."""
        Hn, _ = self.fc.estimate_H(g)
        if Hn is None: return False
        dismissed = self.fc.dismissed
        self.fc.lock(g, Hn)
        self.fc.dismissed = dismissed
        return True

    def check(self, frame, person_boxes=None, t=None):
        if self.fc is None: raise RuntimeError("call setup() first")
        t0 = time.perf_counter()
        g, s = _small(frame, self.size)
        b = None
        if person_boxes is not None:
            b = np.asarray(person_boxes, float).reshape(-1, np.shape(person_boxes)[-1] if np.size(person_boxes) else 5).copy(); b[:, :4] *= s
        r = self._check(g, b, live=True, track=True, t=t)
        if not self.recording and r["checks"]["steady"]["level"] != "ok":
            if self._relock(g):                              # setup: follow the court while the phone is adjusted
                r = self._check(g, b, live=False, track=False)
            else:
                r["checks"]["steady"] = dict(level="hint", fix="Can't find the court any more. Tap the four corners again.", dismissed=False)
                r.update(status="warn" if r["status"] == "warn" else "hint", fix=r["checks"]["steady"]["fix"], fix_check="steady")
        r["ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return r

    def _check(self, g, b, live, track, t=None):
        v = self.fc.check(g, b, live=live, track=track, recording=self.recording)
        if self.recording and track:
            self.track.add(t if t is not None else time.monotonic() - (self._t0 or time.monotonic()), v.quality)
        return dict(schema=SCHEMA, status=v.level, fix=v.fix, fix_check=v.fix_check, blocks_recording=False, checks=v.checks,
                    counts=v.counts, drift_px=v.drift_px, quality=v.quality, ms=0.0)

    def quality(self):
        """vamos.framing_quality.v1 for this recording: pass to app.analyze(framing=...)."""
        return self.track.to_json()
