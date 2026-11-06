"""Framing checker core (pure numpy + OpenCV ORB via ball.geometry.transfer_homography; no learned model).

Coordinates: court metres as in pose.court (x across from the near-left doubles sideline 0..10.97, y from the near
baseline 0..23.77, net at 11.885). H maps court -> image px of the checked frame.

Verdicts (priority order; every active one is listed in Verdict.warnings, the first is Verdict.verdict):
  camera_moved       court fit drifted more than DRIFT_FRAC of the frame diagonal from the fit locked at setup
  court_cut_off      a court corner / line is outside the frame (Verdict.cut lists which corners and sides)
  lines_unclear      the court lines are where the fit says, but the picture barely shows them (smudged lens, glare,
                     dusk, heavy blur): share of in-frame line samples on a bright thin ridge < LINES_MIN
  too_low_or_tilted  horizon roll > ROLL_DEG, or the far half too foreshortened (phone too low) for far-court calls
  far_team_missing   fewer than players_per_side people standing in the far half / near half over the last WINDOW
  near_team_missing  frames (live), or in this frame (single-frame call)
  ok
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..pose.court import COURT_L, COURT_W

NET_Y = COURT_L / 2
CHECK = dict(
    EDGE_FRAC=0.005,        # a corner closer than this (share of width/height) to the frame edge counts as cut off
    DRIFT_FRAC=0.01,        # camera_moved when visible court points moved > 1% of the frame diagonal (~7 px at 640x360)
    ROLL_DEG=4.0,           # horizon roll above this -> level the phone
    ELEV_MIN_DEG=3.5,       # viewing elevation at the far baseline below this -> phone too low (dev_match setup: 5.3-5.6 deg)
    ZONE_X=(-4.0, COURT_W + 4.0), ZONE_Y=(-6.5, COURT_L + 6.5),   # play zone for counting players
    BENCH=(0.5, 3.0),       # not a player: > 0.5 m outside a sideline and within 3 m of the net (courtside benches;
                            # dev_match had people sitting there all match, counted as a near "team" before this)
    WINDOW=20,              # live: frames in the team-presence window (2 s at 10 Hz)
    PRESENT_MIN=0.3,        # live: a side is present if it shows enough players in >= 30% of the window
    DEBOUNCE=3,             # live: a warning must hold this many consecutive checks before it is shown
    TEAM_MIN=1,             # a team is missing when fewer than this many of its players are seen (1 = nobody on that side)
    HALF_VISIBLE_MIN=0.5,   # a half that is mostly out of frame is reported as court_cut_off, not as a missing team
    MOTION="homography",    # drift model: "homography" (ORB + RANSAC 8-dof, sees pitch/yaw) or "similarity" (ball.geometry)
    ORB_N=1500, MIN_INLIERS=25,
    LINE_IN_MIN=0.98,       # a line counts as "in frame" when at least this share of it is inside the picture
    LINES_MIN=0.35,         # lines_unclear when under this share of in-frame line samples sit on a visible white line
                            # (dev_match clean frames with the tracked fit: ~0.8; see docs/framing_check.md)
    TOPHAT_K=7, TOPHAT_T=14,  # line ridge filter at 640 px wide (kernel scales with width)
    BASELINE_MISSING=0.5,   # a baseline is "missing" when under half of it is in the picture (both missing -> firm warn)
    TEAM_WARN_CHECKS=20,    # live: a team missing in >= TEAM_WARN_SHARE of the last 20 checks (~20 s at 1 Hz) -> firm warn;
    TEAM_WARN_SHARE=0.8,    # a stray detection (ball kid, ghost box) doesn't reset it. Less than that -> hint at most
    SHARP_MIN=0.5,          # the picture lost > half its detail vs setup -> blur / smudge (lines_unclear), never "moved"
)
VERDICTS = ("camera_moved", "court_cut_off", "lines_unclear", "too_low_or_tilted", "far_team_missing", "near_team_missing", "ok")
CORNERS = dict(near_left=(0.0, 0.0), near_right=(COURT_W, 0.0), far_right=(COURT_W, COURT_L), far_left=(0.0, COURT_L))
LINES = dict(near_baseline=((0.0, 0.0), (COURT_W, 0.0)), far_baseline=((0.0, COURT_L), (COURT_W, COURT_L)),
             left_sideline=((0.0, 0.0), (0.0, COURT_L)), right_sideline=((COURT_W, 0.0), (COURT_W, COURT_L)),
             left_singles=((1.37, 0.0), (1.37, COURT_L)), right_singles=((COURT_W - 1.37, 0.0), (COURT_W - 1.37, COURT_L)),
             near_service=((1.37, 5.485), (COURT_W - 1.37, 5.485)), far_service=((1.37, COURT_L - 5.485), (COURT_W - 1.37, COURT_L - 5.485)),
             centre_service=((COURT_W / 2, 5.485), (COURT_W / 2, COURT_L - 5.485)))
SIDES = dict(near_baseline=("near_left", "near_right"), far_baseline=("far_left", "far_right"),
             left_sideline=("near_left", "far_left"), right_sideline=("near_right", "far_right"))


def normalise_H(H):
    """Scale H so w > 0 in front of the camera (court centre), and H[2,2]-free of sign flips. DLT fits can come out with
    w < 0 everywhere, which breaks 'q[2] > 0' visibility tests."""
    H = np.asarray(H, float)
    w = (H @ np.array([COURT_W / 2, NET_Y, 1.0]))[2]
    return H / w * abs(w) / np.abs(H).max() if w != 0 else H


def _proj(H, pts):
    P = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(H, float).T
    return P[:, :2] / P[:, 2:3], P[:, 2]


def court_corners_px(H):
    xy, w = _proj(normalise_H(H), list(CORNERS.values()))
    return {k: (tuple(xy[i]), bool(w[i] > 0)) for i, k in enumerate(CORNERS)}


def _court_grid(n=12):
    X, Y = np.meshgrid(np.linspace(0, COURT_W, n), np.linspace(0, COURT_L, 2 * n))
    return np.c_[X.ravel(), Y.ravel()]


def horizon_roll_deg(H):
    """Roll of the ground plane's vanishing line (the horizon) in degrees, independent of where the phone points."""
    a, b, _ = np.linalg.inv(normalise_H(H)).T @ np.array([0.0, 0.0, 1.0])
    ang = np.degrees(np.arctan2(-a, b))
    return float((ang + 90.0) % 180.0 - 90.0)


def far_elevation_deg(H):
    """Camera elevation seen at the far baseline centre, from foreshortening: a small ground patch looks
    sin(elevation) times shorter along the line of sight than across it. Independent of resolution and zoom."""
    c = [(COURT_W / 2, COURT_L - 0.5), (COURT_W / 2, COURT_L + 0.5), (COURT_W / 2 - 0.5, COURT_L), (COURT_W / 2 + 0.5, COURT_L)]
    xy, w = _proj(normalise_H(H), c)
    if (w <= 0).any(): return 0.0
    dy, dx = np.hypot(*(xy[1] - xy[0])), np.hypot(*(xy[3] - xy[2]))
    return float(np.degrees(np.arcsin(min(1.0, dy / max(dx, 1e-9)))))


def geometry_warnings(H, size, p=None):
    """-> (warnings list, cut dict(corners=[...], sides=[...], edges=[...]), info)."""
    p = {**CHECK, **(p or {})}
    W, Hh = size; mx, my = p["EDGE_FRAC"] * W, p["EDGE_FRAC"] * Hh
    out_c, edges, over = [], set(), {}
    for k, ((u, v), front) in court_corners_px(H).items():
        if not front:
            e = "top" if k.startswith("far") else "bottom"
            out_c.append(k); edges.add(e); over[k] = {e: float("inf")}; continue
        o = {}
        if u < mx: o["left"] = mx - u
        if u > W - mx: o["right"] = u - (W - mx)
        if v < my: o["top"] = my - v
        if v > Hh - my: o["bottom"] = v - (Hh - my)
        if o: out_c.append(k); edges.update(o); over[k] = {e: round(float(x), 1) for e, x in o.items()}
    sides = [s for s, (a, b) in SIDES.items() if a in out_c or b in out_c]
    warn = []
    if out_c: warn.append("court_cut_off")
    roll = horizon_roll_deg(H); elev = far_elevation_deg(H)
    low = elev < p["ELEV_MIN_DEG"]
    if abs(roll) > p["ROLL_DEG"] or low: warn.append("too_low_or_tilted")
    return warn, dict(corners=out_c, sides=sides, edges=sorted(edges), over_px=over), dict(roll_deg=round(roll, 2), elevation_deg=round(elev, 2),
                                                                              too_low=bool(low), tilted=bool(abs(roll) > p["ROLL_DEG"]))


def line_mask(gray, p=None):
    """Bright thin ridges (court paint) as a 0/1 mask: white top-hat, threshold, 3x3 dilate. Same filter as the per-point
    court refit (tools/ball_track_v2/court_refit.py), sized for a 640 px wide check frame."""
    import cv2
    p = {**CHECK, **(p or {})}
    g = np.asarray(gray)
    if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    k = max(5, int(round(p["TOPHAT_K"] * g.shape[1] / 640)) | 1)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.dilate((th > p["TOPHAT_T"]).astype(np.uint8), np.ones((3, 3), np.uint8))


def line_report(H, size, gray=None, n=40, p=None):
    """Per court line: share inside the frame, and (with a frame) share of those in-frame samples on visible paint.
    -> dict(lines={name: dict(in_frame, support)}, support=overall support or None)."""
    W, Hh = size; Hn = normalise_H(H)
    m = None if gray is None else line_mask(gray, p)
    out = {}; hit = tot = 0
    s = np.linspace(0, 1, n)[:, None]
    for name, (a, b) in LINES.items():
        q, w = _proj(Hn, (1 - s) * np.array(a) + s * np.array(b))
        ok = (w > 0) & (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < Hh)
        sup = None
        if m is not None and ok.any():
            qq = q[ok].astype(int); on = int(m[qq[:, 1], qq[:, 0]].sum()); hit += on; tot += len(qq); sup = round(on / len(qq), 3)
        out[name] = dict(in_frame=round(float(ok.mean()), 3), support=sup)
    return dict(lines=out, support=round(hit / tot, 3) if tot else None)


def far_roi(H, size, pad_m=(2.0, 3.0), min_w=160):
    """Crop rect (x0, y0, x1, y1) around the far half of the court, for a second detector pass: far players are ~20 px
    tall at 640x360 and a full-frame detector misses many (same idea as the Apple Vision far-ROI pass). None if not visible."""
    W, Hh = size
    pts = [(-pad_m[0], NET_Y), (COURT_W + pad_m[0], NET_Y), (COURT_W + pad_m[0], COURT_L + pad_m[1]), (-pad_m[0], COURT_L + pad_m[1])]
    q, w = _proj(normalise_H(H), pts)
    if (w <= 0).any(): return None
    x0, y0 = q.min(0); x1, y1 = q.max(0)
    y0 -= 0.6 * (y1 - y0)                               # room for the players' bodies above their feet
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(np.ceil(x1))), min(Hh, int(np.ceil(y1)))
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    if x1 - x0 < min_w:
        c = (x0 + x1) / 2; x0, x1 = max(0, int(c - min_w / 2)), min(W, int(c + min_w / 2))
    return x0, y0, x1, y1


def _boxes(boxes):
    """Any box list / array -> (N, >=4) float array of finite boxes (NaN rows dropped)."""
    if boxes is None or np.size(boxes) == 0: return np.zeros((0, 4))
    b = np.asarray(boxes, float)
    b = b.reshape(-1, b.shape[-1])
    return b[np.isfinite(b[:, :4]).all(1)]


def players_by_side(H, boxes, p=None):
    """Count people whose feet (box bottom centre) stand in the play zone, per half. boxes: (N, >=4) x0,y0,x1,y1[,conf]."""
    p = {**CHECK, **(p or {})}
    b = _boxes(boxes)
    if not len(b): return dict(near=0, far=0)
    feet = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
    Hi = np.linalg.inv(normalise_H(H))
    c, w = _proj(Hi, feet)
    ok = (w != 0) & (c[:, 0] >= p["ZONE_X"][0]) & (c[:, 0] <= p["ZONE_X"][1]) & (c[:, 1] >= p["ZONE_Y"][0]) & (c[:, 1] <= p["ZONE_Y"][1])
    bench = ((c[:, 0] < -p["BENCH"][0]) | (c[:, 0] > COURT_W + p["BENCH"][0])) & (np.abs(c[:, 1] - NET_Y) < p["BENCH"][1])
    ok &= ~bench
    return dict(near=int((ok & (c[:, 1] < NET_Y)).sum()), far=int((ok & (c[:, 1] >= NET_Y)).sum()))


def drift_px(H_a, H_b, size):
    """Mean image displacement of the court points visible in frame under H_a, between fits H_a and H_b (px)."""
    g = _court_grid()
    a, wa = _proj(normalise_H(H_a), g); b, wb = _proj(normalise_H(H_b), g)
    W, Hh = size
    vis = (wa > 0) & (wb > 0) & (a[:, 0] >= 0) & (a[:, 0] < W) & (a[:, 1] >= 0) & (a[:, 1] < Hh)
    if not vis.any(): return float("inf")
    return float(np.hypot(*(a[vis] - b[vis]).T).mean())


def _sharpness(gray):
    """Picture detail: mean gradient magnitude at 320 px wide. Lens smudge, fog or focus loss lower it; zooming in or
    moving the phone barely does (dev_match: blur sigma 4 -> 0.41x, haze to 20% contrast -> 0.2x, zoom 1.5x -> 0.94x).
    Laplacian variance was tried first and rejected: sigma-1 blur alone drops it to 0.04x (sensor noise dominates)."""
    import cv2
    g = np.asarray(gray)
    if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (320, max(1, int(round(320 * g.shape[0] / g.shape[1])))), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.mean(np.hypot(cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1))))


def _to_bgr(gray):
    g = np.asarray(gray)
    return np.repeat(g[..., None], 3, axis=2) if g.ndim == 2 else g


def _hint(warnings, cut, info, counts, pps):
    """One short instruction for the most important warning."""
    if not warnings: return "looks good"
    w = warnings[0]
    if w == "camera_moved": return "phone moved: put it back or tap re-lock"
    if w == "court_cut_off":
        e = set(cut["edges"])
        if {"left", "right"} <= e or {"top", "bottom"} <= e: return "move phone back (or switch to 0.5x)"
        if "top" in e: return "tilt up"
        if "bottom" in e: return "tilt down"
        if "left" in e: return "turn phone left"
        if "right" in e: return "turn phone right"
    if w == "lines_unclear": return "wipe the lens / check the light"
    if w == "too_low_or_tilted":
        if info.get("tilted"):
            return f"level the phone (rotate {abs(info['roll_deg']):.0f}° {'clockwise' if info['roll_deg'] > 0 else 'anticlockwise'})"
        return "raise phone (higher = better far-court calls)"
    if w == "far_team_missing": return "far players not in view: check the far side is in frame"
    if w == "near_team_missing": return "near players not in view: tilt down or move phone back"
    return "check framing"


@dataclass
class Verdict:
    verdict: str
    warnings: list
    hint: str
    cut: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    drift_px: float = None
    drift_frac: float = None
    info: dict = field(default_factory=dict)
    shown: list = field(default_factory=list)     # live: debounced warnings to actually display
    checks: dict = field(default_factory=dict)    # the five app-facing checks: {name: {"status": "ok"|"warn", "fix": str|None}}
    fix: str = None                               # the one plain-language fix to show now (None when all ok)
    level: str = "ok"                             # overall: "ok" | "hint" (quiet, dismissable) | "warn" (firm); never blocks recording
    fix_check: str = None                         # which check the fix belongs to (so the UI can dismiss that hint)
    quality: dict = field(default_factory=dict)   # framing quality signal for the engine (framing/quality.py)
    H: object = None                              # court fit used for this check (court m -> check-frame px), for drawing

    def to_dict(self):
        return dict(verdict=self.verdict, warnings=list(self.warnings), hint=self.hint, cut=self.cut, counts=self.counts,
                    drift_px=self.drift_px, drift_frac=self.drift_frac, info=self.info, shown=list(self.shown),
                    checks=self.checks, fix=self.fix, level=self.level, fix_check=self.fix_check, quality=self.quality)


class FramingChecker:
    """Stateful checker: lock() at setup, check() per frame. track=False skips the drift estimate (setup screen before
    the fit is locked: geometry + players only)."""

    def __init__(self, size, players_per_side=2, p=None):
        self.size = tuple(int(v) for v in size); self.pps = int(players_per_side)
        self.p = {**CHECK, **(p or {})}
        self.H_lock = None; self.ref = None
        self.hist = deque(maxlen=self.p["WINDOW"]); self.streak = {}
        self.dismissed = set()                       # checks whose hints the user dismissed (kept across re-locks)
        self.team_hist = deque(maxlen=self.p["TEAM_WARN_CHECKS"])

    def dismiss(self, check):
        """The user dismissed this check's hint: don't show its hints again this session (firm warnings still show)."""
        self.dismissed.add(check)

    def lock(self, ref_gray, H):
        self.H_lock = normalise_H(H); self.ref = None if ref_gray is None else _to_bgr(ref_gray)
        self._ref_feat = None; self.hist.clear(); self.streak = {}; self.team_hist.clear()
        self._ref_sharp = None if ref_gray is None else _sharpness(ref_gray)
        return self

    def estimate_H(self, gray, boxes=None):
        """Current court fit from the locked frame (players masked so moving people do not vote). None if not reliable."""
        img = _to_bgr(gray)
        mask = np.full(img.shape[:2], 255, np.uint8)
        for b in _boxes(boxes):
            x0, y0, x1, y1 = (int(round(v)) for v in b[:4]); mask[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = 0
        if self.p["MOTION"] == "similarity":
            from ..ball.geometry import transfer_homography
            return transfer_homography(self.ref, img, self.H_lock, mask=mask, min_inliers=self.p["MIN_INLIERS"])
        return self._orb_homography(img, mask)

    def _ref_weak(self):
        """True when the setup frame has too few features (< 2 x MIN_INLIERS) for 'no match' to mean 'moved'."""
        if self.p["MOTION"] == "similarity" or self.ref is None: return False
        if getattr(self, "_ref_feat", None) is None: self._orb_homography(self.ref, None)
        return len(self._ref_feat[0]) < 2 * self.p["MIN_INLIERS"]

    def _orb_homography(self, img, mask):
        import cv2
        if getattr(self, "_ref_feat", None) is None:
            orb = cv2.ORB_create(self.p["ORB_N"])
            g1 = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
            self._orb = orb; self._ref_feat = orb.detectAndCompute(g1, None)
        k1, d1 = self._ref_feat
        g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k2, d2 = self._orb.detectAndCompute(g2, mask)
        if mask is not None:                                # players cover so much that too little setup texture is left
            vis = sum(1 for k in k1 if mask[min(int(k.pt[1]), mask.shape[0] - 1), min(int(k.pt[0]), mask.shape[1] - 1)])
            if vis < 2 * self.p["MIN_INLIERS"]: return None, dict(inliers=0, untrackable=True)
        if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10: return None, dict(inliers=0)
        m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
        good = [a for a, *b in m if b and a.distance < 0.75 * b[0].distance]
        if len(good) < self.p["MIN_INLIERS"]: return None, dict(inliers=0, matches=len(good))
        p1 = np.float32([k1[a.queryIdx].pt for a in good]); p2 = np.float32([k2[a.trainIdx].pt for a in good])
        G, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        if G is None or n < self.p["MIN_INLIERS"]: return None, dict(inliers=n, matches=len(good))
        return G @ self.H_lock, dict(inliers=n, matches=len(good))

    def check(self, gray, boxes=None, live=True, track=True, recording=False):
        p = self.p; diag = float(np.hypot(*self.size))
        H = self.H_lock; d = None; moved = False; tinfo = {}; blurred = False
        if track and self.ref is not None:
            Hn, tinfo = self.estimate_H(gray, boxes)
            sharp = _sharpness(gray) / max(self._ref_sharp or 1e-9, 1e-9)
            tinfo = {**tinfo, "sharpness_ratio": round(sharp, 3)}
            if sharp < p["SHARP_MIN"]:
                blurred = True                                  # the picture went soft: feature matches are not trustworthy;
            elif Hn is None and (tinfo.get("untrackable") or self._ref_weak()):   # can't tell; don't call it a move
                tinfo = {**tinfo, "untrackable": True}          # setup frame has too little texture to track: can't tell, don't nag
            elif Hn is None:
                moved = True; d = float("inf")                  # the court is not where it was: treat as moved
            else:
                d = drift_px(self.H_lock, Hn, self.size); moved = d > p["DRIFT_FRAC"] * diag; H = normalise_H(Hn)
        warns, cut, info = geometry_warnings(H, self.size, p)
        lr = line_report(H, self.size, gray, p=p)
        info["lines"] = lr["lines"]; info["line_support"] = lr["support"]
        if blurred or (not moved and lr["support"] is not None and lr["support"] < p["LINES_MIN"]): warns.append("lines_unclear")
        counts = players_by_side(H, boxes, p)
        if boxes is not None: self.hist.append(counts)     # boxes=None: the detector did not run this time; no player call
        need = min(self.pps, p["TEAM_MIN"])
        if boxes is None and (not live or len(self.hist) < 3):
            far_missing = near_missing = False
        elif live and len(self.hist) >= 3:
            pres = {s: np.mean([c[s] >= need for c in self.hist]) for s in ("far", "near")}
            far_missing, near_missing = pres["far"] < p["PRESENT_MIN"], pres["near"] < p["PRESENT_MIN"]
        else:
            far_missing, near_missing = counts["far"] < need, counts["near"] < need
        # a half that is mostly out of frame already explains missing players: report the cause, not both
        from ..robust.framing import side_visible_fraction
        vis = {s: side_visible_fraction(H, self.size, s) for s in ("far", "near")}
        info["half_visible"] = {k: round(v, 3) for k, v in vis.items()}
        if far_missing and vis["far"] >= p["HALF_VISIBLE_MIN"]: warns.append("far_team_missing")
        if near_missing and vis["near"] >= p["HALF_VISIBLE_MIN"]: warns.append("near_team_missing")
        if moved: warns.insert(0, "camera_moved")
        warns = [v for v in VERDICTS if v in warns]
        for v in VERDICTS[:-1]:
            self.streak[v] = self.streak.get(v, 0) + 1 if v in warns else 0
        shown = [v for v in warns if self.streak.get(v, 0) >= p["DEBOUNCE"]] if live else list(warns)
        if live and boxes is not None:
            self.team_hist.append(dict(far="far_team_missing" in warns, near="near_team_missing" in warns))
        full = live and len(self.team_hist) == self.team_hist.maxlen
        sustained = {s: full and float(np.mean([h[s] for h in self.team_hist])) >= p["TEAM_WARN_SHARE"] for s in ("far", "near")}
        info.update(transfer=tinfo)
        v = Verdict(verdict=warns[0] if warns else "ok", warnings=warns, hint=_hint(warns, cut, info, counts, self.pps),
                    cut=cut, counts=counts, drift_px=None if d is None else round(d, 2),
                    drift_frac=None if d is None else (round(d / diag, 5) if np.isfinite(d) else None), info=info, shown=shown, H=H)
        from .quality import frame_quality
        from .status import summarise
        v.quality = frame_quality(H, self.size, info, moved=moved, blurred=blurred, tracked=bool(track and self.ref is not None and not tinfo.get("untrackable")))
        v.checks, v.fix, v.level, v.fix_check = summarise(v, live=live, p=p, recording=recording, sustained=sustained,
                                                          dismissed=self.dismissed)
        return v
