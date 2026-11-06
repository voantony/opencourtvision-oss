"""Court geometry for the ball engine: image <-> court homography, the live-ball court gate and the ball size prior.

Court frame (same as `pose.court`): x across from the near-left doubles sideline (0..10.97 m), y from the near baseline
(0..23.77 m), metres, ground plane. A ball in the air is NOT on that plane, so mapping its image point through the ground
homography lands it farther from the camera than it is. Both gates below are built to tolerate that:

* `CourtGate.contains(px)`: the image region a live ball can occupy = the court footprint expanded by `margin` metres,
  swept straight up by `max_height` metres (a vertical prism, tested per height so it stays non-convex). "Up" in the
  image is approximated as image -y with the local ground scale (px per metre), which holds for a roughly level phone
  camera.
* `CourtGate.expected_diameter_px(px)`: 6.54-6.86 cm (ITF) times the local ground scale at the ball's image point. For an
  airborne ball that point is farther than the ball, so the prior is an under-estimate; `size_ok` accepts
  [lo, hi] x the prior plus an absolute slack (sub-pixel balls at the far baseline, compression, motion blur). Only the
  blob's minor axis is compared, because motion blur stretches the major axis.
"""
import numpy as np

from ..pose.court import COURT_L, COURT_W, apply, homography

BALL_D_MIN, BALL_D_MAX = 0.0654, 0.0686       # m, ITF ball diameter range


def _hull(points):
    """Convex hull (Andrew's monotone chain), counter-clockwise, as (N, 2) float array."""
    p = sorted(map(tuple, np.asarray(points, float)))
    if len(p) <= 2: return np.asarray(p)

    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0: hi.pop()
        hi.append(q)
    return np.asarray(lo[:-1] + hi[:-1])


def point_in_poly(pts, poly):
    """Even-odd rule; pts (N, 2), poly (M, 2) -> bool (N,)."""
    pts = np.atleast_2d(np.asarray(pts, float)); poly = np.asarray(poly, float)
    x, y = pts[:, 0], pts[:, 1]; inside = np.zeros(len(pts), bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xi)
    return inside


class CourtGate:
    """Built from a court->image homography (3x3) or from >= 4 (court_m, image_px) point pairs via `from_points`."""

    def __init__(self, H_court2img, margin=(3.66, 6.40), max_height=8.0, width=COURT_W, length=COURT_L, image_size=None):
        self.H = np.asarray(H_court2img, float)
        self.Hinv = np.linalg.inv(self.H)
        self.width, self.length = width, length
        mx, my = margin if np.ndim(margin) else (margin, margin)
        self.margin = (float(mx), float(my))
        self.max_height = float(max_height)
        self.image_size = image_size
        # The near run-off can reach behind or just below the camera, where the ground homography blows up: keep the
        # near edge where it is in front of the camera and within 2 image heights of the frame (when image_size is known).
        self._lift_cache = {}
        y_near = -my
        for yn in np.linspace(-my, 0.0, 33):
            p = self.H @ np.array([width / 2, yn, 1.0])
            if p[2] > 0 and (image_size is None or p[1] / p[2] <= 2.0 * image_size[1]): y_near = yn; break
        else:
            y_near = 0.0
        self.y_near = float(y_near)
        xs = np.linspace(-mx, width + mx, 17); ys = np.linspace(y_near, length + my, 17)
        ring = np.array([(x, y_near) for x in xs] + [(width + mx, y) for y in ys] + [(x, length + my) for x in xs[::-1]] + [(-mx, y) for y in ys[::-1]])
        g = apply(self.H, ring)
        self.footprint = g                                # ordered ring (projective image of a rectangle: convex)
        s = self.ball_scale(g)
        self.poly = _hull(np.r_[g, g - np.c_[np.zeros(len(g)), s * self.max_height]])   # loose outer bound (drawing only)

    @classmethod
    def from_points(cls, court_m, image_px, **kw):
        return cls(homography(court_m, image_px), **kw)

    def to_court(self, px):
        return apply(self.Hinv, px)

    def to_image(self, m):
        return apply(self.H, m)

    def ground_point(self, px):
        """Image points -> court metres, clamped to the expanded footprint. Points at or above the horizon (the ground
        homography maps them behind the camera) are put on the far edge of the footprint."""
        px = np.atleast_2d(np.asarray(px, float))
        p = np.c_[px, np.ones(len(px))] @ self.Hinv.T
        w = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = p[:, :2] / w[:, None]
        mx, my = self.margin
        far = (w <= 0) | ~np.isfinite(c).all(1)
        c[far] = [self.width / 2, self.length + my]
        c[:, 0] = np.clip(c[:, 0], -mx, self.width + mx)
        c[:, 1] = np.clip(c[:, 1], getattr(self, "y_near", -my), self.length + my)
        return c

    def px_per_m(self, px):
        """Local ground scale (image px per metre) at image points (N, 2): mean of the x/y unit-step lengths at the
        (clamped) ground point, so balls high in the air or near the horizon get a finite, plausible scale."""
        c = self.ground_point(px)
        g = self.to_image(c)
        dx = self.to_image(c + [1.0, 0.0]) - g
        dy = self.to_image(c + [0.0, 1.0]) - g
        return 0.5 * (np.hypot(dx[:, 0], dx[:, 1]) + np.hypot(dy[:, 0], dy[:, 1]))

    def ground_speed_points(self, px):
        """Image points -> court metres on the ground plane, NaN where the ray does not meet the ground in front of
        the camera (at or above the horizon)."""
        px = np.atleast_2d(np.asarray(px, float))
        p = np.c_[px, np.ones(len(px))] @ self.Hinv.T
        with np.errstate(divide="ignore", invalid="ignore"):
            c = p[:, :2] / p[:, 2:]
        c[p[:, 2] <= 0] = np.nan
        return c

    def ball_scale(self, px):
        """Unforeshortened local scale (px per metre) at image points: the larger of the x/y unit-step lengths at the
        clamped ground point. A sphere's apparent size and vertical image distances follow this scale (the court
        direction pointing away from the camera is foreshortened, the one across the view is not)."""
        c = self.ground_point(px)
        g = self.to_image(c)
        dx = self.to_image(c + [1.0, 0.0]) - g
        dy = self.to_image(c + [0.0, 1.0]) - g
        return np.maximum(np.hypot(dx[:, 0], dx[:, 1]), np.hypot(dy[:, 0], dy[:, 1]))

    def contains(self, px, heights=None, diam_px=None, size_tol=1.4, slack_px=0.5):
        """True where an image point can be a ball over our court: for some height z in [0, max_height] it lies inside
        the footprint ring lifted to height z (`_lifted`), so the test follows the court's vertical prism (non-convex
        in the image), unlike a convex hull.

        From one camera a ball on the next court is often on the same ray as a ball 2-3 m above our side margin; the
        apparent size breaks the tie. With diam_px (measured minor axis per point, NaN = unknown) a height hypothesis
        also needs the size to match a 6.54-6.86 cm ball at that ground point within size_tol (x or /) + slack_px."""
        px = np.atleast_2d(np.asarray(px, float))
        zs = np.linspace(0.0, self.max_height, 33) if heights is None else np.asarray(heights, float)
        d = None if diam_px is None else np.asarray(diam_px, float).reshape(-1)

        def size_fits(g):
            if d is None: return np.ones(len(px), bool)
            s = self.ball_scale(g)
            lo, hi = BALL_D_MIN * s / size_tol - slack_px, BALL_D_MAX * s * size_tol + slack_px
            return np.isnan(d) | ((d >= lo) & (d <= hi))

        ok = np.zeros(len(px), bool)
        for z in zs:
            ring = self._lifted(z)
            inside = point_in_poly(px, ring)
            if not inside.any(): continue
            if d is None:
                ok |= inside
            else:                                  # ground point under p at this height, for the size check
                ok |= inside & size_fits(apply(self._lift_h(z, inverse=True), px))
        return ok

    def _lift_h(self, z, inverse=False):
        """Homography ground-image -> lifted-image at height z (DLT on the ring; the plane z = const is a homography)."""
        key = (round(float(z), 4), inverse)
        if key not in self._lift_cache:
            a, b = self.footprint, self._lifted(z)
            self._lift_cache[key] = homography(b, a) if inverse else homography(a, b)
        return self._lift_cache[key]

    def _lifted(self, z):
        """Footprint ring as seen at height z: each ground point moved image-up by z x the local ball scale (an
        approximation of the plane z = const for a roughly level camera)."""
        key = round(float(z), 4)
        if key not in self._lift_cache:
            g = self.footprint
            self._lift_cache[key] = g - np.c_[np.zeros(len(g)), z * self.ball_scale(g)]
        return self._lift_cache[key]

    def on_court(self, px, buffer_m=2.0):
        """Ground point (e.g. a bounce) within the court rectangle + buffer_m (tennis-vision's bounce locality rule)."""
        c = np.atleast_2d(self.to_court(px))
        return (c[:, 0] >= -buffer_m) & (c[:, 0] <= self.width + buffer_m) & (c[:, 1] >= -buffer_m) & (c[:, 1] <= self.length + buffer_m)

    def expected_diameter_px(self, px):
        """(lo, hi) expected ball diameter in px at image points (ground-plane assumption, see module doc)."""
        s = self.ball_scale(px)
        return BALL_D_MIN * s, BALL_D_MAX * s

    def size_ok(self, px, minor_px, lo=0.5, hi=2.5, slack_px=1.5):
        """Blob minor axis consistent with a 6.54-6.86 cm ball at that depth: lo*d_min - slack <= minor <= hi*d_max + slack."""
        dmin, dmax = self.expected_diameter_px(px)
        m = np.asarray(minor_px, float)
        return (m >= lo * dmin - slack_px) & (m <= hi * dmax + slack_px)


def transfer_homography(ref_bgr, frame_bgr, H_ref, mask=None, min_inliers=40):
    """Carry a court->image homography from a reference frame to another frame of the same camera that was digitally
    cropped / zoomed / slightly moved (e.g. a SwingVision export re-framing per point, or a bumped phone): ORB matches,
    RANSAC similarity (scale + rotation + shift; valid for the whole scene under a digital crop). mask: uint8, 0 where
    features are ignored (overlays). -> (H_court2img for frame, info dict) or (None, info) when it is not reliable."""
    import cv2
    g1 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(g1, mask); k2, d2 = orb.detectAndCompute(g2, mask)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10: return None, dict(inliers=0)
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [a for a, *b in m if b and a.distance < 0.75 * b[0].distance]
    if len(good) < min_inliers: return None, dict(inliers=0, matches=len(good))
    p1 = np.float32([k1[a.queryIdx].pt for a in good]); p2 = np.float32([k2[a.trainIdx].pt for a in good])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    n = int(inl.sum()) if inl is not None else 0
    if A is None or n < min_inliers: return None, dict(inliers=n, matches=len(good))
    S = np.vstack([A, [0, 0, 1]])
    scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    return S @ np.asarray(H_ref, float), dict(inliers=n, matches=len(good), scale=round(scale, 3), shift=[round(float(A[0, 2]), 1), round(float(A[1, 2]), 1)])
