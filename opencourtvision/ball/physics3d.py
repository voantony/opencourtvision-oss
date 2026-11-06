"""Physics ball tracker: every reported track is a real 3D ball flight seen through a calibrated camera.

Plain version: we work out where the camera is from the court lines, then only accept a run of ball detections if a
real tennis ball (gravity + air drag, optional spin lift) could have flown that path in 3D and produced those pixels.
Bounces are where a flight meets the ground (z = 0), so the bounce spot is solved, not guessed from pixel wiggles.
Anything that cannot be explained (racket heads, shoes, line glints, people on other courts) is dropped.

Pipeline (own code, numpy only, no third-party model code):
  Camera.from_homography   full P = K[R|t] from the court homography (square pixels, centred principal point, focal
                           length from the two orthonormality constraints), Monte-Carlo uncertainty from tap noise.
  ballistic_dlt            closed-form 3D flight from >= 3 pixels: with a known acceleration, x(t) = p0 + v0 t + a t^2/2
                           is LINEAR in (p0, v0), and so is each projection constraint (u P3 - P1) X = 0. Drag is
                           handled by re-linearising a around the current velocity (2-3 rounds).
  fit_flight               Levenberg-Marquardt on the true ODE (gravity + quadratic drag [+ Magnus lift]), RK4,
                           robust (Huber IRLS) reprojection error; returns covariance (J^T W J)^-1 sigma^2.
  propose                  RANSAC-style seeds: candidate triples at frames (f, f+s, f+2s), solved in one batched
                           6x6 linear solve, physically gated (speed, height, court volume), then grown frame by frame
                           with the physics prediction as the gate.
  select                   global optimum over the segment-and-event structure: weighted interval scheduling (exact
                           DP) over the proposed flights; segment score = detection log-likelihood ratio minus an MDL
                           charge for its parameters; transition terms reward a solved bounce (shared ground point,
                           restitution in range) or a hit (contact near a player), penalise unexplained breaks.
  refine_bounces           joint fit of the flight before and after each bounce with the ground point shared: bounce
                           time, position (and covariance) are outputs of the fit; P(in) from that covariance.
  chi-square test          each flight's reprojection chi^2 vs its dof; failures are rejected, not reported.
Coordinates: court metres, x across (0..10.97), y along (0..23.77, net at 11.885), z up. Image px of the frames the
candidates come from (1280x720 analysis frames by default).
"""
import math

import numpy as np

G = np.array([0.0, 0.0, -9.81])
COURT_W, COURT_L, NET_Y = 10.97, 23.77, 11.885
RHO_AIR = 1.2


class Ball:
    """Ball constants. kd = rho Cd A / (2 m) [1/m]; drag accel = -kd |v| v. lift_max bounds |Magnus| / drag."""

    def __init__(self, mass=0.057, radius=0.0335, cd=0.55, e_range=(0.55, 0.9), mu_range=(0.45, 1.0), lift_max=0.6):
        self.mass, self.radius, self.cd = mass, radius, cd
        self.kd = RHO_AIR * cd * math.pi * radius ** 2 / (2 * mass)
        self.e_range, self.mu_range, self.lift_max = e_range, mu_range, lift_max


TENNIS = Ball()
PADEL = Ball(mass=0.057, radius=0.033, cd=0.55, e_range=(0.5, 0.85))


# ----------------------------------------------------------------------------------------------------------- camera
class Camera:
    """Pinhole camera P = K [R | t] (world = court metres, z up)."""

    def __init__(self, K, R, t, image_size, f_std=None, centre_std=None):
        self.K, self.R, self.t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float).reshape(3)
        self.P = self.K @ np.c_[self.R, self.t]
        self.image_size = tuple(image_size)
        self.f_std, self.centre_std = f_std, centre_std

    @property
    def f(self):
        return float(self.K[0, 0])

    @property
    def centre(self):
        return -np.linalg.solve(self.R, self.t)

    def project(self, X):
        X = np.asarray(X, float)
        h = X @ self.P[:, :3].T + self.P[:, 3]
        return h[..., :2] / h[..., 2:3]

    def depth(self, X):
        return np.asarray(X, float) @ self.P[2, :3] + self.P[2, 3]

    def in_image(self, uv, margin=0):
        w, h = self.image_size
        return (uv[..., 0] >= -margin) & (uv[..., 0] < w + margin) & (uv[..., 1] >= -margin) & (uv[..., 1] < h + margin)

    def ground_to_image(self, xy):
        xy = np.asarray(xy, float)
        return self.project(np.c_[xy, np.zeros(len(xy))])

    @staticmethod
    def _from_H(H, image_size):
        w, h = image_size
        cx, cy = w / 2.0, h / 2.0
        T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
        Hp = T @ np.asarray(H, float)
        a1, b1, c1 = Hp[:, 0]
        a2, b2, c2 = Hp[:, 1]
        # w = 1/f^2:  w (a1 a2 + b1 b2) + c1 c2 = 0 ;  w (a1^2 + b1^2 - a2^2 - b2^2) + (c1^2 - c2^2) = 0
        al = np.array([a1 * a2 + b1 * b2, a1 ** 2 + b1 ** 2 - a2 ** 2 - b2 ** 2])
        be = np.array([c1 * c2, c1 ** 2 - c2 ** 2])
        wv = -float(al @ be) / float(al @ al)
        if not wv > 0:
            raise ValueError("homography gives no real focal length (court too frontal or bad taps)")
        f = 1.0 / math.sqrt(wv)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        A = np.linalg.solve(K, np.asarray(H, float))
        lam = 1.0 / np.linalg.norm(A[:, 0])
        r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
        if t[2] < 0:                                     # court must be in front of the camera
            r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)
        U, _, Vt = np.linalg.svd(np.c_[r1, r2, r3])
        R = U @ Vt
        cam = Camera(K, R, t, image_size)
        if cam.centre[2] < 0:                            # z must point up (camera above the court)
            R = R @ np.diag([1, 1, -1.0])
            cam = Camera(K, R, t, image_size)
        return cam

    @staticmethod
    def from_homography(H_court2img, image_size, taps_img=None, taps_court=None, tap_sigma_px=1.5, n_mc=200, seed=0):
        """H_court2img: 3x3 court metres (x, y) -> image px. With the tap points it also returns Monte-Carlo std of f
        and of the camera centre under tap noise (px sigma)."""
        cam = Camera._from_H(H_court2img, image_size)
        if taps_img is not None and taps_court is not None and len(taps_img) >= 4:
            rng = np.random.default_rng(seed)
            fs, cs = [], []
            for _ in range(n_mc):
                Hm = fit_homography(np.asarray(taps_court, float), np.asarray(taps_img, float) + rng.normal(0, tap_sigma_px, (len(taps_img), 2)))
                try:
                    c = Camera._from_H(Hm, image_size)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                fs.append(c.f); cs.append(c.centre)
            if fs:
                cam.f_std, cam.centre_std = float(np.std(fs)), np.std(np.array(cs), 0)
        return cam

    @staticmethod
    def look_at(centre, target, f, image_size):
        """synthetic cameras for tests."""
        c, tg = np.asarray(centre, float), np.asarray(target, float)
        z = tg - c; z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        w, h = image_size
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
        return Camera(K, R, -R @ c, image_size)


def fit_homography(src, dst):
    """DLT homography src (n,2) -> dst (n,2), Hartley-normalised."""
    def norm(p):
        m = p.mean(0); s = math.sqrt(2) / max(1e-9, np.mean(np.linalg.norm(p - m, axis=1)))
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    Ts, Td = norm(src), norm(dst)
    s = (np.c_[src, np.ones(len(src))] @ Ts.T)[:, :2]; d = (np.c_[dst, np.ones(len(dst))] @ Td.T)[:, :2]
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u]); A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    H = np.linalg.solve(Td, H @ Ts)
    return H / H[2, 2]


# ------------------------------------------------------------------------------------------------------- dynamics
def accel(v, kd, lift):
    """v (B, 3), lift (B,) -> (B, 3): gravity + quadratic drag + Magnus lift in the vertical plane through v
    (lift > 0 = backspin / float, < 0 = topspin dip), |lift accel| = lift * kd |v|^2."""
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    s2 = vx * vx + vy * vy + vz * vz
    sp = np.sqrt(s2)
    # e = unit vector of (z - (z.v_hat) v_hat): perpendicular to v, in the vertical plane through v
    ez = 1.0 - vz * vz / np.maximum(s2, 1e-18)
    en = np.sqrt(np.maximum(ez, 0.0))                      # |z - (z.vh) vh| = sqrt(1 - vh_z^2)
    with np.errstate(invalid="ignore", over="ignore"):
        c = np.where(en > 1e-6, lift * kd * s2 / np.maximum(en, 1e-9), 0.0)
    k = kd * sp
    out = np.empty_like(v)
    zv = vz / np.maximum(s2, 1e-18)
    out[:, 0] = -k * vx + c * (-zv * vx)
    out[:, 1] = -k * vy + c * (-zv * vy)
    out[:, 2] = -9.81 - k * vz + c * ez
    return out


def rk4(p, v, h, kd, lift, n=1):
    """n RK4 steps of size h for states p, v (B, 3), lift (B,)."""
    for _ in range(n):
        a1 = accel(v, kd, lift)
        v2 = v + 0.5 * h * a1; a2 = accel(v2, kd, lift)
        v3 = v + 0.5 * h * a2; a3 = accel(v3, kd, lift)
        v4 = v + h * a3; a4 = accel(v4, kd, lift)
        p = p + h * (v + h / 6.0 * (a1 + a2 + a3))
        v = v + h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
    return p, v


def taylor(v, kd, lift, eps=1e-3):
    """acceleration a, jerk j = da/dt and snap s = d2a/dt2 along the flight, for states v (B, 3) (finite differences
    of accel along the motion; all vectorised)."""
    a = accel(v, kd, lift)
    ap, am = accel(v + eps * a, kd, lift), accel(v - eps * a, kd, lift)
    j = (ap - am) / (2 * eps)
    h = 0.02
    vp, vm = v + a * h + 0.5 * j * h * h, v - a * h + 0.5 * j * h * h
    s = (accel(vp, kd, lift) - 2 * a + accel(vm, kd, lift)) / (h * h)
    return a, j, s


def propagate(p, v, lift, tau, kd, hmax=0.1):
    """positions and velocities at times tau (B, M) [s, either sign] from states p, v (B, 3) at 0: a 4th-order Taylor
    integrator with steps <= hmax (0.1 s = 3 frames at 30 fps; <= 4 mm off RK4 over a 1.7 s, 40 m/s flight), so a
    1 s flight costs 10 steps instead of 120 RK4 sub-steps. -> X, V (B, M, 3)."""
    p, v = np.asarray(p, float), np.asarray(v, float)
    tau = np.asarray(tau, float)
    B, M = tau.shape
    X = np.empty((B, M, 3)); V = np.empty((B, M, 3))
    for sgn in (1.0, -1.0):
        m = (tau >= 0) if sgn > 0 else (tau < 0)
        if not m.any():
            continue
        q, w = p.copy(), v.copy()
        at = np.abs(tau)
        n_ch = int(np.ceil(at[m].max() / hmax + 1e-9)) if at[m].max() > 0 else 1
        for c in range(n_ch):
            lo, hi = c * hmax, (c + 1) * hmax
            sel = m & (at >= lo - 1e-12) & ((at <= hi + 1e-12) if c == n_ch - 1 else (at < hi))
            aa, jj, ss = taylor(w, kd, lift)
            if sel.any():
                bi, mi = np.nonzero(sel)
                dt = (sgn * (at[bi, mi] - lo))[:, None]
                X[bi, mi] = q[bi] + w[bi] * dt + aa[bi] * dt ** 2 / 2 + jj[bi] * dt ** 3 / 6 + ss[bi] * dt ** 4 / 24
                V[bi, mi] = w[bi] + aa[bi] * dt + jj[bi] * dt ** 2 / 2 + ss[bi] * dt ** 3 / 6
            h = sgn * hmax
            q = q + w * h + aa * h ** 2 / 2 + jj * h ** 3 / 6 + ss * h ** 4 / 24
            w = w + aa * h + jj * h ** 2 / 2 + ss * h ** 3 / 6
    return X, V


def simulate(theta, k_lo, k_hi, fps, kd, exact=False):
    """theta (B, 6|7) = (p, v[, lift]) at frame 0 (relative). -> X, V (B, k_hi - k_lo + 1, 3) at frames k_lo..k_hi.
    Taylor integrator by default; exact=True uses RK4 with 4 sub-steps per frame (reference / final check)."""
    th = np.atleast_2d(np.asarray(theta, float))
    B = len(th)
    lift = th[:, 6] if th.shape[1] > 6 else np.zeros(B)
    ks = np.arange(k_lo, k_hi + 1)
    if not exact:
        return propagate(th[:, :3], th[:, 3:6], lift, np.broadcast_to(ks / fps, (B, len(ks))), kd)
    n = len(ks)
    X = np.empty((B, n, 3)); V = np.empty((B, n, 3))
    i0 = -k_lo
    X[:, i0], V[:, i0] = th[:, :3], th[:, 3:6]
    for direction, rng in ((1, range(i0 + 1, n)), (-1, range(i0 - 1, -1, -1))):
        p, v = th[:, :3].copy(), th[:, 3:6].copy()
        h = direction / (fps * 4)
        for i in rng:
            p, v = rk4(p, v, h, kd, lift, n=4)
            X[:, i], V[:, i] = p, v
    return X, V


# --------------------------------------------------------------------------------------------- closed-form init
def ballistic_dlt(cam, t, uv, a=None, w=None, depth=None, anchor=None):
    """Closed-form flight through pixels uv (n, 2) at times t (n,) [s, relative] under constant acceleration a.
    Linear in (p0, v0): rows (u P3 - P1) and (v P3 - P2) applied to p0 + v0 t + a t^2 / 2. Rows are divided by the
    depth estimate (when given) so the algebraic error approximates pixels. -> theta (6,)."""
    a = G if a is None else np.asarray(a, float)
    t = np.asarray(t, float); uv = np.asarray(uv, float)
    P = cam.P
    M = np.concatenate([uv[:, :1] * P[2] - P[0], uv[:, 1:2] * P[2] - P[1]], 0)       # (2n, 4)
    tt = np.r_[t, t]
    A3 = M[:, :3]
    A = np.c_[A3, A3 * tt[:, None]]
    b = -(M[:, 3] + 0.5 * tt ** 2 * (A3 @ a))
    s = np.ones(len(tt))
    if depth is not None:
        s = s / np.maximum(np.r_[depth, depth], 0.3)
    if w is not None:
        s = s * np.sqrt(np.r_[w, w])
    A, b = A * s[:, None], b * s
    if anchor is not None:                      # (t_b, X_b, weight): the flight passes through a known 3D point
        tb, Xb, wa = anchor
        Aa = np.c_[np.eye(3), tb * np.eye(3)] * wa
        A = np.r_[A, Aa]; b = np.r_[b, wa * (np.asarray(Xb, float) - 0.5 * tb ** 2 * a)]
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    return th


def ballistic_dlt_drag(cam, t, uv, kd, w=None, rounds=3, anchor=None, return_a=False):
    """ballistic_dlt with drag re-linearised around the mid-time velocity (constant-accel approximation)."""
    th = ballistic_dlt(cam, t, uv, w=w, anchor=anchor)
    a = G
    for _ in range(rounds):
        X = th[:3] + np.outer(t, th[3:]) + 0.5 * np.outer(t ** 2, a)
        tm = 0.5 * (t.min() + t.max())
        vm = th[3:] + a * tm
        a = G - kd * np.linalg.norm(vm) * vm
        th = ballistic_dlt(cam, t, uv, a=a, w=w, depth=cam.depth(X), anchor=anchor)
    return (th, a) if return_a else th


def batch_triplet_solve(cam, t3, uv3):
    """N exact 3-point ballistic solves at once. t3 (3,) times, uv3 (N, 3, 2) -> theta (N, 6), ok (N,)."""
    P = cam.P
    N = len(uv3)
    A = np.empty((N, 6, 6)); b = np.empty((N, 6))
    for j in range(3):
        for r, (coord, row) in enumerate(((0, 0), (1, 1))):
            m = uv3[:, j, coord:coord + 1] * P[2] - P[row]                # (N, 4)
            k = 2 * j + r
            A[:, k, :3] = m[:, :3]; A[:, k, 3:] = m[:, :3] * t3[j]
            b[:, k] = -(m[:, 3] + 0.5 * t3[j] ** 2 * (m[:, :3] @ G))
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-12 * np.max(np.abs(A).reshape(N, -1), 1) ** 6
    th = np.zeros((N, 6))
    if ok.any():
        th[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    return th, ok


# ------------------------------------------------------------------------------------------------- robust fit (LM)
class Flight:
    """One ballistic segment: theta at reference frame kref (absolute), its inliers and fit statistics."""

    def __init__(self, kref, theta, ks, uv, peaks, cidx):
        self.kref, self.theta = int(kref), np.asarray(theta, float)
        self.ks, self.uv, self.peaks, self.cidx = np.asarray(ks, int), np.asarray(uv, float), np.asarray(peaks, float), np.asarray(cidx, int)
        self.cov = None; self.short = False; self.chi2 = np.nan; self.dof = 0; self.rms = np.nan; self.score = 0.0; self.ok = True

    @property
    def k0(self):
        return int(self.ks.min())

    @property
    def k1(self):
        return int(self.ks.max())

    def states(self, k_lo, k_hi, fps, kd):
        lo, hi = min(k_lo - self.kref, 0), max(k_hi - self.kref, 0)
        X, V = simulate(self.theta[None], lo, hi, fps, kd)
        s = (k_lo - self.kref) - lo
        return X[0, s:s + k_hi - k_lo + 1], V[0, s:s + k_hi - k_lo + 1]


def fit_flight(cam, ks, uv, kref, theta0, fps, kd, sigma=1.5, w=None, spin=False, iters=8, huber=2.5, lift_max=0.6, lift_sd=0.25):
    """LM on the RK4 ODE. ks absolute frames, uv (n,2). -> (theta, cov, residual px (n,2), robust weights)."""
    ks = np.asarray(ks, int); uv = np.asarray(uv, float)
    th = np.r_[np.asarray(theta0, float)[:6], (theta0[6] if len(theta0) > 6 else 0.0)] if spin else np.asarray(theta0, float)[:6].copy()
    npar = len(th)
    rel = ks - kref
    lo, hi = min(int(rel.min()), 0), max(int(rel.max()), 0)
    idx = rel - lo
    eps = np.r_[np.full(3, 1e-4), np.full(3, 1e-4), np.full(npar - 6, 1e-4)]
    w0 = np.ones(len(ks)) if w is None else np.asarray(w, float)
    lam = 1e-3

    def resid(thb):
        X, _ = simulate(thb, lo, hi, fps, kd)
        return cam.project(X[:, idx]) - uv[None]                           # (B, n, 2)

    r = resid(th[None])[0]
    rw = np.ones(len(ks))
    for _ in range(iters):
        nr = np.hypot(r[:, 0], r[:, 1]) / sigma
        rw = np.where(nr <= huber, 1.0, huber / np.maximum(nr, 1e-9))
        W = np.repeat(w0 * rw, 2)
        cost = float(np.sum(W * r.reshape(-1) ** 2))
        B = th[None] + np.diag(eps)
        J = ((resid(B) - r[None]) / eps[:, None, None]).reshape(npar, -1).T   # (2n, npar)
        H = J.T @ (J * W[:, None]); g = J.T @ (W * r.reshape(-1))
        if spin:
            H[6, 6] += (sigma ** 2) / lift_sd ** 2                          # weak prior: lift ~ N(0, lift_sd)
            g[6] += (sigma ** 2) / lift_sd ** 2 * th[6]
        improved = False
        for _ in range(6):
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), g)
            tn = th + step
            if spin:
                tn[6] = float(np.clip(tn[6], -lift_max, lift_max))
            rn = resid(tn[None])[0]
            cn = float(np.sum(W * rn.reshape(-1) ** 2))
            if cn < cost:
                th, r, lam, improved = tn, rn, max(lam / 3, 1e-7), True
                break
            lam *= 4
        if not improved or np.max(np.abs(step)) < 1e-6:
            break
    B = th[None] + np.diag(eps)
    J = ((resid(B) - r[None]) / eps[:, None, None]).reshape(npar, -1).T
    W = np.repeat(w0 * rw, 2)
    try:
        cov = np.linalg.inv(J.T @ (J * W[:, None])) * sigma ** 2
    except np.linalg.LinAlgError:
        cov = np.full((npar, npar), np.inf)
    return th, cov, r, rw


# ---------------------------------------------------------------------------------------------------- proposals
class Config:
    def __init__(self, fps=30.0, sigma=1.5, gate_px=6.0, gate_grow=1.0, seed_min_peak=0.25, min_inliers=6, min_inliers_short=4, max_miss=5,
                 steps=(2, 3), speed=(3.0, 75.0), z_max=10.0, xy_margin=8.0, p_detect=0.75, clutter_per_frame=2.0,
                 min_score=0.0, chi2_p=1e-3, bounce_bonus=4.0, hit_bonus=1.5, break_penalty=2.0, max_hyp=400,
                 spin=True, lift_sd=0.25, ball=TENNIS, max_join=12, join_px=18.0, bounce_tol=0.1, min_extent_px=8.0, min_step_px=1.0,
                 terminal=True, tail_gate_px=12.0, tail_frames=6, tail_min=2, tail_box=False, tail_net_m=1.2, tail_last_only=True):
        self.__dict__.update(locals()); del self.__dict__["self"]


def _plausible(cam, th, cfg):
    sp = np.linalg.norm(th[:, 3:6], axis=1)
    p = th[:, :3]
    ok = (sp >= cfg.speed[0]) & (sp <= cfg.speed[1]) & (p[:, 2] > -0.3) & (p[:, 2] < cfg.z_max)
    ok &= (p[:, 0] > -cfg.xy_margin) & (p[:, 0] < COURT_W + cfg.xy_margin) & (p[:, 1] > -cfg.xy_margin) & (p[:, 1] < COURT_L + cfg.xy_margin)
    ok &= cam.depth(p) > 0.5
    return ok


def _nearest(cands, k, uvp, gate, used=None):
    """best candidate of frame k within gate px of uvp -> (j, dist) or (-1, inf)."""
    c = cands[k]
    d = np.hypot(c[:, 0] - uvp[0], c[:, 1] - uvp[1])
    d = np.where(np.isfinite(d), d, np.inf)
    if used is not None:
        for j in range(len(c)):
            if (k, j) in used:
                d[j] = np.inf
    j = int(np.argmin(d))
    return (j, float(d[j])) if d[j] <= gate else (-1, np.inf)


def _grow(cam, cands, f, cfg):
    """grow a seeded flight forward and backward, gated by the (drag-relinearised) ballistic prediction; the gate
    widens with each missed frame. -> Flight or None."""
    T = len(cands); kd = cfg.ball.kd
    for _ in range(2):
        th, acc = ballistic_dlt_drag(cam, (f.ks - f.kref) / cfg.fps, f.uv, kd, rounds=2, return_a=True)
        f.theta = th
        have = set(f.ks.tolist())
        for direction in (1, -1):
            k = (f.k1 if direction > 0 else f.k0)
            miss = 0
            while 0 <= k + direction < T and miss < cfg.max_miss:
                k += direction
                t = (k - f.kref) / cfg.fps
                Xk = f.theta[:3] + f.theta[3:6] * t + 0.5 * acc * t * t
                if Xk[2] < -0.15 or cam.depth(Xk) < 0.5:
                    break
                uvp = cam.project(Xk)
                if not cam.in_image(uvp, 20):
                    break
                if k in have:
                    continue
                j, d = _nearest(cands, k, uvp, cfg.gate_px + cfg.gate_grow * miss)
                if j < 0:
                    miss += 1
                    continue
                miss = 0
                have.add(k)
                f.ks = np.r_[f.ks, k]; f.uv = np.r_[f.uv, cands[k, j, None, :2]]; f.peaks = np.r_[f.peaks, cands[k, j, 2]]; f.cidx = np.r_[f.cidx, j]
                if len(f.ks) % 4 == 0:
                    f.theta, acc = ballistic_dlt_drag(cam, (f.ks - f.kref) / cfg.fps, f.uv, kd, rounds=1, return_a=True)
        o = np.argsort(f.ks)
        f.ks, f.uv, f.peaks, f.cidx = f.ks[o], f.uv[o], f.peaks[o], f.cidx[o]
    if len(f.ks) < cfg.min_inliers_short:
        return None
    # re-centre the reference frame (Taylor expansions are best in the middle of the span)
    km = int(round(0.5 * (f.k0 + f.k1)))
    t = (km - f.kref) / cfg.fps
    f.theta = np.r_[f.theta[:3] + f.theta[3:6] * t + 0.5 * acc * t * t, f.theta[3:6] + acc * t]
    f.kref = km
    return f


def propose(cam, cands, cfg):
    """cands (T, K, 3) [u, v, peak] NaN-padded -> list of Flight hypotheses (grown, not yet refined)."""
    T, K, _ = cands.shape
    valid = np.isfinite(cands[..., 0]) & (cands[..., 2] >= cfg.seed_min_peak)
    seeds = []
    for s in cfg.steps:
        t3 = np.array([-s, 0, s]) / cfg.fps
        for f in range(T - 2 * s):
            js = [np.nonzero(valid[f + q * s])[0] for q in range(3)]
            if not all(len(j) for j in js):
                continue
            combo = np.array(np.meshgrid(*js, indexing="ij")).reshape(3, -1).T
            uv3 = np.stack([cands[f + q * s, combo[:, q], :2] for q in range(3)], 1)
            th, ok = batch_triplet_solve(cam, t3, uv3)
            ok &= _plausible(cam, th, cfg)
            for c, t_ in zip(combo[ok], th[ok]):
                pk = sum(cands[f + q * s, c[q], 2] for q in range(3))
                seeds.append((pk, f + s, s, c, t_))
    # quick support count with the constant-accel model in a +-8 frame window, best first
    scored = []
    for pk, kref, s, c, th in seeds:
        ks = np.arange(max(0, kref - 8), min(T, kref + 9))
        tt = (ks - kref) / cfg.fps
        X = th[:3] + np.outer(tt, th[3:]) + 0.5 * np.outer(tt ** 2, G)
        uvp = cam.project(X)
        d = np.hypot(cands[ks, :, 0] - uvp[:, None, 0], cands[ks, :, 1] - uvp[:, None, 1])
        sup = int(np.sum(np.nanmin(np.where(np.isfinite(d), d, np.inf), 1) <= cfg.gate_px))
        scored.append((sup + 0.01 * pk, kref, s, c, th))
    scored.sort(key=lambda r: -r[0])
    used, out = set(), []
    for sup, kref, s, c, th in scored[: cfg.max_hyp]:
        trip = [(kref - s, int(c[0])), (kref, int(c[1])), (kref + s, int(c[2]))]
        if sum(t in used for t in trip) >= 2 or sup < 4:
            continue
        ks = np.array([kref - s, kref, kref + s]); uv = np.stack([cands[k, j, :2] for k, j in trip])
        f = Flight(kref, th, ks, uv, [cands[k, j, 2] for k, j in trip], [j for _, j in trip])
        f = _grow(cam, cands, f, cfg)
        if f is None:
            continue
        mine = set(zip(f.ks.tolist(), f.cidx.tolist()))
        if len(mine & used) > 0.5 * len(mine):
            continue
        used |= mine
        out.append(f)
    return out


def _chi2_sf(x, k):
    """chi-square survival function (Wilson-Hilferty), no scipy."""
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def refine(cam, cands, f, cfg):
    """LM on the ODE, drop gross outliers, chi-square test, score. Mutates and returns f."""
    kd = cfg.ball.kd
    for _ in range(2):
        th, cov, r, rw = fit_flight(cam, f.ks, f.uv, f.kref, f.theta, cfg.fps, kd, cfg.sigma, spin=cfg.spin, lift_max=cfg.ball.lift_max, lift_sd=cfg.lift_sd)
        e = np.hypot(r[:, 0], r[:, 1])
        keep = e <= 3.5 * cfg.sigma
        f.theta, f.cov = th, cov
        if keep.all() or keep.sum() < cfg.min_inliers_short:
            break
        f.ks, f.uv, f.peaks, f.cidx = f.ks[keep], f.uv[keep], f.peaks[keep], f.cidx[keep]
    # chi-square on the exact ODE (RK4), so the Taylor integrator used in the search never decides acceptance
    lo, hi = min(f.k0 - f.kref, 0), max(f.k1 - f.kref, 0)
    Xe, _ = simulate(f.theta[None], lo, hi, cfg.fps, kd, exact=True)
    e = np.hypot(*(cam.project(Xe[0, f.ks - f.kref - lo]) - f.uv).T)
    npar = len(f.theta)
    f.chi2 = float(np.sum((e / cfg.sigma) ** 2)); f.dof = 2 * len(f.ks) - npar; f.rms = float(np.sqrt(np.mean(e ** 2)))
    f.p_fit = _chi2_sf(f.chi2, f.dof)
    X, V = f.states(f.k0, f.k1, cfg.fps, kd)
    sp = np.linalg.norm(V, axis=1)
    # degeneracy: a flight straight along a camera ray barely moves in the image and can explain a static glint;
    # its depth is unobservable, so we demand real image motion (as the 2D filter does) before believing it
    ext = float(np.hypot(*(f.uv.max(0) - f.uv.min(0))))
    step = float(np.median(np.hypot(*np.diff(f.uv, axis=0).T) / np.maximum(np.diff(f.ks), 1))) if len(f.ks) > 1 else 0.0
    f.short = len(f.ks) < cfg.min_inliers           # only usable as the flight right after a solved bounce
    # z is checked loosely: a lone monocular flight's depth (hence height) is weakly fixed until a bounce anchors it
    f.ok = bool(len(f.ks) >= cfg.min_inliers_short and f.p_fit >= cfg.chi2_p and X[:, 2].min() > -1.0 and sp.min() >= cfg.speed[0]
                and sp.max() <= cfg.speed[1] and X[:, 2].max() < cfg.z_max
                and ext >= cfg.min_extent_px and step >= cfg.min_step_px)
    # score: log-likelihood ratio (ball vs clutter) of the inliers + misses inside the span, minus the MDL charge of the
    # parameters (each pixel pair pays for 2 of them: npar/2 inliers' worth of evidence)
    w, h = cam.image_size
    lr_in = math.log(cfg.p_detect * w * h / (2 * math.pi * cfg.sigma ** 2 * max(cfg.clutter_per_frame, 1e-3)))
    gains = lr_in - 0.5 * (e / cfg.sigma) ** 2 + np.log(np.clip(f.peaks, 1e-3, 1))
    span = np.arange(f.k0, f.k1 + 1)
    uvp = cam.project(X)
    vis = cam.in_image(uvp)
    n_miss = int(np.sum(vis[np.setdiff1d(span, f.ks) - f.k0])) if len(span) > len(f.ks) else 0
    f.score = float(np.sum(gains) + n_miss * math.log(1 - cfg.p_detect) - 0.5 * npar * lr_in)
    return f


# ----------------------------------------------------------------------------------------------- events + select
def _ground_root(X, k0):
    """first index where z crosses 0 going down (linear interp) -> fractional absolute frame or None."""
    z = X[:, 2]
    idx = np.nonzero((z[:-1] > 0) & (z[1:] <= 0))[0]
    if not len(idx):
        return None
    i = int(idx[0]); a = z[i] / (z[i] - z[i + 1])
    return k0 + i + a


def _img_meet(cam, a, b, cfg):
    """closest image distance between the two flights' predictions over the join window -> (px, frame)."""
    lo, hi = min(a.k1, b.k0) - 2, max(a.k1, b.k0) + 2
    Xa, _ = a.states(lo, hi, cfg.fps, cfg.ball.kd); Xb, _ = b.states(lo, hi, cfg.fps, cfg.ball.kd)
    d = np.hypot(*(cam.project(Xa) - cam.project(Xb)).T)
    i = int(np.argmin(d))
    return float(d[i]), lo + i, Xa[i]


def _bounce_model(th, a, kmax, fps, kd):
    """batched. th (B, 11): flight a = (p, v, lift) at a.kref until its first ground root kb after a.k1 - 4, then a
    new flight from (pb, th[7:10], lift th[10]) starting AT kb. -> dict(X (B, n, 3) for frames a.kref+lo..kmax, lo,
    kb (B,), pb (B, 3), va (B, 3), ok (B,)), frames as absolute = a.kref + lo + index."""
    th = np.atleast_2d(th)
    B = len(th)
    lo = min(int(a.ks.min()) - a.kref, 0)
    hi = max(int(kmax) - a.kref + 1, 1)
    X, V = simulate(th[:, :7], lo, hi, fps, kd)
    n = X.shape[1]
    s0 = max(0, int(a.k1) - a.kref - lo - 4)
    z = X[:, :, 2]
    cross = np.zeros((B, n - 1), bool)
    cross[:, s0:] = (z[:, s0:-1] > 0) & (z[:, s0 + 1:] <= 0)
    ok = cross.any(1)
    i = np.where(ok, np.argmax(cross, 1), 0)
    r = np.arange(B)
    fr = np.where(ok, z[r, i] / np.where(ok, z[r, i] - z[r, i + 1], 1.0), 0.0)
    pb = X[r, i] + fr[:, None] * (X[r, i + 1] - X[r, i]); pb[:, 2] = 0.0
    va = V[r, i] + fr[:, None] * (V[r, i + 1] - V[r, i])
    kb = a.kref + lo + i + fr                                  # fractional absolute frame
    absk = a.kref + lo + np.arange(n)
    tau = (absk[None, :] - kb[:, None]) / fps                 # time since the bounce
    Xb, _ = propagate(pb, th[:, 7:10], th[:, 10], np.maximum(tau, 0.0), kd)
    X = np.where((tau > 0)[..., None], Xb, X)
    return dict(X=X, lo=lo, kb=kb, pb=pb, va=va, ok=ok, kf=np.floor(kb).astype(int) + 1)


def fit_bounce(cam, a, b, cfg, iters=8, meet=None):
    """joint fit of flight a + the flight after its ground contact, which starts AT a's z = 0 root (shared point, so
    the bounce anchors the depth that a single monocular flight leaves loose). 11 params: a (p, v, lift) + outgoing
    (v, lift). -> dict(theta, cov, rms, chi2, dof, kb, pb, e, mu, kf, theta_b) or None."""
    kd, fps = cfg.ball.kd, cfg.fps
    ks = np.r_[a.ks, b.ks]; uv = np.r_[a.uv, b.uv]
    kmax = int(ks.max())
    inits = []
    ta = np.r_[a.theta[:6], a.theta[6] if len(a.theta) > 6 else 0.0]
    m0 = _bounce_model(np.r_[ta, np.zeros(4)][None], a, kmax, fps, kd)
    if m0["ok"][0]:
        _, Vb = b.states(int(m0["kf"][0]), int(m0["kf"][0]) + 1, fps, kd)
        inits.append(np.r_[ta, Vb[0], b.theta[6] if len(b.theta) > 6 else 0.0])
    # image-anchored inits: the bounce is where the two image tracks meet; that pixel on the ground plane is exact
    if meet is not None:
        Hi = np.linalg.inv(cam.P[:, [0, 1, 3]])
        for dk in (0.0,):
            km = meet[1] + dk
            g = Hi @ np.r_[meet[0], 1.0]; pb = np.r_[g[:2] / g[2], 0.0]
            tha = ballistic_dlt_drag(cam, (a.ks - a.kref) / fps, a.uv, kd, rounds=2, anchor=((km - a.kref) / fps, pb, 50.0))
            thb = ballistic_dlt_drag(cam, (b.ks - b.kref) / fps, b.uv, kd, rounds=2, anchor=((km - b.kref) / fps, pb, 50.0))
            vb = thb[3:6] + G * (km - b.kref) / fps
            inits.append(np.r_[tha, 0.0, vb, 0.0])
    if not inits:
        return None
    best = None
    for th0 in inits[::-1]:                              # image-anchored init first; stop once it explains the data
        out = _lm_bounce(cam, a, ks, uv, kmax, th0, cfg, iters)
        if out is not None and (best is None or out["rms"] < best["rms"]):
            best = out
        if best is not None and _bounce_ok(best, a, b, cfg):
            break
    return best


def _bounce_ok(jb, a, b, cfg):
    """acceptance test for a joint bounce fit: chi-square, rms, timing inside the gap, e and mu in the ball's range."""
    er, mr = cfg.ball.e_range, cfg.ball.mu_range
    return bool(jb["p_fit"] >= cfg.chi2_p and jb["rms"] <= 2.0 * cfg.sigma and a.k1 - 4 <= jb["kb"] <= b.k0 + 1
                and er[0] - cfg.bounce_tol <= jb["e"] <= er[1] + cfg.bounce_tol and mr[0] - cfg.bounce_tol <= jb["mu"] <= mr[1] + cfg.bounce_tol)


def _lm_bounce(cam, a, ks, uv, kmax, th, cfg, iters, emu_prior=None):
    """emu_prior (e0, sd_e, mu0, sd_mu): Gaussian prior on restitution and horizontal speed ratio, for outgoing
    flights with too few detections to fix their velocity (terminal bounces)."""
    kd, fps, sig, ls = cfg.ball.kd, cfg.fps, cfg.sigma, cfg.lift_sd
    n = len(ks)
    eps = 1e-4

    def res(T):
        m = _bounce_model(T, a, kmax, fps, kd)
        r = (cam.project(m["X"][:, ks - a.kref - m["lo"]]) - uv[None]).reshape(len(T), -1)
        r = np.c_[r, sig * T[:, 6] / ls, sig * T[:, 10] / ls]
        if emu_prior is not None:
            va = m["va"]; e0, se, m0, sm = emu_prior
            ee = -T[:, 9] / np.minimum(va[:, 2], -0.1)
            mm = np.hypot(T[:, 7], T[:, 8]) / np.maximum(np.hypot(va[:, 0], va[:, 1]), 1e-3)
            r = np.c_[r, sig * (ee - e0) / se, sig * (mm - m0) / sm]
        r[~m["ok"]] = np.inf
        return r, m

    R, m = res(th[None])
    r = R[0]
    if not np.isfinite(r).all():
        return None
    lam = 1e-3
    for _ in range(iters):
        Rj, _ = res(th[None] + eps * np.eye(11))
        J = ((Rj - r[None]) / eps).T
        if not np.isfinite(J).all():
            return None
        H = J.T @ J; g = J.T @ r
        ok = False
        for _ in range(5):
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), g)
            tn = th + step; tn[[6, 10]] = np.clip(tn[[6, 10]], -cfg.ball.lift_max, cfg.ball.lift_max)
            Rn, mn = res(tn[None])
            if np.isfinite(Rn).all() and Rn[0] @ Rn[0] < r @ r:
                th, r, m, lam, ok = tn, Rn[0], mn, lam / 3, True
                break
            lam *= 4
        if not ok or np.max(np.abs(step)) < 1e-6:
            break
    Rj, _ = res(th[None] + eps * np.eye(11))
    J = ((Rj - r[None]) / eps).T
    if not np.isfinite(J).all():
        return None
    e2 = r[:2 * n].reshape(n, 2)
    rms = float(np.sqrt(np.mean(np.sum(e2 ** 2, 1))))
    chi2 = float(np.sum(e2 ** 2) / sig ** 2); dof = 2 * n - 11
    va, vb = m["va"][0], th[7:10]
    kb, pb = float(m["kb"][0]), m["pb"][0]
    kf = int(m["kf"][0])
    # state of the outgoing flight at frame kf (its reference)
    Xf, Vf = propagate(pb[None], vb[None], th[10:11], np.array([[(kf - kb) / fps]]), kd)
    try:
        cov = np.linalg.pinv(J.T @ J) * sig ** 2
    except np.linalg.LinAlgError:
        return None
    return dict(theta=th, cov=cov, rms=rms, chi2=chi2, dof=dof, p_fit=_chi2_sf(chi2, dof), kb=kb, pb=pb, va=va,
                e=float(-vb[2] / va[2]) if va[2] < -0.1 else np.nan,
                mu=float(np.linalg.norm(vb[:2]) / max(1e-6, np.linalg.norm(va[:2]))), kf=kf, theta_b=np.r_[Xf[0, 0], Vf[0, 0], th[10]])


def classify_join(cam, a, b, cfg, boxes_at=None):
    """event between flight a (earlier) and b. -> (kind, bonus, info) with kind bounce | hit | break, or None when
    they cannot follow each other. A bounce is accepted only when ONE joint fit through the shared ground point
    explains both flights (chi-square), with restitution and friction in the ball's range."""
    if b.k0 - a.k1 > cfg.max_join or b.k0 < a.k0 + 2 or b.k1 <= a.k1:
        return None
    meet_px, k_meet, X_meet = _img_meet(cam, a, b, cfg)
    if meet_px <= 3 * cfg.join_px:
        g_uv = 0.5 * (cam.project(a.states(k_meet, k_meet, cfg.fps, cfg.ball.kd)[0][0]) + cam.project(b.states(k_meet, k_meet, cfg.fps, cfg.ball.kd)[0][0]))
        jb = fit_bounce(cam, a, b, cfg, meet=(g_uv, float(k_meet)))
        if jb is not None:
            if _bounce_ok(jb, a, b, cfg):
                return "bounce", cfg.bounce_bonus, dict(k=jb["kb"], xy=jb["pb"][:2].tolist(), e=jb["e"], mu=jb["mu"], fit=jb)
    if meet_px <= cfg.join_px:
        near = False
        if boxes_at is not None:
            uv = cam.project(X_meet)
            for bx in boxes_at(k_meet) or ():
                x0, y0, x1, y1 = bx[:4]; ww, hh = x1 - x0, y1 - y0
                if x0 - 0.6 * ww <= uv[0] <= x1 + 0.6 * ww and y0 - 0.4 * hh <= uv[1] <= y1:
                    near = True
        return "hit", cfg.hit_bonus + (1.0 if near else 0.0), dict(k=float(k_meet), uv_meet=cam.project(X_meet).tolist(), near_player=near)
    return "break", -cfg.break_penalty, dict(k=float(0.5 * (a.k1 + b.k0)))


def select(flights, cfg, cam, boxes_at=None):
    """exact weighted-interval DP over the accepted flights. -> (chain [Flight], events [dict])."""
    F = sorted([f for f in flights if f.ok and f.score > cfg.min_score], key=lambda f: (f.k1, f.k0))
    n = len(F)
    if not n:
        return [], []
    best = np.array([-np.inf if getattr(f, "short", False) else f.score for f in F], float); prev = [-1] * n; ev = [None] * n
    for j in range(n):
        for i in range(j):
            if F[i].k1 > F[j].k0 + 1 or not np.isfinite(best[i]):
                continue
            c = classify_join(cam, F[i], F[j], cfg, boxes_at) if F[j].k0 - F[i].k1 <= cfg.max_join else ("gap", 0.0, {})
            if c is None or (getattr(F[j], "short", False) and c[0] != "bounce"):
                continue
            v = best[i] + F[j].score + c[1]
            if v > best[j]:
                best[j], prev[j], ev[j] = v, i, c
    j = int(np.argmax(best)); chain, events = [], []
    if not np.isfinite(best[j]) or best[j] <= 0:
        return [], []
    while j >= 0:
        chain.append(F[j])
        if ev[j] is not None and ev[j][0] != "gap":
            events.append(dict(kind=ev[j][0], a=len(chain), **ev[j][2]))
        j = prev[j]
    chain = chain[::-1]
    for e in events:                                   # index of the flight BEFORE the event in the forward chain
        e["a"] = len(chain) - e["a"] - 1
    return chain, events[::-1]


def terminal_bounces(cam, cands, chain, events, cfg, boxes_at=None):
    """bounce whose outgoing flight has only tail_min..3+ detections (the last bounce of a point, or a bounce hit
    again at once). A single flight's depth is loose, so the contact is found in the image: candidates in the next
    tail_frames frames that leave flight a's image track (gate grows with time since a's last inlier) are the outgoing
    ball; the contact pixel is a's image position at the trial bounce frame, back-projected onto z = 0. Accepted only
    if ONE joint fit (same test as fit_bounce) explains a plus those detections, each within 2.5 sigma. Mutates chain
    and events (inserts the short outgoing flight)."""
    kd, fps = cfg.ball.kd, cfg.fps
    T = len(cands)
    has_b = {e["a"] for e in events if e["kind"] in ("bounce", "hit")}      # a hit join already explains the end of a
    Hi = np.linalg.inv(cam.P[:, [0, 1, 3]])
    for ia in range(len(chain) - 1, -1, -1):
        if ia in has_b or (cfg.tail_last_only and ia != len(chain) - 1):
            continue
        a = chain[ia]
        k_next = chain[ia + 1].k0 if ia + 1 < len(chain) else T
        k_hi = min(k_next, T, a.k1 + 1 + cfg.tail_frames)
        if k_hi - (a.k1 + 1) < cfg.tail_min:
            continue
        Xa, _ = a.states(a.k1, k_hi, fps, kd)
        uva = cam.project(Xa)                                      # a's image track, frames a.k1..k_hi
        hit = []
        for k in range(a.k1 + 1, k_hi):
            # the contact lies on a's image track before frame k, and the ball leaves it from there: gate on the
            # distance to that track (densely sampled), growing with the time since a's last inlier
            dk = k - a.k1
            poly = np.concatenate([np.linspace(uva[q], uva[q + 1], 8) for q in range(dk)])
            c = cands[k]
            d = np.min(np.hypot(c[:, None, 0] - poly[None, :, 0], c[:, None, 1] - poly[None, :, 1]), 1)
            d = np.where(np.isfinite(d) & (c[:, 2] >= cfg.seed_min_peak), d, np.inf)
            j = int(np.argmin(d))
            if d[j] <= cfg.tail_gate_px * (1 + dk):
                hit.append((k, j))
        if len(hit) < cfg.tail_min:
            continue
        hk = np.array([h[0] for h in hit]); huv = np.array([cands[k, j, :2] for k, j in hit]); hp = np.array([cands[k, j, 2] for k, j in hit])
        hj = np.array([h[1] for h in hit])
        ta = np.r_[a.theta[:6], a.theta[6] if len(a.theta) > 6 else 0.0]
        best = None
        for km in np.arange(a.k1 + 0.5, hk[-cfg.tail_min], 0.5):
            pre, post = hk < km, hk > km                   # tail detections before the contact still belong to a
            if post.sum() < cfg.tail_min:
                continue
            q = km - a.k1
            u = uva[int(q)] + (q - int(q)) * (uva[int(q) + 1] - uva[int(q)])
            if pre.any():                                  # contact pixel: between a's last image point and the next
                u = huv[pre][-1] + (km - hk[pre][-1]) / max(1e-6, hk[post][0] - hk[pre][-1]) * (huv[post][0] - huv[pre][-1])
            g = Hi @ np.r_[u, 1.0]
            if abs(g[2]) < 1e-12:
                continue
            pb = np.r_[g[:2] / g[2], 0.0]
            aks, auv = np.r_[a.ks, hk[pre]], np.r_[a.uv, huv[pre]]
            tha = ballistic_dlt_drag(cam, (aks - a.kref) / fps, auv, kd, rounds=2, anchor=((km - a.kref) / fps, pb, 50.0))
            thb = ballistic_dlt_drag(cam, np.r_[0.0, (hk[post] - km) / fps], np.r_[[u], huv[post]], kd, rounds=2, anchor=(0.0, pb, 50.0))
            if not (np.all(np.isfinite(tha)) and np.all(np.isfinite(thb))):
                continue
            a2 = Flight(a.kref, a.theta, aks, auv, np.r_[a.peaks, hp[pre]], np.r_[a.cidx, hj[pre]])
            b = Flight(hk[post][0], thb, hk[post], huv[post], hp[post], hj[post]); b.short = True
            for th0 in (np.r_[tha, 0.0, thb[3:6], 0.0], np.r_[ta, thb[3:6], 0.0]):
                er, mr = cfg.ball.e_range, cfg.ball.mu_range
                prior = (0.5 * (er[0] + er[1]), 0.5 * (er[1] - er[0]), 0.5 * (mr[0] + mr[1]), 0.5 * (mr[1] - mr[0]))
                jb = _lm_bounce(cam, a2, np.r_[aks, b.ks], np.r_[auv, b.uv], int(hk.max()), th0, cfg, 8, emu_prior=prior)
                if jb is not None and _bounce_ok(jb, a2, b, cfg) and (best is None or jb["rms"] < best[0]["rms"]):
                    best = (jb, b, a2)
        if best is None:
            continue
        jb, b, a2 = best
        m = _bounce_model(jb["theta"][None], a2, int(hk.max()), fps, kd)
        e2 = np.hypot(*(cam.project(m["X"][0, b.ks - a.kref - m["lo"]]) - b.uv).T)
        if e2.max() > 2.5 * cfg.sigma or jb["kb"] >= b.k0:
            continue
        if abs(jb["pb"][1] - COURT_L / 2) < cfg.tail_net_m:  # a ball dropping off the net tape kinks like a bounce
            continue
        if cfg.tail_box and boxes_at is not None:          # a ground contact is not up at a player's racket
            uvb = cam.project(jb["pb"][None])[0]
            if any(x0 - 0.3 * (x1 - x0) <= uvb[0] <= x1 + 0.3 * (x1 - x0) and uvb[1] <= y1 - 0.25 * (y1 - y0)
                   for x0, y0, x1, y1 in (bx[:4] for bx in (boxes_at(int(round(jb["kb"]))) or ()))):
                continue
        chain[ia] = a = a2
        chain.insert(ia + 1, b)
        for e in events:                                # hit / break that followed a now follows b
            if e["a"] >= ia:
                e["a"] += 1
        events.append(dict(kind="bounce", a=ia, k=jb["kb"], xy=jb["pb"][:2].tolist(), e=jb["e"], mu=jb["mu"], fit=jb, terminal=True))
    events.sort(key=lambda e: e["a"])

def bounce_stats(cam, a, jb, cfg, n_mc=200, seed=0, lines=None):
    """bounce covariance (Monte Carlo over flight a's joint-fit posterior, ground root re-solved per sample) and
    P(in) against the lines box (x0, x1, y0, y1) in court metres (default: doubles court)."""
    kd, fps = cfg.ball.kd, cfg.fps
    rng = np.random.default_rng(seed)
    Ca = jb["cov"][:7, :7]
    S = rng.multivariate_normal(jb["theta"][:7], (Ca + Ca.T) / 2, size=n_mc, method="svd")
    lo = min(int(a.ks.min()) - a.kref, 0); hi = int(math.ceil(jb["kb"])) - a.kref + 4
    X, _ = simulate(S, lo, hi, fps, kd)
    pts = []
    for q in range(n_mc):
        s0 = max(0, int(a.k1) - a.kref - lo - 4)
        kq = _ground_root(X[q, s0:], a.kref + lo + s0)
        if kq is None:
            continue
        i = int(math.floor(kq - a.kref - lo)); fr = kq - a.kref - lo - i
        if i + 1 < X.shape[1]:
            pts.append(X[q, i, :2] + fr * (X[q, i + 1, :2] - X[q, i, :2]))
    pts = np.array(pts)
    if len(pts) <= 10:
        return {}
    x0, x1, y0, y1 = lines or (0.0, COURT_W, 0.0, COURT_L)
    cv = np.cov(pts.T)
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return dict(cov_xy=cv.tolist(), sd_xy=np.sqrt(np.diag(cv)).tolist(), p_in=float(inside.mean()))


# ------------------------------------------------------------------------------------------------------- top level
def track(cam, cands, cfg=None, boxes_at=None, bounce_refine=True, lines=None):
    """cands (T, K, 3) [u, v, peak] (or (T, 2) positions) -> dict(pos (T, 2) px, NaN where no accepted flight; conf
    (T,) 1 inlier / 0.6 physics-filled; xyz (T, 3) court metres; flights; events (bounce: k, xy, sd_xy, p_in, e, mu,
    uv; hit: k, near_player); rejected flights (chi-square / physics failures); stats). Only frames inside an
    accepted flight's span are reported, so every reported pixel lies on a fitted 3D flight."""
    # divergent LM trial steps / hypotheses produce inf states on purpose (they score inf and are rejected)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _track(cam, cands, cfg, boxes_at, bounce_refine, lines)


def _track(cam, cands, cfg, boxes_at, bounce_refine, lines):
    cfg = cfg or Config()
    cands = np.asarray(cands, float)
    if cands.ndim == 2:
        cands = cands[:, None, :]
    if cands.shape[-1] == 2:
        cands = np.concatenate([cands, np.ones(cands.shape[:-1] + (1,))], -1)
    T = len(cands)
    hyp = propose(cam, cands, cfg)
    for f in hyp:
        refine(cam, cands, f, cfg)
    chain, events = select(hyp, cfg, cam, boxes_at)
    if cfg.terminal and chain:
        terminal_bounces(cam, cands, chain, events, cfg, boxes_at)
    kd = cfg.ball.kd
    # bounces: the joint fit is the better 3D for both flights (depth anchored at the ground contact)
    for e in events:
        if e["kind"] != "bounce":
            continue
        jb = e.pop("fit")
        a, b = chain[e["a"]], chain[e["a"] + 1]
        a.theta = jb["theta"][:7].copy(); b.kref, b.theta = jb["kf"], jb["theta_b"].copy()
        a.cov = jb["cov"][:7, :7]
        if bounce_refine:
            e.update(bounce_stats(cam, a, jb, cfg, lines=lines))
        e["uv"] = cam.ground_to_image(np.array([e["xy"]]))[0].tolist()
        e["joint_rms_px"] = jb["rms"]
    for e in events:
        e.pop("fit", None)
    pos = np.full((T, 2), np.nan); xyz = np.full((T, 3), np.nan); conf = np.zeros(T)
    for i, f in enumerate(chain):
        k1 = f.k1
        if i + 1 < len(chain) and any(e["kind"] == "bounce" and e["a"] == i for e in events):
            k1 = min(chain[i + 1].k0, int(math.floor(next(e["k"] for e in events if e["a"] == i))))
        X, _ = f.states(f.k0, max(k1, f.k0), cfg.fps, kd)
        uv = cam.project(X)
        pos[f.k0:f.k0 + len(X)] = uv; xyz[f.k0:f.k0 + len(X)] = X; conf[f.k0:f.k0 + len(X)] = 0.6
        conf[f.ks] = 1.0
    rejected = [f for f in hyp if not f.ok]
    stats = dict(frames=T, candidates=int(np.isfinite(cands[..., 0]).sum()), hypotheses=len(hyp), rejected=len(rejected),
                 flights=len(chain), bounces=sum(e["kind"] == "bounce" for e in events), hits=sum(e["kind"] == "hit" for e in events),
                 tracked_frames=int(np.isfinite(pos[:, 0]).sum()), inlier_frames=int((conf == 1.0).sum()))
    return dict(pos=pos, conf=conf, xyz=xyz, flights=chain, events=events, rejected=rejected, stats=stats)
