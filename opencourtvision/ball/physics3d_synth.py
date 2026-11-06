"""Synthetic rallies with known 3D truth for the physics ball tracker (tests and A/B benches; own code, numpy only).

rally()      a tennis rally: serve, then alternating groundstrokes, each flight integrated with gravity + quadratic drag
             (+ optional Magnus lift) at 600 Hz, bounces with restitution e and friction factor mu, sampled at fps.
detections() per-frame candidates (T, K, 3) [u, v, peak]: the true ball with pixel noise, missed detections, an
             occlusion gap, plus clutter (uniform false peaks, static glints, a 'racket' that follows a player).
"""
import numpy as np

from . import physics3d as P3


def _fly(p, v, fps_hi, kd, lift, stop):
    """integrate until stop(p, v) -> list of (t, p, v)."""
    out = [(0.0, p.copy(), v.copy())]
    h = 1.0 / fps_hi
    t = 0.0
    lift = np.array([lift])
    while True:
        pn, vn = P3.rk4(p[None], v[None], h, kd, lift)
        p, v = pn[0], vn[0]; t += h
        out.append((t, p.copy(), v.copy()))
        if stop(p, v) or t > 4.0:
            return out


def _aim(p0, target, speed, kd, lift, tries=12):
    """initial velocity of the given speed from p0 that lands at target (x, y) (drag-aware shooting)."""
    d = np.r_[target, 0.0] - p0
    T = np.linalg.norm(d[:2]) / (0.8 * speed)
    v = d / T - 0.5 * P3.G * T
    for _ in range(tries):
        tr = _fly(p0, v, 600.0, kd, lift, lambda p, v: p[2] <= 0 and v[2] < 0)
        land = tr[-1][1][:2]; Tn = tr[-1][0]
        v = v + np.r_[(target - land) / max(Tn, 0.2), 0.0]
        v[2] += (0 - tr[-1][1][2]) / max(Tn, 0.2)
    return v


def rally(rng, n_shots=4, fps=30.0, ball=P3.TENNIS, e=0.75, mu=0.7, lift_sd=0.15):
    """-> dict(t, X (T,3) truth at frame times, events [(kind, t, xyz)], bounces [(t, x, y)])."""
    kd = ball.kd
    near_y, far_y = -1.0, 24.8
    p = np.array([rng.uniform(3.0, 5.0), -0.5, 2.7])
    target = np.array([rng.uniform(5.6, 8.0), rng.uniform(15.0, 17.5)])
    speed = rng.uniform(32, 45)
    events, samples, t0 = [("hit", 0.0, p.copy())], [], 0.0
    side = +1
    for shot in range(n_shots):
        lift = rng.normal(-0.1, lift_sd)
        v = _aim(p, target, speed, kd, lift)
        tr = _fly(p, v, 600.0, kd, lift, lambda q, w: q[2] <= 0 and w[2] < 0)
        samples += [(t0 + t, q) for t, q, _ in tr]
        tb, pb, vb = tr[-1]
        t0 += tb
        pb = pb.copy(); pb[2] = 0.0
        events.append(("bounce", t0, pb.copy()))
        vb = np.array([vb[0] * mu, vb[1] * mu, -vb[2] * e])
        hit_y = far_y if side > 0 else near_y
        tr = _fly(pb, vb, 600.0, kd, lift * 0.5, lambda q, w: (q[1] >= hit_y if side > 0 else q[1] <= hit_y) or (q[2] <= 0 and w[2] < 0))
        samples += [(t0 + t, q) for t, q, _ in tr]
        t0 += tr[-1][0]
        p = tr[-1][1].copy(); p[2] = max(p[2], 0.4)
        events.append(("hit", t0, p.copy()))
        side = -side
        target = np.array([rng.uniform(1.5, 9.5), rng.uniform(17.5, 22.5) if side > 0 else rng.uniform(1.5, 6.0)])
        speed = rng.uniform(22, 34)
    ts = np.array([s[0] for s in samples]); Xs = np.array([s[1] for s in samples])
    tf = np.arange(0, ts[-1], 1.0 / fps)
    X = np.stack([np.interp(tf, ts, Xs[:, i]) for i in range(3)], 1)
    return dict(t=tf, X=X, events=events, bounces=[(t, q[0], q[1]) for k, t, q in events if k == "bounce"], fps=fps)


def detections(rng, cam, X, p_detect=0.8, noise_px=1.0, false_share=0.5, gap=(0.45, 8), k=6, static_n=2):
    """-> cands (T, k, 3), truth uv (T, 2) (NaN when out of image). false_share: expected share of false peaks among
    all peaks. gap: (start as a share of T, length in frames) occlusion with no ball detection."""
    T = len(X)
    w, h = cam.image_size
    uv = cam.project(X)
    vis = cam.in_image(uv) & (cam.depth(X) > 0)
    uv = np.where(vis[:, None], uv, np.nan)
    cands = np.full((T, k, 3), np.nan)
    g0 = int(gap[0] * T) if gap else -1
    n_true = 0
    rows = [[] for _ in range(T)]
    for i in range(T):
        if vis[i] and rng.random() < p_detect and not (g0 <= i < g0 + (gap[1] if gap else 0)):
            rows[i].append((uv[i, 0] + rng.normal(0, noise_px), uv[i, 1] + rng.normal(0, noise_px), rng.uniform(0.3, 0.95)))
            n_true += 1
    lam = false_share / max(1e-6, 1 - false_share) * n_true / T
    statics = rng.uniform([0, h * 0.3], [w, h], (static_n, 2))
    racket = np.array([w * 0.5, h * 0.8]); rv = np.zeros(2)
    for i in range(T):
        n_f = rng.poisson(lam)
        for _ in range(n_f):
            r = rng.random()
            if r < 0.4:
                s = statics[rng.integers(static_n)]
                rows[i].append((s[0] + rng.normal(0, 0.7), s[1] + rng.normal(0, 0.7), rng.uniform(0.2, 0.8)))
            elif r < 0.7:
                rows[i].append((racket[0] + rng.normal(0, 3), racket[1] + rng.normal(0, 3), rng.uniform(0.2, 0.9)))
            else:
                rows[i].append((rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.15, 0.7)))
        rv = 0.8 * rv + rng.normal(0, 6, 2); racket = np.clip(racket + rv, [0, h * 0.4], [w, h])
        rows[i].sort(key=lambda c: -c[2])
        for j, c in enumerate(rows[i][:k]):
            cands[i, j] = c
    return cands, uv
