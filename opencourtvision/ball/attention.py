"""Player-attention / contact prior for picking the live ball among ball candidates (research, ball tracker v2).

Idea (Antony): the live ball is the one every player watches and that the players hit in turn. From 2D body pose only
(COCO-17 keypoints of the 4 players; no faces, no identity, no biometrics are stored or derived):
  1. gaze: head direction (ear midpoint -> nose) and torso facing point toward the live ball; a candidate scores by how
     many players face it. Players seen from behind (no nose) fall back to "faces the far side".
  2. swing timing: a wrist-speed peak of a player within a few frames of the candidate being within reach of that
     player's wrist marks a contact; candidates there are boosted (and those far from any reachable player at a swing
     peak are not penalised, only not boosted).
  3. stray balls: a candidate that does not move over +-S frames (ball in a pocket, on the ground, on another court)
     is down-weighted.
  4. hitter alternation is a trajectory-level check (contacts must alternate between the two sides of the net):
     `alternation_ok` scores a hit sequence for the fused tracker / learned event model.
When the ball is hidden (no candidate), `gaze_point` gives a weak pseudo-candidate where the players' gaze rays meet.

All coordinates are analysis px (1280x720). Pose comes from pw_tracks npz (kp at pose_k frames, 640x360 px; far_kp in
source px).
"""
import numpy as np

NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12


class Config:
    def __init__(self, kp_min=0.3, gaze_w=0.35, contact_w=0.6, static_w=0.7, static_px=2.5, static_span=6,
                 reach_box=0.9, swing_win=3, swing_min=0.06, hide_peak=0.18, hide_min_players=2, k_out=8):
        self.__dict__.update(locals()); del self.__dict__["self"]


def players_from_npz(d, n, t_off=0, src_w=3840, aw=1280):
    """per analysis frame i (0..n-1): list of (kp (17, 3) analysis px, box (5,)) from the nearest pose frame (<=3 frames
    away). Near-court players from kp (640x360), far-court players from far_kp (source px) when present (they replace a
    near-res duplicate within 15 px)."""
    pk = np.asarray(d["pose_k"]) if "pose_k" in d else np.zeros(0, int)
    if not len(pk):
        return [[] for _ in range(n)]
    s1 = aw / 640.0; s2 = aw / float(src_w)
    per_pose = []
    for j in range(len(pk)):
        ps = []
        for b, k in zip(d["boxes"][j], d["kp"][j]):
            if np.isfinite(b[0]) and b[4] >= 0.4:
                kk = k.copy(); kk[:, :2] *= s1; bb = b.copy(); bb[:4] *= s1; ps.append((kk, bb))
        if "far_kp" in d:
            for b, k in zip(d["far_boxes"][j], d["far_kp"][j]):
                if np.isfinite(b[0]) and b[4] >= 0.4:
                    kk = k.copy(); kk[:, :2] *= s2; bb = b.copy(); bb[:4] *= s2
                    c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
                    ps = [p for p in ps if np.hypot((p[1][0] + p[1][2]) / 2 - c[0], (p[1][1] + p[1][3]) / 2 - c[1]) > 15]
                    ps.append((kk, bb))
        per_pose.append(ps)
    out = []
    for i in range(n):
        q = i + t_off
        j = int(np.argmin(np.abs(pk - q)))
        out.append(per_pose[j] if abs(int(pk[j]) - q) <= 3 else [])
    return out


def _pt(kp, idx, th):
    v = [kp[i, :2] for i in idx if kp[i, 2] >= th and np.isfinite(kp[i, 0])]
    return np.mean(v, 0) if v else None


def head_ray(kp, box, th=0.3, y_net=None):
    """(origin, unit direction) of where a player faces in the image, or None. Nose vs ear midpoint when the nose is
    visible; seen from behind (ears/shoulders but no nose) -> towards the far side (up the image) if the player stands
    below the net row, else down."""
    nose = kp[NOSE, :2] if kp[NOSE, 2] >= th else None
    ears = _pt(kp, (LEAR, REAR), th); sh = _pt(kp, (LSH, RSH), th)
    origin = _pt(kp, (NOSE, LEYE, REYE, LEAR, REAR), th)
    if origin is None:
        origin = sh
    if origin is None:
        return None
    if nose is not None and ears is not None and np.hypot(*(nose - ears)) > 0.5:
        d = nose - ears
        d = d + np.array([0.0, -0.6 if (y_net is None or origin[1] > y_net) else 0.6]) * np.hypot(*d)   # look over the net
    else:
        d = np.array([0.0, -1.0 if (y_net is None or origin[1] > y_net) else 1.0])
    nd = np.hypot(*d)
    return (origin, d / nd) if nd > 0 else None


def wrist_speed(players_seq, i, th=0.3):
    """per player at frame i: (max wrist speed in box heights / frame over i-1..i+1, wrist positions) matched to the
    same player at i-3/i+3 by box centre."""
    out = []
    cur = players_seq[i]
    for kp, bb in cur:
        h = max(4.0, bb[3] - bb[1]); c = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        wr = [kp[w, :2] for w in (LWR, RWR) if kp[w, 2] >= th]
        sp = 0.0
        for di in (-3, 3):
            j = i + di
            if 0 <= j < len(players_seq) and players_seq[j]:
                prev = min(players_seq[j], key=lambda p: np.hypot((p[1][0] + p[1][2]) / 2 - c[0], (p[1][1] + p[1][3]) / 2 - c[1]))
                for w in (LWR, RWR):
                    if kp[w, 2] >= th and prev[0][w, 2] >= th:
                        sp = max(sp, np.hypot(*(kp[w, :2] - prev[0][w, :2])) / 3.0 / h)
        out.append((sp, wr, c, h))
    return out


def features(cands, players_seq, cfg=None, y_net=None):
    """cands (T, K, 3) -> feats (T, K, 4): [gaze (0..1 share of players facing it), contact (0..1), static (0..1),
    nearest-player distance in box heights]. NaN rows where the candidate is NaN."""
    cfg = cfg or Config()
    T, K, _ = cands.shape
    F = np.full((T, K, 4), np.nan)
    rays = [[r for r in (head_ray(kp, bb, cfg.kp_min, y_net) for kp, bb in players_seq[i]) if r is not None] for i in range(T)]
    swings = [wrist_speed(players_seq, i, cfg.kp_min) for i in range(T)]
    for i in range(T):
        for k in range(K):
            u = cands[i, k, :2]
            if not np.isfinite(u[0]):
                continue
            g = [max(0.0, float(np.dot(d, (u - o) / max(1e-6, np.hypot(*(u - o)))))) for o, d in rays[i]]
            gaze = float(np.mean([x > 0.8 for x in g])) if g else np.nan
            con, dmin = 0.0, np.inf
            for di in range(-cfg.swing_win, cfg.swing_win + 1):
                j = i + di
                if not (0 <= j < T):
                    continue
                for sp, wr, c, h in swings[j]:
                    dmin = min(dmin, np.hypot(*(u - c)) / h) if di == 0 else dmin
                    if sp < cfg.swing_min or not wr:
                        continue
                    dw = min(np.hypot(*(u - w)) for w in wr) / h
                    if dw <= cfg.reach_box:
                        con = max(con, min(1.0, sp / (3 * cfg.swing_min)) * (1 - dw / cfg.reach_box * 0.5))
            # static: same spot (within static_px) found in most frames of +-static_span
            hits = 0; tot = 0
            for di in range(-cfg.static_span, cfg.static_span + 1):
                j = i + di
                if di == 0 or not (0 <= j < T):
                    continue
                tot += 1
                cj = cands[j, :, :2]; okj = np.isfinite(cj[:, 0])
                if okj.any() and np.min(np.hypot(*(cj[okj] - u).T)) <= cfg.static_px:
                    hits += 1
            F[i, k] = (gaze, con, hits / max(1, tot), dmin)
    return F


def gaze_point(rays):
    """least-squares meeting point of >= 2 image rays (origin, unit dir), only in front of each origin; None if parallel."""
    if len(rays) < 2:
        return None
    A = np.zeros((2, 2)); b = np.zeros(2)
    for o, d in rays:
        P = np.eye(2) - np.outer(d, d); A += P; b += P @ o
    if abs(np.linalg.det(A)) < 1e-3:
        return None
    x = np.linalg.solve(A, b)
    if sum(np.dot(x - o, d) > 0 for o, d in rays) < len(rays) - 0:
        return None
    return x


def reweight(cands, feats, players_seq=None, cfg=None, use=("gaze", "contact", "static"), hide=False, y_net=None, size=(1280, 720)):
    """peaks re-scored by the prior, neutral without evidence (a lone candidate keeps its peak):
      gaze    relative within the frame: p *= 1 - gaze_w * (best gaze share in the frame - this one's)
      contact boost only: p *= 1 + contact_w * contact
      static  p *= 1 - static_w * static share
    best first again. hide=True adds a weak gaze-meeting pseudo-candidate in frames with no candidate. -> (T, k_out, 3)."""
    cfg = cfg or Config()
    T, K, _ = cands.shape
    out = np.full((T, cfg.k_out, 3), np.nan)
    for i in range(T):
        rows = []
        ok = np.isfinite(cands[i, :, 0])
        g_all = feats[i, ok, 0] if ok.any() else np.zeros(0)
        gmax = np.nanmax(g_all) if len(g_all) and np.isfinite(g_all).any() else np.nan
        for k in range(K):
            if not ok[k]:
                continue
            g, c, s, _ = feats[i, k]
            m = 1.0
            if "gaze" in use and np.isfinite(g) and np.isfinite(gmax):
                m *= 1 - cfg.gaze_w * (gmax - g)
            if "contact" in use:
                m *= 1 + cfg.contact_w * c
            if "static" in use:
                m *= 1 - cfg.static_w * s
            rows.append((cands[i, k, 0], cands[i, k, 1], cands[i, k, 2] * m))
        if hide and not rows and players_seq is not None:
            rays = [r for r in (head_ray(kp, bb, cfg.kp_min, y_net) for kp, bb in players_seq[i]) if r is not None]
            if len(rays) >= cfg.hide_min_players:
                x = gaze_point(rays)
                if x is not None and 0 <= x[0] < size[0] and 0 <= x[1] < size[1]:
                    rows.append((x[0], x[1], cfg.hide_peak))
        rows.sort(key=lambda r: -r[2])
        for j, r in enumerate(rows[:cfg.k_out]):
            out[i, j] = r
    return out


def alternation_ok(hits):
    """hits: list of (frame, side) with side +1 near / -1 far -> share of consecutive hit pairs on opposite sides."""
    if len(hits) < 2:
        return None
    s = [h[1] for h in sorted(hits)]
    return float(np.mean([a != b for a, b in zip(s, s[1:])]))
