"""Rally structure from the whole-point ball track + audio hits (point winner S0 v3, engineering_notes section 2.9).

The ball's image row, normalised so 0 = net line and 1 = near baseline (negative = far side), is a sawtooth during a
rally: a near-team contact is a local maximum (the ball turns back up-court), a far-team contact a local minimum.
Shots = turning points with a real swing on both sides; the rally = the longest chain of alternating shots at rally
pace. Post-point ball feeding gives short, slow, non-alternating chains and is not picked. Audio hits refine shot
times (snap within 0.25 s) and extend the rally by one terminal shot when the ball reaches the other side and a hit
is heard there (the last contact often has no clean turning point: the ball goes into the net or out of view).
Pure numpy. All times in seconds of the source video.
"""
import numpy as np

NET_Y, COURT_W = 11.885, 10.97


def norm_rows(d, view, scale=6.0, peak=0.5):
    """(t, y_norm, x_px, y_px) of good ball detections, y_norm: 0 at the net line, 1 at the near baseline."""
    ball = d["ball"].astype(float) * scale
    t = float(d["t0"]) + np.arange(len(ball)) / float(d["fps"])
    g = np.isfinite(ball[:, 0]) & (d["ball_peak"] >= peak)
    ynet = 0.5 * (view.c2i(0, NET_Y)[1] + view.c2i(COURT_W, NET_Y)[1])
    ynear = view.c2i(COURT_W / 2, 0)[1]
    return t[g], (ball[g, 1] - ynet) / max(ynear - ynet, 50.0), ball[g, 0], ball[g, 1]


def despike(t, y, k=2, tol=0.25):
    """drop isolated detections that disagree with both neighbours' median (TrackNet false positives)."""
    keep = np.ones(len(y), bool)
    for i in range(len(y)):
        lo, hi = max(0, i - k), min(len(y), i + k + 1)
        nb = np.r_[y[lo:i], y[i + 1:hi]]
        tt = np.r_[t[lo:i], t[i + 1:hi]]
        nb = nb[np.abs(tt - t[i]) <= 0.25]
        if len(nb) >= 2 and abs(y[i] - np.median(nb)) > tol:
            keep[i] = False
    return keep


def turning_points(t, y, min_swing=0.3, max_gap=1.0, max_dep=0.9):
    """local extrema of the (despiked) row series with a swing >= min_swing on both sides inside one continuous piece.
    Returns list of (t, y, kind) with kind +1 = near contact (maximum), -1 = far contact (minimum)."""
    out = []
    if len(t) < 5:
        return out
    pieces = np.split(np.arange(len(t)), np.nonzero(np.diff(t) > max_gap)[0] + 1)
    for idx in pieces:
        if len(idx) < 5:
            continue
        tt, yy = t[idx], y[idx]
        # zig-zag: emit a pivot whenever the series reverses by >= min_swing from its running extreme
        piv = []; mode = 0; imax = imin = 0
        for i in range(1, len(yy)):
            if mode == 0:                                    # direction not known yet: track both extremes
                if yy[i] > yy[imax]: imax = i
                if yy[i] < yy[imin]: imin = i
                if yy[imax] - yy[imin] >= min_swing:
                    mode = 1 if imax > imin else -1          # rising after the min / falling after the max
                    piv.append((imin, -1, tt[i] - tt[imin]) if mode == 1 else (imax, 1, tt[i] - tt[imax]))
                continue
            if mode == 1:                                    # rising: track the max
                if yy[i] >= yy[imax]: imax = i
                elif yy[imax] - yy[i] >= min_swing: piv.append((imax, 1, tt[i] - tt[imax])); mode = -1; imin = i
            else:                                            # falling: track the min
                if yy[i] <= yy[imin]: imin = i
                elif yy[i] - yy[imin] >= min_swing: piv.append((imin, -1, tt[i] - tt[imin])); mode = 1; imax = i
        # the first pivot only marks where the zig-zag started (its swing before is unknown): keep it only if it is a
        # genuine end of a descent/ascent inside the piece (not the piece's first sample)
        for i, k, dep in piv:
            if i == 0 or dep > max_dep:                      # a contact sends the ball away fast; a dying ball drifts
                continue
            out.append((float(tt[i]), float(yy[i]), k))
    return out


def near_player(d, view, t, bxy, kind, scale=6.0, max_rel=1.6):
    """True if, at the pose frame nearest t, a player box on the kind's side (+1 near, -1 far half) is within max_rel
    box heights of the ball (image px). Unknown (no pose frame within 0.2 s or no ball x) -> True (do not reject)."""
    if bxy is None or not np.isfinite(bxy[0]):
        return True
    tk = float(d["t0"]) + d["pose_k"] / float(d["fps"])
    j = int(np.argmin(np.abs(tk - t)))
    if abs(tk[j] - t) > 0.2:
        return True
    for b in d["boxes"][j]:
        if not np.isfinite(b[0]) or b[4] < 0.3:
            continue
        x0, y0, x1, y1 = b[:4] * scale; h = max(y1 - y0, 20.0)
        side = 1 if view.i2c([((x0 + x1) / 2, y1)])[0][1] < NET_Y else -1
        if side != kind:
            continue
        cx, cy = (x0 + x1) / 2, y0 + 0.35 * h
        if np.hypot(bxy[0] - cx, bxy[1] - cy) / h <= max_rel:
            return True
    return False


def snap(tp, hits, tol=0.25):
    o = []
    for (t, y, k) in tp:
        near = [h for h in hits if abs(h - t) <= tol]
        o.append((min(near, key=lambda h: abs(h - t)) if near else t, y, k, bool(near)))
    return o


def chains(shots, gmin=0.45, gmax=2.8):
    """split shots into chains of alternating kinds at rally pace."""
    out, cur = [], []
    for s in shots:
        if cur and (s[2] == cur[-1][2] or not (gmin <= s[0] - cur[-1][0] <= gmax)):
            out.append(cur); cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def rally_score(c, swing=0.45):
    """how rally-like a cluster is: alternating near/far contacts at rally pace with a real cross-court swing count 1
    each, audio-confirmed shots 0.5. Post-point feeding (small, slow, same-side moves) scores low."""
    sc = 0.5 * sum(q[3] for q in c)
    for a, b in zip(c[:-1], c[1:]):
        if a[2] * b[2] == -1 and 0.5 <= b[0] - a[0] <= 2.8 and np.isfinite(a[1]) and np.isfinite(b[1]) and abs(a[1] - b[1]) >= swing:
            sc += 1.0
    return sc


def clusters(shots, gmax=3.0):
    """group shots separated by <= gmax s (kinds may repeat: a missed turning point must not split the rally)."""
    out, cur = [], []
    for s in shots:
        if cur and s[0] - cur[-1][0] > gmax:
            out.append(cur); cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def rally(d, view, hits, start, end, min_swing=0.3, max_dep=0.6, gmax=3.0, use_players=True, mode="cluster", fuse_audio=False, extend_audio=False):
    """Returns dict(shots=[(t, y, kind, audio)], t_last, last_kind (+1 near / -1 far / 0), n_shots, terminal, t_first,
    first_kind) for the main rally, or None when the track has no rally structure."""
    t, y, xp, yp = norm_rows(d, view)
    m = (t >= start - 0.5) & (t <= end + 0.5)
    t, y, xp, yp = t[m], y[m], xp[m], yp[m]
    if len(t) < 5:
        return None
    k = despike(t, y); t, y, xp, yp = t[k], y[k], xp[k], yp[k]
    hp = [h for h in hits if start - 0.5 <= h <= end + 0.5]
    tp = turning_points(t, y, min_swing, max_dep=max_dep)
    if use_players:
        pos = {float(a): (b, c) for a, b, c in zip(t, xp, yp)}
        tp = [q for q in tp if near_player(d, view, q[0], pos.get(q[0]), q[2])]
    shots = snap(tp, hp)
    if fuse_audio:                                           # audio hits with no turning point within 0.25 s join as shots
        for h in hp:                                         # (kind from the ball's direction after the hit when seen)
            if any(abs(h - q[0]) <= 0.25 for q in shots):
                continue
            a_ = (t > h + 0.05) & (t <= h + 0.6)
            kd = 0
            if a_.sum() >= 3:
                sl = np.polyfit(t[a_] - h, y[a_], 1)[0]
                kd = 1 if sl < -0.15 else (-1 if sl > 0.15 else 0)
            shots.append((h, float("nan"), kd, True))
        shots.sort(key=lambda q: q[0])
    if not shots:
        return None
    ch = clusters(shots, gmax) if mode == "cluster" else chains(shots, gmax=gmax)
    best = max(ch, key=lambda c: (rally_score(c), -c[0][0]))
    if extend_audio:                                         # audio hits chained on after the cluster while the ball is in play
        moving = lambda h: ((np.abs(t - h) <= 0.6).sum() >= 6 and np.ptp(y[np.abs(t - h) <= 0.6]) >= 0.25)
        for h in hp:
            if best[-1][0] + 0.45 <= h <= best[-1][0] + 2.8 and moving(h):
                best = best + [(h, float("nan"), 0, True)]
    last = best[-1]; terminal = False
    # terminal shot: after the last turning point the ball heads to the other side; a hit heard there 0.45-2.2 s
    # later, with the ball seen on that side (or lost on its way), is the real last contact
    toward = -last[2]
    for h in (hp if last[2] != 0 else []):
        if not (last[0] + 0.45 <= h <= last[0] + 2.2):
            continue
        w = (t >= h - 0.35) & (t <= h + 0.1)
        if w.any() and np.sign(np.median(y[w]) + 0.05 * toward) != np.sign(toward) and abs(np.median(y[w])) > 0.15:
            continue                                         # ball clearly still on the hitter's side at that hit
        best = best + [(h, float(np.median(y[w])) if w.any() else np.nan, toward, True)]; terminal = True
        break
    return dict(shots=best, t_last=best[-1][0], last_kind=best[-1][2], n_shots=len(best), terminal=terminal,
                t_first=best[0][0], first_kind=best[0][2], all_shots=shots)
