"""Live-ball filter: keep only a ball that has been hit and is in flight (Antony, 18 Oct 2026: "track only a ball that's
been hit and is moving, never one lying on the ground or rolling in from another court").

A candidate track (tracklet = consecutive detected frames, gaps <= MAX_GAP) is LIVE only if all of these hold:

1. moving       not static (path extent > STATIC_EXTENT_D ball diameters and > STATIC_MIN_PX px),
2. ballistic    (per segment: only the passing segments of a live tracklet are kept, so one bad stretch, e.g. a
                jump to a wrong blob, does not discard the whole rally) split at velocity reversals (bounces / contacts) into segments of >= MIN_SEG frames; inside each segment
                sliding windows of FIT_WIN frames fit a quadratic x(t), y(t) (a parabola under perspective is only
                locally quadratic, so a whole long approach does not fit one), a window passes when its RMS residual
                <= max(RES_MIN_PX, RES_D x ball diameter) + RES_STEP x its median step (blur centroids wander more on
                fast balls), and >= FIT_PASS of the windows pass; at least one window is in flight: image-down
                acceleration within [G_LO, G_HI] x g at that depth, or court-scale speed >= FAST_MPS (flat fast balls
                bend too little over a short span to measure g),
3. not rolling  median ground-plane speed (3-frame displacement through the homography, so detector jitter does not
                count as motion; exact for a rolling ball, an over-estimate for one in the air) >= MIN_SPEED_MPS,
4. contact      an audio hit within +-CONTACT_S of a tracklet frame (or up to LOOKBACK_S before it, at the
                back-extrapolated position, since the detector often picks the ball up a few frames late) with the ball
                near a player (NEAR_PLAYER_M), or, without player boxes, the hit alone (flagged); or a velocity reversal
                inside the track, or the tracklet continues a live one that ended <= CONTINUE_S earlier,
5. on our court >= IN_GATE_FRAC of its points inside `CourtGate.contains` (court + margin, swept up to max height), so a
                ball from the adjacent court is rejected even when it is in flight (with measured blob sizes the
                height hypothesis must also match the apparent ball size; without them a next-court ball that is on
                the same camera ray as a ball 2-3 m above our side margin cannot be told apart).

Thresholds are first guesses to be tuned on labelled frames (labels/ball/*.csv); every decision carries its reason.
"""
from dataclasses import dataclass, field

import numpy as np

MAX_GAP = 3
MIN_LEN = 5
MIN_SEG = 5
STATIC_EXTENT_D, STATIC_MIN_PX = 3.0, 4.0
RES_MIN_PX, RES_D, RES_STEP = 1.5, 0.6, 0.15
FIT_WIN, FIT_PASS = 7, 0.6
G = 9.81
G_LO, G_HI = 0.2, 4.0
FAST_MPS = 12.0
MIN_SPEED_MPS = 4.0
CONTACT_S = 0.080
CONTINUE_S = 0.5       # a tracklet starting this soon after a live one ended inherits its contact (fragmented track)
LOOKBACK_S = 0.6        # a hit up to this long before the first detection counts via back-extrapolation
NEAR_PLAYER_M = 2.0
IN_GATE_FRAC = 0.8
REVERSAL_DEG = 100.0
SIZE_RELIABLE_PX = 5.0  # measured blob diameters below this are too quantised to rank depth (960 px proxy: almost all)


@dataclass
class Tracklet:
    i0: int
    i1: int                      # inclusive
    idx: np.ndarray              # frame indices with a detection
    xy: np.ndarray               # (n, 2)
    live: bool = False
    keep: np.ndarray = None      # bool per detection: part of a passing (ballistic, in-flight) segment
    reasons: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def tracklets(pos, max_gap=MAX_GAP, min_len=1):
    """Split per-frame positions (N, 2; NaN = none) into tracklets over gaps <= max_gap and jumps < 120 px/frame."""
    ok = np.where(~np.isnan(pos[:, 0]))[0]
    out, cur = [], []
    for i in ok:
        if cur and (i - cur[-1] > max_gap + 1 or np.hypot(*(pos[i] - pos[cur[-1]])) > 120 * (i - cur[-1])):
            out.append(cur); cur = []
        cur.append(i)
    if cur: out.append(cur)
    return [Tracklet(c[0], c[-1], np.array(c), pos[c].copy()) for c in out if len(c) >= min_len]


def _quad_fit(t, v):
    A = np.c_[t ** 2, t, np.ones_like(t)]
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return coef, v - A @ coef


def reversals(idx, xy, min_deg=REVERSAL_DEG, min_step_px=1.0):
    """Indices k into the tracklet where the direction of motion turns by >= min_deg (bounce or contact) or image-y
    velocity flips from down to up (ground bounce)."""
    if len(idx) < 5: return []
    t = idx.astype(float)
    v = np.gradient(xy, t, axis=0)
    out = []
    for k in range(2, len(idx) - 2):
        a, b = xy[k] - xy[k - 2], xy[k + 2] - xy[k]
        na, nb = np.hypot(*a), np.hypot(*b)
        if na < min_step_px or nb < min_step_px: continue
        ang = np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))
        bounce = v[k - 1, 1] > 0.3 and v[k + 1, 1] < -0.3 and xy[k, 1] >= max(xy[k - 1, 1], xy[k + 1, 1])
        if ang >= min_deg or bounce:
            if not out or k - out[-1] >= 3: out.append(k)
    return out


def assess(tr, fps, gate=None, hits=(), t0=0.0, players=None, ball_d_px=None, prev_live_end=None, sizes=None):
    """Fill tr.live / tr.reasons / tr.info. gate: CourtGate (None: court checks skipped and flagged). hits: audio hit
    times (s, same clock as t0 + frame/fps). players: per-frame list of (x0, y0, x1, y1) boxes or None."""
    idx, xy = tr.idx, tr.xy
    t = idx / fps
    reasons, info = [], {}
    if len(idx) < MIN_LEN:
        reasons.append("short")
    ppm = float(np.median(gate.px_per_m(xy))) if gate is not None else None
    d_px = ball_d_px if ball_d_px is not None else (0.067 * float(np.median(gate.ball_scale(xy))) if gate is not None else 3.0)
    extent = float(np.hypot(*(xy.max(0) - xy.min(0))))
    info["extent_px"] = round(extent, 1)
    if extent <= max(STATIC_MIN_PX, STATIC_EXTENT_D * d_px):
        reasons.append("static")
    step = np.hypot(*np.diff(xy, axis=0).T) / np.diff(idx)
    k3 = min(3, len(idx) - 1)
    disp3 = np.hypot(*(xy[k3:] - xy[:-k3]).T) / (idx[k3:] - idx[:-k3]) if k3 > 0 else np.zeros(0)
    speed_px = float(np.median(disp3)) if len(disp3) else 0.0
    if gate is not None and k3 > 0:
        # ground-plane speed: exact for a ball on the ground (the roller we want to reject) and an over-estimate for a
        # ball in the air (its ray meets the ground farther away), so a live ball is never mistaken for a slow one
        c = gate.ground_speed_points(xy)
        gs = np.hypot(*(c[k3:] - c[:-k3]).T) / (idx[k3:] - idx[:-k3]) * fps
        speed_mps = float(np.median(np.where(np.isfinite(gs), gs, np.inf)))
    else:
        speed_mps = speed_px * fps / ppm if ppm else None
    info["speed_mps"] = round(speed_mps, 2) if speed_mps is not None else None
    if speed_mps is not None and speed_mps < MIN_SPEED_MPS and "static" not in reasons:
        reasons.append("slow_or_rolling")
    rev = reversals(idx, xy, min_step_px=max(2.0, d_px))
    info["reversals"] = [int(idx[k]) for k in rev]
    cuts = [0] + rev + [len(idx) - 1]
    res_th = max(RES_MIN_PX, RES_D * d_px)
    g_px = G * (ppm or d_px / 0.067) / fps ** 2           # g in image px / frame^2 at this depth
    segs, keep = [], np.zeros(len(idx), bool)
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a + 1 < MIN_SEG: continue
        n = b - a + 1
        w = min(FIT_WIN, n)
        npass = nwin = 0; rms_max = 0.0; ay_list = []; v_max = 0.0; flight = False
        for s0 in range(a, b - w + 2):
            sl = slice(s0, s0 + w)
            ts = (idx[sl] - idx[s0]).astype(float)
            cx, rx = _quad_fit(ts, xy[sl, 0]); cy, ry = _quad_fit(ts, xy[sl, 1])
            rms = float(np.sqrt(np.mean(rx ** 2 + ry ** 2)))
            st = step[s0:s0 + w - 1]
            th = res_th + RES_STEP * (float(np.median(st)) if len(st) else 0.0)
            nwin += 1; npass += rms <= th; rms_max = max(rms_max, rms)
            ay = 2 * cy[0]; ay_list.append(ay / g_px)
            vseg = np.hypot(cx[1] + 2 * cx[0] * ts.mean(), cy[1] + 2 * cy[0] * ts.mean()) * fps / (ppm or d_px / 0.067)
            v_max = max(v_max, float(vseg))
            if rms <= th and (G_LO * g_px <= ay <= G_HI * g_px or vseg >= FAST_MPS): flight = True
        fit = npass >= FIT_PASS * nwin
        segs.append(dict(f0=int(idx[a]), f1=int(idx[b]), windows=nwin, pass_frac=round(npass / nwin, 2), rms_max=round(rms_max, 2),
                         ay_g_median=round(float(np.median(ay_list)), 2), v_max_mps=round(v_max, 1), fit=bool(fit), flight=bool(flight)))
        if fit and flight: keep[a:b + 1] = True
    # short pieces between reversals (bounce / contact neighbourhoods) are kept when a neighbour segment passes
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a + 1 < MIN_SEG and ((a > 0 and keep[a]) or (b < len(idx) - 1 and keep[b])): keep[a:b + 1] = True
    tr.keep = keep
    info["kept_frac"] = round(float(keep.mean()), 2)
    info["segments"] = segs
    if not segs: reasons.append("no_fit_segment")
    elif not any(sg["fit"] for sg in segs): reasons.append("not_ballistic")
    elif not keep.any(): reasons.append("no_flight")
    # contact
    contact = None
    for h in hits:
        if not (t0 + t[0] - LOOKBACK_S - CONTACT_S <= h <= t0 + t[-1] + CONTACT_S): continue
        k = int(np.argmin(np.abs(t0 + t - h)))
        if abs(t0 + t[k] - h) <= CONTACT_S:
            p = xy[k]
        else:                                              # hit just before the track starts: extrapolate back
            fr = (h - t0) * fps
            n = min(len(idx), 6)
            cx, _ = _quad_fit(idx[:n].astype(float), xy[:n, 0]); cy, _ = _quad_fit(idx[:n].astype(float), xy[:n, 1])
            p = np.array([np.polyval(cx, fr), np.polyval(cy, fr)])
            k = 0
        if players is None:
            contact = dict(kind="audio_no_player", t=float(h)); break
        fi = int(round((h - t0) * fps))
        boxes = players[fi] if 0 <= fi < len(players) else []
        near = (NEAR_PLAYER_M * ppm) if ppm else 60.0
        for (x0, y0, x1, y1) in boxes or []:
            dx = max(x0 - p[0], 0, p[0] - x1); dy = max(y0 - p[1], 0, p[1] - y1)
            if np.hypot(dx, dy) <= near:
                contact = dict(kind="audio_player", t=float(h)); break
        if contact: break
    if contact is None and rev:
        contact = dict(kind="velocity_reversal", frame=info["reversals"][0])
    if contact is None and prev_live_end is not None and 0 <= (idx[0] - prev_live_end) / fps <= CONTINUE_S:
        contact = dict(kind="continues_live", gap_s=round(float((idx[0] - prev_live_end) / fps), 3))
    info["contact"] = contact
    if contact is None: reasons.append("no_contact")
    if gate is not None:
        dm = None
        if sizes is not None:
            dm = np.asarray(sizes, float)[idx]
            dm = np.where(dm >= SIZE_RELIABLE_PX, dm, np.nan)
        frac = float(np.mean(gate.contains(xy, diam_px=dm)))
        info["in_gate"] = round(frac, 2)
        if frac < IN_GATE_FRAC: reasons.append("off_court")
    else:
        info["in_gate"] = None
    tr.reasons, tr.info = reasons, info
    tr.live = not reasons
    return tr


def filter_live(pos, fps, gate=None, hits=(), t0=0.0, players=None, sizes=None):
    """-> (live_pos with non-live frames set to NaN, [Tracklet]). Tracklets are judged in time order so a fragment
    that starts within CONTINUE_S of a live one inherits its contact; inside a live tracklet only the passing
    segments are kept. sizes: optional per-frame measured ball diameter (px, NaN = unknown) for the court test."""
    trs, out, last_end = [], np.full_like(pos, np.nan), None
    for tr in tracklets(pos):
        assess(tr, fps, gate, hits, t0, players, prev_live_end=last_end, sizes=sizes)
        trs.append(tr)
        if tr.live:
            k = tr.idx[tr.keep]; out[k] = tr.xy[tr.keep]; last_end = int(k[-1])
    return out, trs
