"""Virtual-camera caps, zoom target and wide mode for the 9:16 reel (editorial spec §5.2, Media Director answers v1).

v4.6 (Engineer, 18 Oct; MD review of 0c59fa6 for reel v4.5 + the far-partner concern from 37302d7): pre-serve phase
(clip start -> serve contact) keeps the server in the crop from the first frame (backward server track, soft keep-in),
pans at <= 0.25 cw/s with no reversal and ignores the bounced ball; the gate after a serve leads to the returner 0.8 s
early; far-side players get a soft keep-in (no widening, away from gates); a clip whose role player touches the source
frame edge is flagged reject (source_edge_check); hitters only from person-sized boxes near the ball, a verified side
picks the server; a clip the tight v4.5 zoom cannot hold is re-planned with the v4.1 relevant span. Checks:
preserve_qa / framing_checks; bench: benchmarks/reel_camera_bench.py.

v4.5 (Engineer, 18 Oct; md-v4 crop_height_check): gate_spread() replaces the v4.1 relevant-span zoom need (the hitter
and receiver were unioned across the flight, so the calm zoom sat at ~full height all rally), held over spread_hold_s
(hysteresis); widen_request() widens a missed gate (soft receivers too) by <= widen_step per re-plan; no ball-lost
player-framing widen when gates frame the players; partner_stats() = far-partner / sliced stats. dev_match reel, MD
check: mean crop height 0.932 -> 0.834, >= 95 % frames 41.9 -> 24.0 % (pass), roles 0/28 + jerk + caps unchanged.

v4.3 (Media Director camera patch on 303246d, adopted 2026-10-18): clips with verified contacts are planned in one shot
by plan_clip() -> gate_plan() (contact gates, LP under pan 0.8/0.5 cw/s and 1.35 cw/s^2, minimum peak jerk, arrive on
the next hitter 0.35 s early, widen just enough where a hitter gate cannot be held, auto-wide / reject flag when
infeasible). limit_1d / follow_pan below remain only for clips without verified contacts and for the tilt. The text
below describes the 303246d follower; the CAMERA defaults are the v4.3 values.

Deliberately small and isolated so a dynamic-zoom camera can replace it: `zoom_target()` says how much of the source
height each frame should show, `limit_1d()` enforces the rate caps; `pan_caps()` / `zoom_caps()` wrap it, and
`vertical_follow()` places the crop vertically with the same caps.

Hard caps (defaults in CAMERA, overridable from the reel config's "camera" block):
  pan    <= 0.5 crop-widths/s, up to 0.8 only while the ball is in flight between two verified contacts
         (<= 0.6 crop-widths/s^2 always); lifted only on the big-shot ramp of a "punch_in" moment
Pan planning (offline, plan_pan): between verified contacts the target is an eased move from one contact position to
the next, and the whole target is read pan_lead_s ahead, so the camera is already moving toward the next contact
before the hit instead of chasing it. Where the capped pan would let the ball/players leave the crop, the planner asks
for a wider crop first (need_h, read ahead by the time the 15 %/s zoom needs) and pans only for what the extra width
does not cover; it only widens for exits predicted < exit_lead_s (0.5 s) ahead, and never above the soft ceiling
h_ceiling (90 %) unless the player spread needs it or the ball is in a verified fast flight. Mean crop height and the
share of frames >= 95 % are a soft check (warning). Smoothness beats coverage: the path is low-pass filtered and re-capped, never sped up to keep the ball.
  zoom   <= 15 %/s change of the crop size; up to 25 %/s only on the big-shot ramp of a "punch_in" moment
  tilt   vertical follow: tilt_max 0.5 crop-heights/s, tilt_acc 0.6 crop-heights/s^2 (pre-v4.3 limits; no in-flight
         boost), dead zone 15 % of crop height
  all    no direction reversal within 0.5 s of the previous direction change
Crop height (fraction of source height, 9:16 width follows): 0.75 default (1620 px from 2160), widening to 1.0 when
the camera frames players (ball lost), the players spread wider than the default crop, or the vertical player box
(far players' heads to near players' feet) does not fit; tightening to 0.65 only for verified big shots, easing in
from 1.3 s before the contact so it arrives on the shot (lookahead; the ramp stays under the zoom-rate cap). Wide mode: the whole 16:9 frame letterboxed over a blurred fill, at most `max_wide` moments
per reel, only when the players do not fit even at full height for most of a fast (net-heavy) exchange.
jerk_p95 is measured on the FLOAT crop-centre path (before rounding to whole pixels; the rounded path's value is kept as a
debug field) and passes when <= jerk_ratio x the calibration baseline; with no baseline the result is unknown.
The baseline should be the v3 reel's float path. That path cannot be rebuilt (its ball track was not kept), so the
example config uses the cross-check value: v3's rendered crop centre smoothed by a 9-frame centred mean (jerk_baseline_source).
"""
import time

import numpy as np

CAMERA = dict(
    h_default=0.75, h_wide=1.0, h_tight=0.65,        # crop height / source height
    pan_max=0.5, pan_acc=1.35,                        # crop-widths per s, per s^2 (v4.2: 0.6 -> 1.35, jerk-limited below)
    tilt_max=0.5, tilt_acc=0.6,                       # crop-heights per s, per s^2: the pre-v4.3 tilt limits (v4.3 raised the
                                                      # shared acceleration to 1.35; MD 18 Oct: restore the old tilt, keep the new pan)
    pan_fast=0.8, max_flight_s=2.5,                   # in flight between two verified contacts <= max_flight_s apart
    pan_lead_s=0.5, pan_smooth_s=0.1, exit_margin=0.03,   # target read-ahead, low-pass sigma, keep-inside margin (of cw)
    exit_lead_s=0.5,                                  # widen only for exits predicted less than this far ahead
    h_ceiling=0.9,                                    # soft ceiling: above it only for player spread or verified fast flight
    h_fast_ceiling=0.95,                              # v4.1 patch: verified fast flight widens to at most 95 %
    rel_pre_s=0.5, rel_post_s=0.25,                   # v4.1 patch: a contact's hitter must be framed from 0.5 s before to 0.25 s after it
    rel_land_s=0.9,                                   # v4.1 patch: predicted landing time after the last contact (no next contact)
    rel_hold_s=0.5,                                   # v4.1 patch: hysteresis, the framed set is held >= 0.5 s (no pumping on a switch)
    rel_margin=0.03,                                  # v4.1 patch: hitter / receiver kept 3 % inside the frame at contact
    calm_close_s=2.0, calm_ease_s=0.35,               # v4.1 patch: skip tightenings shorter than 2 s; ease zoom targets (sigma, s)
    h_mean_target=(0.75, 0.85), ge95_max=0.25,        # soft check (warning): mean crop height, share of frames >= 95 %
    doubles_h_mean_max=0.93, doubles_ge95_max=0.50,   # doubles (not padel): mean <= 0.93, >= 95 % on <= 50 % of (non-wide) frames
    calm_shape_close_s=2.0, calm_shape_sigma_s=0.5, calm_shape_dil=2.0,   # v4.3 zoom taste pass, as run on the MD reel
    dip_fill=0.07,                                    # v4.3: zoom-in dips shallower than 7 % are filled (no breathing)
    spread_mode="gate",                               # v4.5 spread fix: "gate" = only what must be in frame at the same time
                                                      # (each role over its +-0.1 s gate window); "relevant" = v4.1 union span
    spread_hold_s=0.5,                                # v4.5 hysteresis: a gate's spread need is held over this centred window
    widen_soft=True,                                  # v4.5: a missed soft gate (predicted receiver) also widens, early
    widen_step=0.1,                                   # v4.5: widen a missed gate by <= 10 % per re-plan (just enough)
    fallback_wide_with_gates=False,                   # v4.5: with verified contacts the gates frame the players, so the
                                                      # 303246d ball-lost player-framing widen (to h_ceiling) is not applied
    w_ball=3.0,                                       # v4.5: gate-LP weight of the soft ball keep-in (was 1.0)
    preserve=True,                                    # v4.6 (MD reel v4.5 review): clip start -> serve contact frames the server
    preserve_min_s=1.0,                               # ... when the clip starts at least this long before the first contact
    preserve_pan_max=0.25,                            # ... with a slow pan (crop-widths/s) and no direction reversal at all
    w_server=20.0,                                    # ... server keep-in weight per frame (px; far partner 4, cuts 3, ball 3): a
                                                      # 1 s conflict (60 frames) still costs less than a missed gate (1000 / px)
    serve_lead_s=0.8,                                 # v4.6: after a serve, lead to the returner this early (gates: lead 0.35 s)
    far_keep=True, w_far=10.0, far_guard_s=0.4,       # v4.6 far-partner framing: soft keep-in of each far-side player that fits the
                                                      # crop, not within far_guard_s of a hard gate (gates first)
    far_out_max=0.40,                                 # far_partner_check: far player out of frame on <= 40 % of frames
    source_edge_px=2.0, source_edge_pad_s=0.5,        # source_edge_check: a role player at the 4K frame edge (analysis px) near their gate
    source_edge_min_frames=6,                         # ... for at least 0.1 s
    infeasible_action="wide",                         # contacts too far apart for the pan caps: "wide" (if allowed) else "reject"
    zoom_rate=0.22, zoom_rate_punch=0.25,             # max relative crop-size change per s (v4.2: 15 -> 22 %/s, eased; punch_in 25 %/s)
    hold_s=0.5,                                       # min time between direction changes (pan, tilt and zoom)
    tight_lead_s=1.3, tight_hold_s=0.8,               # ease in from 1.3 s before a verified big shot, hold 0.8 s after it
    v_dead_zone=0.15, v_margin=0.04,                  # vertical dead zone (fraction of crop height), box margin (of SH)
    center_y=0.5,                                     # vertical centre only when there are no player boxes
    spread_margin=0.9,                                # widen when the players' union is wider than 90 % of the crop
    wide_spread_frac=0.6, wide_min_fast=3, wide_fast_gap_s=0.8, max_wide=1,
    jerk_baseline=None, jerk_ratio=0.5,               # px/frame^3 of the v3 reel; pass <= 0.5 x baseline
    jerk_baseline_source=None,                        # where the baseline came from (e.g. "cross-check: ...")
    plan_budget_s=20.0,                               # v4.7: wall-time budget for plan_clip (all re-plans included); past it
                                                      # the clip falls back to safe_plan (never hangs: P059/P062/P082 600 s+)
    locked_wide=False,                                # v4.7 broadcast grammar (16:9): the full source frame, no pans at all
    safe_hold_s=0.6,                                  # v4.7 safe_plan: a framing key is held this long around its contact
    planner="l1",                                     # v4.8: "l1" = l1_plan (one convex LP, Grundmann L1-optimal path);
                                                      # "gate" = the v4.3-v4.6 gate_plan loop (kept for comparison)
    l1_w=(10.0, 1.0, 100.0), l1_w_target=0.05,        # v4.8: |v|, |a|, |jerk| weights (Grundmann et al. 2011) + target pull
    l1_w_gate=1000.0,                                 # v4.8: per-px penalty for a framing constraint that cannot hold
)


class PlanTimeout(RuntimeError):
    """plan_clip ran past cfg['_deadline'] (time.monotonic()); callers fall back to safe_plan / locked wide."""


def _deadline_left(cfg):
    d = cfg.get("_deadline")
    if d is None: return None
    left = d - time.monotonic()
    if left <= 0: raise PlanTimeout(f"camera planner over its {cfg.get('plan_budget_s', '?')} s budget")
    return left


def crop_size(h_frac, SH=2160):
    """Crop (w, h) in source px for a height fraction, 9:16, even sizes."""
    ch = (np.round(np.asarray(h_frac) * SH / 2) * 2).astype(int)
    return (np.round(ch * 9 / 16 / 2) * 2).astype(int), ch


def limit_1d(x, vmax, amax=None, hold=0, x0=None):
    """Follow x under per-frame velocity cap vmax (scalar or array, np.inf = uncapped), optional acceleration cap amax,
    and no direction reversal within `hold` frames of the previous direction change (the camera stops instead)."""
    x = np.asarray(x, float); n = len(x)
    vmax = np.broadcast_to(np.asarray(vmax, float), (n,)).copy()
    if amax is not None and n > 1:      # a cap that drops (e.g. end of a fast in-flight window) is met by braking early
        for i in range(n - 2, -1, -1): vmax[i] = min(vmax[i], vmax[i + 1] + amax)
    out = np.empty(n); out[0] = x[0] if x0 is None else x0
    v, direction, changed = 0.0, 0, -10 ** 9
    for i in range(1, n):
        want = x[i] - out[i - 1]
        if amax is not None and np.isfinite(vmax[i]):
            want = float(np.sign(want) * min(abs(want), np.sqrt(2 * amax * abs(want))))   # brake in time: no overshoot
            want = float(np.clip(want, v - amax, v + amax))
        want = float(np.clip(want, -vmax[i], vmax[i]))
        s = int(np.sign(want)) if abs(want) > 1e-9 else 0
        if s and direction and s != direction and i - changed < hold:    # reversal too soon: brake instead
            want = direction * max(0.0, abs(v) - amax) if amax is not None and np.isfinite(vmax[i]) else 0.0
            s = int(np.sign(want)) if abs(want) > 1e-9 else 0
        if s and s != direction: direction, changed = s, i
        v = want; out[i] = out[i - 1] + v
    return out


def pan_caps(cx, cw, fps, punch_in=None, cfg=CAMERA, fast=None, rate=None, acc=None):
    """Crop-centre path under the pan caps. cw: crop width per frame. punch_in: bool mask of exempt frames.
    fast: bool mask of in-flight frames (between verified contacts) allowed pan_fast instead of pan_max.
    rate: override the base cap (crop-sizes/s), e.g. a separate cap for the vertical follow; acc: override pan_acc."""
    cw = np.asarray(cw, float)
    base = cfg["pan_max"] if rate is None else rate
    vmax = base * cw / fps
    if fast is not None: vmax = np.where(fast, max(base, cfg["pan_fast"]) * cw / fps, vmax)
    if punch_in is not None: vmax = np.where(punch_in, np.inf, vmax)
    acc = cfg["pan_acc"] if acc is None else acc
    return limit_1d(cx, vmax, acc * float(np.median(cw)) / fps ** 2, int(round(cfg["hold_s"] * fps)))


def flight_mask(n, fps, t0, contacts, cfg=CAMERA):
    """Frames strictly between two consecutive VERIFIED contacts (times in s) at most max_flight_s apart."""
    m = np.zeros(n, bool)
    ts = t0 + np.arange(n) / fps
    c = sorted(contacts)
    for a, b in zip(c, c[1:]):
        if 0 < b - a <= cfg["max_flight_s"]: m |= (ts > a) & (ts < b)
    return m


def _gauss(x, sigma):
    x = np.asarray(x, float)
    if sigma < 0.5 or len(x) < 3: return x.copy()
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, "valid")


def _fwd_max(x, k):
    x = np.asarray(x, float)
    return np.array([x[i:i + k + 1].max() for i in range(len(x))]) if k > 0 else x


def plan_pan(target, fps, t0=0.0, contacts=(), contact_x=(), cw_def=912, SH=2160, cfg=CAMERA):
    """Offline pan plan. target: per-frame framing target x (ball, or players when the ball is lost), source px.
    contacts / contact_x: verified contact times (s) and ball x at each. Returns dict:
      planned  eased contact-to-contact target read pan_lead_s ahead and low-passed (what the camera aims for)
      fast     in-flight mask (pan_fast allowed)
      need_h   crop-height fraction needed so the target stays inside the default-capped camera, read ahead by the
               time the zoom cap needs to get there (feed into zoom_target: widen first, then pan)"""
    tg = np.asarray(target, float).copy(); n = len(tg)
    ts = t0 + np.arange(n) / fps
    cs = sorted(zip(contacts, contact_x))
    for (a, xa), (b, xb) in zip(cs, cs[1:]):
        if 0 < b - a <= cfg["max_flight_s"]:
            sel = (ts >= a) & (ts <= b); u = (ts[sel] - a) / (b - a)
            tg[sel] = xa + (xb - xa) * u * u * u * (u * (6 * u - 15) + 10)          # smootherstep a -> b
    L = int(round(cfg["pan_lead_s"] * fps))
    planned = _gauss(np.r_[tg[L:], np.full(min(L, n), tg[-1])][:n], cfg["pan_smooth_s"] * fps)
    fast = flight_mask(n, fps, t0, contacts, cfg)
    cw = np.full(n, float(cw_def))
    cam = pan_caps(planned, cw, fps, None, cfg, fast)
    dev = np.abs(np.asarray(target, float) - cam) + cfg["exit_margin"] * cw_def
    need_w = np.where(dev > cw_def / 2, 2 * dev, 0.0)
    need_h = need_w * 16 / 9 / SH
    need_h = _fwd_max(need_h, int(round(cfg["exit_lead_s"] * fps)))                   # only exits < exit_lead_s ahead
    need_h = np.minimum(need_h, np.where(fast, cfg.get("h_fast_ceiling", 1.0), cfg["h_ceiling"]))   # 90 % / 95 % in fast flight (v4.1)
    return dict(planned=planned, fast=fast, need_h=np.minimum(need_h, cfg["h_wide"]), exit=need_w > 0, exit_frames=int((need_w > 0).sum()))


def follow_pan(planned, cw, cw_def, fps, fast=None, punch_in=None, cfg=CAMERA):
    """Final pan: the crop centre only has to keep the default-width window around the plan inside the (possibly
    wider) crop, so with extra width the camera pans only for the remainder. Then caps, low-pass, caps again."""
    planned = np.asarray(planned, float); cw = np.asarray(cw, float)
    slack = np.maximum(0.0, (cw - cw_def) / 2)
    guess = _gauss(planned, 1.0 * fps)                                                   # the laziest smooth path
    tgt = np.clip(guess, planned - slack, planned + slack)
    out = pan_caps(tgt, cw, fps, punch_in, cfg, fast)
    out = _gauss(out, cfg["pan_smooth_s"] * 2 * fps)
    return pan_caps(out, cw, fps, punch_in, cfg, fast)


def zoom_caps(h_target, fps, cfg=CAMERA, rate=None):
    """Height fraction under the zoom-rate cap (in log space, so in and out are symmetric) and the reversal rule.
    rate: optional per-frame max rate (fraction per s), e.g. from zoom_rates(); default cfg zoom_rate everywhere."""
    r = cfg["zoom_rate"] if rate is None else np.asarray(rate, float)
    lz = limit_1d(np.log(np.asarray(h_target, float)), np.log1p(r) / fps, None, int(round(cfg["hold_s"] * fps)))
    return np.exp(lz)


def _lead(cfg, punch):
    """Ramp length (s) into a big shot: tight_lead_s, or for punch_in the time the 25 %/s cap needs."""
    if not punch: return cfg["tight_lead_s"]
    return min(cfg["tight_lead_s"], np.log(cfg["h_default"] / cfg["h_tight"]) / np.log1p(cfg["zoom_rate_punch"]))


def zoom_rates(n, fps, t0=0.0, big_shots=(), punch_in=False, cfg=CAMERA):
    """Per-frame zoom-rate cap: zoom_rate, raised to zoom_rate_punch only on the big-shot ramps of a punch_in moment."""
    r = np.full(n, cfg["zoom_rate"])
    if punch_in:
        ts = t0 + np.arange(n) / fps; L = _lead(cfg, True)
        for t in big_shots: r[(ts >= t - L - 0.05) & (ts <= t + cfg["tight_hold_s"])] = cfg["zoom_rate_punch"]
    return r


def zoom_target(n, fps, t0=0.0, fallback=None, spread_px=None, big_shots=(), cfg=CAMERA, SH=2160, need_h=None, punch_in=False):
    """Per-frame crop-height fraction: default; tight for verified big shots (times in s) with a lookahead ramp that
    starts tight_lead_s before the contact and arrives on it (linear in log scale, so its rate stays under the cap);
    wider where `fallback` (player framing, up to the soft ceiling h_ceiling), as wide as the players' spread physically
    needs (`spread_px`, source px; may exceed the ceiling), or the vertical player box / exit plan asks (`need_h`).
    Widening wins over tightening: never cut players off."""
    lh = np.full(n, np.log(cfg["h_default"]))
    ts = t0 + np.arange(n) / fps
    lt, ld = np.log(cfg["h_tight"]), np.log(cfg["h_default"])
    L = _lead(cfg, punch_in)
    for t in big_shots:
        ramp = ld + (lt - ld) * np.clip((ts - (t - L)) / L, 0, 1)
        prof = np.where(ts <= t + cfg["tight_hold_s"], ramp, ld)
        lh = np.minimum(lh, prof)
    h = np.exp(lh)
    if fallback is not None:                       # player framing: wider, but under the soft ceiling
        h = np.where(np.asarray(fallback, bool), np.maximum(h, cfg["h_ceiling"]), h)
    if spread_px is not None:                      # what the player spread physically needs (may exceed the ceiling)
        need_sp = np.nan_to_num(np.asarray(spread_px, float)) / cfg["spread_margin"] * 16 / 9 / SH
        h = np.maximum(h, np.minimum(need_sp, cfg["h_wide"]))
    if need_h is not None: h = np.maximum(h, np.nan_to_num(np.asarray(need_h, float)))
    return np.clip(h, cfg["h_tight"], cfg["h_wide"])


def vertical_box(boxes, scale, n=None):
    """Per frame (top, bottom) in source px of the player box: far players' heads (min y0) to near players' feet
    (max y1). Frames without boxes are interpolated; all NaN when no frame has y extents (old 3-tuple blobs)."""
    n = len(boxes) if n is None else n
    top, bot = np.full(n, np.nan), np.full(n, np.nan)
    for i, bb in enumerate(boxes[:n]):
        bb = [b for b in bb if len(b) >= 5]
        if bb: top[i] = min(b[3] for b in bb) * scale; bot[i] = max(b[4] for b in bb) * scale
    g = ~np.isnan(top)
    if g.any():
        t = np.arange(n); top = np.interp(t, t[g], top[g]); bot = np.interp(t, t[g], bot[g])
    return top, bot


def need_height(top, bot, fps, SH=2160, cfg=CAMERA, lookahead_s=1.0):
    """Height fraction needed to fit the vertical player box plus margins, with a 1 s look-ahead max so widening
    starts before the players spread (the zoom-rate cap makes it lag otherwise). NaN-free; 0 where unknown."""
    need = np.nan_to_num((bot - top + 2 * cfg["v_margin"] * SH) / SH)
    k = max(1, int(round(lookahead_s * fps)))
    return np.clip(np.array([need[i:i + k].max() for i in range(len(need))]) if len(need) else need, 0, 1)


def vertical_follow(top, bot, ch, fps, SH=2160, cfg=CAMERA, punch_in=None):
    """Crop top y per frame: follow the centre of the player box with a dead zone of v_dead_zone x crop height, the
    same smoothing as the horizontal path, the pan caps (in crop-heights) and the reversal rule. Without player boxes,
    centre_y fallback."""
    ch = np.asarray(ch, float); n = len(ch)
    if n == 0: return np.zeros(0, int)
    if np.isnan(top).all():
        cy = np.full(n, cfg["center_y"] * SH)
    else:
        tgt = (top + bot) / 2
        cy = np.empty(n); c = tgt[0]
        for i in range(n):                        # dead-zone follower (same gain as the horizontal path)
            d = tgt[i] - c; dz = cfg["v_dead_zone"] * ch[i] / 2
            if abs(d) > dz: c += (d - np.sign(d) * dz) * 0.07
            cy[i] = c
        if n > 91:
            from scipy.signal import savgol_filter
            cy = savgol_filter(cy, 91, 2)
        cy = np.clip(cy, ch / 2, SH - ch / 2)
        cy = pan_caps(cy, ch, fps, punch_in, cfg, rate=cfg.get("tilt_max", cfg["pan_max"]), acc=cfg.get("tilt_acc", cfg["pan_acc"]))
    return np.clip(np.round(cy - ch / 2), 0, SH - ch).astype(int)


def spread(boxes, scale):
    """Per frame: width (source px) of the union of all player blobs, NaN when none."""
    return np.array([(max(b[1] for b in bb) - min(b[0] for b in bb)) * scale if bb else np.nan for bb in boxes])


def wide_trigger(spread_px, hits, fps, SH=2160, cfg=CAMERA):
    """-> (bool, reason). Wide mode only when players do not fit even at full height on >= wide_spread_frac of the
    frames AND the moment has a fast exchange (>= wide_min_fast consecutive hits < wide_fast_gap_s apart, i.e. net play)."""
    sp = np.asarray(spread_px, float)
    full_w = crop_size(1.0, SH)[0]
    frac = float(np.mean(np.nan_to_num(sp) > full_w)) if len(sp) else 0.0
    hs = sorted(hits); run = best = 1 if hs else 0
    for p, q in zip(hs, hs[1:]):
        run = run + 1 if q - p < cfg["wide_fast_gap_s"] else 1; best = max(best, run)
    ok = frac >= cfg["wide_spread_frac"] and best >= cfg["wide_min_fast"]
    return ok, f"players wider than the full-height crop on {frac:.0%} of frames, fastest run {best} hits"


def wide_filter(fps, nsplit, out_w=1080, out_h=1920, blur=30):
    """ffmpeg filter head for wide mode: full 16:9 frame letterboxed into 9:16 over a blurred, cropped fill."""
    return (f"[0:v]split=2[wbg][wfg];[wbg]scale=-2:{out_h},crop={out_w}:{out_h},boxblur={blur}:3[wbb];"
            f"[wfg]scale={out_w}:-2[wfs];[wbb][wfs]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps},split={nsplit}")


def crop_render(x, y, cw, ch, SW, SH, out_w=1080, out_h=1920):
    """Per-frame ffmpeg plan for a crop whose SIZE changes (zoom). Changing crop w/h mid-stream drops every frame on
    ffmpeg 7.1, so: fixed full-height window (x only) -> per-frame scale (eval=frame) -> fixed out_w x out_h crop.
    Returns dict of int arrays win_x, sw, sh, ox, oy and the window width cw0."""
    x, y, cw, ch = (np.asarray(v, float) for v in (x, y, cw, ch))
    cw0 = int(crop_size(1.0, SH)[0])
    win_x = np.clip(np.round(x + cw / 2 - cw0 / 2), 0, SW - cw0).astype(int)
    k = out_h / ch
    sw = np.maximum(out_w, 2 * np.round(cw0 * k / 2)).astype(int)
    sh = np.maximum(out_h, 2 * np.round(SH * k / 2)).astype(int)
    ox = np.clip(np.round((x - win_x) * k), 0, sw - out_w).astype(int)
    oy = np.clip(np.round(y * k), 0, sh - out_h).astype(int)
    return dict(win_x=win_x, sw=sw, sh=sh, ox=ox, oy=oy, cw0=cw0)


def crop_render_cmds(plan, fps):
    return [f"{i / fps:.4f} crop@win x {plan['win_x'][i]}, scale@zoom w {plan['sw'][i]}, scale@zoom h {plan['sh'][i]}, "
            f"crop@out x {plan['ox'][i]}, crop@out y {plan['oy'][i]};" for i in range(len(plan["win_x"]))]


def crop_render_head(plan, cmd_file, SH, fps, nsplit, out_w=1080, out_h=1920):
    p = {k: (v[0] if hasattr(v, "__len__") else v) for k, v in plan.items()}
    return (f"[0:v]sendcmd=f={cmd_file},crop@win=w={p['cw0']}:h={SH}:x={p['win_x']}:y=0:exact=1,"
            f"scale@zoom=w={p['sw']}:h={p['sh']}:flags=lanczos:eval=frame,"
            f"crop@out=w={out_w}:h={out_h}:x={p['ox']}:y={p['oy']}:exact=1,setsar=1,fps={fps},split={nsplit}")


QUANT_FLOOR_PX = 2.0   # p95 |d3| of ANY moving crop rounded to whole pixels: why the metric uses the float path


def crop_height_qa(h, cfg=CAMERA, fmt=None, sport=None):
    """Soft check (warning only, never a failure). Singles and padel: mean crop height in h_mean_target and >= 95 % on
    < ge95_max frames. Doubles (tennis): mean <= doubles_h_mean_max and >= 95 % on <= doubles_ge95_max frames (two
    framed roles across a wide court sit near full height far more often). Pass non-wide frames for the doubles gate."""
    h = np.asarray(h, float)
    if not len(h): return {"mean_crop_height": None, "pct_frames_ge95": None, "warnings": []}
    mean, ge = float(h.mean()), float((h >= 0.95 - 1e-9).mean())
    w = []
    if fmt == "doubles" and sport != "padel":
        hi, gmax = cfg["doubles_h_mean_max"], cfg["doubles_ge95_max"]
        if mean > hi + 1e-9: w.append(f"mean crop height {mean:.3f} above doubles target {hi:.2f}")
        if ge > gmax + 1e-9: w.append(f"{ge:.1%} of frames at >= 95 % height (doubles target <= {gmax:.0%})")
    else:
        lo, hi = cfg["h_mean_target"]
        if not lo - 1e-9 <= mean <= hi + 1e-9: w.append(f"mean crop height {mean:.3f} outside target {lo:.2f}-{hi:.2f}")
        if ge >= cfg["ge95_max"]: w.append(f"{ge:.1%} of frames at >= 95 % height (target < {cfg['ge95_max']:.0%})")
    return {"mean_crop_height": round(mean, 3), "pct_frames_ge95": round(100 * ge, 1), "warnings": w}


def crop_height_target(cfg=CAMERA, fmt=None, sport=None):
    if fmt == "doubles" and sport != "padel":
        return f"doubles: mean <= {cfg['doubles_h_mean_max']}, >= 95 % on <= {cfg['doubles_ge95_max']:.0%} of non-wide frames"
    return f"mean {cfg['h_mean_target'][0]}-{cfg['h_mean_target'][1]}, >= 95 % on < {cfg['ge95_max']:.0%} of frames"


def jerk_qa(jerk_px, cfg=CAMERA):
    """float-path jerk (px/frame^3) vs the calibration baseline: dict(pass=True/False/None, limit, baseline, source)."""
    base, src = cfg.get("jerk_baseline"), cfg.get("jerk_baseline_source")
    if base is None or jerk_px is None:
        return {"pass": None, "baseline": base, "limit": None, "source": src,
                "note": "no calibration baseline (v3 reel jerk_p95) configured"}
    lim = cfg["jerk_ratio"] * base
    return {"pass": bool(jerk_px <= lim), "baseline": base, "limit": float(f"{lim:.6g}"), "source": src}


# ---------------------------------------------------------------------------------------------------------------
# v4.1 patch (local, on top of 303246d): framing roles. In doubles the four players almost never fit a 9:16 crop, so
# widening for the union of all players pins the crop at full height. Only the players who matter are framed: the
# hitter around each verified contact, the receiver (the next contact's hitter, or the receiving side's player nearest
# the predicted landing) around their contact, and the ball. The other partner may leave the frame (broadcast framing).
# Boxes: per frame [(x0, x1, area, y0, y1[, side])] analysis px; side 0 = near, 1 = far (inferred from y1 if absent).

MIN_PERSON_H = 16     # v4.6: analysis px (720p); thinner motion blobs (a static strip at the frame edge) are not players


def _person(b):
    return len(b) >= 5 and b[4] - b[3] >= MIN_PERSON_H


def _side(b, ymid):
    return int(b[5]) if len(b) >= 6 else int((b[3] + b[4]) / 2 < ymid)


def _track(boxes, i0, ref, side, lo, hi, ymid):
    """Follow one player from frame i0 (box ref) over [lo, hi): nearest same-side box centre each frame, held through gaps."""
    out = {}
    for rng in (range(i0, hi), range(i0 - 1, lo - 1, -1)):
        cur = ref
        for i in rng:
            if i < 0 or i >= len(boxes): continue
            cand = [b for b in boxes[i] if len(b) >= 5 and _side(b, ymid) == side]
            cand = [b for b in cand if _person(b)]
            if cand and not (i == i0 and rng.start == i0):      # v4.6: the start frame keeps the chosen box
                c = min(cand, key=lambda b: abs((b[0] + b[1]) / 2 - (cur[0] + cur[1]) / 2) + abs(b[4] - cur[4]))
                if abs((c[0] + c[1]) / 2 - (cur[0] + cur[1]) / 2) < 3 * (cur[4] - cur[3]) + 20: cur = c
            out[i] = cur
    return out


def _nearest_frame_boxes(boxes, i, k=8):
    for d in range(k + 1):
        for j in (i - d, i + d):
            if 0 <= j < len(boxes) and any(len(b) >= 5 for b in boxes[j]): return j
    return None


def _pick_server(boxes, bxy, i, want, ymid, fps, search_s=0.3, toss_s=1.2):
    """v4.6.1 server identity (MD P101 review): the serving side's person boxes at the serve contact (nearest frame within
    search_s that has any), cross-checked with the raw ball detections over the toss (toss_s before the contact to just
    after it): the server is the side box whose centre is nearest the median toss x, and the ball must lie within
    max(1.5 box widths, 0.5 box height) of that box. With no toss detections: the deepest box of the side (ball_ok None).
    -> (box or None, frame, check dict(ok, side, ball_x, ball_ok, frame, box))."""
    n = len(boxes); k = int(round(search_s * fps)); j = None; cand = []
    for d in range(k + 1):
        for q in (i - d, i + d):
            if 0 <= q < n:
                c = [b for b in boxes[q] if _person(b) and _side(b, ymid) == want]
                if c: j, cand = q, c; break
        if cand: break
    q0, q1 = max(0, i - int(round(toss_s * fps))), min(n, i + int(round(0.05 * fps)) + 1)
    bxs = np.asarray(bxy, float)[q0:q1, 0]; bxs = bxs[np.isfinite(bxs)]
    mx = float(np.median(bxs)) if len(bxs) else None
    chk = dict(ok=False, side=("near", "far")[want], ball_x=None if mx is None else round(mx, 1), ball_ok=None, frame=j, box=None)
    if not cand: return None, j, chk
    if mx is None:
        hb = max(cand, key=lambda b: b[4]) if want == 0 else min(cand, key=lambda b: b[3])
    else:
        hb = min(cand, key=lambda b: abs((b[0] + b[1]) / 2 - mx))
        w, h = hb[1] - hb[0], hb[4] - hb[3]
        tol = max(1.5 * w, 0.5 * h)
        chk["ball_ok"] = bool(hb[0] - tol <= mx <= hb[1] + tol)
    chk.update(ok=chk["ball_ok"] is not False, box=[round(float(v), 1) for v in hb[:5]])
    return hb, j, chk


def contact_roles(boxes, ball_xy, contacts, t0, fps, cfg=CAMERA, ymid=None, sides=None, kinds=None):
    """-> list of roles, one per verified contact: dict(t, i, hitter={frame: box}, side, receiver={frame: box} or None,
    recv_t). Hitter = player box nearest the ball at the contact, among the verified side's players when `sides`
    (per contact 'near'/'far'/None) is given; receiver = the predicted next hitter: the next verified contact's hitter
    when it follows within max_flight_s, else the receiving side's player nearest the ball's predicted landing
    (rel_land_s after), whichever of the two partners that is (a poaching net player included)."""
    n = len(boxes); bxy = np.asarray(ball_xy, float); fr = np.arange(n); ok = ~np.isnan(bxy[:, 0])
    bi = np.stack([np.interp(fr, fr[ok], bxy[ok, 0]), np.interp(fr, fr[ok], bxy[ok, 1])], 1) if ok.any() else None
    if ymid is None:
        ys = [(b[3] + b[4]) / 2 for bb in boxes for b in bb if len(b) >= 5]
        ymid = float(np.median(ys)) if ys else 0.0
    pre, post = int(round(cfg["rel_pre_s"] * fps)), int(round(cfg["rel_post_s"] * fps))
    roles = []
    sd_of = dict(zip(contacts, sides)) if sides is not None else {}
    kd_of = dict(zip(contacts, kinds)) if kinds is not None else {}
    for t in sorted(contacts):
        i = int(np.clip(round((t - t0) * fps), 0, n - 1)); j = _nearest_frame_boxes(boxes, i)
        if j is None or bi is None: roles.append(dict(t=t, i=i, hitter={}, side=None, receiver=None, recv_t=None)); continue
        bx_, by_ = bi[i]
        def dist(b):
            h = max(1.0, b[4] - b[3]); dx = max(b[0] - bx_, 0, bx_ - b[1]); dy = max(b[3] - by_, 0, by_ - b[4])
            return np.hypot(dx, dy) / h
        want = {'near': 0, 'far': 1}.get(sd_of.get(t))
        if kd_of.get(t) == 'serve' and want is not None:     # v4.6.1: server identity from side + toss / contact ball
            hb, j2, chk = _pick_server(boxes, bxy, i, want, ymid, fps)
            if chk["ok"]:
                roles.append(dict(t=t, i=i, side=want, hitter=_track(boxes, j2, hb, want, max(0, i - pre - fps), min(n, i + post + fps), ymid),
                                  receiver=None, recv_t=None, server_check=chk)); continue
            srv_chk = chk                                      # not identified: plan the contact as a normal hitter, flag it
        else:
            srv_chk = None
        allb = [b for b in boxes[j] if _person(b)] or [b for b in boxes[j] if len(b) >= 5]
        side_b = [b for b in allb if want is None or _side(b, ymid) == want]
        # v4.6: only boxes within reach of the ball (3 box heights) can be the hitter: a thin blob at the frame edge or a
        # merged blob whose centre falls on the other half must not win just by its side or depth
        near_ball = [b for b in side_b if dist(b) <= 3.0] or [b for b in allb if dist(b) <= 3.0]
        pool = near_ball or side_b or allb
        hb = min(pool, key=dist)
        sd = want if want is not None else _side(hb, ymid)  # v4.6: a verified side wins over a merged blob's centre y
        r_ = dict(t=t, i=i, side=sd, hitter=_track(boxes, j, hb, sd, max(0, i - pre - fps), min(n, i + post + fps), ymid),
                  receiver=None, recv_t=None)
        if srv_chk is not None: r_["server_check"] = srv_chk
        roles.append(r_)
    for k, r in enumerate(roles):
        nxt = roles[k + 1] if k + 1 < len(roles) else None
        if nxt is not None and 0 < nxt['t'] - r['t'] <= cfg["max_flight_s"]:
            r['receiver'], r['recv_t'] = nxt['hitter'], nxt['t']; continue
        if r['side'] is None or bi is None: continue
        il = int(min(n - 1, r['i'] + round(cfg["rel_land_s"] * fps)))
        j = _nearest_frame_boxes(boxes, il)
        cand = [b for b in boxes[j] if len(b) >= 5 and _side(b, ymid) != r['side']] if j is not None else []
        if cand:
            rb = min(cand, key=lambda b: abs((b[0] + b[1]) / 2 - bi[il][0]))
            r['receiver'] = _track(boxes, j, rb, 1 - r['side'], max(0, il - pre - fps), min(n, il + post + fps), ymid)
            r['recv_t'] = t0 + il / fps
    return roles


def relevant_span(n, roles, ball_x, t0, fps, cfg=CAMERA):
    """Per frame (left, right) in analysis px of what must be in frame: the ball, each contact's hitter over
    [t - rel_pre_s, t + rel_post_s], each receiver over the same window around their reception. NaN where only the
    ball matters (the ball is framed by the pan target itself between contacts). Also returns the per-frame mask of 'must keep' player frames."""
    L, R = np.full(n, np.nan), np.full(n, np.nan); keep = np.zeros(n, bool)
    bxa = np.asarray(ball_x, float); okb = ~np.isnan(bxa)
    near_c = np.zeros(n, bool)                     # the ball joins the framed set only right at a contact (+-0.15 s): an
    for r in roles:                                # incoming ball 0.5 s out is still across the court and would force a widen
        near_c[max(0, r['i'] - int(0.15 * fps)):r['i'] + int(0.15 * fps) + 1] = True
    sel = okb & near_c
    L[sel] = bxa[sel]; R[sel] = bxa[sel]
    pre, post = cfg["rel_pre_s"], cfg["rel_post_s"]
    def add(track, tc):
        if not track or tc is None: return
        for i, b in track.items():
            t = t0 + i / fps
            if tc - pre <= t <= tc + post:
                L[i] = b[0] if np.isnan(L[i]) else min(L[i], b[0]); R[i] = b[1] if np.isnan(R[i]) else max(R[i], b[1]); keep[i] = True
    for r in roles:
        add(r['hitter'], r['t']); add(r['receiver'], r['recv_t'])
    # hysteresis: the framed set is held for rel_hold_s (running min / max over a centred window), so a switch of the
    # predicted receiver between partners cannot make the zoom pump
    k = max(1, int(round(cfg["rel_hold_s"] * fps / 2)))
    kp = keep.copy()
    Lh, Rh = L.copy(), R.copy()
    for i in range(n):
        sl = slice(max(0, i - k), min(n, i + k + 1))
        if kp[sl].any():
            keep[i] = True
            Lh[i] = np.nanmin(np.where(kp[sl], L[sl], np.nan)); Rh[i] = np.nanmax(np.where(kp[sl], R[sl], np.nan))
            if not np.isnan(L[i]): Lh[i] = min(Lh[i], L[i]); Rh[i] = max(Rh[i], R[i])
    return Lh, Rh, keep


def calm_zoom(h_target, fps, cfg=CAMERA):
    """Tasteful zoom target: closing in log space (a tightening that would last < calm_close_s (2 s) is skipped, so the
    camera never dips in and straight back out), then a Gaussian ease (calm_ease_s). The zoom_rate cap and the 0.5 s
    no-reversal rule are applied afterwards by zoom_caps, unchanged."""
    lh = np.log(np.asarray(h_target, float)); n = len(lh)
    k = max(1, int(round(cfg["calm_close_s"] * fps / 2)))
    mx = np.array([lh[max(0, i - k):i + k + 1].max() for i in range(n)])
    cl = np.array([mx[max(0, i - k):i + k + 1].min() for i in range(n)])
    cl = np.maximum(cl, lh)
    sg = cfg["calm_ease_s"] * fps; d = int(round(2 * sg))
    dil = np.array([cl[max(0, i - d):i + d + 1].max() for i in range(n)])      # dilate first so the ease never undercuts a need
    return np.exp(np.maximum(_gauss(dil, sg), cl))


# ---------------------------------------------------------------------------------------------------------------
# v4.3 patch: gate planning. Every verified contact is a gate: the crop centre must keep that hitter (and, at their own
# contact, the receiver) rel_margin inside the frame. The plan moves gate to gate with a jerk-limited S-curve
# (smootherstep) that starts right after a contact and arrives on the next hitter early (leading toward where the next
# hitter is, so a pan never frames empty court). The path is the minimum-jerk curve through the gates (an S-curve by
# construction), then the pan caps; where a gate cannot be held under the caps the caller widens just enough, early,
# and re-plans.

def contact_gates(roles, boxes_scale, cw, n, fps, cfg=CAMERA, win=6):
    """-> list of (frame, lo, hi, centre, width, hard) per verified contact hitter (hard) and per predicted receiver
    without a later verified contact (soft: there is no contact to clip, e.g. a winner or a verified no-swing); lo/hi bound the crop centre (source px) so the player's box (union over +-win frames) stays rel_margin
    inside a crop of width cw[frame]."""
    gates = []
    def g(track, i):
        if not track or i is None or i < 0 or i >= n: return None
        bb = [track[j] for j in range(i - win, i + win + 1) if j in track]
        if not bb: return None
        X0 = min(b[0] for b in bb) * boxes_scale; X1 = max(b[1] for b in bb) * boxes_scale
        m = cfg["rel_margin"] * cw[i]
        return (i, X1 - cw[i] / 2 + m, X0 + cw[i] / 2 - m, (X0 + X1) / 2, X1 - X0)
    for k, r in enumerate(roles):
        x = g(r["hitter"], r["i"])
        if x: gates.append(x + (True,))                  # hard: a verified contact
        nxt = roles[k + 1] if k + 1 < len(roles) else None
        if r["receiver"] is not None and r["recv_t"] is not None and (nxt is None or r["receiver"] is not nxt["hitter"]):
            x = g(r["receiver"], int(round(r["i"] + (r["recv_t"] - r["t"]) * fps)))   # predicted reception
            if x: gates.append(x + (False,))             # soft: no verified contact (a winner / let-go / no swing)
    return sorted(gates)


def gate_plan(gates, n, fps, cw, SW, fast=None, cfg=CAMERA, lead_s=0.35, win=6, w_slack=1000.0, w_pref=0.05,
              w_jmax=200.0, w_jerk=1.0, w_vel=0.2, ball_x=None, w_ball=1.0, ball_in=40.0, safety=2.0, w_lead=30.0,
              mono=None, _round=0, w_soft=40.0, cuts=None, w_cut=3.0, keep=None, w_keep=1000.0, lead_by_gate=None, slow=None):
    """Crop-centre path (source px) through the contact gates, solved as one linear programme (scipy HiGHS):
      hard: |v| <= pan_fast (between verified contacts) / pan_max (elsewhere) * cw, |a| <= pan_acc * median(cw),
            crop inside the image;
      soft (heavily weighted slack): the centre stays inside each gate over the +-0.1 s contact window and, for every
            gate after the first, over a lead window of lead_s before it (arrive on the next hitter early instead of
            framing empty court mid-pan); margins are waived at the image border;
      objective: slack, then the peak jerk (a jerk-limited S-curve: trapezoidal acceleration), total |jerk|, travel,
            and a weak pull toward the player's centre at the gates; ball_x (source px, NaN = unknown) adds a soft
            term keeping the detected ball ball_in px inside the crop (weighted far below the gates).
    v4.6: lead_by_gate {gate index: lead s} overrides lead_s (a returner after a serve); slow [(f0, f1, cw/s)] caps the
    pan over frames f0..f1 (the pre-serve phase); mono [(f0, f1, sign)] forbids moves against sign over f0..f1.
    Returns (planned, reached[list of bool], slack px per gate)."""
    import scipy.sparse as sps
    from scipy.optimize import linprog
    cw = np.asarray(cw, float)
    if not gates and not keep: return np.full(n, SW / 2.0), [], []
    rows = []                                          # (frame, lo, hi, centre, gate index)
    prev_i = None
    for gk, (i, glo, ghi, c, w, hard) in enumerate(gates):
        ls_ = (lead_by_gate or {}).get(gk, lead_s)
        f0 = max(0, i - win) if prev_i is None else max(0, min(i - win, max(prev_i + win + int(0.2 * fps), i - int(ls_ * fps))))
        for q in range(f0, min(n, i + win + 1)):
            mq = cfg["rel_margin"] * cw[q]                 # per-frame bounds: the zoom may change inside the window
            lq = c + w / 2 - cw[q] / 2 + mq + safety; hq = c - w / 2 + cw[q] / 2 - mq - safety
            rows.append((q, min(lq, SW - cw[q] / 2), max(hq, cw[q] / 2), c, gk, q >= i - win, None))
        prev_i = i
    for (q, X0, X1, *wk) in (keep or []):              # v4.3: subject keep-in rows (fully in frame, rel_margin inside);
        if 0 <= q < n:                                 # v4.6: optional per-row weight (server, far partner)
            mq = cfg["rel_margin"] * cw[q]
            lq = X1 - cw[q] / 2 + mq + safety; hq = X0 + cw[q] / 2 - mq - safety
            rows.append((q, min(lq, SW - cw[q] / 2), max(hq, cw[q] / 2), (X0 + X1) / 2, -1, False, float(wk[0]) if wk else w_keep))
    m = len(rows)
    fm = np.zeros(n, bool) if fast is None else np.asarray(fast, bool).copy()
    fm[1:-1] |= fm[:-2] & fm[2:]                         # the contact frame itself sits between two flights
    vmax = np.where(fm, cfg["pan_fast"], cfg["pan_max"]) * cw / fps
    for (f0, f1, rate) in (slow or []):
        vmax[max(0, f0):max(0, min(n, f1))] = np.minimum(vmax[max(0, f0):max(0, min(n, f1))], rate * cw[max(0, f0):max(0, min(n, f1))] / fps)
    amax = cfg["pan_acc"] * cw / fps ** 2               # v4.6.1: crop-widths/s^2 of the frame's own crop (was the median)
    nu, nz = n - 3, n - 1
    ox, os_, ow, ou, oz = 0, n, n + m, n + 2 * m, n + 2 * m + nu
    bi = [] if ball_x is None else [q for q in range(n) if np.isfinite(ball_x[q])]
    oJ = oz + nz; ob = oJ + 1; N = ob + len(bi)
    ri, ci, vv, b = [], [], [], []
    def add(cols, vals, rhs):
        r = len(b); ri.extend([r] * len(cols)); ci.extend(cols); vv.extend(vals); b.append(rhs)
    cuts = list(cuts or []); oc = N; N += len(cuts)
    G = len(gates); og = N; N += G                       # one slack per gate for its contact window: an
    for r, (q, lo, hi, c, gk, ct, wk) in enumerate(rows):    # unavoidable miss lands on as few contacts as possible
        sv = og + gk if ct else os_ + r
        add([ox + q, sv], [-1, -1], -lo)                 # x >= lo - s
        add([ox + q, sv], [1, -1], hi)                   # x <= hi + s
        add([ox + q, ow + r], [1, -1], c)                # |x - c| <= w
        add([ox + q, ow + r], [-1, -1], -c)
    for i in range(n - 1):
        add([ox + i + 1, ox + i], [1, -1], vmax[i]); add([ox + i + 1, ox + i], [-1, 1], vmax[i])
        add([ox + i + 1, ox + i, oz + i], [1, -1, -1], 0); add([ox + i + 1, ox + i, oz + i], [-1, 1, -1], 0)
    for i in range(n - 2):
        add([ox + i + 2, ox + i + 1, ox + i], [1, -2, 1], amax[i + 1]); add([ox + i + 2, ox + i + 1, ox + i], [-1, 2, -1], amax[i + 1])
    for i in range(n - 3):
        cols = [ox + i + 3, ox + i + 2, ox + i + 1, ox + i]
        add(cols + [ou + i], [1, -3, 3, -1, -1], 0); add(cols + [ou + i], [-1, 3, -3, 1, -1], 0)
        add(cols + [oJ], [1, -3, 3, -1, -1], 0); add(cols + [oJ], [-1, 3, -3, 1, -1], 0)
    for k_, q in enumerate(bi):
        add([ox + q, ob + k_], [-1, -1], -(ball_x[q] - cw[q] / 2 + ball_in))
        add([ox + q, ob + k_], [1, -1], ball_x[q] + cw[q] / 2 - ball_in)
    for k_, (q, ge, val, *_w) in enumerate(cuts):           # soft: don't slice a non-role player with the crop edge
        add([ox + q, oc + k_], [-1, -1] if ge else [1, -1], -val if ge else val)
    for (f0, f1, sg) in (mono or []):                   # no reversal within hold_s: keep the previous direction (or stop)
        for i in range(max(0, f0), min(n - 1, f1)):
            add([ox + i + 1, ox + i], [-sg, sg], 0.0)
    Aub = sps.csr_matrix((vv, (ri, ci)), shape=(len(b), N))
    cost = np.zeros(N); cost[os_:os_ + m] = w_lead; cost[og:og + G] = [w_slack if g_[5] else w_soft for g_ in gates]; cost[ow:ow + m] = w_pref; cost[ou:ou + nu] = w_jerk
    cost[oz:oz + nz] = w_vel; cost[oJ] = w_jmax * n; cost[ob:ob + len(bi)] = w_ball; cost[oc:oc + len(cuts)] = [cu[3] if len(cu) > 3 else w_cut for cu in cuts]
    for r, row in enumerate(rows):
        if row[6] is not None: cost[os_ + r] = row[6]
    bounds = [(cw[q] / 2, SW - cw[q] / 2) for q in range(n)] + [(0, None)] * (N - n)
    left = _deadline_left(cfg)                           # v4.7: the LP never runs past the clip's planning budget
    res = linprog(cost, A_ub=Aub, b_ub=np.array(b), bounds=bounds, method="highs",
                  options=({"time_limit": max(0.2, left)} if left is not None else None))
    if res.status != 0:
        if left is not None: _deadline_left(cfg)
        if res.status == 1 and left is not None: raise PlanTimeout("gate LP hit the planning budget")
        raise RuntimeError(f"gate LP failed: {res.message}")
    x = res.x[:n]; slack = [float(v) for v in res.x[og:og + G]]
    _, _, quick, segs = cap_report(x, cw, fps, fast, cfg, detail=True)
    if quick and _round < 6:                             # forbid each too-early reversal and re-solve
        return gate_plan(gates, n, fps, cw, SW, fast, cfg, lead_s, win, w_slack, w_pref, w_jmax, w_jerk, w_vel, ball_x,
                         w_ball, ball_in, safety, w_lead, list(mono or []) + segs, _round + 1, w_soft, cuts, w_cut, keep, w_keep,
                         lead_by_gate, slow)
    return x, [s_ < 0.5 for s_ in slack], slack


def cap_report(x, cw, fps, fast=None, cfg=CAMERA, detail=False):
    """-> (max |v|/cap, max |a|/cap, reversals sooner than hold_s after the previous one), |v| > 2 % cw/s."""
    cw = np.asarray(cw, float)
    fm = np.zeros(len(x), bool) if fast is None else np.asarray(fast, bool).copy()
    fm[1:-1] |= fm[:-2] & fm[2:]
    v = np.diff(x); vmax = np.where(fm, cfg["pan_fast"], cfg["pan_max"])[:-1] * cw[:-1] / fps
    a = np.diff(x, 2); amax = cfg["pan_acc"] * np.asarray(cw, float)[1:-1] / fps ** 2
    sgn = np.where(np.abs(v) > 0.02 * cw[:-1] / fps, np.sign(v), 0)
    last, d, quick, segs = None, 0, [], []
    for i, s_ in enumerate(sgn):
        if s_ and s_ != d:
            if d and last is not None and i - last < cfg["hold_s"] * fps: quick.append(i); segs.append((last, last + int(cfg["hold_s"] * fps) + 1, int(d)))
            d, last = s_, i
    out = (float(np.max(np.abs(v) / vmax)) if len(v) else 0.0, float(np.max(np.abs(a) / amax)) if len(a) else 0.0, quick)
    return out + (segs,) if detail else out


def jerk_limited(planned, cw, fps, fast=None, cfg=CAMERA):
    """v4.2: the gate LP already enforces the pan caps and minimises peak jerk, so its path is used as is (the
    303246d limit_1d follower treats every frame as a new destination and its brake-in-time rule lags a moving
    target badly); cap_report() verifies the caps on the final path."""
    return np.asarray(planned, float).copy()


def calm_shape(h, fps, close_s=2.5, sigma_s=0.6, dil=2.67):
    """v4.2 zoom taste pass on top of the requirement curve h (crop-height fraction): (1) a morphological closing of
    log h over close_s removes every tight phase too short to be worth it (no in-out pumping); (2) a dilation by
    2.67 sigma then a Gaussian of sigma_s turns every remaining change into one eased move lasting ~4 sigma (>= 2 s
    at 0.6 s) that is complete before a widening is needed and starts only after it ends; the result never drops
    below the requirement. Peak rate for a 0.75 -> 1.0 move at sigma 0.6 s: ~21 %/s."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
    lh = np.log(np.asarray(h, float)); W = max(1, int(close_s * fps)) | 1
    lc = minimum_filter1d(maximum_filter1d(lh, W, mode="nearest"), W, mode="nearest")
    lc = np.maximum(lc, lh)
    ld = maximum_filter1d(lc, (2 * int(dil * sigma_s * fps)) | 1, mode="nearest")
    lf = np.maximum(gaussian_filter1d(ld, sigma_s * fps, mode="nearest"), lc)
    return np.minimum(np.exp(lf), 1.0)


def edge_cuts(x, cw, boxes_src, SW, pad=6.0, weight=None, force=None, force_mask=None):
    """Players (source-px boxes per frame, (X0, X1)) sliced by a crop edge on the path x -> soft constraints for a
    re-solve: mostly inside -> keep fully in; mostly outside -> keep fully out. Image-border cases are skipped."""
    out = []
    for q, bb in enumerate(boxes_src):
        L, R = x[q] - cw[q] / 2, x[q] + cw[q] / 2
        for (X0, X1) in bb:
            if X1 <= L or X0 >= R or (X0 >= L and X1 <= R) or X1 - X0 < 4: continue
            inside = (min(X1, R) - max(X0, L)) / (X1 - X0)
            if X1 - X0 > 0.6 * cw[q]: continue
            if force is not None and force_mask is not None and force_mask[q]: inside = 1.0 if force == 'in' else 0.0
            if inside >= 0.5:
                if X0 < L: out.append((q, False, X0 + cw[q] / 2 - pad))      # x <= X0 + cw/2 - pad (left edge left of X0)
                else: out.append((q, True, X1 - cw[q] / 2 + pad))           # x >= X1 - cw/2 + pad
            else:
                if X0 < L: out.append((q, True, X1 + cw[q] / 2 + pad))       # left edge right of X1
                else: out.append((q, False, X0 - cw[q] / 2 - pad))          # right edge left of X0
    if weight is not None: out = [cu + (float(weight[cu[0]]),) for cu in out]   # v4.3: per-frame weight (editorial no-cut windows)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Integration (OpenCourtVision, on top of the MD's v4.3 patch): the per-clip orchestration that lived in the MD's
# reel_v43.py, minus its editorial extras (subject tracking, no-cut windows), plus the infeasibility rule.

def _dip_fill(h, fps, cfg, rate):
    """Fill zoom-in dips shallower than dip_fill (7 %) between two wider phases (no breathing), re-capped."""
    lh = np.log(zoom_caps(h, fps, cfg, rate)); n = len(lh); k3 = int(1.5 * fps)
    mx = np.array([lh[max(0, q - k3):q + k3 + 1].max() for q in range(n)])
    cl = np.maximum(np.array([mx[max(0, q - k3):q + k3 + 1].min() for q in range(n)]), lh)
    seg = (cl - lh) > 1e-4; fill = np.zeros(n, bool); q0 = None
    for q in range(n + 1):
        if q < n and seg[q]:
            if q0 is None: q0 = q
        elif q0 is not None:
            if (cl[q0:q] - lh[q0:q]).max() < np.log(1 / (1 - cfg["dip_fill"])): fill[q0:q] = True
            q0 = None
    return zoom_caps(np.exp(np.where(fill, cl, lh)), fps, cfg, rate)


def calm_height(h_target, fps, cfg=CAMERA, rate=None):
    """v4.3 zoom: skip < calm_close_s tightenings, fill < 7 % dips, eased S-curve moves under the zoom_rate cap."""
    h1 = _dip_fill(calm_zoom(h_target, fps, cfg), fps, cfg, rate)
    return zoom_caps(calm_shape(h1, fps, cfg["calm_shape_close_s"], cfg["calm_shape_sigma_s"], cfg["calm_shape_dil"]), fps, cfg, rate)


def gate_win(fps):
    """+-0.1 s contact window in frames (the MD's code hard-coded 6, i.e. 60 fps only)."""
    return max(1, int(round(0.1 * fps)))


def infeasible_contacts(gates, n, fps, SW, SH=2160, cfg=CAMERA, t0=0.0, contacts=()):
    """Contacts the motion limits cannot connect even at full height: consecutive HARD gates whose keep-in intervals
    (at the widest crop, margin waived at the image border) are further apart than pan_cap x crop width x the time
    between their contact windows. A necessary condition for the LP, so it never flags a clip the planner could hold.
    -> list of (t_from, t_to, gap_px, reach_px)."""
    cwf = float(crop_size(cfg["h_wide"], SH)[0]); m = cfg["rel_margin"] * cwf; win = gate_win(fps)
    fm = flight_mask(n, fps, t0, contacts, cfg)
    def iv(c, w):
        lo, hi = c + w / 2 - cwf / 2 + m, c - w / 2 + cwf / 2 - m
        return min(lo, SW - cwf / 2), max(hi, cwf / 2)
    hard = [g for g in gates if g[5]]; out = []
    for g1, g2 in zip(hard, hard[1:]):
        (l1, h1), (l2, h2) = iv(g1[3], g1[4]), iv(g2[3], g2[4])
        gap = max(0.0, l2 - h1, l1 - h2, l1 - h1, l2 - h2)       # also a single player wider than the widest crop
        dt = max(0.0, (g2[0] - g1[0] - 2 * win) / fps)
        v = (cfg["pan_fast"] if fm[min(n - 1, (g1[0] + g2[0]) // 2)] else cfg["pan_max"]) * cwf
        if gap > v * dt + 1.0:
            out.append((round(t0 + g1[0] / fps, 2), round(t0 + g2[0] / fps, 2), round(gap, 1), round(v * dt, 1)))
    return out


def gate_spread(n, roles, ball_x, t0, fps, scale=1.0, cfg=CAMERA):
    """v4.5 spread fix (Engineer, replaces MD's undelivered v4.1 spread diff). Per frame, the width (source px) that has
    to be inside the crop AT THE SAME TIME: each role's own box over its gate window (+-0.1 s around the hitter's contact
    and around the receiver's reception) plus the ball at the contact when it is within racket reach (one box height). The v4.1 relevant span unioned the hitter and
    the receiver over overlapping 0.75 s windows plus 0.5 s hold, i.e. near hitter to far receiver across the court,
    which the calm zoom then held at ~full height for whole rallies (mean crop height 0.93-0.99). The gate planner pans
    from hitter to receiver, so only what one gate holds is a zoom need; a gate the pan cannot hold is widened by
    widen_request(). Hysteresis: each need is held over spread_hold_s (running max, centred), so a short gap or a
    receiver switch between partners cannot make the zoom pump. -> width per frame (0 where nothing is required)."""
    win = gate_win(fps); L, R = np.full(n, np.nan), np.full(n, np.nan)
    bxa = np.asarray(ball_x, float) if ball_x is not None else np.full(n, np.nan)
    def add(track, ic, with_ball):
        if not track or ic is None: return
        for i in range(max(0, ic - win), min(n, ic + win + 1)):
            if i not in track: continue
            b = track[i]; x0, x1 = b[0] * scale, b[1] * scale
            reach = (b[4] - b[3]) * scale if len(b) >= 5 else 0.0     # the ball at a contact is within racket reach;
            if with_ball and np.isfinite(bxa[i]) and x0 - reach <= bxa[i] <= x1 + reach:   # further away = a misdetection
                x0, x1 = min(x0, bxa[i]), max(x1, bxa[i])
            L[i] = x0 if np.isnan(L[i]) else min(L[i], x0); R[i] = x1 if np.isnan(R[i]) else max(R[i], x1)
    for r in roles:
        add(r["hitter"], r["i"], True)
        if r["receiver"] is not None and r["recv_t"] is not None:
            add(r["receiver"], int(round(r["i"] + (r["recv_t"] - r["t"]) * fps)), False)
    w = np.nan_to_num(R - L)
    k = max(0, int(round(cfg.get("spread_hold_s", 0.5) * fps / 2)))
    return np.array([w[max(0, i - k):i + k + 1].max() for i in range(n)]) if k and n else w


def widen_request(gates, cx, cw, h, n, fps, SW, SH=2160, cfg=CAMERA, t0=0.0, soft=None):
    """Gates the planned path misses over their +-0.1 s window -> (req, viol, infeasible, missed): req is the log crop
    height (per frame, -inf = no request) that would hold each missed gate, applied +-0.5 s around it; infeasible lists
    hard gates even the widest crop cannot hold. soft (default cfg widen_soft): also widen for missed soft gates (a
    predicted receiver with no later contact), which the roles check counts like a hitter."""
    soft = cfg.get("widen_soft", False) if soft is None else soft
    win = gate_win(fps); cwf = crop_size(cfg["h_wide"], SH)[0]
    req = np.full(n, -np.inf); viol = 0; infeasible = []; missed = []
    for (gi, lo, hi, c, w, hard) in gates:
        if not hard and not soft: continue
        for q in range(max(0, gi - win), min(n, gi + win + 1)):
            mq = cfg["rel_margin"] * cw[q]
            lq = min(c + w / 2 - cw[q] / 2 + mq, SW - cw[q] / 2 - 1); hq = max(c - w / 2 + cw[q] / 2 - mq, cw[q] / 2 + 1)
            if not (lq - 0.5 <= cx[q] <= hq + 0.5):
                missed.append(round(t0 + gi / fps, 2))
                rw = 2 * (abs(cx[q] - c) + w / 2) / (1 - 2 * cfg["rel_margin"]) * 1.02
                if hard and rw > cwf and h[q] >= cfg["h_wide"] - 1e-6: infeasible.append(round(t0 + gi / fps, 2))
                rh = min(cfg["h_wide"], rw * 16 / 9 / SH)
                st = cfg.get("widen_step")                  # v4.5: step toward rh (the LP re-pans with the extra width,
                if st: rh = min(rh, h[q] * (1 + st))         # so the width at the current miss overestimates the need)
                k = int(round(0.5 * fps)); lo_ = max(0, gi - k); req[lo_:min(n, gi + k + 1)] = np.maximum(req[lo_:min(n, gi + k + 1)], np.log(rh)); viol += 1
                break
    return req, viol, infeasible, missed


def _ymid(boxes):
    ys = [(b[3] + b[4]) / 2 for bb in boxes for b in bb if len(b) >= 5]
    return float(np.median(ys)) if ys else 0.0


def partner_stats(boxes, x, cw, scale, wide=False, SW=3840):
    """Far-partner / bystander stats on the rendered crop (MD v4 stats, now in the package). Per frame bools:
    far_out = a far-side player (box side 1, else inferred from the box centre y like contact_roles) not fully inside the crop; any_out = any player not fully inside;
    sliced = a player cut by a crop edge (not at the image border). Wide frames count as all in."""
    n = len(boxes)
    ymid = _ymid(boxes)                            # v4.6: blobs without a side field (package motion boxes) by box centre y
    far_out = np.array([any(len(b) >= 5 and _side(b, ymid) == 1 and (b[0] * scale < x[i] or b[1] * scale > x[i] + cw[i]) for b in bb) for i, bb in enumerate(boxes)], bool)
    any_out = np.array([any(len(b) >= 2 and (b[0] * scale < x[i] or b[1] * scale > x[i] + cw[i]) for b in bb) for i, bb in enumerate(boxes)], bool)
    sliced = np.array([any(len(b) >= 2 and b[1] - b[0] >= 2 and ((x[i] > 1 and b[0] * scale < x[i] < b[1] * scale) or
                                                              (x[i] + cw[i] < SW - 2 and b[0] * scale < x[i] + cw[i] < b[1] * scale)) for b in bb)
                       for i, bb in enumerate(boxes)], bool)
    if wide: far_out[:] = False; any_out[:] = False; sliced[:] = False
    return dict(far_out=far_out.reshape(n), any_out=any_out.reshape(n), sliced=sliced.reshape(n))


def serve_role(roles, t0, cfg=CAMERA, kinds=None):
    """v4.6: the clip's serve -> its role (the first verified contact) when the clip starts >= preserve_min_s before it
    and it is not labelled as another stroke; None otherwise (a clip cut into a rally has no pre-serve phase)."""
    if not cfg.get("preserve", True) or not roles or not roles[0]["hitter"]: return None
    if kinds is not None and len(kinds) and kinds[0] not in (None, "serve"): return None
    if not roles[0].get("server_check", {"ok": True})["ok"]: return None     # v4.6.1: never plan on an unverified server
    return roles[0] if roles[0]["t"] - t0 >= cfg.get("preserve_min_s", 1.0) else None


def server_track(boxes, role, ymid=None):
    """The server from the clip's first frame to the serve contact: the serve hitter's box followed backwards from the
    contact (same side, box continuity; _track). -> {frame: box} for frames <= the serve frame."""
    if role is None or not role["hitter"]: return {}
    ymid = _ymid(boxes) if ymid is None else ymid
    k0 = min(role["hitter"], key=lambda k: abs(k - role["i"]))
    tr = _track(boxes, k0, role["hitter"][k0], role["side"], 0, k0 + 1, ymid)
    return {i: b for i, b in tr.items() if i <= role["i"]}


def far_keep_rows(boxes, cw, scale, cfg=CAMERA, ymid=None, safety=2.0, gates=(), fps=60):
    """v4.6 far-partner framing: per frame, one soft keep-in row (q, X0, X1, w_far) per far-side player whose box fits
    the crop with the rel_margin (a far pair wider than the crop keeps whoever the LP can hold; no widening for it).
    No rows within far_guard_s of a hard gate, so the summed far-partner pull never competes with a hitter."""
    ymid = _ymid(boxes) if ymid is None else ymid; rows = []
    guard = np.zeros(len(boxes), bool); k = int(round(cfg.get("far_guard_s", 0.6) * fps))
    for g in gates:
        if g[5]: guard[max(0, g[0] - k):g[0] + k + 1] = True
    for q, bb in enumerate(boxes):
        if guard[q]: continue
        for b in bb:
            if not _person(b) or _side(b, ymid) != 1: continue
            X0, X1 = b[0] * scale, b[1] * scale
            if X1 - X0 <= (1 - 2 * cfg["rel_margin"]) * cw[q] - 2 * safety: rows.append((q, X0, X1, cfg["w_far"]))
    return rows


def source_edge_check(roles, scale, SW, fps, t0, cfg=CAMERA):
    """v4.6 (MD reel v4.5, P70): a role player (hitter around their contact, receiver around their reception, padded by
    source_edge_pad_s) whose box touches the left/right edge of the SOURCE frame cannot be framed by any crop or wide
    view. -> sorted list of (t_from, t_to) source-time intervals (empty = pass)."""
    AW = SW / scale; e = cfg.get("source_edge_px", 2.0); pad = cfg.get("source_edge_pad_s", 0.5)
    hit = set(); last_t = max((r["t"] for r in roles), default=None)
    def scan(track, tc, post):
        if not track or tc is None: return
        for i, b in track.items():
            t = t0 + i / fps
            if tc - cfg["rel_pre_s"] - pad <= t <= tc + cfg["rel_post_s"] + post and (b[0] <= e or b[1] >= AW - e): hit.add(i)
    for r in roles:                     # the hitter is still in play after the contact while the rally goes on (a running
        scan(r["hitter"], r["t"], pad if r["t"] < last_t else 0.0)   # get); after the last contact only rel_post_s
        scan(r["receiver"], r["recv_t"], 0.0)
    out, run = [], []
    def close():
        if len(run) >= cfg.get("source_edge_min_frames", 6): out.append((round(t0 + run[0] / fps, 2), round(t0 + run[-1] / fps, 2)))
    for i in sorted(hit):
        if run and i - run[-1] > int(0.2 * fps): close(); run = []
        run.append(i)
    if run: close()
    return out


def preserve_qa(plan, x, y, cw, ch, scale, fps, t0, wide=False, cfg=CAMERA):
    """v4.6 automated checks for clip start -> serve contact (MD reel v4.5 rule): pan reversals on the float centre path
    (|v| > 2 % cw/s, as cap_report), swing (px), opening frame contains the server (x and y), server fully in the crop
    on every pre-serve frame. -> dict, or None when the clip has no pre-serve phase."""
    i_s, st = plan.get("serve_i"), plan.get("server") or {}
    if i_s is None: return None
    chk = plan.get("server_check")                       # the planner's own server identity (no separate path)
    ident = bool(st) and (chk is None or chk.get("ok", False))
    cx = np.asarray(plan["cx"], float)[:i_s + 1]; cwa = np.asarray(cw, float)
    v = np.diff(cx); sgn = np.where(np.abs(v) > 0.02 * cwa[:len(v)] / fps, np.sign(v), 0)
    nz = sgn[sgn != 0]; rev = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    def inside(i):
        if wide: return True
        b = st.get(i)
        if b is None: return None
        X0, X1, Y0, Y1 = b[0] * scale, b[1] * scale, b[3] * scale, b[4] * scale
        return bool(X0 >= x[i] - 1 and X1 <= x[i] + cw[i] + 1 and Y0 >= y[i] - 2 and Y1 <= y[i] + ch[i] + 2)
    ins = [inside(i) for i in range(i_s + 1)]
    known = [v_ for v_ in ins if v_ is not None]
    out_fr = [i for i, v_ in enumerate(ins) if v_ is False]
    first = next((v_ for v_ in ins[:max(1, int(round(0.1 * fps)))] if v_ is not None), None)
    if not ident: first = None                           # an unidentified server cannot pass the opening / in-crop checks
    return dict(serve_t=round(t0 + i_s / fps, 2), pre_s=round(i_s / fps, 2), reversals=rev, server_identified=ident,
                server_check=chk, server_coverage=round(len(known) / max(1, i_s + 1), 3),
                swing_px=round(float(cx.max() - cx.min()), 1) if len(cx) else 0.0, opening_has_server=first,
                server_out_frames=len(out_fr), server_known_frames=len(known),
                server_out_span=[round(t0 + out_fr[0] / fps, 2), round(t0 + out_fr[-1] / fps, 2)] if out_fr else None)



def plan_clip(boxes, ball_xy, ball_x, contacts, t0, fps, SW=3840, SH=2160, cfg=CAMERA, scale=3.0, fallback=None,
              need_v=None, big_shots=(), punch_in=False, sides=None, kinds=None, cut_passes=4, max_iter=12, w_ball=None):
    """One-shot horizontal pan plan for a clip (replaces follow_pan / limit_1d when there are verified contacts):
    framing roles -> zoom target on the relevant span -> calm zoom -> contact gates -> gate LP (+ re-solves so
    bystanders are not sliced by the crop edge) -> widen just enough, early, where a hard gate is missed -> repeat.
    boxes: per frame [(x0, x1, area, y0, y1[, side])] analysis px; ball_xy analysis px; ball_x source px (NaN = none).
    -> dict(h, cw, ch, cx, fast, roles, gates, reached, caps=(v/cap, a/cap, quick), iters, infeasible[...], feasible)."""
    ball_x_in, need_v_in = ball_x, need_v
    n = len(boxes); C = sorted(contacts); win = gate_win(fps)
    roles = contact_roles(boxes, ball_xy, C, t0, fps, cfg, sides=sides, kinds=kinds)
    ymid = _ymid(boxes)
    srole = serve_role(roles, t0, cfg, kinds)                         # v4.6 pre-serve phase: clip start -> serve contact
    i_s = srole["i"] if srole is not None else None
    strk = server_track(boxes, srole, ymid) if srole is not None else {}
    srv_check = roles[0].get("server_check") if roles else None
    serve_i_qa = i_s if i_s is not None else (roles[0]["i"] if srv_check is not None and roles[0]["t"] - t0 >= cfg.get("preserve_min_s", 1.0) else None)
    slow, mono_pre = [], []
    if i_s is not None:
        bx_pre = np.asarray(ball_x, float).copy()                      # pre-serve: only a ball within the server's reach
        for q in range(max(0, i_s - win)):                              # (bounce, toss) stays a framing target; others
            b_ = strk.get(q)                                            # (stray balls, misdetections) are ignored
            if b_ is None or not np.isfinite(bx_pre[q]): bx_pre[q] = np.nan; continue
            rch = (b_[4] - b_[3]) * scale
            if not (b_[0] * scale - rch <= bx_pre[q] <= b_[1] * scale + rch): bx_pre[q] = np.nan
        ball_x = bx_pre
        slow = [(0, i_s - win, cfg["preserve_pan_max"])]
        ks = sorted(strk)
        if ks:
            c0 = (strk[ks[0]][0] + strk[ks[0]][1]) / 2 * scale; c1 = (strk[ks[-1]][0] + strk[ks[-1]][1]) / 2 * scale
            dir_ = 1 if c1 >= c0 else -1                                  # one direction (or still) up to the serve
            k_s = next((k for k, r_ in enumerate(roles) if r_ is srole), None)    # v4.6.1: drift toward the returner (P101:
            nx = roles[k_s + 1] if k_s is not None and k_s + 1 < len(roles) else None   # far server right, near returner
            hb_n = nx["hitter"].get(nx["i"]) if nx is not None and nx["hitter"] else None   # left, 1.1 s later: entering the
            if hb_n is not None and nx["t"] - srole["t"] <= cfg["max_flight_s"]:          # serve at rest cannot reach him)
                cn = (hb_n[0] + hb_n[1]) / 2 * scale
                if abs(c1 - c0) < 0.1 * float(np.median(crop_size(cfg["h_default"], SH)[0])): dir_ = 1 if cn >= c1 else -1
            mono_pre = [(0, i_s, dir_)]
    if cfg.get("spread_mode", "relevant") == "gate":                 # v4.5 spread fix: per-gate need, held (hysteresis)
        sp = gate_spread(n, roles, ball_x, t0, fps, scale, cfg)
    else:                                                             # v4.1: union of the relevant roles
        Lr, Rr, _ = relevant_span(n, roles, np.asarray(ball_xy, float)[:, 0], t0, fps, cfg)
        sp = np.nan_to_num((Rr - Lr) * scale)
    fast = flight_mask(n, fps, t0, C, cfg)
    zr = zoom_rates(n, fps, t0, big_shots, punch_in, cfg)
    dlog = np.log1p(cfg["zoom_rate"]) / fps
    need_v = np.zeros(n) if need_v is None else np.asarray(need_v, float)
    fb = np.zeros(n, bool) if (fallback is None or not cfg.get("fallback_wide_with_gates", True)) else fallback
    w_ball = cfg.get("w_ball", 1.0) if w_ball is None else w_ball
    bsrc = [[(b[0] * scale, b[1] * scale) for b in bb] for bb in boxes]
    extra = np.zeros(n); infeasible = []; lp_error = None
    for it in range(max_iter):
        _deadline_left(cfg)                                           # v4.7: never hang (raises PlanTimeout)
        ht = zoom_target(n, fps, t0, fb, sp, big_shots, cfg, SH, np.maximum(need_v, extra), punch_in)
        must = extra > cfg["h_fast_ceiling"] + 1e-6
        ht = np.where(fast & ~must, np.minimum(ht, cfg["h_fast_ceiling"]), ht)
        h = calm_height(ht, fps, cfg, zr)
        cw, ch = crop_size(h, SH)
        gates = contact_gates(roles, scale, cw, n, fps, cfg, win=win)
        keep = [(q, b[0] * scale, b[1] * scale, cfg["w_server"]) for q, b in strk.items()]
        if cfg.get("far_keep"):                                       # the server owns the pre-serve phase
            keep += [r_ for r_ in far_keep_rows(boxes, cw, scale, cfg, ymid, gates=gates, fps=fps) if i_s is None or r_[0] > i_s]
        lead = {}
        if i_s is not None:                                           # the gate after the serve: lead to the returner early
            gi = next((k for k, g_ in enumerate(gates) if g_[5] and abs(g_[0] - i_s) <= win), None)
            if gi is not None and gi + 1 < len(gates): lead[gi + 1] = cfg["serve_lead_s"]
        kw = dict(win=win, ball_x=ball_x, w_ball=w_ball, keep=keep or None, lead_by_gate=lead, slow=slow, mono=mono_pre or None)
        try:
            x, reached, _ = gate_plan(gates, n, fps, cw, SW, fast, cfg, **kw)
            cuts = []
            for _ in range(cut_passes):
                new = [cu for cu in edge_cuts(x, cw, bsrc, SW) if (cu[0], cu[1]) not in {(c_[0], c_[1]) for c_ in cuts}]
                if not new: break
                cuts += new
                x, reached, _ = gate_plan(gates, n, fps, cw, SW, fast, cfg, cuts=cuts, **kw)
        except PlanTimeout:
            raise
        except RuntimeError as e:                             # LP failure: treat the clip as unplannable
            lp_error = str(e); x = np.full(n, SW / 2.0); reached = [False] * len(gates)
            break
        cx = np.clip(x, cw / 2, SW - cw / 2)
        req, viol, infeasible, missed = widen_request(gates, cx, cw, h, n, fps, SW, SH, cfg, t0)
        if not viol or (infeasible and len(infeasible) == viol): break
        for q in range(n - 2, -1, -1): req[q] = max(req[q], req[q + 1] - dlog * 0.8)   # start early enough for the zoom cap
        extra = np.maximum(extra, np.exp(req))
    pre = infeasible_contacts(gates, n, fps, SW, SH, cfg, t0, C)
    caps = cap_report(cx, cw, fps, fast, cfg)
    bad = sorted(set(infeasible)) or ([p[1] for p in pre]) or ([] if lp_error is None else ["lp"]) or sorted(set(missed))
    # ^ last: a hard gate (hitter at a verified contact) still outside the crop after max_iter widenings is a framing
    #   risk too, even when the widest crop could in principle hold it
    if cfg.get("far_keep") and (bad or not all(r_ for r_, g_ in zip(reached, gates) if g_[5])):   # gates beat the far partner
        out = plan_clip(boxes, ball_xy, ball_x_in, contacts, t0, fps, SW, SH, dict(cfg, far_keep=False), scale, fallback, need_v_in,
                        big_shots, punch_in, sides, kinds, cut_passes, max_iter, w_ball)
        out["far_keep_dropped"] = True
        return out
    if bad and cfg.get("spread_fallback", True) and cfg.get("spread_mode") == "gate":
        # v4.6: the v4.5 tight (per-gate) zoom could not hold every role within max_iter widenings (merged motion blobs,
        # back-to-back contacts): re-plan this clip with the v4.1 relevant-span zoom, widened in one step, and keep it
        # when it holds more roles (the 0c59fa6 behaviour, which held all of them on the MD set)
        alt = plan_clip(boxes, ball_xy, ball_x_in, contacts, t0, fps, SW, SH,
                        dict(cfg, spread_mode="relevant", widen_step=None, spread_fallback=False), scale, fallback, need_v_in,
                        big_shots, punch_in, sides, kinds, cut_passes, max_iter, w_ball)
        if len(alt["infeasible"]) < len(bad):
            alt["spread_fallback"] = True
            return alt
    edge = source_edge_check(roles, scale, SW, fps, t0, cfg)          # v4.6: a role player outside the 4K source frame
    return dict(h=h, cw=cw, ch=ch, cx=cx, fast=fast, roles=roles, gates=gates, reached=reached, caps=caps, iters=it + 1,
                infeasible=bad, infeasible_pairs=pre, lp_error=lp_error, feasible=not bad, source_edge=edge,
                serve_i=serve_i_qa, server=strk, server_check=srv_check)


def _roles_and_server(boxes, ball_xy, contacts, t0, fps, cfg, sides, kinds):
    roles = contact_roles(boxes, ball_xy, sorted(contacts), t0, fps, cfg, sides=sides, kinds=kinds)
    ymid = _ymid(boxes)
    srole = serve_role(roles, t0, cfg, kinds)
    strk = server_track(boxes, srole, ymid) if srole is not None else {}
    chk = roles[0].get("server_check") if roles else None
    i_s = srole["i"] if srole is not None else (roles[0]["i"] if chk is not None and roles[0]["t"] - t0 >= cfg.get("preserve_min_s", 1.0) else None)
    return roles, strk, chk, i_s


def _box_at(track, i):
    if not track: return None
    k = min(track, key=lambda q: abs(q - i))
    return track[k] if abs(k - i) <= 12 else None


def safe_plan(boxes, ball_xy, contacts, t0, fps, SW=3840, SH=2160, cfg=CAMERA, scale=3.0, sides=None, kinds=None,
              why="fallback"):
    """v4.7 safe 9:16 plan (no LP, bounded time): full-height crop, so the framing can only gain width, never clip by
    zooming. Keys: the server from the first frame to the serve; at each contact the hitter and the receiver (their
    union when it fits, else the hitter kept fully inside with the crop slid toward the receiver), held safe_hold_s;
    eased between keys, then the pan / acceleration caps, a low-pass for jerk and the caps again; clamped to the source
    frame. -> the plan_clip dict shape (feasible = no hard key needed more than the crop width)."""
    n = len(boxes)
    roles, strk, chk, i_s = _roles_and_server(boxes, ball_xy, contacts, t0, fps, cfg, sides, kinds)
    h = np.ones(n); cw, ch = crop_size(h, SH); c0 = float(cw[0]); m = cfg["rel_margin"] * c0
    keys, over = [], []
    def key(i, lo, hi):                                   # [lo, hi] = must-see span (source px) at frame i
        if hi - lo <= c0 - 2 * m: keys.append((i, (lo + hi) / 2)); return True
        keys.append((i, None)); return False
    if strk:
        ks = sorted(strk); b0 = strk[ks[0]]
        keys.append((0, (b0[0] + b0[1]) / 2 * scale))
        if i_s is not None:
            bs = _box_at(strk, i_s) or b0; keys.append((i_s, (bs[0] + bs[1]) / 2 * scale))
    for r in roles:
        hb = _box_at(r["hitter"], r["i"])
        if hb is None: continue
        lo, hi = hb[0] * scale, hb[1] * scale
        if r.get("receiver") is not None and r.get("recv_t") is not None:
            ir = int(round(r["i"] + (r["recv_t"] - r["t"]) * fps)); rb = _box_at(r["receiver"], ir)
            if rb is not None:
                L, R = min(lo, rb[0] * scale), max(hi, rb[1] * scale)
                if not key(r["i"], L, R):                  # too wide: hitter inside, slid toward the receiver
                    keys.pop(); toward = 1 if rb[0] * scale > lo else -1
                    c = (lo + c0 / 2 - m) if toward > 0 else (hi - c0 / 2 + m)
                    keys.append((r["i"], c)); over.append(round(t0 + r["i"] / fps, 2))
                    keys.append((min(n - 1, ir), (rb[0] + rb[1]) / 2 * scale))
                continue
        keys.append((r["i"], (lo + hi) / 2))
    keys = sorted([(i, c) for i, c in keys if c is not None and 0 <= i < n])
    if not keys:
        tgt = np.full(n, SW / 2.0)
    else:
        hold = int(round(cfg["safe_hold_s"] * fps / 2))
        xs, ys = [], []
        for i, c in keys:
            for q in (max(0, i - hold), min(n - 1, i + hold)): xs.append(q); ys.append(c)
        order = np.argsort(xs, kind="stable"); xs = np.asarray(xs)[order]; ys = np.asarray(ys)[order]
        tgt = np.interp(np.arange(n), xs, ys)
    fast = flight_mask(n, fps, t0, sorted(contacts), cfg)
    x = pan_caps(_gauss(tgt, 0.25 * fps), cw, fps, None, cfg, fast)
    x = pan_caps(_gauss(x, 0.35 * fps), cw, fps, None, cfg, fast)
    cx = np.clip(x, cw / 2, SW - cw / 2)
    caps = cap_report(cx, cw, fps, fast, cfg)
    edge = source_edge_check(roles, scale, SW, fps, t0, cfg)
    return dict(h=h, cw=cw, ch=ch, cx=cx, fast=fast, roles=roles, gates=[], reached=[], caps=caps, iters=0,
                infeasible=over, infeasible_pairs=[], lp_error=None, feasible=not over, source_edge=edge,
                serve_i=i_s, server=strk, server_check=chk, planner="safe_plan", fallback=why)


def locked_wide_plan(boxes, ball_xy, contacts, t0, fps, SW=3840, SH=2160, cfg=CAMERA, scale=3.0, sides=None, kinds=None):
    """v4.7 broadcast grammar for 16:9 (Antony, 19 Oct: Wimbledon / US Open): a steady, locked wide shot of the whole
    source frame during the rally; no pans, no zoom, no planner (so no timeout, no jerk, no reversal, nobody clipped by
    the crop). Roles and the server are still computed so the same QA runs on what is shown."""
    n = len(boxes)
    roles, strk, chk, i_s = _roles_and_server(boxes, ball_xy, contacts, t0, fps, cfg, sides, kinds)
    cw, ch = np.full(n, SW), np.full(n, SH)
    cx = np.full(n, SW / 2.0)
    return dict(h=np.ones(n), cw=cw, ch=ch, cx=cx, fast=np.zeros(n, bool), roles=roles, gates=[], reached=[],
                caps=(0.0, 0.0, []), iters=0, infeasible=[], infeasible_pairs=[], lp_error=None, feasible=True,
                source_edge=source_edge_check(roles, scale, SW, fps, t0, cfg), serve_i=i_s, server=strk,
                server_check=chk, planner="locked_wide")


def rts_smooth(z, fps, q=2.0e5, r=400.0):
    """Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother over a 1-D series with gaps (NaN = no
    measurement): the robust target for l1_plan (gap filling included). q: process noise (px/s^2)^2, r: measurement
    noise px^2. -> smoothed positions (n,), the first/last measurement held outside the measured span."""
    z = np.asarray(z, float); n = len(z); ok = np.isfinite(z)
    if not ok.any(): return np.full(n, np.nan)
    dt = 1.0 / fps
    F = np.array([[1, dt], [0, 1]]); Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]); H = np.array([[1.0, 0.0]])
    xs = np.zeros((n, 2)); Ps = np.zeros((n, 2, 2)); xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    x = np.array([z[ok][0], 0.0]); P = np.diag([r, 1e4])
    for k in range(n):
        if k: x = F @ x; P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if ok[k]:
            S_ = H @ P @ H.T + r; K = P @ H.T / S_
            x = x + (K * (z[k] - H @ x)).ravel(); P = (np.eye(2) - K @ H) @ P
        xs[k], Ps[k] = x, P
    for k in range(n - 2, -1, -1):
        C_ = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C_ @ (xs[k + 1] - xp[k + 1]); Ps[k] = Ps[k] + C_ @ (Ps[k + 1] - Pp[k + 1]) @ C_.T
    return xs[:, 0]


def frame_constraints(roles, strk, i_s, n, fps, scale, SW, cw, cfg=CAMERA):
    """Per-frame hard framing intervals for the crop centre (source px): the hitter at each contact and the receiver
    at each reception (+-gate_win, rel_margin inside, exactly the role_clip_qa test), the server on every pre-serve
    frame. -> list of (frame, lo, hi, kind) before intersection."""
    win = gate_win(fps); m = cfg["rel_margin"] * cw + 2.0; out = []          # +2 px: whole-pixel rendering
    def add(track, ic, kind, margin):
        if not track: return
        for i in range(ic - win, ic + win + 1):
            if 0 <= i < n and i in track:
                b = track[i]; X0, X1 = max(0.0, b[0] * scale), min(float(SW), b[1] * scale)
                out.append((i, X1 - cw / 2 + margin, X0 + cw / 2 - margin, kind))
    for r in roles:
        add(r["hitter"], r["i"], "hitter", m)
        if r.get("receiver") is not None and r.get("recv_t") is not None:
            add(r["receiver"], int(round(r["i"] + (r["recv_t"] - r["t"]) * fps)), "receiver", m)
    if i_s is not None:
        for i, b in strk.items():
            if 0 <= i <= i_s:
                out.append((i, b[1] * scale - cw / 2 + 1, b[0] * scale + cw / 2 - 1, "server"))
    return out


# --- v4.9: shot context (Antony 19 Oct: "nobody can see where the ball went") -------------------------------------
# Each exchange shot gets a CONTEXT window just before its bounce: the receiver and the landing area must be in the
# crop then, so the viewer sees the target space the hitter chose. When the pan from the hitter (at contact) to that
# context is beyond the camera caps, long_shot_mode decides: "cut" = a hard cut to the receiving side once the ball
# leaves the hitter's frame (default, Antony); "fast_pan" = the pan caps are lifted while the ball is in flight only
# (experimental, for Media Director's comparison); "off" = v4.8 behaviour (no context gate, no cut).
CAMERA.update(
    long_shot_mode="cut",
    ctx_pre_s=0.30,            # context window: the last 0.30 s before the bounce
    ctx_min_flight_s=0.50,     # shorter exchanges (volleys) are covered by the receiver gate itself
    bounce_frac=0.80,          # no bounce event and no ball kink: bounce estimated at 80 % of contact -> reception
    land_pad_frac=0.08,
    land_reach_frac=0.60,      # a landing x more than 60 % of the crop width from the receiver is treated as a false detection        # landing area = ball x at the bounce +- 8 % of the crop width
    ctx_w=300.0,               # per-px penalty for a context gate (below the hitter / receiver gates)
    min_shot_s=0.50,           # cut mode: no two cuts (or clip start / cut) closer than this
    long_shot_ratio=0.90,      # a shot is "long" when its pan needs > 90 % of what the caps allow
    pan_fast_flight=2.4,       # fast_pan: in-flight pan cap, crop-widths / s (pan_fast 0.8 elsewhere)
    pan_acc_flight=8.0,        # fast_pan: in-flight acceleration cap, crop-widths / s^2
)


def ball_kink(ball_xy, i0, i1, lo=0.35, hi=0.95, min_px=4.0):
    """Bounce frame from the ball's image track: the ball is lowest relative to its contact -> reception chord at the
    bounce (height 0), so the frame with the largest positive y residual against the straight chord, in the middle
    of the flight. -> frame or None (too few detections / no kink)."""
    if ball_xy is None or i1 - i0 < 6: return None
    seg = np.asarray(ball_xy, float)[i0:i1 + 1, 1]; ok = np.isfinite(seg)
    if ok.sum() < 6: return None
    k = np.arange(len(seg)); A = np.vstack([k[ok], np.ones(ok.sum())]).T
    m_, c_ = np.linalg.lstsq(A, seg[ok], rcond=None)[0]
    res = np.where(ok, seg - (m_ * k + c_), -np.inf)
    a, b = int(lo * len(seg)), max(int(lo * len(seg)) + 1, int(hi * len(seg)))
    j = a + int(np.argmax(res[a:b]))
    return i0 + j if np.isfinite(res[j]) and res[j] >= min_px else None


def shot_context(roles, ball_xs, n, fps, t0, bounces=(), cfg=CAMERA, ball_xy=None, scale=3.0):
    """-> per exchange shot dict(i contact frame, ir reception frame, ib bounce frame, c0 context start frame,
    tb, land_x source px or None, receiver track, bounce 'event' | 'ball kink' | 'estimate')."""
    win = gate_win(fps); out = []; bs = sorted(bounces or ())
    bx = np.asarray(ball_xs, float) if ball_xs is not None else np.full(n, np.nan)
    for r in roles:
        if r.get("receiver") is None or r.get("recv_t") is None or not r.get("hitter"): continue
        tc, tr = r["t"], r["recv_t"]
        if tr - tc < cfg["ctx_min_flight_s"]: continue
        i = r["i"]; ir = int(round(i + (tr - tc) * fps))
        tb = next((b for b in bs if tc + 0.15 < b <= tr), None); src = "event"
        if tb is None:
            kb = ball_kink(ball_xy, i, min(n - 1, ir))
            tb, src = (t0 + kb / fps, "ball kink") if kb is not None else (tc + cfg["bounce_frac"] * (tr - tc), "estimate")
        ib = int(round((tb - t0) * fps))
        c0 = max(i + win + 1, ib - int(round(cfg["ctx_pre_s"] * fps)))
        if not (0 <= c0 <= ib < n): continue
        lx = None                                  # landing x: the raw ball detection nearest the bounce (+-3 frames)
        if ball_xy is not None:
            bxy_ = np.asarray(ball_xy, float)
            for d in (0, -1, 1, -2, 2, -3, 3):
                q = ib + d
                if 0 <= q < len(bxy_) and np.isfinite(bxy_[q, 0]): lx = float(bxy_[q, 0] * scale); break
        if lx is None and np.isfinite(bx[ib]): lx = float(bx[ib])
        rb = _box_at(r["receiver"], ib)              # plausibility: the bounce is within reach of the receiver; a landing
        if lx is not None and rb is not None and abs(lx - (rb[0] + rb[1]) / 2 * scale) > cfg["land_reach_frac"] * crop_size(1.0)[0]:
            lx, src = None, src + " (landing rejected: false detection)"   # point far from them is a false detection
        if src == "estimate": lx = None          # mid-flight x is not a landing point: then the receiver alone is the target
        out.append(dict(i=i, ir=ir, ib=ib, c0=c0, tb=round(tb, 2), land_x=lx, receiver=r["receiver"], bounce=src))
    return out


def context_constraints(shots, scale, SW, cw, cfg=CAMERA):
    """Context gates: receiver box (no margin, +2 px) and the landing area inside the crop on every frame of the context
    window. -> list of (frame, lo, hi, 'context')."""
    out = []; lp = cfg["land_pad_frac"] * cw
    for s in shots:
        m = 1.0 / max(1, s["ib"] + 1 - s["c0"])            # weight per frame: the whole window weighs ctx_w, so a context
        for i in range(s["c0"], s["ib"] + 1):              # never outweighs a single hitter / receiver gate frame
            lo, hi = -1e9, 1e9
            b = _box_at(s["receiver"], i)
            if b is not None:
                X0, X1 = max(0.0, b[0] * scale), min(float(SW), b[1] * scale)
                lo, hi = max(lo, X1 - cw / 2 + 2), min(hi, X0 + cw / 2 - 2)
            if s["land_x"] is not None:
                L0, L1 = max(0.0, s["land_x"] - lp), min(float(SW), s["land_x"] + lp)
                lo, hi = max(lo, L1 - cw / 2), min(hi, L0 + cw / 2)
            if lo > -1e9 or hi < 1e9: out.append((i, lo, hi, "context", m))
    return out


def _interval(cons, frame, kinds):
    lo, hi = -1e9, 1e9
    for c_ in cons:
        i, a, b, k = c_[:4]
        if i == frame and k in kinds: lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if lo > -1e9 or hi < 1e9 else None


def _reach(F, v, a):
    """Max travel in F frames from rest under |v| <= v, |a| <= a (px / frame, px / frame^2)."""
    if F <= 0: return 0.0
    return a * (F / 2) ** 2 if F <= 2 * v / a else v * (F - v / a)


def long_shots(shots, cons, ball_xs, fps, cw, n, cfg=CAMERA):
    """Mark each context shot long / not, and choose its cut frame: the first frame after the hitter's contact window
    where the ball leaves the hitter's crop (so the flight is seen leaving), never later than the context start."""
    win = gate_win(fps); bx = np.asarray(ball_xs, float) if ball_xs is not None else np.full(n, np.nan)
    v = cfg["pan_fast"] * cw / fps; a = cfg["pan_acc"] * cw / fps ** 2
    for s in shots:
        fa = min(n - 1, s["i"] + win); H = _interval(cons, fa, ("hitter",)) or _interval(cons, s["i"], ("hitter",))
        C = _interval(cons, s["c0"], ("context",))
        s["long"], s["need_px"] = False, 0.0
        if not H or not C: continue
        hc = min(max((H[0] + H[1]) / 2, H[0]), H[1]) if H[0] <= H[1] else (H[0] + H[1]) / 2
        gap = max(0.0, C[0] - min(H[1], max(H[0], hc)), max(H[0], min(H[1], hc)) - C[1]) if C[0] <= C[1] else abs((C[0] + C[1]) / 2 - hc)
        s["need_px"] = round(gap, 1)
        s["long"] = gap > cfg["long_shot_ratio"] * _reach(s["c0"] - fa, v, a)
        k = fa + 1
        if not np.isfinite(bx[fa + 1:s["c0"]]).any():          # no ball track (highlights): cut 0.2 s after the contact window
            s["cut_i"] = int(min(fa + 1 + round(0.2 * fps), s["c0"])); continue
        while k < s["c0"] and not (np.isfinite(bx[k]) and abs(bx[k] - hc) > cw / 2 - 40): k += 1
        s["cut_i"] = int(min(k, s["c0"]))
    return shots


def _l1_core(n, fps, cw0, SW, tgt, cons, wkind, vmax, amax, cuts=(), mono=None, budget=20.0, w=(10.0, 1.0, 100.0), wt=0.05):
    """One LP (Grundmann et al. 2011): min w1|dx| + w2|d2x| + w3|d3x| + wt|x - tgt| + sum_k w_kind * slack, with
    hard source bounds, |dx| <= vmax, |d2x| <= amax, an optional monotone pre-serve, and hard cuts (frame k: x may jump
    between k-1 and k; every difference across it is dropped). -> (x, slack per constraint)."""
    import scipy.sparse as sps
    from scipy.optimize import linprog
    cuts = set(int(c) for c in cuts)
    def spans_cut(k, m):                                   # difference over frames k..k+m crosses a cut?
        return any((k + j) in cuts for j in range(1, m + 1))
    n1, n2, n3, G = n - 1, max(0, n - 2), max(0, n - 3), len(cons)
    o1, o2, o3, oe, og = n, n + n1, n + n1 + n2, n + n1 + n2 + n3, n + n1 + n2 + n3 + n
    N = og + G
    rows, cols, vals, b = [], [], [], []; r_ = [0]
    def row(cs, vs, rhs):
        for c_, v_ in zip(cs, vs): rows.append(r_[0]); cols.append(c_); vals.append(v_)
        b.append(rhs); r_[0] += 1
    for k in range(n1):
        if spans_cut(k, 1): continue
        row([k + 1, k, o1 + k], [1, -1, -1], 0); row([k + 1, k, o1 + k], [-1, 1, -1], 0)
        row([k + 1, k], [1, -1], vmax[k + 1]); row([k + 1, k], [-1, 1], vmax[k + 1])
    for k in range(n2):
        if spans_cut(k, 2): continue
        row([k + 2, k + 1, k, o2 + k], [1, -2, 1, -1], 0); row([k + 2, k + 1, k, o2 + k], [-1, 2, -1, -1], 0)
        row([k + 2, k + 1, k], [1, -2, 1], amax[k + 1]); row([k + 2, k + 1, k], [-1, 2, -1], amax[k + 1])
    for k in range(n3):
        if spans_cut(k, 3): continue
        row([k + 3, k + 2, k + 1, k, o3 + k], [1, -3, 3, -1, -1], 0); row([k + 3, k + 2, k + 1, k, o3 + k], [-1, 3, -3, 1, -1], 0)
    for k in range(n):
        row([k, oe + k], [1, -1], tgt[k]); row([k, oe + k], [-1, -1], -tgt[k])
    for g, c_ in enumerate(cons):
        i, lo, hi = c_[0], c_[1], c_[2]
        if lo > -1e9: row([i, og + g], [-1, -1], -lo)
        if hi < 1e9: row([i, og + g], [1, -1], hi)
    if mono is not None:
        i_s, d = mono
        for k in range(min(i_s, n1)): row([k + 1, k], [-d, d], 0)
    A = sps.csr_matrix((vals, (rows, cols)), shape=(r_[0], N))
    cost = np.zeros(N); cost[o1:o2] = w[0]; cost[o2:o3] = w[1]; cost[o3:oe] = w[2]; cost[oe:og] = wt
    cost[og:] = [wkind.get(c[3], 1000.0) * (c[4] if len(c) > 4 else 1.0) for c in cons]
    bounds = [(cw0 / 2, SW - cw0 / 2)] * n + [(0, None)] * (N - n)
    res = linprog(cost, A_ub=A, b_ub=np.array(b), bounds=bounds, method="highs", options={"time_limit": float(budget)})
    if res.status == 1: raise PlanTimeout("l1 LP hit the planning budget")
    if res.status != 0: raise RuntimeError(f"l1 LP failed: {res.message}")
    return res.x[:n], res.x[og:]


def _mode_caps(n, fps, cw0, fast, mode, cfg):
    vmax = np.where(fast, cfg["pan_fast"], cfg["pan_max"]) * cw0 / fps
    amax = np.full(n, cfg["pan_acc"] * cw0 / fps ** 2)
    if mode == "fast_pan":
        vmax = np.where(fast, cfg["pan_fast_flight"] * cw0 / fps, vmax)
        amax = np.where(fast, cfg["pan_acc_flight"] * cw0 / fps ** 2, amax)
    return vmax, amax


def _pick_cuts(shots, mode, fps, cfg):
    if mode != "cut": return []
    cuts, last, gap = [], 0, int(round(cfg["min_shot_s"] * fps))
    for s in sorted(shots, key=lambda s_: s_["i"]):
        if not (s.get("long") and s.get("cut_i") is not None): continue
        k = max(s["cut_i"], last + gap)                  # slide later (never past the context start) to keep shots >= min_shot_s
        if k <= s["c0"]:
            cuts.append(k); last = k; s["cut"] = True; s["cut_i"] = k
    return cuts


def jerk_p95_cuts(cx, frame_w, cuts=()):
    """ball.jerk_p95 per continuous shot: third differences that span a hard cut are not camera motion."""
    cx = np.asarray(cx, float)
    if len(cx) < 4: return None
    d3 = np.abs(np.diff(cx, 3)); keep = np.ones(len(d3), bool)
    for c in cuts:
        keep[max(0, c - 3):max(0, c)] = False
    j = d3[keep] if keep.any() else d3
    p = float(np.percentile(j, 95))
    return dict(px_per_frame3=float(f"{p:.6g}"), norm=float(f"{p / frame_w:.6g}") if frame_w else None, n=int(len(j)), cuts=len(cuts))


def landing_qa(shots, x, cw, scale, SW, fps, t0, wide=False):
    """Media Director check (v4.9): 'landing zone and receiver visible before the bounce'. Per exchange shot: on at
    least one frame of its context window the receiver's box is fully inside the crop and the landing x is >= 20 px
    inside. -> dict(shots, ok, pct, failing=[contact t], pass = every shot ok)."""
    ok, bad = 0, []
    for s in shots:
        good = False
        for i in range(s["c0"], s["ib"] + 1):
            if i >= len(x): break
            L, R = (0, SW) if wide else (x[i], x[i] + cw[i])
            b = _box_at(s["receiver"], i)
            rv = wide or b is None or (b[0] * scale >= L - 1 and b[1] * scale <= R + 1)
            lv = s["land_x"] is None or (L + (0 if wide else 20) <= s["land_x"] <= R - (0 if wide else 20))
            if rv and lv: good = True; break
        ok += good
        if not good: bad.append(round(t0 + s["i"] / fps, 2))
    n = len(shots)
    return dict(shots=n, ok=ok, pct=round(100.0 * ok / n, 1) if n else None, failing=bad, **{"pass": ok == n})


def l1_plan(boxes, ball_xy, ball_x, contacts, t0, fps, SW=3840, SH=2160, cfg=CAMERA, scale=3.0, sides=None, kinds=None,
            bounces=()):
    """v4.8/v4.9 virtual cinematographer for 9:16 (Grundmann, Kwatra & Essa 2011, L1-optimal camera paths; the
    AutoFlip crop formulation): ONE linear program per clip over the crop-centre path x at full source height (the
    vertical never clips; source edges are plain bounds). Hard: source bounds, the pan / acceleration caps, a
    monotone pre-serve. Soft (L1 slack, reported): hitter at contact, receiver at reception, server pre-serve and
    (v4.9) the shot context: receiver + landing area in frame before each bounce. long_shot_mode: cut | fast_pan | off
    (see CAMERA). target = RTS-smoothed ball x (gap-filled). HiGHS time_limit = plan_budget_s."""
    n = len(boxes); mode = cfg.get("long_shot_mode", "cut")
    roles, strk, chk, i_s = _roles_and_server(boxes, ball_xy, contacts, t0, fps, cfg, sides, kinds)
    cw0 = float(crop_size(1.0, SH)[0]); cw = np.full(n, cw0); ch = np.full(n, SH)
    fast = flight_mask(n, fps, t0, sorted(contacts), cfg)
    tgt = rts_smooth(np.asarray(ball_x, float), fps)
    if not np.isfinite(tgt).any(): tgt = np.full(n, SW / 2.0)
    bxs = tgt.copy()
    tgt = np.clip(tgt, cw0 / 2, SW - cw0 / 2)
    cons = frame_constraints(roles, strk, i_s, n, fps, scale, SW, cw0, cfg)
    shots = shot_context(roles, bxs, n, fps, t0, bounces, cfg, ball_xy, scale)
    if mode != "off":
        cons += context_constraints(shots, scale, SW, cw0, cfg)
        long_shots(shots, cons, bxs, fps, cw0, n, cfg)
    cuts = [c for c in _pick_cuts(shots, mode, fps, cfg) if i_s is None or c > i_s + gate_win(fps)]
    vmax, amax = _mode_caps(n, fps, cw0, fast, mode, cfg)
    mono = None
    if i_s is not None and strk:
        ks = sorted(strk); c0 = (strk[ks[0]][0] + strk[ks[0]][1]) / 2 * scale; c1 = (strk[ks[-1]][0] + strk[ks[-1]][1]) / 2 * scale
        mono = (i_s, 1 if c1 >= c0 else -1)
        vmax[:i_s] = np.minimum(vmax[:i_s], cfg["preserve_pan_max"] * cw0 / fps)
    wk = dict(hitter=cfg["l1_w_gate"], receiver=cfg["l1_w_gate"], server=cfg["l1_w_gate"], context=cfg["ctx_w"])
    t_ = time.monotonic()
    x, sl = _l1_core(n, fps, cw0, SW, tgt, cons, wk, vmax, amax, cuts, mono, cfg.get("plan_budget_s", 20.0),
                     cfg["l1_w"], cfg["l1_w_target"])
    bad = sorted({round(t0 + cons[g][0] / fps, 2) for g in range(len(cons)) if sl[g] > 0.5 and cons[g][3] in ("hitter", "receiver")})
    xc = x.copy()                                          # caps are camera motion: remove the jump at each cut
    for c in cuts: xc[c:] -= x[c] - x[c - 1]
    caps = cap_report(xc, cw, fps, fast, cfg if mode != "fast_pan" else dict(cfg, pan_fast=cfg["pan_fast_flight"], pan_acc=cfg["pan_acc_flight"]))
    return dict(h=np.ones(n), cw=cw.astype(int), ch=ch, cx=x, fast=fast, roles=roles, gates=[], reached=[], caps=caps, iters=1,
                infeasible=bad, infeasible_pairs=[], lp_error=None, feasible=not bad,
                source_edge=source_edge_check(roles, scale, SW, fps, t0, cfg), serve_i=i_s, server=strk, server_check=chk,
                planner="l1_lp", lp_s=round(time.monotonic() - t_, 3), n_constraints=len(cons),
                slack_px=round(float(sl.max()) if len(sl) else 0.0, 1), long_shot_mode=mode, shots=shots,
                cut_frames=cuts, cuts=[round(t0 + c / fps, 2) for c in cuts], long_shots=sum(bool(s.get("long")) for s in shots))



def plan_clip_capped(boxes, ball_xy, ball_x, contacts, t0, fps, SW=3840, SH=2160, cfg=CAMERA, scale=3.0, roles_qa=None,
                     **kw):
    """v4.7 entry point: locked wide when cfg['locked_wide']; else plan_clip under plan_budget_s, falling back to
    safe_plan on a timeout or LP failure, and also when plan_clip's result is infeasible and safe_plan frames more of
    the hitters / receivers (roles_qa(plan) -> clipped count; the caller renders x/y with it)."""
    sides, kinds = kw.get("sides"), kw.get("kinds")
    if cfg.get("locked_wide"):
        return locked_wide_plan(boxes, ball_xy, contacts, t0, fps, SW, SH, cfg, scale, sides, kinds)
    t_start = time.monotonic()
    if cfg.get("planner", "l1") == "l1":                 # v4.8: one LP, bounded; fallback safe_plan
        try:
            gp = l1_plan(boxes, ball_xy, ball_x, contacts, t0, fps, SW, SH, cfg, scale, sides, kinds, kw.get("bounces", ()))
        except (PlanTimeout, RuntimeError, ValueError) as e:
            gp = safe_plan(boxes, ball_xy, contacts, t0, fps, SW, SH, cfg, scale, sides, kinds, why=f"l1: {e}")
        gp["plan_s"] = round(time.monotonic() - t_start, 2)
        return gp
    try:
        gp = plan_clip(boxes, ball_xy, ball_x, contacts, t0, fps, SW, SH,
                       dict(cfg, _deadline=t_start + float(cfg.get("plan_budget_s", 20.0))), scale, **kw)
        gp["planner"] = "gate_plan"
    except PlanTimeout as e:
        gp = safe_plan(boxes, ball_xy, contacts, t0, fps, SW, SH, cfg, scale, sides, kinds, why=f"timeout: {e}")
    gp["plan_s"] = round(time.monotonic() - t_start, 2)
    if gp["planner"] == "gate_plan" and (gp.get("lp_error") or not gp["feasible"]) and roles_qa is not None:
        alt = safe_plan(boxes, ball_xy, contacts, t0, fps, SW, SH, cfg, scale, sides, kinds, why="gate plan infeasible")
        if roles_qa(alt) <= roles_qa(gp):
            alt["plan_s"] = round(time.monotonic() - t_start, 2); return alt
    return gp


def infeasible_decision(plan, n_wide, cfg=CAMERA, roles_qa=None):
    """Contact framing at risk -> ('wide', why) when wide mode is allowed and the per-reel limit is not reached, else
    ('reject', why): the clip is flagged for rejection (rendered, but marked in the stats for the editor).
    At risk = the planner cannot hold a hitter at a verified contact under the pan/zoom caps (plan["feasible"] False),
    or, given the rendered x/y crop (roles_qa = role_clip_qa(...)), a hitter or the receiver is clipped within +-0.1 s
    of their contact (this also catches the vertical: the tilt runs under its own, slower caps)."""
    risk = []
    if plan.get("source_edge"):                                       # v4.6: no crop (or wide view) can frame it
        risk.append(f"role player leaves the source frame at {plan['source_edge']}")
    if not plan["feasible"]:
        risk.append(f"contacts at {plan['infeasible']} cannot be held under the pan/zoom caps")
    if roles_qa is not None and not roles_qa["pass"]:
        risk.append(f"contact framing: {roles_qa['contacts_hitter_clipped']} hitter / "
                    f"{roles_qa['receptions_receiver_clipped']} receiver clipped at contact")
    if not risk: return None, ""
    why = "; ".join(risk)
    if plan.get("source_edge"): return "reject", why                  # wide shows the same source frame: reject
    if cfg.get("infeasible_action", "wide") == "wide" and n_wide < cfg["max_wide"]: return "wide", "auto: " + why
    return "reject", why


def role_clip_qa(roles, x, y, cw, ch, scale, SW, fps, t0, cfg=CAMERA, wide=False):
    """Hitter / receiver clipped gates: per contact, is the hitter (receiver) box outside the crop, or inside the
    rel_margin band (waived where the crop sits on the image border), within +-0.1 s of their contact?
    -> dict(contacts_hitter_clipped=k, contacts=N, receptions_receiver_clipped=k, receptions=M, pass)."""
    win = gate_win(fps)
    def clipped(track, ic):
        if wide or not track: return False
        for i in range(ic - win, ic + win + 1):
            if i not in track or i < 0 or i >= len(x): continue
            b = track[i]; X0, X1 = max(0.0, b[0] * scale), min(SW, b[1] * scale); Y0, Y1 = b[3] * scale, b[4] * scale
            mx = cfg["rel_margin"] * cw[i]
            lft = x[i] > 1 and X0 < x[i] + mx; rgt = x[i] + cw[i] < SW - 2 and X1 > x[i] + cw[i] - mx
            if X0 < x[i] - 1 or X1 > x[i] + cw[i] + 1 or lft or rgt or Y0 < y[i] - 2 or Y1 > y[i] + ch[i] + 2: return True
        return False
    hc = sum(clipped(r["hitter"], r["i"]) for r in roles if r["hitter"])
    nh = sum(1 for r in roles if r["hitter"])
    rec = [r for r in roles if r["receiver"] is not None and r["recv_t"] is not None]
    rc = sum(clipped(r["receiver"], int(round(r["i"] + (r["recv_t"] - r["t"]) * fps))) for r in rec)
    return dict(contacts_hitter_clipped=int(hc), contacts=nh, receptions_receiver_clipped=int(rc), receptions=len(rec),
                **{"pass": hc == 0 and rc == 0})


def framing_checks(clips, pct_far_out, cfg=CAMERA, server_out_max=3):
    """v4.6 reel-level automated checks (MD reel v4.5 review + the 37302d7 far-partner concern). clips: {name: dict(
    preserve=preserve_qa(...) or None, source_edge=[...], wide=bool)}. Each check -> dict(pass, detail)."""
    pre = {k: v["preserve"] for k, v in clips.items() if v.get("preserve")}
    rev = {k: p["reversals"] for k, p in pre.items() if p["reversals"]}
    opn = {k: p["opening_has_server"] for k, p in pre.items() if p["opening_has_server"] is not True}
    sic = {k: (p["server_out_frames"], p["server_out_span"]) for k, p in pre.items()
           if p["server_out_frames"] > server_out_max or not p.get("server_identified", True) or p.get("server_coverage", 1.0) < 0.8}
    edge = {k: v["source_edge"] for k, v in clips.items() if v.get("source_edge")}
    return dict(
        preserve_reversals_check=dict(**{"pass": not rev}, rule="no pan reversal from clip start through the serve contact",
                                      clips_with_serve=len(pre), failing=rev),
        opening_frame_server_check=dict(**{"pass": not opn}, rule="the first frame contains the server (x and y)", failing=opn),
        server_in_crop_preserve_check=dict(**{"pass": not sic}, rule=f"identified server (serve side + toss ball) tracked on >= 80 % and fully in the crop on every pre-serve frame (<= {server_out_max} frames out)",
                                           failing=sic),
        source_edge_check=dict(**{"pass": not edge}, rule="no role player at the left/right edge of the source frame near their gate (clip flagged reject)",
                               failing=edge),
        far_partner_check=dict(**{"pass": pct_far_out is not None and pct_far_out <= 100 * cfg["far_out_max"] + 1e-9},
                               rule=f"far-side player out of frame on <= {cfg['far_out_max']:.0%} of frames", pct=pct_far_out))


# ---------------------------------------------------------------------------------------------------------------------
# Broadcast grammar for Highlights only (Antony, 19 Oct: Wimbledon / US Open). No ball track, no player ID: positions
# come from low-res regional motion energy (app/highlights.py motion pass), contacts from audio hits.
#   16:9  rally: locked full-court wide shot (no pans at all) from the clip start to last hit + split_after_last_s;
#         then a hard cut to a tight reaction crop (react_zoom) on the most animated spot after the point (a proxy for
#         the winner: point-winner logic is not run in this mode) until the clip end; then, only on a starred point
#         (R-SLOW: <= 2 per reel, user-verified, never slower than 0.45x), a slow-mo replay of the last shot at
#         replay_zoom (2-4x) around the hitter; then back to wide for the next serve.
#   9:16  rally: smooth follow of the hitter at full height (contacts = audio hits, contact x = motion hot spot at the
#         hit) through plan_pan / follow_pan, i.e. the existing pan, acceleration and reversal caps; then the same
#         reaction close-up and starred replay.
BROADCAST = dict(split_after_last_s=1.0, react_zoom=2.0, replay_zoom=2.5, replay_pre_s=1.2, replay_post_s=0.6,
                 replay_speed=0.5, replay_gain=0.35, key_win=(-0.6, 0.3), react_win=(0.5, 2.5), blur_cells=4.0,
                 max_replays=2, min_zoom=2.0, max_zoom=4.0)


def hot_spot(E, blur=BROADCAST["blur_cells"]):
    """(x, y) fractions of the strongest motion blob in an energy map E (h, w) or a stack (k, h, w); None if flat."""
    from scipy.ndimage import gaussian_filter
    E = np.asarray(E, float)
    if E.ndim == 3: E = E.sum(0)
    if E.size == 0 or E.max() <= 0: return None
    S = gaussian_filter(E, blur)
    y, x = np.unravel_index(int(np.argmax(S)), S.shape)
    return (x + 0.5) / S.shape[1], (y + 0.5) / S.shape[0]


def window_spot(times, maps, t0, t1):
    """Hot spot of the motion maps whose sample time is in [t0, t1] (None when no sample falls inside)."""
    times = np.asarray(times, float)
    m = (times >= t0) & (times <= t1)
    return hot_spot(np.asarray(maps)[m]) if m.any() else None


def tight_window(cx, cy, zoom, SW, SH, aspect="16:9"):
    """Crop (x, y, w, h) in source px for a zoom (relative to the full-height shot of that aspect), centred on the
    fractions cx, cy and clamped inside the frame. Even sizes."""
    zoom = float(np.clip(zoom, 1.0, BROADCAST["max_zoom"]))
    if aspect == "9:16":
        h = SH / zoom; w = h * 9 / 16
    else:
        w = SW / zoom; h = w * 9 / 16
        if h > SH: h = SH; w = h * 16 / 9
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    x = int(np.clip(cx * SW - w / 2, 0, SW - w)); y = int(np.clip(cy * SH - h / 2, 0, SH - h))
    return x, y, w, h


def follow_path(sample_t, sample_x, hits, t0, t1, fps, SW, SH, cfg=CAMERA, info=None):
    """Per-frame 9:16 crop centre (source px) for [t0, t1] at full height. v4.9: the same L1 LP as l1_plan on motion
    targets (no player ID in Highlights only): the hitter's hot spot at each audio hit, and before each bounce the
    receiving side (the next hitter's hot spot) so the viewer sees where the ball goes; long_shot_mode cut | fast_pan |
    off as in l1_plan. sample_t / sample_x: motion hot-spot x (fraction) per motion sample; hits: audio contact times.
    info (dict, optional) receives cuts / long shots."""
    n = max(2, int(round((t1 - t0) * fps)))
    ts = t0 + np.arange(n) / fps
    cw = int(crop_size(1.0, SH)[0])
    st, sx = np.asarray(sample_t, float), np.asarray(sample_x, float) * SW
    if len(st) == 0:
        return np.full(n, SW / 2.0), cw
    target = np.clip(np.interp(ts, st, sx), cw / 2, SW - cw / 2)
    hs = [h for h in hits if t0 <= h <= t1]
    hx = [float(np.interp(h, st, sx)) for h in hs]
    mode = cfg.get("long_shot_mode", "cut")
    if mode == "off" or len(hs) < 1:
        plan = plan_pan(target, fps, t0, hs, hx, cw_def=cw, SH=SH, cfg=cfg)
        x = follow_pan(plan["planned"], np.full(n, float(cw)), cw, fps, plan["fast"], None, cfg)
        return np.clip(x, cw / 2, SW - cw / 2), cw
    win = gate_win(fps); pw = 0.04 * SW; cons, shots = [], []
    for k, (h, x_) in enumerate(zip(hs, hx)):
        i = int(round((h - t0) * fps))
        for q in range(i - win, i + win + 1):
            if 0 <= q < n: cons.append((q, x_ + pw - cw / 2, x_ - pw + cw / 2, "hitter"))
        if k + 1 < len(hs) and cfg["ctx_min_flight_s"] <= hs[k + 1] - h <= cfg["max_flight_s"]:
            tb = h + cfg["bounce_frac"] * (hs[k + 1] - h); ib = min(n - 1, int(round((tb - t0) * fps)))
            c0 = max(i + win + 1, ib - int(round(cfg["ctx_pre_s"] * fps)))
            if c0 <= ib:
                m = 1.0 / (ib + 1 - c0)
                for q in range(c0, ib + 1): cons.append((q, hx[k + 1] + pw - cw / 2, hx[k + 1] - pw + cw / 2, "context", m))
                shots.append(dict(i=i, ib=ib, c0=c0, tb=round(tb, 2), land_x=None, receiver=None))
    fast = flight_mask(n, fps, t0, hs, cfg)
    long_shots(shots, cons, None, fps, cw, n, cfg)
    cuts = _pick_cuts(shots, mode, fps, cfg)
    vmax, amax = _mode_caps(n, fps, cw, fast, mode, cfg)
    try:
        x, _ = _l1_core(n, fps, cw, SW, target, cons, dict(hitter=cfg["l1_w_gate"], context=cfg["ctx_w"]), vmax, amax, cuts,
                        None, cfg.get("plan_budget_s", 20.0), cfg["l1_w"], cfg["l1_w_target"])
    except (PlanTimeout, RuntimeError, ValueError):
        plan = plan_pan(target, fps, t0, hs, hx, cw_def=cw, SH=SH, cfg=cfg); cuts = []
        x = follow_pan(plan["planned"], np.full(n, float(cw)), cw, fps, plan["fast"], None, cfg)
    if info is not None:
        info.update(mode=mode, cuts=[round(t0 + c / fps, 2) for c in cuts], long_shots=sum(bool(s.get("long")) for s in shots),
                    shots=len(shots))
    return np.clip(x, cw / 2, SW - cw / 2), cw


def broadcast_plan(clip, motion, aspect, SW, SH, fps, starred=False, cfg=BROADCAST, camera_cfg=CAMERA):
    """Framing segments for one whole-point clip: list of dicts(kind, t0, t1, speed, window | path).
    motion: dict(t=sample times, x=hot-spot x per sample, maps=(k, h, w) energy maps) or None (then wide / centred)."""
    a, b, last = clip["a"], clip["b"], clip["last_hit"]
    split = min(b - 0.8, last + cfg["split_after_last_s"])
    segs = []
    if aspect == "9:16" and motion is not None:
        fi = {}
        path, cw = follow_path(motion["t"], motion["x"], clip.get("hits_t", ()), a, split, fps, SW, SH, camera_cfg, fi)
        segs.append(dict(kind="follow", t0=a, t1=split, speed=1.0, path=path, cw=cw, long_shot=fi))
    elif aspect == "9:16":
        segs.append(dict(kind="tight", t0=a, t1=split, speed=1.0, window=tight_window(0.5, 0.5, 1.0, SW, SH, aspect)))
    else:
        segs.append(dict(kind="wide", t0=a, t1=split, speed=1.0))
    react = window_spot(motion["t"], motion["maps"], last + cfg["react_win"][0], last + cfg["react_win"][1]) if motion else None
    rx, ry = react or (0.5, 0.6)
    segs.append(dict(kind="reaction", t0=split, t1=b, speed=1.0, window=tight_window(rx, ry, cfg["react_zoom"], SW, SH, aspect)))
    if starred:
        key = window_spot(motion["t"], motion["maps"], last + cfg["key_win"][0], last + cfg["key_win"][1]) if motion else None
        kx, ky = key or (rx, ry)
        speed = max(0.45, float(cfg["replay_speed"]))
        z = float(np.clip(cfg["replay_zoom"], cfg["min_zoom"], cfg["max_zoom"]))
        segs.append(dict(kind="replay", t0=max(a, last - cfg["replay_pre_s"]), t1=min(b, last + cfg["replay_post_s"]),
                         speed=speed, window=tight_window(kx, ky, z, SW, SH, aspect), gain=cfg["replay_gain"]))
    return segs


def broadcast_check(clips, cfg=BROADCAST):
    """R-SLOW for replays: <= max_replays (and <= 2) per reel, only on starred (user-verified) points, >= 0.45x."""
    reps = [(c, s) for c in clips for s in c.get("framing", ()) if s["kind"] == "replay"]
    if len(reps) > min(2, cfg["max_replays"]): raise ValueError(f"R-SLOW: {len(reps)} replays > 2")
    for c, s in reps:
        if not c.get("starred"): raise ValueError("R-SLOW: replay on a point the user did not star")
        if s["speed"] < 0.45 - 1e-9: raise ValueError("R-SLOW: replay slower than 0.45x")
    return True
