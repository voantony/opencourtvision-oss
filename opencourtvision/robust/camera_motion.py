"""Camera-motion handling (engine-v2): global motion from the static background -> shake / drift / bump detection ->
automatic court re-fit with accept / retry -> per-sample vision reliability.

* Global motion: phase correlation of each proxy frame against the reference frame (absolute offset, so slow drift does
  not accumulate error), on a Hann-windowed, mean-removed image. Players are small next to court lines, fences and
  trees, so the peak is the background's shift.
* Shake: frame-to-frame offset change > SHAKE_PX, or the 2 s RMS of that change > SHAKE_RMS -> samples unreliable.
* Bump: the offset jumps by > BUMP_PX within BUMP_WIN s and stays (median of the next HOLD_S s). Drift: the offset moves
  > DRIFT_PX away from the last accepted fit. Either triggers a re-fit.
* Re-fit (N1 auto-fit, automatic accept): H = translate(offset) @ H_ref, accepted when the last stable frame before
  the move, shifted by the measured change, matches the current frame (normalised cross-correlation on the overlap >= NCC_MIN). Otherwise retry on the
  next RETRY samples; still rejected -> the vision side is low-confidence until a fit is accepted (in the app: the N1
  prompt to re-tap the 4 corners on the raw frame; no dragging of drawn lines).
* v4.5 continuous re-alignment (CONTINUOUS, default): the fit follows the smoothed measured offset sample by sample
  while the camera moves, verified every VERIFY_EVERY samples by NCC of the current frame against a recent keyframe moved
  by the measured change (keyframes advance every KEY_S s, re-anchor after RETRY failures). Shake is jitter around the
  smoothed path (JIT_*), so a slow pan stays reliable instead of waiting for the camera to stop; event re-fits are kept
  as CONTINUOUS=False for comparison.
* Low confidence never drops a point: the motion cues return "no decision" there and the edit falls back to audio plus
  the score prior. In the US build none of this sets a clip boundary (motion is a keep/drop gate only).
"""
import numpy as np

CAM = dict(SHAKE_PX=0.6, SHAKE_RMS=0.35, BUMP_PX=3.0, BUMP_WIN=1.0, HOLD_S=3.0, DRIFT_PX=2.0, NCC_MIN=0.5, RETRY=12,
           SMOOTH_S=1.0,
           CONTINUOUS=True,          # v4.5: track the court frame to frame while the camera moves (pans), not one re-fit per move
           JIT_PX=0.8, JIT_RMS=0.35,  # v4.5 shake = jitter around the smoothed path (a smooth pan is not shake)
           VERIFY_EVERY=3, KEY_S=10.0,  # v4.5 verify the tracked offset against a keyframe every 3 samples; new keyframe every 10 s
           PAN_PX_S=0.5,             # v4.5 an event is a "pan" when the camera moves faster than this for the whole move
           PEAK_KEY=0.08, PEAK_ABS=0.2)  # v4.5 keyframe chaining: advance the key only on a clear peak; snap to the match
                                         # reference when its own peak is strong (no slow random-walk drift)             # v4.5 an event is a "pan" when the camera moves faster than this for the whole move


def _prep(img):
    a = np.asarray(img, np.float64)
    a = a - a.mean()
    h, w = a.shape
    return a * np.outer(np.hanning(h), np.hanning(w))


def phase_shift(ref_f, img, F=None):
    """Shift (dx, dy) px that moves the reference onto img; ref_f = conj FFT of the prepared reference. F: the
    image's FFT if already computed."""
    F = np.fft.fft2(_prep(img)) if F is None else F
    R = F * ref_f
    R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    h, w = r.shape
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)

    def para(a, b, d):
        den = a - 2 * b + d
        return 0.5 * (a - d) / den if abs(den) > 1e-12 else 0.0
    dy = iy + para(r[(iy - 1) % h, ix], r[iy, ix], r[(iy + 1) % h, ix])
    dx = ix + para(r[iy, (ix - 1) % w], r[iy, ix], r[iy, (ix + 1) % w])
    if dy > h / 2: dy -= h
    if dx > w / 2: dx -= w
    return float(dx), float(dy), float(r.max())


def ncc(a, b, mask=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if mask is not None: a, b = a[mask], b[mask]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _shifted(img, dx, dy):
    from scipy.ndimage import shift
    return shift(np.asarray(img, np.float32), (dy, dx), order=1, mode="constant", cval=np.nan)


def analyze(frames, t, ref=None, p=None):
    """frames: (n, h, w) proxy frames at times t. -> dict(offset (n,2) px, unstable (n,), events [(t, kind, px)],
    refits [(t, (dx, dy), accepted, ncc)], fit_offset (n,2) px of the accepted fit in force, reliable (n,))."""
    p = {**CAM, **(p or {})}
    frames = np.asarray(frames, np.float32); t = np.asarray(t, float); n = len(t)
    ref = frames[0] if ref is None else np.asarray(ref, np.float32)
    ref_f = np.conj(np.fft.fft2(_prep(ref)))
    hz = 1.0 / max(1e-6, float(np.median(np.diff(t)))) if n > 1 else 6.0
    if p["CONTINUOUS"]:
        off = _track_offsets(frames, t, ref_f, p)
    else:
        off = np.array([phase_shift(ref_f, f)[:2] for f in frames]) if n else np.zeros((0, 2))
    step = np.r_[0.0, np.hypot(*np.diff(off, axis=0).T)] if n else np.zeros(0)
    k = max(1, int(round(2.0 * hz)))
    rms = np.sqrt(np.convolve(step ** 2, np.ones(k) / k, "same")) if n else step
    unstable = (step > p["SHAKE_PX"]) | (rms > p["SHAKE_RMS"])
    from scipy.ndimage import median_filter
    sm = median_filter(off, size=(max(1, int(round(p["SMOOTH_S"] * hz))) | 1, 1)) if n else off
    if p["CONTINUOUS"]:
        return _continuous(frames, t, off, sm, hz, k, p)
    events, refits = [], []
    fit = np.zeros(2); fit_off = np.zeros((n, 2)); reliable = ~unstable.copy()
    hold = max(1, int(round(p["HOLD_S"] * hz))); win = max(1, int(round(p["BUMP_WIN"] * hz)))
    pending, tries, anchor = None, 0, 0
    i = 0
    while i < n:
        cur = sm[i]
        if pending is None and not unstable[i]:
            j0 = max(0, i - win - k)                  # a bump is flagged unstable for ~k samples first
            jump = np.hypot(*(cur - sm[j0]))
            far = np.hypot(*(cur - fit))
            after = sm[i:min(n, i + hold)]
            stays = len(after) and np.hypot(*(np.median(after, 0) - fit)) > p["DRIFT_PX"]
            if far > p["DRIFT_PX"] and stays:
                events.append((float(t[i]), "bump" if jump > p["BUMP_PX"] else "drift", round(float(far), 2)))
                pending, tries = cur.copy(), 0
                anchor = max(0, j0 - 1)                 # last stable frame before the move: the scene there is current
        if pending is not None and not unstable[i]:
            cand = sm[i]
            dlt = cand - sm[anchor]                    # check the move against a recent frame, not the match start
            r = _shifted(frames[anchor], dlt[0], dlt[1]); m = ~np.isnan(r)
            c = ncc(r, frames[i], m) if m.sum() > 50 else 0.0
            ok = c >= p["NCC_MIN"]
            refits.append((float(t[i]), (round(float(cand[0]), 2), round(float(cand[1]), 2)), bool(ok), round(c, 3)))
            if ok: fit = cand.copy(); pending = None
            else:
                tries += 1
                if tries >= p["RETRY"]: pending = cand.copy(); tries = 0      # keep trying, stay low-confidence
        if pending is not None: reliable[i] = False
        fit_off[i] = fit
        i += 1
    return dict(offset=off, unstable=unstable, events=events, refits=refits, fit_offset=fit_off, reliable=reliable)


def _track_offsets(frames, t, ref_f, p):
    """v4.5 frame-to-frame tracking: phase correlation against a recent keyframe (the scene and light there are current)
    plus that keyframe's offset; the keyframe advances every KEY_S s on a clear peak and snaps to the match reference
    when the reference still correlates strongly. Late-match light changes no longer break the absolute match-start
    correlation, and a pan never drifts out of overlap with the reference."""
    n = len(t); off = np.zeros((n, 2))
    key_f, key_off, key_t = ref_f, np.zeros(2), (t[0] if n else 0.0)
    for i in range(n):
        F = np.fft.fft2(_prep(frames[i]))
        dx, dy, pk = phase_shift(key_f, None, F)
        off[i] = key_off + (dx, dy)
        if t[i] - key_t >= p["KEY_S"] and pk >= p["PEAK_KEY"] and (i == 0 or np.hypot(*(off[i] - off[i - 1])) <= p["JIT_PX"]):
            ax, ay, apk = phase_shift(ref_f, None, F)
            if apk >= p["PEAK_ABS"]: off[i] = (ax, ay)
            key_f, key_off, key_t = np.conj(F), off[i].copy(), t[i]
    return off


def _continuous(frames, t, off, sm, hz, k, p):
    """v4.5: frame-to-frame court tracking (see module doc). Same return keys as analyze()."""
    n = len(t)
    jit = np.hypot(*(off - sm).T) if n else np.zeros(0)
    jrms = np.sqrt(np.convolve(jit ** 2, np.ones(k) / k, "same")) if n else jit
    unstable = (jit > p["JIT_PX"]) | (jrms > p["JIT_RMS"])
    fit = np.zeros(2); fit_off = np.zeros((n, 2)); reliable = np.zeros(n, bool)
    events, refits = [], []
    win = max(1, int(round(p["BUMP_WIN"] * hz))); key = None; ok = False; fails = 0; logged = np.zeros(2); last_ev = None
    for i in range(n):
        if unstable[i]:
            fit_off[i] = fit; continue
        if key is None:
            key = i; fit = sm[i].copy(); logged = fit.copy(); ok = True
        elif (i - key) % p["VERIFY_EVERY"] == 0 or np.hypot(*(sm[i] - fit)) > p["DRIFT_PX"]:
            d = sm[i] - sm[key]
            r = _shifted(frames[key], d[0], d[1]); m = ~np.isnan(r)
            c = ncc(r, frames[i], m) if m.sum() > 50 else 0.0
            ok = c >= p["NCC_MIN"]
            if ok:
                fails = 0
                moved = np.hypot(*(sm[i] - logged))
                if moved > p["DRIFT_PX"]:                     # log a move once it exceeds DRIFT_PX (not every sample)
                    j0 = max(0, i - win - k); jump = np.hypot(*(sm[i] - sm[j0]))
                    speed = jump / max(1e-6, (i - j0) / hz)
                    kind = "bump" if jump > p["BUMP_PX"] else ("pan" if speed > p["PAN_PX_S"] else "drift")
                    if last_ev is None or i - last_ev > win or events[-1][1] != kind:
                        events.append((float(t[i]), kind, round(float(moved), 2)))
                    last_ev = i
                    refits.append((float(t[i]), (round(float(sm[i][0]), 2), round(float(sm[i][1]), 2)), True, round(c, 3)))
                    logged = sm[i].copy()
                if (t[i] - t[key]) >= p["KEY_S"]: key = i
            else:
                fails += 1
                refits.append((float(t[i]), (round(float(sm[i][0]), 2), round(float(sm[i][1]), 2)), False, round(c, 3)))
                if fails >= p["RETRY"]: key, fails = i, 0       # re-anchor on the current frame (stays low-confidence until verified)
        if ok: fit = sm[i].copy()
        reliable[i] = ok; fit_off[i] = fit
    return dict(offset=off, unstable=unstable, events=events, refits=refits, fit_offset=fit_off, reliable=reliable)


def realign_energy(E, fit_offset_px, frame_size, order=1):
    """Undo the accepted camera offset on the motion grid so each cell keeps meaning the same patch of court.
    order: 1 = bilinear (sub-cell), 0 = whole cells (no smearing of player energy across cells)."""
    from scipy.ndimage import shift
    E = np.asarray(E, np.float32); n, R, C = E.shape; w, h = frame_size
    out = E.copy()
    for k in range(n):
        dx, dy = fit_offset_px[k]
        if dx or dy: out[k] = shift(E[k], (-dy * R / h, -dx * C / w), order=order, mode="constant", cval=0.0)
    return out


def refit_homography(H_ref, offset_px):
    """court -> image homography carried to the current camera position (similarity = translation here)."""
    S = np.array([[1, 0, offset_px[0]], [0, 1, offset_px[1]], [0, 0, 1.0]])
    return S @ np.asarray(H_ref, float)
