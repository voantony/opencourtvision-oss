"""Smart dead space (v4.0): evidence-based clip boundaries and non-point rejection on top of E1/E3 point candidates.

Why (eng notes §2.8.7c, dev_match doubles, 101 labelled points): a fixed 4 s lead and a 2.5-6 s tail, applied to a 6 s-gap
hit group, put 12.8 s of walk-back, ball pick-up and pre-serve bouncing into every point, cut the reaction on fast
endings, and showed first-serve faults and between-point hits as their own clips. A quarter of serves (mostly the far
server) are 12-14 dB onsets, under the 14 dB hit threshold, so the "first hit" was often the return.

Per candidate point (a group of audio hit onsets):
1. Anatomy. Split the group into runs at pauses > RUN_GAP s. The played run is the run with the most hits (ties: later).
   Runs before it are a first-serve fault (ends >= FAULT_SEP s before the played run) or pre-serve bouncing (dropped);
   runs after it are pick-up / walk-back (dropped).
2. Start. Back-search for a quiet onset (>= QUIET_TH dB) up to BACK s before the first loud hit: the far server's serve
   under the hit threshold. Start = min(first loud hit - LEAD_LOUD, quiet onset - LEAD_QUIET): just before the toss.
3. End. last contact + TAIL, extended only when the broadband level after the last contact rises >= REACT_DB above the
   level in the 3 s before the serve (cheering, shouting, high-fives); then it runs to the end of the reaction + 0.5 s,
   capped at last contact + CAP. A pick-up hit after the point stops the base tail (not a reaction).
4. Non-points. A candidate whose played run is short (<= FAULT_HITS hits, <= FAULT_DUR s) and is followed by the next
   candidate's serve within FAULT_NEXT s is a first-serve fault: it is merged into that point as a short leading part
   (shown as a jump cut), never as a clip of its own. A short run within PICKUP_GAP s after the previous point's last
   contact is pick-up (dropped). A short run with >= BREAK_GAP s of silence on both sides is changeover / set-break
   hitting (dropped).
   Optional live gate (motion, both builds): drop a clip only if the court is idle; never moves a boundary.

Audio only: every boundary here comes from audio onsets and loudness. US build: motion may only gate live play
(`live_gate`). EU build may additionally use motion cues for admission (`motion.apply_cues`). No ball tracking anywhere.
"""
import numpy as np

from ..detection.audio import detect_hits

SMART = dict(RUN_GAP=3.0, EXT_GAP=3.5, FAULT_SEP=5.5, QUIET_TH=9.0, QUIET_RISE=3.0, BACK=2.2, LEAD_LOUD=1.6, LEAD_QUIET=0.7,
             TAIL=2.8, REACT_DB=6.0, REACT_WIN=5.0, CAP=4.0, FAULT_HITS=3, FAULT_DUR=2.0, FAULT_NEXT=16.0,
             PICKUP_GAP=7.0, PICKUP_HITS=2, BREAK_HITS=3, BREAK_GAP=40.0, FAULT_PART=(0.55, 1.5), MIN_GAP=0.2,
             LEAD_NOSERVE=1.6, FAULT_MAX=0.8, POINT_GAP=None, KEEP_HITS=4, CHAIN_MAX=6.0, PAUSE_PAD=(1.0, 1.0))
# 18 Oct: FAULT_PART/FAULT_MAX = the fault lead-in is a 0.8 s jump cut (toss, serve, call) instead of 2.7+ s.
# Rally chaining (on): a later run of >= KEEP_HITS loud hits starting <= CHAIN_MAX s after the played run is the rest of
# the rally (split by a lob / quiet far hits) or the real point after a played-out fault; the pause between them
# (> EXT_GAP, minus PAUSE_PAD after / before the hits) is skipped by a jump cut, so the extra recall costs no dead time.
# Split-half (eng notes 2.8.7d): tuned on points 1-50; on 51-101 recall 80.4 -> 84.3 %, dead 6.87 -> 6.76 s, cut-offs
# 5/4 -> 5/3. Audio only (both builds). POINT_GAP (split at an inter-point gap) never fires on dev_match; kept, off.
FRAME = 100.0


def _runs(times, gap):
    out = []
    for v in times:
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def anatomy(times, p, quiet=()):
    """(fault_run or None, played_run, dropped runs) for one candidate's loud hit times. Runs are split on loud + quiet
    onsets together (far-court hits are often under the loud threshold, so loud-only gaps inside a rally reach 3.5-5.5 s
    while loud + quiet gaps stay under 3 s); each run is then reduced to its loud hits."""
    loud = sorted(times)
    if not loud:
        raise ValueError("candidate without hits")
    allt = sorted(set(loud) | {q for q in quiet if loud[0] - 0.01 <= q <= loud[-1] + 0.01})
    L = set(loud)
    rs = [[v for v in r if v in L] for r in _runs(allt, p["RUN_GAP"])]
    rs = [r for r in rs if r]
    # fault: a short run ending >= FAULT_SEP s before the next run; the played rally starts at the first run after the
    # last fault, and continues through runs separated by <= EXT_GAP s (a lob or a missed far hit), then stops.
    k0, fault = 0, None
    for i in range(len(rs) - 1):
        if len(rs[i]) <= p["FAULT_HITS"] and rs[i + 1][0] - rs[i][-1] >= p["FAULT_SEP"]:
            fault, k0 = rs[i], i + 1
        else:
            break
    k1 = k0
    while k1 + 1 < len(rs) and rs[k1 + 1][0] - rs[k1][-1] <= p["EXT_GAP"]:
        k1 += 1
    if p.get("KEEP_HITS"):          # rally chaining: a later run with >= KEEP_HITS hits inside the same point (no
        j = k1 + 1                  # inter-point gap in between) is the rest of the rally after a pause, not pick-up
        while j < len(rs):
            if rs[j][0] - rs[k1][-1] > p["CHAIN_MAX"]:
                break
            if len(rs[j]) >= p["KEEP_HITS"]:
                k1 = j
                while k1 + 1 < len(rs) and rs[k1 + 1][0] - rs[k1][-1] <= p["EXT_GAP"]: k1 += 1
                j = k1 + 1
            else:
                j += 1
    run = [v for r in rs[k0:k1 + 1] for v in r]
    return fault, run, rs[:k0 - (1 if fault else 0)] + rs[k1 + 1:]


class Audio:
    """Onsets and loudness for one match (per-10 ms band loudness le, its background bg, broadband lb)."""

    def __init__(self, le, bg, lb, p):
        self.lb = np.asarray(lb, float)
        self.qt, self.qa = detect_hits(le, bg, th=p["QUIET_TH"], min_rise=p["QUIET_RISE"])
        k = 25
        self.lbs = np.convolve(self.lb, np.ones(k) / k, "same")     # 0.25 s smoothed broadband level

    def quiet_before(self, t, back):
        m = (self.qt >= t - back) & (self.qt <= t - 0.2)
        return float(self.qt[m].min()) if m.any() else None

    def level(self, a, b, q=50):
        s = self.lbs[max(0, int(a * FRAME)):max(0, int(b * FRAME))]
        return float(np.percentile(s, q)) if len(s) else None

    def reaction_end(self, last, serve, p, stop=None):
        """End of a post-point reaction (s) or None: smoothed broadband >= REACT_DB above the pre-serve level."""
        base = self.level(serve - 3.0, serve - 0.3)
        if base is None:
            return None, 0.0
        b = min(last + p["REACT_WIN"], stop) if stop else last + p["REACT_WIN"]
        i0, i1 = int((last + 0.4) * FRAME), int(b * FRAME)
        s = self.lbs[i0:i1] - base
        if not len(s) or s.max() < p["REACT_DB"]:
            return None, float(s.max()) if len(s) else 0.0
        on = np.where(s >= p["REACT_DB"] - 3.0)[0]
        return (i0 + on[-1]) / FRAME, float(s.max())


def smart_points(points, le, bg, lb, sets=None, p=None):
    """Candidate points (dicts with `times`, from E1/E3) -> points with evidence-based clip bounds A/B (+ optional
    leading fault part) and non-points removed. Returns (clips, rejected)."""
    p = {**SMART, **(p or {})}
    au = Audio(le, bg, lb, p)
    C = []
    pts = []
    for q in sorted(points, key=lambda x: x["t0"]):   # rally chaining: an inter-point gap (>= POINT_GAP s of silence)
        ts = sorted(q["times"])                         # inside one candidate separates two points
        if p.get("POINT_GAP") and len(ts) > 1:
            cut = [0] + [i for i in range(1, len(ts)) if ts[i] - ts[i - 1] >= p["POINT_GAP"]] + [len(ts)]
            for a_, b_ in zip(cut, cut[1:]):
                pts.append({**q, "times": ts[a_:b_], "t0": ts[a_], "t1": ts[b_ - 1]})
        else:
            pts.append(q)
    for q in pts:
        f, run, _ = anatomy(q["times"], p, au.qt)
        C.append(dict(src=q, fault=f, run=run, first=run[0], last=run[-1], hits=len(run), dur=run[-1] - run[0]))
    short = lambda c, n: c["hits"] <= n and c["dur"] <= p["FAULT_DUR"]  # noqa: E731
    rej, keep = [], []
    for i, c in enumerate(C):
        nxt = C[i + 1] if i + 1 < len(C) else None
        prv = keep[-1] if keep else None
        if short(c, p["FAULT_HITS"]) and nxt is not None and nxt["first"] - c["last"] <= p["FAULT_NEXT"] and nxt["fault"] is None:
            nxt["fault"] = c["run"]; rej.append((c, "fault->next")); continue
        if prv is not None and short(c, p["PICKUP_HITS"]) and c["first"] - prv["last"] <= p["PICKUP_GAP"]:
            rej.append((c, "pickup")); continue
        gp = c["first"] - (C[i - 1]["last"] if i else -1e9)
        gn = (nxt["first"] - c["last"]) if nxt is not None else 1e9
        if short(c, p["BREAK_HITS"]) and gp >= p["BREAK_GAP"] and gn >= p["BREAK_GAP"]:
            rej.append((c, "break")); continue
        keep.append(c)
    out = []
    for i, c in enumerate(keep):
        qb = au.quiet_before(c["first"], p["BACK"])
        serve = qb if qb is not None else c["first"]
        A = c["first"] - (p["LEAD_LOUD"] if qb is not None else p["LEAD_NOSERVE"])
        if qb is not None:
            A = min(A, qb - p["LEAD_QUIET"])
        nxt_first = keep[i + 1]["first"] if i + 1 < len(keep) else None
        stop = nxt_first - p["LEAD_LOUD"] - p["MIN_GAP"] if nxt_first else None
        rend, rdb = au.reaction_end(c["last"], serve, p, stop)
        B = c["last"] + p["TAIL"]
        if rend is not None:
            B = max(B, min(rend + 0.5, c["last"] + p["CAP"]))
        if stop is not None:
            B = min(B, max(c["last"] + 0.5, stop))
        if out and A < out[-1]["B"] + p["MIN_GAP"]:
            A = max(A, out[-1]["B"] + p["MIN_GAP"])
        pauses = []                                   # chained pauses (> EXT_GAP between loud hits): skipped by a jump cut
        if p.get("KEEP_HITS"):
            for u, v in zip(c["run"], c["run"][1:]):
                a_, b_ = u + p["PAUSE_PAD"][0], v - p["PAUSE_PAD"][1]
                if v - u > p["EXT_GAP"] and b_ - a_ >= 0.5: pauses.append((a_, b_))
        parts = split_pauses(A, B, pauses)
        if c["fault"] is not None:
            fa = c["fault"][0] - p["FAULT_PART"][0]
            fb = min(c["fault"][-1] + p["FAULT_PART"][1], A)
            if p.get("FAULT_MAX"): fb = min(fb, fa + p["FAULT_MAX"])   # just the serve and the call: a short jump cut
            if out: fa = max(fa, out[-1]["B"] + p["MIN_GAP"])
            if fb > fa: parts.insert(0, (fa, fb))
        src = c["src"]
        out.append({**src, "t0": c["first"], "t1": c["last"], "hits": c["hits"], "A": A, "B": B, "parts": parts,
                    "serve_est": serve, "pauses": pauses, "quiet_serve": qb is not None, "fault_first": c["fault"] is not None,
                    "reaction_db": round(rdb, 1), "reaction_ext": rend is not None,
                    "sources": set(src.get("sources") or {"audio"})})
    return out, rej


def split_pauses(A, B, pauses):
    """[A, B] minus the chained pauses -> list of spans (one span when there are none)."""
    out, a = [], A
    for p0, p1 in sorted(pauses):
        if a < p0 < p1 < B: out.append((a, p0)); a = p1
    return out + [(a, B)]


def finish(clips, KF=None, dur=None):
    """Fields the pipeline / renderer expect (fault_like, serve_only, gap_after, rallies), optional outward keyframe snap
    that never crosses a neighbour, and the end clamp to the video duration. The fault part (if any) stays in `parts`
    for the reel / EDL; the lossless renderer cuts [A, B] minus the chained `pauses` (jump cuts over a mid-rally pause)."""
    import bisect
    out = [dict(c) for c in clips]
    for i, s in enumerate(out):
        if dur is not None: s["B"] = min(s["B"], dur)
        if KF:
            k = bisect.bisect_right(KF, s["A"]) - 1
            A = KF[max(0, k)]
            if i == 0 or A >= out[i - 1]["B"]: s["A"] = A
            j = bisect.bisect_left(KF, s["B"])
            B = KF[j] if j < len(KF) else (dur if dur is not None else s["B"])
            if i + 1 == len(out) or B <= out[i + 1]["A"]: s["B"] = B
        if KF and s.get("pauses"):                  # resume on a keyframe at or before the next hit's lead (stream copy)
            ps = []
            for p0, p1 in s["pauses"]:
                k = bisect.bisect_right(KF, p1) - 1
                p1 = KF[k] if k >= 0 else p1
                if p1 - p0 >= 0.5: ps.append((p0, p1))
            s["pauses"] = ps
        s["parts"] = [p for p in s.get("parts", []) if p[1] <= s["A"] + 1e-9] + split_pauses(s["A"], s["B"], s.get("pauses", ()))
    for i, s in enumerate(out):
        s["fault_like"] = False                     # faults are merged into their point, never their own clip
        s["serve_only"] = s["hits"] == 1
        s["gap_after"] = out[i + 1]["A"] - s["B"] if i + 1 < len(out) else 0.0
        s["rallies"] = [dict(t0=s["t0"], t1=s["t1"], hits=s["hits"], amax=s.get("amax", 0.0), times=list(s.get("times", [])),
                             pause_after=s["gap_after"])]
    return out
