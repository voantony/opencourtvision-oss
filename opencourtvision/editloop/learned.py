"""Learned onset and stretch classifiers for point segmentation (v4.1; eng notes §2.8.7d).

Small and on-device friendly (N1): a 60-tree depth-3 GBDT (or logistic regression) in numpy, no third-party runtime.
Audio + timing only, so the same models run in the US build; no ball, player or motion input anywhere here.

1. Onsets: every audio onset >= ONSET_TH dB (well under the 14 dB hit threshold, so quiet far-side serves are kept).
   Per onset: level, rise, spectral shape of the first 60 ms (band energies, centroid, decay) and timing context
   (gaps to neighbours, onset density, time since the last >= 6 s reset, local level rank).
2. Onset models: p_inplay (between the serve and the last contact of a played rally: hits and in-play bounces, versus
   pre-serve bouncing, pick-up, walk-back, feeds and other-court hits) and p_serve.
3. Stretches: in-play onsets chained across gaps <= CHAIN s. Start = most serve-like onset at or just before the
   stretch; end = last in-play onset. Stretch model: point / first-serve fault or let / junk (changeover hits etc.).
   Faults are attached to the next point as an optional lead-in part; junk is dropped. Boundaries are audio-derived.
4. Hybrid (`hybrid_points`): keep the v4.0 smart audio candidates and use the onset models only to re-derive serve /
   last contact inside each one, plus an optional candidate-level point / non-point filter.

STATUS: experimental, not wired into the pipeline. Trained on the one labelled match (dev_match), split-half, the models do
not generalise to the other half (eng notes §2.8.7d): the standalone path loses recall, the hybrid filter loses recall,
the hybrid refine cuts dead time but not cut-offs. Weights are derived from private labels and are never committed;
train with benchmarks/train_point_classifiers.py --out <private dir>.
"""
import json

import numpy as np

from ..detection.audio import detect_hits

ONSET_TH, ONSET_RISE, SR = 8.0, 3.0, 16000
SEG = dict(p_in=0.5, p_serve=0.35, chain=3.0, serve_back=2.5, lead=1.2, tail=2.8, react_db=6.0, cap=4.0,
           p_point=0.5, fault_next=20.0, fault_part=(1.0, 1.5), min_gap=0.2)
ONSET_FEATURES = ["amp", "rise", "lb_rel", "b1", "b2", "b3", "b4", "cent", "decay", "dt_prev", "dt_next",
                  "dt_prev_loud", "dt_next_loud", "n_before3", "n_after3", "n_win8", "since_reset", "until_reset",
                  "amp_rank5", "amp_rel5", "loud", "amp_prev", "amp_next", "amp_prev2", "amp_next2", "dt_prev2", "dt_next2",
                  "loud_before3", "loud_after3"]


def onsets(le, bg):
    t, a = detect_hits(le, bg, th=ONSET_TH, min_rise=ONSET_RISE)
    return np.asarray(t, float), np.asarray(a, float)


def _spectral(x, t):
    """Band energy shares (0.2-1, 1-2, 2-4, 4-8 kHz), centroid (kHz) and decay (dB, 20-60 ms vs 0-20 ms) per onset."""
    out = np.zeros((len(t), 6))
    if x is None:
        return out
    freqs = np.fft.rfftfreq(960, 1 / SR)
    bands = [(200, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    win = np.hanning(960)
    for k, v in enumerate(t):
        i = int((v - 0.01) * SR)
        seg = x[max(0, i):max(0, i) + 960]
        if len(seg) < 960:
            continue
        P = np.abs(np.fft.rfft(seg * win)) ** 2 + 1e-12
        tot = P[(freqs >= 200)].sum()
        for j, (a, b) in enumerate(bands):
            out[k, j] = P[(freqs >= a) & (freqs < b)].sum() / tot
        out[k, 4] = float((P * freqs).sum() / P.sum()) / 1000
        e0 = (seg[:320] ** 2).mean() + 1e-12; e1 = (seg[320:960] ** 2).mean() + 1e-12
        out[k, 5] = 10 * np.log10(e1 / e0)
    return out


def onset_features(le, bg, lb, x=None, t=None, a=None):
    """(t, amp, X) with X[:, ONSET_FEATURES]."""
    if t is None:
        t, a = onsets(le, bg)
    n = len(t)
    if n == 0:
        return t, a, np.zeros((0, len(ONSET_FEATURES)))
    fr = lambda v: np.clip((np.asarray(v) * 100).astype(int), 0, len(le) - 1)  # noqa: E731
    rise = le[fr(t)] - np.maximum.reduce([le[fr(t - d)] for d in (0.03, 0.04, 0.05)])
    lbs = np.convolve(lb, np.ones(1000) / 1000, "same")
    loud = (a >= 14.0).astype(float)
    tl = t[loud > 0]
    dprev = np.r_[30.0, np.diff(t)]; dnext = np.r_[np.diff(t), 30.0]
    il = np.searchsorted(tl, t)
    dpl = np.where(il > 0, t - tl[np.maximum(il - 1, 0)], 30.0)
    jl = np.searchsorted(tl, t, side="right")
    dnl = np.where(jl < len(tl), tl[np.minimum(jl, len(tl) - 1)] - t, 30.0)
    cnt = lambda lo, hi: np.searchsorted(t, t + hi) - np.searchsorted(t, t + lo)  # noqa: E731
    reset_idx = np.where(dprev >= 6.0)[0]
    since = t - t[reset_idx[np.searchsorted(reset_idx, np.arange(n), side="right") - 1]]
    resets_next = np.where(dnext >= 6.0)[0]
    nx = resets_next[np.minimum(np.searchsorted(resets_next, np.arange(n)), len(resets_next) - 1)] if len(resets_next) else np.full(n, n - 1)
    until = t[nx] - t
    rank = np.zeros(n); rel = np.zeros(n)
    for k in range(n):
        lo, hi = np.searchsorted(t, t[k] - 5), np.searchsorted(t, t[k] + 5)
        w = a[lo:hi]; rank[k] = (w > a[k]).sum(); rel[k] = a[k] - w.max()
    sh = lambda v, k, fill: np.r_[np.full(k, fill), v[:-k]] if k > 0 else np.r_[v[-k:], np.full(-k, fill)]  # noqa: E731
    cntl = lambda lo, hi: np.searchsorted(tl, t + hi) - np.searchsorted(tl, t + lo)  # noqa: E731
    nb = np.column_stack([sh(a, 1, 0.0), sh(a, -1, 0.0), sh(a, 2, 0.0), sh(a, -2, 0.0),
                          np.minimum(t - sh(t, 2, -1e9), 30), np.minimum(sh(t, -2, 1e9) - t, 30), cntl(-3, -0.01), cntl(0.01, 3)])
    X = np.column_stack([a, rise, lb[fr(t)] - lbs[fr(t)], _spectral(x, t), np.minimum(dprev, 30), np.minimum(dnext, 30),
                         np.minimum(dpl, 30), np.minimum(dnl, 30), cnt(-3, -0.01), cnt(0.01, 3), cnt(-8, 8),
                         np.minimum(since, 60), np.minimum(until, 60), rank, rel, loud, nb])
    return t, a, X


class LR:
    """Standardised logistic regression with L2 and class balancing (numpy gradient descent)."""

    def __init__(self, l2=1e-2, iters=3000, lr=0.2):
        self.l2, self.iters, self.lr = l2, iters, lr

    @staticmethod
    def expand(X):
        Z = np.asarray(X, float)
        return np.column_stack([Z, np.sign(Z) * np.log1p(np.abs(Z))])

    def fit(self, X, y, expand=True):
        self.ex = expand
        Z = self.expand(X) if expand else np.asarray(X, float)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        Z = (Z - self.mu) / self.sd
        y = np.asarray(y, float)
        wpos = 0.5 / max(y.mean(), 1e-3); wneg = 0.5 / max(1 - y.mean(), 1e-3)
        sw = np.where(y > 0, wpos, wneg)
        self.w = np.zeros(Z.shape[1]); self.b = 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = sw * (p - y)
            self.w -= self.lr * (Z.T @ g / len(y) + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict(self, X):
        Z = self.expand(X) if self.ex else np.asarray(X, float)
        return 1 / (1 + np.exp(-(((Z - self.mu) / self.sd) @ self.w + self.b)))

    def to_json(self):
        return dict(kind="lr", ex=self.ex, mu=self.mu.tolist(), sd=self.sd.tolist(), w=self.w.tolist(), b=self.b)

    @classmethod
    def from_json(cls, d):
        m = cls(); m.ex, m.mu, m.sd, m.w, m.b = d["ex"], np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"]), d["b"]
        return m


class GBDT:
    """Small gradient-boosted trees for binary log-loss (numpy; histogram splits on <= 32 quantile bins). Default
    60 trees of depth 3: a few thousand thresholds, cheap on a phone, no third-party runtime."""

    def __init__(self, n=60, depth=3, lr=0.1, bins=32, min_leaf=20, l2=1.0):
        self.n, self.depth, self.lr, self.bins, self.min_leaf, self.l2 = n, depth, lr, bins, min_leaf, l2

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.bins + 1)[1:-1])) for j in range(X.shape[1])]
        Q = np.column_stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges)])
        w = np.where(y > 0, 0.5 / max(y.mean(), 1e-3), 0.5 / max(1 - y.mean(), 1e-3))
        p0 = float(np.clip(np.average(y, weights=w), 1e-3, 1 - 1e-3)); self.b = np.log(p0 / (1 - p0))
        F = np.full(len(y), self.b); self.trees = []
        for _ in range(self.n):
            p = 1 / (1 + np.exp(-F)); g = w * (p - y); h = w * p * (1 - p)
            tree = self._grow(Q, g, h, np.arange(len(y)), 0)
            F += self.lr * self._apply(tree, Q); self.trees.append(tree)
        return self

    def _grow(self, Q, g, h, idx, d):
        G, H = g[idx].sum(), h[idx].sum()
        leaf = ("leaf", -G / (H + self.l2))
        if d >= self.depth or len(idx) < 2 * self.min_leaf:
            return leaf
        best = (0.0, None)
        base = G * G / (H + self.l2)
        for j in range(Q.shape[1]):
            nb = len(self.edges[j]) + 1
            if nb < 2: continue
            gs = np.bincount(Q[idx, j], g[idx], nb); hs = np.bincount(Q[idx, j], h[idx], nb); cs = np.bincount(Q[idx, j], None, nb)
            gl, hl, cl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1], np.cumsum(cs)[:-1]
            ok = (cl >= self.min_leaf) & (len(idx) - cl >= self.min_leaf)
            if not ok.any(): continue
            gain = gl ** 2 / (hl + self.l2) + (G - gl) ** 2 / (H - hl + self.l2) - base
            gain[~ok] = -1
            k = int(np.argmax(gain))
            if gain[k] > best[0]: best = (float(gain[k]), (j, k))
        if best[1] is None:
            return leaf
        j, k = best[1]
        m = Q[idx, j] <= k
        return ("split", j, k, self._grow(Q, g, h, idx[m], d + 1), self._grow(Q, g, h, idx[~m], d + 1))

    def _apply(self, tree, Q):
        out = np.zeros(len(Q))
        stack = [(tree, np.arange(len(Q)))]
        while stack:
            nd, ii = stack.pop()
            if nd[0] == "leaf": out[ii] = nd[1]; continue
            m = Q[ii, nd[1]] <= nd[2]
            stack += [(nd[3], ii[m]), (nd[4], ii[~m])]
        return out

    def predict(self, X):
        X = np.asarray(X, float)
        Q = np.column_stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges)])
        F = self.b + self.lr * sum(self._apply(tr, Q) for tr in self.trees)
        return 1 / (1 + np.exp(-F))

    def to_json(self):
        enc = lambda nd: ["leaf", nd[1]] if nd[0] == "leaf" else ["split", nd[1], nd[2], enc(nd[3]), enc(nd[4])]  # noqa: E731
        return dict(kind="gbdt", lr=self.lr, b=self.b, edges=[e.tolist() for e in self.edges], trees=[enc(t) for t in self.trees])

    @classmethod
    def from_json(cls, d):
        dec = lambda nd: ("leaf", nd[1]) if nd[0] == "leaf" else ("split", nd[1], nd[2], dec(nd[3]), dec(nd[4]))  # noqa: E731
        m = cls(); m.lr, m.b = d["lr"], d["b"]; m.edges = [np.array(e) for e in d["edges"]]; m.trees = [dec(t) for t in d["trees"]]
        return m


def model_from_json(d):
    return GBDT.from_json(d) if d.get("kind") == "gbdt" else LR.from_json(d)


STRETCH_FEATURES = ["n", "dur", "p_in_mean", "p_in_max", "p_serve", "serve_amp", "amax", "gap_prev", "gap_next",
                    "next_serve_amp_diff", "n_loud", "ioi_med", "since_reset"]


def stretches(t, a, X, p_in, p_serve, s=None):
    """In-play onset stretches with their serve onset. Returns list of dicts (indices into t)."""
    s = {**SEG, **(s or {})}
    idx = np.where(p_in >= s["p_in"])[0]
    out = []
    for k in idx:
        if out and t[k] - t[out[-1]["idx"][-1]] <= s["chain"]:
            out[-1]["idx"].append(int(k))
        else:
            out.append(dict(idx=[int(k)]))
    for st in out:
        i0 = st["idx"][0]
        lo = np.searchsorted(t, t[i0] - s["serve_back"])
        cand = list(range(lo, i0 + 1))
        best = max(cand, key=lambda j: (p_serve[j], -j))
        st["serve"] = best if p_serve[best] >= s["p_serve"] else i0
        st["t0"], st["t1"] = float(t[st["serve"]]), float(t[st["idx"][-1]])
    return out


def stretch_features(S, t, a, X, p_in, p_serve):
    F = []
    for k, st in enumerate(S):
        ii = st["idx"]; ts = t[ii]
        nxt = S[k + 1] if k + 1 < len(S) else None
        F.append([len(ii), st["t1"] - st["t0"], float(p_in[ii].mean()), float(p_in[ii].max()), float(p_serve[st["serve"]]),
                  float(a[st["serve"]]), float(a[ii].max()), min(60.0, st["t0"] - S[k - 1]["t1"]) if k else 60.0,
                  min(60.0, nxt["t0"] - st["t1"]) if nxt else 60.0, float(a[nxt["serve"]] - a[st["serve"]]) if nxt else 0.0,
                  float((a[ii] >= 14).sum()), float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0,
                  float(X[st["serve"], ONSET_FEATURES.index("since_reset")])])
    return np.asarray(F, float).reshape(-1, len(STRETCH_FEATURES))


class Models:
    def __init__(self, m_in, m_serve, m_point, m_fault):
        self.m_in, self.m_serve, self.m_point, self.m_fault = m_in, m_serve, m_point, m_fault

    def save(self, path):
        json.dump({k: getattr(self, k).to_json() for k in ("m_in", "m_serve", "m_point", "m_fault")} |
                  dict(onset_features=ONSET_FEATURES, stretch_features=STRETCH_FEATURES), open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(*(model_from_json(d[k]) for k in ("m_in", "m_serve", "m_point", "m_fault")))


def learned_points(le, bg, lb, x, models, s=None, feats=None):
    """Audio -> clips (A, B, parts, sources={'audio'}) with the learned classifiers. Returns (clips, debug)."""
    from .smart import SMART, Audio
    s = {**SEG, **(s or {})}
    t, a, X = feats if feats is not None else onset_features(le, bg, lb, x)
    p_in, p_sv = models.m_in.predict(X), models.m_serve.predict(X)
    S = stretches(t, a, X, p_in, p_sv, s)
    if not S:
        return [], dict(S=S)
    SF = stretch_features(S, t, a, X, p_in, p_sv)
    pp, pf = models.m_point.predict(SF), models.m_fault.predict(SF)
    au = Audio(le, bg, lb, {**SMART, "QUIET_TH": ONSET_TH, "QUIET_RISE": ONSET_RISE})
    keep, faults = [], {}
    for k, st in enumerate(S):
        st["p_point"], st["p_fault"] = float(pp[k]), float(pf[k])
        if pf[k] >= 0.5 and pf[k] > pp[k] and k + 1 < len(S) and S[k + 1]["t0"] - st["t1"] <= s["fault_next"]:
            faults[k + 1] = st; st["kind"] = "fault"
        elif pp[k] >= s["p_point"]:
            st["kind"] = "point"; keep.append(k)
        else:
            st["kind"] = "junk"
    out = []
    for n, k in enumerate(keep):
        st = S[k]
        A = st["t0"] - s["lead"]
        nxt = S[keep[n + 1]]["t0"] - s["lead"] - s["min_gap"] if n + 1 < len(keep) else None
        rend, rdb = au.reaction_end(st["t1"], st["t0"], {**SMART, "REACT_DB": s["react_db"]}, nxt)
        B = st["t1"] + s["tail"]
        if rend is not None: B = max(B, min(rend + 0.5, st["t1"] + s["cap"]))
        if nxt is not None: B = min(B, max(st["t1"] + 0.5, nxt))
        if out and A < out[-1]["B"] + s["min_gap"]: A = out[-1]["B"] + s["min_gap"]
        parts = [(A, B)]
        f = faults.get(k)
        if f is not None and s["fault_part"]:
            fa, fb = f["t0"] - s["fault_part"][0], min(f["t1"] + s["fault_part"][1], A)
            if out: fa = max(fa, out[-1]["B"] + s["min_gap"])
            if fb > fa: parts.insert(0, (fa, fb))
        ts = [float(v) for v in t[st["idx"]]]
        out.append(dict(t0=st["t0"], t1=st["t1"], hits=len(ts), times=ts, amax=float(a[st["idx"]].max()), A=A, B=B, parts=parts,
                        serve_est=st["t0"], fault_first=f is not None, reaction_db=round(rdb, 1), reaction_ext=rend is not None,
                        p_point=st["p_point"], sources={"audio"}))
    return out, dict(S=S, t=t, a=a, p_in=p_in, p_serve=p_sv)


CAND_FEATURES = ["hits", "dur", "fault_first", "quiet_serve", "reaction_db", "gap_prev", "gap_next", "p_in_max", "p_in_mean",
                 "n_in", "p_serve_max", "span_in", "amax"]
REFINE = dict(p_in=0.5, chain=3.0, serve_back=2.5, p_serve=0.35, lead=1.2, tail=2.8, cap=4.0, min_gap=0.2, margin=3.0,
              union=False)


def _chain(t, p_in, lo, hi, seed0, seed1, s):
    """Maximal chain of in-play onsets (p_in >= s['p_in'], gaps <= chain) that overlaps the seed run [seed0, seed1]."""
    ii = [k for k in range(np.searchsorted(t, lo), np.searchsorted(t, hi)) if p_in[k] >= s["p_in"]]
    if not ii:
        return None
    runs = []
    for k in ii:
        if runs and t[k] - t[runs[-1][-1]] <= s["chain"]: runs[-1].append(k)
        else: runs.append([k])
    ov = [r for r in runs if t[r[0]] <= seed1 + 0.3 and t[r[-1]] >= seed0 - 0.3]
    if not ov:
        return None
    return [k for r in ov for k in r]


def refine(clips, t, a, p_in, p_serve, lb=None, s=None):
    """Re-derive [serve, last contact] of each audio candidate from the onset classifiers (the played-rally choice that
    caused the cut-offs), then re-pad: start = serve - lead, end = last in-play onset + tail (reaction-capped as before).
    Falls back to the candidate's own bounds when no in-play onset overlaps it. Audio only."""
    s = {**REFINE, **(s or {})}
    out = []
    for i, c in enumerate(clips):
        lo = max(c["A"], out[-1]["B"] if out else -1e9) - s["margin"]
        hi = c["B"] + s["margin"]
        if i + 1 < len(clips): hi = min(hi, clips[i + 1]["t0"] - 0.3)
        ch = _chain(t, p_in, lo, hi, c["t0"], c["t1"], s)
        d = dict(c)
        if ch:
            first, last = t[ch[0]], t[ch[-1]]
            j0 = np.searchsorted(t, first - s["serve_back"]); j1 = ch[0] + 1
            cand = range(j0, j1)
            k = max(cand, key=lambda j: (p_serve[j], -j))
            serve = t[k] if p_serve[k] >= s["p_serve"] else first
            ext = c["B"] - (c["t1"] + s["tail"])          # reaction extension the candidate already had
            A_, B_ = float(serve - s["lead"]), float(last + s["tail"] + max(0.0, min(ext, s["cap"] - s["tail"])))
            if s["union"]:                                  # safety: never start later / end earlier than the audio rule
                A_, B_ = min(A_, c["A"]), max(B_, c["B"])
            d.update(t0=float(min(serve, c["t0"]) if s["union"] else serve), t1=float(max(last, c["t1"]) if s["union"] else last),
                     A=A_, B=B_, refined=True,
                     p_in_max=float(p_in[ch].max()), p_in_mean=float(p_in[ch].mean()), n_in=len(ch),
                     p_serve_max=float(p_serve[k]), span_in=float(last - first))
        else:
            d.update(refined=False, p_in_max=0.0, p_in_mean=0.0, n_in=0, p_serve_max=0.0, span_in=0.0)
        if out and d["A"] < out[-1]["B"] + s["min_gap"]:
            d["A"] = out[-1]["B"] + s["min_gap"]
        if d["B"] <= d["A"] + 0.5: d["B"] = d["A"] + 0.5
        parts = list(c.get("parts") or [])[:-1]
        parts = [(pa, min(pb, d["A"])) for pa, pb in parts if min(pb, d["A"]) > pa]
        d["parts"] = parts + [(d["A"], d["B"])]
        out.append(d)
    return out


def cand_features(clips):
    F = []
    for i, c in enumerate(clips):
        F.append([c["hits"], c["t1"] - c["t0"], float(bool(c.get("fault_first"))), float(bool(c.get("quiet_serve"))),
                  float(c.get("reaction_db") or 0.0), min(60.0, c["t0"] - clips[i - 1]["t1"]) if i else 60.0,
                  min(60.0, clips[i + 1]["t0"] - c["t1"]) if i + 1 < len(clips) else 60.0, c["p_in_max"], c["p_in_mean"],
                  c["n_in"], c["p_serve_max"], c["span_in"], float(c.get("amax") or 0.0)])
    return np.asarray(F, float).reshape(-1, len(CAND_FEATURES))


def hybrid_points(clips, t, a, p_in, p_serve, m_cand=None, p_keep=0.5, s=None):
    """Smart audio candidates -> learned refinement -> optional learned point/non-point filter. Returns (kept, dropped)."""
    R = refine(clips, t, a, p_in, p_serve, s=s)
    if m_cand is None or not R:
        return R, []
    pk = m_cand.predict(cand_features(R))
    keep, drop = [], []
    for c, p in zip(R, pk):
        c["p_point"] = float(p)
        (keep if p >= p_keep else drop).append(c)
    for i, c in enumerate(keep):                       # re-clamp neighbours after drops (never overlap)
        if i and c["A"] < keep[i - 1]["B"] + REFINE["min_gap"]:
            c["A"] = keep[i - 1]["B"] + REFINE["min_gap"]; c["parts"] = [p for p in c["parts"][:-1] if p[1] <= c["A"]] + [(c["A"], c["B"])]
    return keep, drop
