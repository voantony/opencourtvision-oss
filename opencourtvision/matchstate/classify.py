"""Sequence-aware match-state classifier (19 Oct).

Binary product gate for one-tap recording: keep `match_point`, reject everything else as non-point
for the edit (warm-up / casual rally / post-match). A missed real point is worse than an extra clip,
so the default threshold is recall-biased.

CEO starter signals (challenged; each gets an ablation weight in score_signals):
  serve_ritual, formation_doubles, server_rotation, cooperative_rhythm, between_point_behaviour,
  audio_callouts, score_state_post_match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATES = ("warm_up", "match_point", "between_points", "changeover", "break_casual", "post_match")

# Feature names (stable order for the linear model)
FEAT = (
    "dur_s",
    "n_hits",
    "hit_rate",           # hits / duration
    "med_interhit",
    "pause_before",       # gap from previous segment end
    "pause_after",
    "cover_frac",         # hit span / duration (warmup high)
    "has_score_before",
    "server_known",
    "server_alternates",  # consistent with previous point's server flip expectation
    "long_gap_changeover",  # pause_before > 60 s
    "early_in_file",      # start in first 8 min -> warmup prior
    "late_in_file",       # after decoded match end proxy
    "short_burst",        # < 6 s with few hits (fault/let/feed)
    "cooperative",        # steady inter-hit 0.6–2.5 s, >= 4 hits, no long pause inside
)


@dataclass
class Seg:
    idx: int
    start: float
    end: float
    kind: str | None = None          # point | non_point | None
    winner: str | None = None
    server: str | None = None
    score_before: dict | None = None
    hits: list | None = None         # absolute times of audio contacts inside [start, end]
    file_dur: float | None = None
    match_end_s: float | None = None  # decoded match end; segments after -> post_match prior


def _hits_in(seg: Seg) -> np.ndarray:
    if not seg.hits:
        return np.array([], float)
    h = np.asarray(seg.hits, float)
    return h[(h >= seg.start - 0.05) & (h <= seg.end + 0.05)]


def features_for_segment(seg: Seg, prev: Seg | None = None, next_seg: Seg | None = None) -> dict:
    dur = max(1e-3, float(seg.end - seg.start))
    h = _hits_in(seg)
    n = len(h)
    ih = np.diff(h) if n >= 2 else np.array([], float)
    med_ih = float(np.median(ih)) if len(ih) else 9.0
    cover = float((h[-1] - h[0]) / dur) if n >= 2 else 0.0
    pause_b = (seg.start - prev.end) if prev is not None else seg.start
    pause_a = (next_seg.start - seg.end) if next_seg is not None else 0.0
    file_dur = seg.file_dur or max(seg.end, 1.0)
    match_end = seg.match_end_s
    server_alt = 0.0
    if prev is not None and seg.server and prev.server and prev.kind == "point":
        # tennis doubles: server usually stays for a game; crude: same server often within game
        server_alt = 1.0 if seg.server == prev.server else 0.5
    cooperative = 1.0 if (n >= 4 and 0.6 <= med_ih <= 2.5 and cover >= 0.35) else 0.0
    return dict(
        dur_s=dur,
        n_hits=float(n),
        hit_rate=n / dur,
        med_interhit=med_ih,
        pause_before=float(pause_b),
        pause_after=float(pause_a),
        cover_frac=cover,
        has_score_before=1.0 if seg.score_before else 0.0,
        server_known=1.0 if seg.server else 0.0,
        server_alternates=server_alt,
        long_gap_changeover=1.0 if pause_b >= 60 else 0.0,
        early_in_file=1.0 if seg.start < min(480.0, 0.15 * file_dur) else 0.0,
        late_in_file=1.0 if (match_end is not None and seg.start >= match_end) or seg.start > 0.95 * file_dur else 0.0,
        short_burst=1.0 if dur < 6.0 and n <= 2 else 0.0,
        cooperative=cooperative,
    )


def rule_prior(feat: dict) -> str:
    """Hand rules -> state. Biased to KEEP real points (false reject near zero).

    A missed real point is worse than an extra clip. Only reject when evidence is strong.
    """
    # Strong rejects only
    if feat["late_in_file"] >= 1.0 and feat["n_hits"] <= 2:
        return "post_match"
    if feat["early_in_file"] >= 1.0 and feat["cover_frac"] >= 0.50 and feat["hit_rate"] >= 0.30 and feat["has_score_before"] < 1:
        return "warm_up"
    if feat["long_gap_changeover"] >= 1.0 and feat["n_hits"] <= 1 and feat["dur_s"] < 20:
        return "changeover"
    # Continuous cooperative hitting without a score state -> casual / warm-up
    if feat["cooperative"] >= 1.0 and feat["cover_frac"] >= 0.55 and feat["has_score_before"] < 1 and feat["pause_before"] < 6:
        return "break_casual"
    # Default keep during a match
    if feat["has_score_before"] >= 1 or feat["server_known"] >= 1 or feat["dur_s"] >= 5:
        return "match_point"
    if feat["short_burst"] >= 1.0:
        return "between_points"
    return "match_point"


def vec(feat: dict) -> np.ndarray:
    return np.array([feat[k] for k in FEAT], float)


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """y in {0,1} = is match_point. Returns weights including bias as last element."""
    Xb = np.c_[X, np.ones(len(X))]
    # simple gradient for logistic
    w = np.zeros(Xb.shape[1])
    for _ in range(80):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / len(y) + lam * np.r_[w[:-1], 0]
        w -= 0.4 * g
    return w


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.c_[X, np.ones(len(X))]
    return 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))


def predict_batch(segs: list[Seg], w: np.ndarray | None = None, thresh: float = 0.35) -> list[dict]:
    """Per segment: features, rule state, optional model P(match_point), keep decision."""
    out = []
    for i, s in enumerate(segs):
        prev = segs[i - 1] if i else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        f = features_for_segment(s, prev, nxt)
        st = rule_prior(f)
        row = dict(idx=s.idx, features=f, rule_state=st, truth_kind=s.kind)
        if w is not None:
            p = float(predict_proba(vec(f)[None, :], w)[0])
            row["p_match_point"] = p
            row["keep"] = bool(p >= thresh)
        else:
            row["keep"] = st == "match_point"
        out.append(row)
    return out


def score_signals(segs: list[Seg]) -> dict:
    """Ablation: each CEO signal as a lone keep-rule; measure non-point rejection and false rejection of points."""
    points = [s for s in segs if s.kind == "point"]
    nons = [s for s in segs if s.kind == "non_point"]

    def metrics(keep_idx: set[int]):
        tp = sum(1 for s in points if s.idx in keep_idx)
        fn = sum(1 for s in points if s.idx not in keep_idx)
        tn = sum(1 for s in nons if s.idx not in keep_idx)
        fp = sum(1 for s in nons if s.idx in keep_idx)
        n_pt, n_np = max(1, len(points)), max(1, len(nons))
        # taps/100 ≈ false rejects to restore + extras to delete, per 100 real points
        taps = 100.0 * (fn + fp) / n_pt
        return dict(
            point_recall=round(tp / n_pt, 3),
            false_reject_points=fn,
            nonpoint_reject_rate=round(tn / n_np, 3),
            false_keep_nonpoints=fp,
            taps_per_100=round(taps, 1),
        )

    feats = []
    for i, s in enumerate(segs):
        prev = segs[i - 1] if i else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        feats.append((s, features_for_segment(s, prev, nxt)))

    rules = {
        "always_keep": metrics({s.idx for s in segs}),
        "serve_ritual_proxy": metrics({s.idx for s, f in feats if f["pause_before"] >= 8 and f["n_hits"] >= 1 and f["short_burst"] < 1}),
        "formation_proxy_dur_hits": metrics({s.idx for s, f in feats if f["dur_s"] >= 8 and f["n_hits"] >= 3}),
        "server_known": metrics({s.idx for s, f in feats if f["server_known"] >= 1}),
        "cooperative_rhythm": metrics({s.idx for s, f in feats if f["cooperative"] >= 1}),
        "between_point_long_gap": metrics({s.idx for s, f in feats if f["long_gap_changeover"] < 1 and f["n_hits"] >= 2}),
        "score_state": metrics({s.idx for s, f in feats if f["has_score_before"] >= 1 and f["late_in_file"] < 1}),
        "reject_warmup_cover": metrics({s.idx for s, f in feats if not (f["early_in_file"] >= 1 and f["cover_frac"] >= 0.45)}),
        "rule_prior_keep": metrics({s.idx for s, f in feats if rule_prior(f) == "match_point"}),
    }
    return rules
