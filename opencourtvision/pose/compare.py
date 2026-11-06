"""Era / group comparison of stroke metrics ("then vs now", player A vs B, before vs after a coaching block).

Method (the general recipe, implemented independently):
1. optionally regress nuisance covariates (camera scale, court position, ...) out of each metric across strokes;
2. one number per session (match) and metric: its mean, so sessions not strokes are the samples;
3. Welch's t-test between the two groups of sessions, Holm correction over the metrics tested;
4. a change is reported only when corrected p < alpha AND |difference| >= min_effect x the stroke-level SD.
"""
import numpy as np
from scipy import stats


def residualize(y, X):
    """y (n,) minus its least-squares fit on covariates X (n, k) (plus intercept), mean added back. NaN rows kept."""
    y = np.asarray(y, float); X = np.asarray(X, float).reshape(len(y), -1)
    ok = np.isfinite(y) & np.isfinite(X).all(1); out = y.copy()
    if ok.sum() > X.shape[1] + 2:
        A = np.c_[np.ones(ok.sum()), X[ok]]; beta = np.linalg.lstsq(A, y[ok], rcond=None)[0]
        out[ok] = y[ok] - A[:, 1:] @ beta[1:]
    return out


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (float("nan"), float("nan"))
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (float(x.mean() - h), float(x.mean() + h))


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def compare(rows, metrics, group_key="group", session_key="session", a=None, b=None, covariates=(), alpha=0.05, min_effect=0.4):
    """rows: per-stroke dicts with group, session and metric values. a/b: the two group values (default: the first
    and last in sorted order). Returns a list of per-metric result dicts."""
    groups = sorted({r[group_key] for r in rows}); a = groups[0] if a is None else a; b = groups[-1] if b is None else b
    res = []
    for m in metrics:
        rr = [r for r in rows if r.get(m) is not None and np.isfinite(r[m])]
        if not rr: continue
        y = np.array([r[m] for r in rr], float)
        if covariates:
            y = residualize(y, [[r.get(c, np.nan) for c in covariates] for r in rr])
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else float("nan")
        sess = {}
        for r, v in zip(rr, y): sess.setdefault((r[group_key], r[session_key]), []).append(v)
        A = np.array([np.mean(v) for (g, _), v in sess.items() if g == a]); B = np.array([np.mean(v) for (g, _), v in sess.items() if g == b])
        if len(A) >= 2 and len(B) >= 2:
            t, p = stats.ttest_ind(B, A, equal_var=False)
            p = 1.0 if not np.isfinite(p) else p                 # identical groups -> no evidence of change
        else:
            t, p = float("nan"), 1.0
        res.append(dict(metric=m, a=a, b=b, n_sessions_a=len(A), n_sessions_b=len(B), mean_a=float(A.mean()) if len(A) else None,
                        mean_b=float(B.mean()) if len(B) else None, ci_a=ci95(A), ci_b=ci95(B), diff=float(B.mean() - A.mean()) if len(A) and len(B) else None,
                        stroke_sd=sd, t=float(t), p=float(p)))
    if res:
        for r, pa in zip(res, holm([r["p"] for r in res])):
            r["p_holm"] = float(pa); r["stars"] = stars(pa)
            r["changed"] = bool(pa < alpha and r["diff"] is not None and np.isfinite(r["stroke_sd"]) and abs(r["diff"]) >= min_effect * r["stroke_sd"])
    return res
