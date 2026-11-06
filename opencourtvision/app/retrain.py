"""Personal on-device adapter fine-tune (overnight while charging) + held-out eval gate + auto-rollback.

v1 trains a tiny logistic / linear adapter over frozen embeddings of the user's fixes. No cloud GPU. Private labels and
the resulting personal weights stay in session/models/ unless the user opts in to share (share.py).
Eval gate: accuracy on a held-out 20% of fixes must not drop vs the previous adapter; otherwise we keep the old one.
"""
import os
import time

import numpy as np


def _examples(session):
    from .review import list_fixes
    return [e for e in list_fixes(session) if e.get("kind") in ("point_winner", "bounce", "player_id")]


def _embed(ex):
    """Deterministic bag-of-features embedding (no faces). Real Core ML / MLX adapters replace this on device."""
    kind = {"point_winner": 0, "bounce": 1, "player_id": 2}.get(ex.get("kind"), 3)
    tags = ex.get("struggle_tags") or []
    tag_vec = [1.0 if t in tags else 0.0 for t in
               ("low_light", "shaky", "behind_fence", "portrait_phone", "narrow_fov", "low_resolution", "no_audio")]
    t = float(ex.get("t") or 0.0)
    return np.array([kind, t / 600.0, len(tags), *tag_vec], dtype=np.float64)


def _label(ex):
    v = ex.get("value")
    if ex.get("kind") == "point_winner":
        return 1.0 if v == "A" else 0.0 if v == "B" else 0.5
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, dict) and "x" in v: return float(v["x"]) / 3840.0
    return 0.0


def train_adapter(session, min_examples=8):
    """Fit a ridge-linear adapter. Returns dict(ok, path, n, eval, rolled_back)."""
    ex = _examples(session)
    if len(ex) < min_examples:
        return dict(ok=False, reason=f"need >={min_examples} fixes, have {len(ex)}", n=len(ex))
    X = np.stack([_embed(e) for e in ex]); y = np.array([_label(e) for e in ex])
    # held-out split (last 20%, time-ordered by created)
    n = len(ex); k = max(1, n // 5)
    Xtr, Xte, ytr, yte = X[:-k], X[-k:], y[:-k], y[-k:]
    # ridge
    A = Xtr.T @ Xtr + 1e-2 * np.eye(Xtr.shape[1]); w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    mae = float(np.mean(np.abs(pred - yte)))
    # accuracy proxy for classification-like kinds
    acc = float(np.mean((pred > 0.5) == (yte > 0.5)))
    metrics = dict(mae=round(mae, 4), acc=round(acc, 4), n_train=int(n - k), n_test=int(k))
    prev = session.get("personal_model") or {}
    prev_acc = (prev.get("eval") or {}).get("acc")
    if prev_acc is not None and acc + 1e-9 < float(prev_acc):
        # auto-rollback: keep previous weights
        return dict(ok=True, rolled_back=True, reason="held-out acc dropped; kept previous adapter",
                    n=n, eval=metrics, path=prev.get("path"), previous_eval=prev.get("eval"))
    out = os.path.join(session["paths"]["models"], f"adapter_{int(time.time())}.npz")
    np.savez(out, w=w, metrics=metrics, n=n, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    session["personal_model"] = dict(path=out, last_trained=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     n_examples=n, eval=metrics)
    return dict(ok=True, rolled_back=False, n=n, eval=metrics, path=out)
