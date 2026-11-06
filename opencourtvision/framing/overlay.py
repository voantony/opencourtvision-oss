"""Debug / docs overlay for a framing verdict (the app draws its own in SwiftUI). Boxes only: no faces, nothing saved
unless the caller writes the returned image."""
import numpy as np

from .check import LINES, _proj, normalise_H
from .status import CHECKS

GREEN, RED, AMBER, WHITE, BLUE = (80, 200, 80), (60, 60, 230), (40, 170, 240), (255, 255, 255), (230, 160, 60)


def draw(bgr, v, boxes=None):
    """-> BGR image: court lines from the checked fit (amber = hint, red = firm warning), person boxes, the fix banner on
    top and one chip per check at the bottom."""
    import cv2
    img = np.asarray(bgr).copy()
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    lv = lambda k: v.checks.get(k, {}).get("level", "ok")
    col = dict(ok=GREEN, hint=AMBER, warn=RED)
    court = "warn" if "warn" in (lv("lines"), lv("baselines")) else "hint" if "hint" in (lv("lines"), lv("baselines")) else "ok"
    if v.H is not None:
        Hn = normalise_H(v.H); s = np.linspace(0, 1, 60)[:, None]
        for a, b in LINES.values():
            q, wq = _proj(Hn, (1 - s) * np.array(a) + s * np.array(b))
            q = q[wq > 0]
            if len(q) > 1:
                cv2.polylines(img, [np.round(q).astype(np.int32)], False, col[court], 2, cv2.LINE_AA)
    for b in (boxes if boxes is not None else []):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), BLUE, 1)
    text = ({"warn": "! ", "hint": "Tip: "}.get(v.level, "") + (v.fix or "Looks good. Ready to record.")).replace("°", " deg")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (w, th + 14), (30, 30, 30), -1)
    cv2.putText(img, text, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col.get(v.level, GREEN) if v.fix else GREEN, 1, cv2.LINE_AA)
    x = 6
    for k in CHECKS:
        st = lv(k)
        (cw, ch), _ = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(img, (x, h - ch - 12), (x + cw + 10, h - 4), col[st], -1)
        cv2.putText(img, k, (x + 5, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        x += cw + 16
    return img
