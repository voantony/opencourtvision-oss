"""Match predicted events to ground truth and score them.

Matching: per event type, one-to-one, highest-confidence predictions first, each to the nearest unmatched ground-truth
event within the tolerance (±80 ms for contact/bounce, ±150 ms otherwise).
Reports: precision/recall/F1 per event type and per label value (field=value), a stroke confusion matrix
(serve / forehand / backhand / unknown; a 'none' row for spurious and column for missed), and a confidence calibration table with ECE.
"""
import glob
import json
import os
from collections import defaultdict

from ..events import TIGHT_TYPES, Event
from ..labels.schema import ShotLabel

TOL_TIGHT, TOL_DEFAULT = 0.080, 0.150
STROKES = ("serve", "forehand", "backhand", "unknown")


def tolerance(etype):
    return TOL_TIGHT if etype in TIGHT_TYPES else TOL_DEFAULT


def _ev(x):
    return x if isinstance(x, Event) else Event.from_dict(x)


def match_events(pred, gt):
    """-> (pairs [(p, g)], unmatched_pred, unmatched_gt)."""
    pred = [_ev(p) for p in pred]; gt = [_ev(g) for g in gt]
    pairs, used = [], set()
    for p in sorted(pred, key=lambda e: -e.confidence):
        tol = tolerance(p.type); best, bd = None, None
        for j, g in enumerate(gt):
            if j in used or g.type != p.type: continue
            d = abs(g.t - p.t)
            if d <= tol + 1e-9 and (bd is None or d < bd): best, bd = j, d
        if best is not None: used.add(best); pairs.append((p, gt[best]))
    mp = {id(p) for p, _ in pairs}
    return pairs, [p for p in pred if id(p) not in mp], [g for j, g in enumerate(gt) if j not in used]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / (p + r), 4) if p + r else 0.0)


def stroke_of(label):
    lab = label if isinstance(label, ShotLabel) or label is None else ShotLabel.from_dict(label)
    if lab is None: return "unknown"
    if lab.is_serve: return "serve"
    return str(lab.side) if lab.side else "unknown"


def _label_items(e):
    return {f"{k}={v}" for k, v in (e.label.to_dict().items() if e.label else []) if k != "hand"}


def calibration(pred_conf_correct, bins=5):
    """[(conf, correct)] -> table rows and expected calibration error."""
    rows, n, ece = [], len(pred_conf_correct), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, k) for c, k in pred_conf_correct if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not sel:
            rows.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=0, mean_conf=None, accuracy=None)); continue
        mc = sum(c for c, _ in sel) / len(sel); acc = sum(k for _, k in sel) / len(sel)
        ece += len(sel) / n * abs(mc - acc)
        rows.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=len(sel), mean_conf=round(mc, 3), accuracy=round(acc, 3)))
    return rows, round(ece, 4)


def evaluate(pred, gt, bins=5):
    """pred, gt: lists of Event or dicts. -> report dict."""
    pred = [_ev(p) for p in pred]; gt = [_ev(g) for g in gt]
    pairs, up, ug = match_events(pred, gt)
    types = sorted({e.type for e in pred + gt})
    per_type = {t: prf(sum(p.type == t for p, _ in pairs), sum(p.type == t for p in up), sum(g.type == t for g in ug)) for t in types}
    lab = defaultdict(lambda: [0, 0, 0])
    for p, g in pairs:
        pi, gi = _label_items(p), _label_items(g)
        for k in pi & gi: lab[k][0] += 1
        for k in pi - gi: lab[k][1] += 1
        for k in gi - pi: lab[k][2] += 1
    for p in up:
        for k in _label_items(p): lab[k][1] += 1
    for g in ug:
        for k in _label_items(g): lab[k][2] += 1
    per_label = {k: prf(*v) for k, v in sorted(lab.items())}
    # stroke confusion over contact-like events that carry labels
    has_lab = lambda e: e.label is not None and (e.label.shot_type is not None or e.label.side is not None)
    # rows = truth, cols = predicted; "none" row = spurious predictions, "none" column = missed ground truth
    keys = STROKES + ("none",)
    cm = {r: {c: 0 for c in keys} for r in keys}
    for p, g in pairs:
        if has_lab(g) or has_lab(p): cm[stroke_of(g.label)][stroke_of(p.label)] += 1
    for g in ug:
        if has_lab(g): cm[stroke_of(g.label)]["none"] += 1
    for p in up:
        if has_lab(p): cm["none"][stroke_of(p.label)] += 1
    # calibration: a prediction is "correct" if matched and (when labelled on both sides) its stroke agrees
    mp = {id(p): g for p, g in pairs}
    cc = []
    for p in pred:
        g = mp.get(id(p)); ok = g is not None and (not (has_lab(p) and has_lab(g)) or stroke_of(p.label) == stroke_of(g.label))
        cc.append((min(1.0, max(0.0, p.confidence)), int(ok)))
    cal, ece = calibration(cc, bins)
    errs = [abs(p.t - g.t) * 1000 for p, g in pairs]
    return dict(n_pred=len(pred), n_gt=len(gt), overall=prf(len(pairs), len(up), len(ug)), per_type=per_type, per_label=per_label,
                stroke_confusion=cm, calibration=cal, ece=ece,
                timing_ms=dict(mean=round(sum(errs) / len(errs), 1) if errs else None, max=round(max(errs), 1) if errs else None),
                tolerance_ms=dict(contact_bounce=TOL_TIGHT * 1000, other=TOL_DEFAULT * 1000))


def load_events(path):
    d = json.load(open(path))
    return [Event.from_dict(x) for x in (d["events"] if isinstance(d, dict) else d)]


def evaluate_dirs(pred_path, labels_dir):
    """Pair files by name: labels_dir/*.json (ground truth) with pred_path/<same name> or a single pred file."""
    gts = sorted(glob.glob(os.path.join(labels_dir, "*.json"))) if os.path.isdir(labels_dir) else [labels_dir]
    P, G, used = [], [], []
    for g in gts:
        pp = os.path.join(pred_path, os.path.basename(g)) if os.path.isdir(pred_path) else pred_path
        if not os.path.exists(pp): continue
        P += load_events(pp); G += load_events(g); used.append(os.path.basename(g))
    rep = evaluate(P, G); rep["files"] = used
    return rep


def format_report(rep):
    L = [f"files: {', '.join(rep.get('files', [])) or '-'}   pred {rep['n_pred']}  gt {rep['n_gt']}   "
         f"tolerance ±{rep['tolerance_ms']['contact_bounce']:.0f} ms contact/bounce, ±{rep['tolerance_ms']['other']:.0f} ms other",
         f"overall  P {rep['overall']['precision']:.3f}  R {rep['overall']['recall']:.3f}  F1 {rep['overall']['f1']:.3f}   "
         f"timing mean {rep['timing_ms']['mean']} ms max {rep['timing_ms']['max']} ms", "", "per event type"]
    L += [f"  {t:<14} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}  (tp {v['tp']} fp {v['fp']} fn {v['fn']})" for t, v in rep["per_type"].items()]
    L += ["", "per label"] + [f"  {k:<28} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}" for k, v in rep["per_label"].items()]
    cm = rep["stroke_confusion"]; cols = list(next(iter(cm.values())).keys())
    L += ["", "stroke confusion (rows = truth, cols = predicted; none = spurious row / missed column)", "  " + " " * 10 + "".join(f"{c:>10}" for c in cols)]
    L += ["  " + f"{r:<10}" + "".join(f"{cm[r][c]:>10}" for c in cols) for r in cm]
    L += ["", f"calibration (ECE {rep['ece']})"] + [f"  {r['bin']}  n {r['n']:>4}  conf {r['mean_conf']}  acc {r['accuracy']}" for r in rep["calibration"]]
    return "\n".join(L)
