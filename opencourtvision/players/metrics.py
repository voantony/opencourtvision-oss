"""ID metrics against (proxy) truth. Recall: share of the 2 expected people per side that got an id (false ones
excluded). False players: ids put on someone who is not one of the side's 2 truth people (frames with full truth).
 truth: {frame_index: {det_index: id}}; preds: list (per frame) of {det: slot}."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

IDS = ("A1", "A2", "B1", "B2")


def evaluate(frames, preds, truth, frame_filter=None, n_points=None):
    fidx = [i for i in range(len(frames)) if frame_filter is None or frame_filter(frames[i])]
    corr = defaultdict(int); tot = defaultdict(int); seq = defaultdict(list); false = 0; nfs = 0; rec = {0: [], 1: []}
    for fi in fidx:
        f = frames[fi]; pr = preds[fi]; tr = truth.get(fi, {})
        for d, s in tr.items():
            tot[s] += 1; corr[s] += int(pr.get(d) == s); seq[s].append(pr.get(d))
        for s in (0, 1):
            outs = [d for d in pr if f["dets"][d]["side"] == s]
            tside = [d for d in tr if f["dets"][d]["side"] == s]
            bad = [d for d in outs if d not in tr and len(tside) == 2]       # only measurable where the side has full truth
            false += len(bad); nfs += len(tside) == 2; rec[s].append(min(2, len(outs) - len(bad)) / 2)
    sw = 0
    for s, v in seq.items():
        v = [x for x in v if x is not None]
        sw += sum(1 for a, b in zip(v, v[1:]) if a != b)
    npts = n_points or len({frames[i]["point"] for i in fidx})
    allc = sum(corr.values()); allt = sum(tot.values())
    return dict(frames=len(fidx), truth_dets=allt, id_acc=allc / allt if allt else float("nan"),
                per_player={s: (corr[s] / tot[s] if tot[s] else float("nan")) for s in IDS},
                switches_per_100pts=100.0 * sw / max(1, npts), switches=sw,
                false_per_100_frames=100.0 * false / max(1, nfs / 2),               # per 100 frames with full truth

                recall_near=float(np.mean(rec[0])) if rec[0] else float("nan"), recall_far=float(np.mean(rec[1])) if rec[1] else float("nan"))
