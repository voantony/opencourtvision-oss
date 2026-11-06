"""Per-point court refit for camera drift (the dev_match phone crept up to ~107 px mid-match, PR #29).

Each view has one court fit (prelabels courts_by_view, view 0 replaced by the manual tap). For every point we grab a
1280x720 frame near its end and estimate the drift from the view's reference frame with ORB + RANSAC homography D
(reference proxy px -> point proxy px). The point's fit is H_pt = H_view . S^-1 . D^-1 . S (S: source px -> proxy).
The reference frame of a view is the frame on which the stale fit has the best court-line support; a refit is kept only
when its line support is higher than the stale fit's on that point's frame.

  court_refit.py --frames DIR --labels L.json --prelabels P.json [--court_override F.json] --out refit.json
  (frames DIR/p{idx:03d}b.jpg, as ballbench grab.sh writes them; private, never committed)

Output (no labels, no pixels): per point view, drift_px (max court-corner shift, proxy px), inliers, line support of
the stale and refit H, and the chosen H_img2court (source px). pw_court_refit.py feeds it to pw_stages.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

SCALE = 3.0       # source (3840x2160) / proxy (1280x720)
W, LEN = 10.97, 23.77
LINES = [((0, 0), (W, 0)), ((0, LEN), (W, LEN)), ((0, 0), (0, LEN)), ((W, 0), (W, LEN)), ((1.37, 0), (1.37, LEN)),
         ((9.6, 0), (9.6, LEN)), ((1.37, 5.485), (9.6, 5.485)), ((1.37, 18.285), (9.6, 18.285)),
         ((5.485, 5.485), (5.485, 18.285))]
S = np.diag([1 / SCALE, 1 / SCALE, 1.0])
SI = np.linalg.inv(S)


def apply_court_override(cbv, path):
    cbv = [dict(c) for c in cbv]
    if path:
        ov = json.load(open(path)); ov = ov if isinstance(ov, list) else [ov]
        for o in ov:
            cbv[o["view"]].update(H_img2court=o["H_img2court"], manual=True)
    return cbv


def court_to_proxy(H_i2c, pts):
    Hi = S @ np.linalg.inv(np.asarray(H_i2c, float))
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hi.T
    return p[:, :2] / p[:, 2:3]


def line_mask(gray):
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    return cv2.dilate((th > 18).astype(np.uint8), np.ones((3, 3), np.uint8))


def line_support(H_i2c, mask):
    """Share of in-image samples along the court model lines that sit on a bright thin line."""
    h, w = mask.shape; tot = hit = 0
    for a, b in LINES:
        s = np.linspace(0, 1, 60)[:, None]
        q = court_to_proxy(H_i2c, (1 - s) * np.array(a) + s * np.array(b))
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        q = q[ok].astype(int); tot += len(q); hit += int(mask[q[:, 1], q[:, 0]].sum())
    return hit / tot if tot else 0.0


def drift(ref, img, orb, bf):
    k1, d1 = orb.detectAndCompute(ref, None); k2, d2 = orb.detectAndCompute(img, None)
    if d1 is None or d2 is None:
        return None, 0
    m = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 30:
        return None, len(good)
    p1 = np.float32([k1[g.queryIdx].pt for g in good]); p2 = np.float32([k2[g.trainIdx].pt for g in good])
    D, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    return D, int(inl.sum()) if inl is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--prelabels", required=True); ap.add_argument("--court_override")
    ap.add_argument("--suffix", default="b"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(a.prelabels)); cbv = apply_court_override(P["courts_by_view"], a.court_override)
    pts = [p for p in json.load(open(a.labels))["points"] if p.get("kind") == "point"]
    orb = cv2.ORB_create(6000); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    corners = [(0, 0), (W, 0), (W, LEN), (0, LEN)]
    frames, by_view = {}, {}
    for p in pts:
        f = os.path.join(a.frames, f"p{p['idx']:03d}{a.suffix}.jpg")
        if not os.path.exists(f):
            continue
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        v = next((c["view"] for c in cbv if c["t0"] <= p["end"] < c["t1"]), cbv[-1]["view"])
        frames[p["idx"]] = (g, line_mask(g), v); by_view.setdefault(v, []).append(p["idx"])
    out = dict(note="per-point court refit for camera drift; proxy px 1280x720, H_img2court in source px", views={}, points=[])
    for v, idxs in sorted(by_view.items()):
        H = np.asarray(cbv[v]["H_img2court"], float)
        ls = {i: line_support(H, frames[i][1]) for i in idxs}
        ref = max(ls, key=ls.get)
        out["views"][str(v)] = dict(ref_point=ref, ref_line_support=round(ls[ref], 3), n=len(idxs), manual=bool(cbv[v].get("manual")))
        c_ref = court_to_proxy(H, corners)
        for i in idxs:
            g, mk, _ = frames[i]
            D, ninl = (np.eye(3), 0) if i == ref else drift(frames[ref][0], g, orb, bf)
            row = dict(idx=i, view=v, inliers=ninl, ls_stale=round(ls[i], 3))
            if D is None:
                row.update(drift_px=None, ls_refit=None, refit=False, H_img2court=H.tolist())
            else:
                Hn = H @ SI @ np.linalg.inv(D) @ S; Hn = Hn / Hn[2, 2]
                c_new = court_to_proxy(Hn, corners)
                lsn = line_support(Hn, mk)
                keep = lsn > ls[i] + 0.02
                row.update(drift_px=round(float(np.max(np.linalg.norm(c_new - c_ref, axis=1))), 1), ls_refit=round(lsn, 3),
                           refit=bool(keep), H_img2court=(Hn if keep else H).tolist(), H_refit=Hn.tolist())
            out["points"].append(row)
    json.dump(out, open(a.out, "w"), indent=1)
    d = np.array([r["drift_px"] for r in out["points"] if r["drift_px"] is not None])
    print(f"{len(out['points'])} points, drift px median {np.median(d):.1f} p90 {np.percentile(d, 90):.1f} max {d.max():.1f}; "
          f"refit kept {sum(r['refit'] for r in out['points'])}; views {out['views']}", file=sys.stderr)


if __name__ == "__main__":
    main()
