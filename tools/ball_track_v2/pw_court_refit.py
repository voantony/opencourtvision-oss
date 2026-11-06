"""Point-winner stages with per-point court fits (camera drift, court_refit.py) in place of the per-view fits.

Runs tools/point_winner/pw_stages.py unchanged except view_at(views, p["end"]): for a point listed in REFIT.json the
view keeps its t0/t1/ok but uses that point's H_img2court. By default only refits that court_refit kept (line support
improved) are used; --all uses every drift-based refit. Other args pass through.

  pw_court_refit.py --pw_stages PATH/pw_stages.py --refit REFIT.json [--all] -- <pw_stages args>
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np


def main():
    argv = sys.argv[1:]; cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(); ap.add_argument("--pw_stages", required=True); ap.add_argument("--refit", required=True)
    ap.add_argument("--all", action="store_true", help="use every drift-based refit, not only those that improve line support")
    ap.add_argument("--log_rules", help="write the bounce-rule verdict with the stale and the refit court per point (JSON)")
    a = ap.parse_args(argv[:cut]); rest = argv[cut + 1:]
    R = json.load(open(a.refit))
    labels = json.load(open(rest[rest.index("--labels") + 1]))["points"]
    by_end = {round(p["end"], 3): p["idx"] for p in labels}
    H_by = {}
    for r in R["points"]:
        if r.get("drift_px") is not None and (r["refit"] or a.all):
            H_by[r["idx"]] = r["H_refit"] if a.all and "H_refit" in r else r["H_img2court"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.pw_stages)))
    spec = importlib.util.spec_from_file_location("pw_stages", a.pw_stages); pw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pw)
    orig = pw.view_at; used = [0]

    def view_at(views, t):
        v = orig(views, t); i = by_end.get(round(t, 3))
        if i not in H_by:
            return v
        nv = object.__new__(pw.View); nv.__dict__.update(v.__dict__)
        nv.H = np.asarray(H_by[i], float); nv.Hi = np.linalg.inv(nv.H); nv._stale = v; nv._idx = i; used[0] += 1
        return nv

    pw.view_at = view_at
    log = {}

    def logged(name, fn):
        def f(r, x, view, *k, **kw):
            res = fn(r, x, view, *k, **kw)
            if hasattr(view, "_stale") and r:
                st = fn(r, x, view._stale, *k, **kw)
                log.setdefault(view._idx, dict(idx=view._idx))[name] = dict(stale=list(st), refit=list(res))
            return res
        return f

    pw.RU.rule_bounce = logged("bounce", pw.RU.rule_bounce); pw.RU.rule_net = logged("net", pw.RU.rule_net)
    sys.argv = [a.pw_stages] + rest
    pw.main()
    print(f"court refit used on {used[0]} points", file=sys.stderr)
    if a.log_rules:
        ch = {n: sum(v[n]["stale"] != v[n]["refit"] for v in log.values() if n in v) for n in ("bounce", "net")}
        json.dump(dict(points=sorted(log.values(), key=lambda v: v["idx"]), changed=ch), open(a.log_rules, "w"), indent=1)


if __name__ == "__main__":
    main()
