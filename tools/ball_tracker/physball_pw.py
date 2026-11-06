"""Point-winner S1-S6 with the physics tracker's bounce P(in) in the in/out hard rules (research wrapper).

Runs tools/point_winner/pw_stages.py unchanged, with two hooks:
  * rule_bounce: for a physics bounce with a P(in) (track npz 'bounce_pin', aligned with 'bounces'), the first bounce
    0.2-2.5 s after the last shot on the receiving half calls "in" (hitter won) when P(in) >= q and "out" (hitter lost)
    when P(in) <= 1 - q; it abstains in between. Bounces without a P(in) keep the original line-margin rule.
  * activation gate: the in/out rules may override S5 only if, on the GATE points (default 1-50, never the test half),
    each reaches precision >= 0.95 with >= 10 fires. Otherwise they are measured and reported but abstain, as before.
Needs a pw_stages.py with --hard_rules (and its pw_rules.py); the copy on this branch predates them.
Prints and stores per-half fire tables so the gate is auditable. Summary numbers only; labels stay private.

  python tools/ball_tracker/physball_pw.py --pw_stages tools/point_winner/pw_stages.py --q 0.95 -- <pw_stages args>
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np


def load(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("pw_stages_mod", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def pin_table(tracks):
    """round(t, 2) -> P(in) over every track npz that carries bounce_pin."""
    T = {}
    for f in glob.glob(os.path.join(tracks, "*.npz")):
        d = np.load(f)
        if "bounce_pin" not in d.files or "bounces" not in d.files:
            continue
        for b, p in zip(np.atleast_2d(d["bounces"]).reshape(-1, 3), d["bounce_pin"]):
            if np.isfinite(p):
                T[round(float(b[0]), 2)] = float(p)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw_stages", required=True); ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--gate_max_point", type=int, default=50); ap.add_argument("--min_fires", type=int, default=10)
    ap.add_argument("--min_precision", type=float, default=0.95)
    ap.add_argument("--force_active", action="store_true", help="DIAGNOSTIC: activate the in/out rules regardless of the gate")
    a, rest = ap.parse_known_args()
    rest = [x for x in rest if x != "--"]
    PW = load(a.pw_stages)
    RU = PW.RU
    tracks = rest[rest.index("--tracks") + 1]
    PIN = pin_table(tracks)
    orig = RU.rule_bounce
    q = a.q

    def rule_bounce(r, bounces, view, ok_views=True, out_m=0.4, in_m=0.6):
        if not r or r["last_kind"] == 0 or not view.ok or not ok_views:
            return None, None
        k = r["last_kind"]; t0 = r["t_last"]
        bo = [b for b in bounces if t0 + 0.2 < b["t"] <= t0 + 2.5]
        for b in bo:
            p = PIN.get(round(float(b["t"]), 2))
            if p is None:
                return orig(r, [b], view, ok_views, out_m, in_m)
            x, yc = view.i2c([b["x"], b["y"]])[0]
            recv = (yc > RU.NET_Y) if k > 0 else (yc < RU.NET_Y)
            if not recv:
                continue
            if p >= q:
                return k, "bounce_in"
            if p <= 1 - q:
                return -k, "bounce_out"
            return None, None
        return None, None

    RU.rule_bounce = rule_bounce
    orig_hard = PW.hard_layer
    gate_info = {}

    def hard_layer(rows, bounces, gap_s):
        h = orig_hard(rows, bounces, gap_s)
        split = {}
        for r, c in zip(rows, h["calls"]):
            half = "gate" if r["p"]["point_no"] <= a.gate_max_point else "test"
            for w, nm in c:
                t = split.setdefault(nm, {}).setdefault(half, dict(fires=0, correct=0))
                t["fires"] += 1; t["correct"] += int(w == r["p"]["winner"])
        for nm in ("bounce_in", "bounce_out"):
            g = split.get(nm, {}).get("gate", dict(fires=0, correct=0))
            ok = g["fires"] >= a.min_fires and g["correct"] / max(1, g["fires"]) >= a.min_precision
            if ok or a.force_active:
                PW.ACTIVE.add(nm); PW.RULE_CONF[nm] = max(PW.RULE_CONF.get(nm, 0.9), a.min_precision)
            gate_info[nm] = dict(gate=g, test=split.get(nm, {}).get("test", dict(fires=0, correct=0)), active=nm in PW.ACTIVE)
        h["table"] = {**h["table"], "_split": split, "_gate": gate_info, "_q": q}
        print("in/out gate:", json.dumps(gate_info))
        return h

    PW.hard_layer = hard_layer
    sys.argv = [a.pw_stages] + rest
    PW.main()


if __name__ == "__main__":
    main()
