"""Point-winner stages with the LEARNED last contact (EventNet hits) in place of the audio rally anchor.

Runs tools/point_winner/pw_stages.py unchanged, except that the rally's t_last (the last-contact anchor used with
--anchor rally) is taken from EventNet's learned hits (eventnet.py --write_tc) for points that have one:
  --mode override   t_last = the point's last learned hit
  --mode snap       t_last moves to the nearest learned hit within --snap_s of the audio anchor (else unchanged)
Points without a learned hit keep the audio anchor. Every other pw_stages argument is passed through.

  pw_learned_tc.py --pw_stages PATH/pw_stages.py --tc TC.json [TC2.json ...] --mode snap -- <pw_stages args>
"""
import argparse
import importlib.util
import json
import os
import sys


def main():
    argv = sys.argv[1:]; cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(); ap.add_argument("--pw_stages", required=True); ap.add_argument("--tc", nargs="+", required=True)
    ap.add_argument("--mode", choices=("override", "snap"), default="snap"); ap.add_argument("--snap_s", type=float, default=0.3)
    a = ap.parse_args(argv[:cut]); rest = argv[cut + 1:]
    learned = {}
    for f in a.tc:
        for c in json.load(open(f))["clips"].values():
            learned[c["idx"]] = c
    labels = json.load(open(rest[rest.index("--labels") + 1]))["points"]
    by_start = {round(p["start"], 3): p["idx"] for p in labels}
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.pw_stages)))
    spec = importlib.util.spec_from_file_location("pw_stages", a.pw_stages); pw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pw)
    orig = pw.RL.rally; used = dict(override=0, snap=0, kept=0)

    def rally(d, view, hits, start, end, *k, **kw):
        r = orig(d, view, hits, start, end, *k, **kw)
        c = learned.get(by_start.get(round(start, 3)))
        if r is None or not c or not c["hits"]:
            used["kept"] += 1; return r
        if a.mode == "override":
            r = dict(r, t_last=c["t_last"]); used["override"] += 1
        else:
            near = min(c["hits"], key=lambda h: abs(h - r["t_last"]))
            if abs(near - r["t_last"]) <= a.snap_s:
                r = dict(r, t_last=near); used["snap"] += 1
            else:
                used["kept"] += 1
        return r

    pw.RL.rally = rally
    sys.argv = [a.pw_stages] + rest
    pw.main()
    print("learned tc:", used, file=sys.stderr)


if __name__ == "__main__":
    main()
