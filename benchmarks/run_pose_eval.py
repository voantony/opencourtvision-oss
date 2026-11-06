"""Pose-stage benchmark on synthetic stick figures (no footage): stroke labelling accuracy and mechanics error
versus keypoint noise, plus syllable purity. Usage: python benchmarks/run_pose_eval.py [--json out.json]"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opencourtvision.pose import metrics as M  # noqa: E402
from opencourtvision.pose import strokes as S  # noqa: E402
from opencourtvision.pose import syllables as SY  # noqa: E402
from opencourtvision.pose import synthetic as syn  # noqa: E402
from opencourtvision.pose.court import Court  # noqa: E402


def run(noise, seeds=5):
    acc, err = [], {"stance_deg": [], "contact_cm": [], "finish_cm": []}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        plan = [dict(t=2.0 + 2.4 * i, kind=k, stance=float(rng.uniform(10, 70)), contact=float(rng.uniform(45, 70)), finish=float(rng.uniform(-25, 10)))
                for i, k in enumerate(rng.choice(["forehand", "backhand", "serve"], 8))]
        r = syn.rally(plan, noise_px=noise, seed=seed); court = Court(r["court"])
        st = S.strokes(r["kp"], r["fps"], r["hits"])
        by_t = {round(s["t"], 3): s for s in st}
        for p in plan:
            s = by_t.get(round(p["t"], 3)); acc.append(bool(s and s["kind"] == p["kind"]))
            if s and p["kind"] != "serve":
                m = M.stroke_metrics(r["kp"], r["fps"], s, court)
                err["stance_deg"].append(abs(m["stance_deg"] - p["stance"])); err["contact_cm"].append(abs(m["contact_cm"] - np.hypot(p["contact"], 5)))
                err["finish_cm"].append(abs(m["finish_cm"] - p["finish"]))
    return dict(noise_px=noise, stroke_acc=round(float(np.mean(acc)), 3), **{f"{k}_mae": round(float(np.mean(v)), 2) for k, v in err.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); A = ap.parse_args()
    rows = [run(n) for n in (0.0, 1.0, 2.0, 4.0)]
    kp, truth, _ = syn.movement_sequence(seed=0); pur = round(float(SY.purity(SY.fit(kp, 30.0, k=10)["labels"], truth)), 3)
    print(" | ".join(rows[0])); [print(" | ".join(str(v) for v in r.values())) for r in rows]
    print(f"\nsyllable purity (7 synthetic movement types, k=10): {pur}")
    if A.json: json.dump(dict(strokes=rows, syllable_purity=pur), open(A.json, "w"), indent=1)


if __name__ == "__main__":
    main()
