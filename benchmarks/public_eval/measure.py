#!/usr/bin/env python3
"""Scoreboard for the two fixes in front of us: far-side players, and p(A) calibration.

This public snapshot is a work in progress. This script is how a change is measured.

  python benchmarks/public_eval/measure.py
  python benchmarks/public_eval/measure.py --check

The default run is synthetic. It validates labels, prints trivial baselines inside the point-winner
eval, and reports Wilson or bootstrap intervals. --check exits 1 if a primary detector recall or the
report-bundle ECE gets worse than the frozen floor in baseline.json by more than that floor's
interval half-width. A higher far-player recall is not a failure. The biased p(A) probe must still
lose to the best trivial baseline: that gate is the calibration tripwire.

No private match is read. No image or video is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from run_eval import evaluate  # noqa: E402
from statsutil import bootstrap_ece, ece_regressed, recall_regressed  # noqa: E402
from synthetic import BUNDLES  # noqa: E402

from benchmarks.detector_bench import run as detector_run  # noqa: E402

BASELINE_PATH = HERE / "baseline.json"
PRIMARY = (
    "ball_recall_far_half",
    "ball_recall_fast_blurry",
    "far_player_recall",
    "bounce_recall",
    "point_end_recall",
)


def _winners(labels):
    pts = sorted(
        [p for p in labels["points"] if p.get("winner") in ("A", "B")],
        key=lambda p: float(p["start"]),
    )
    return [p["winner"] for p in pts]


def score(n_boot: int = 2000, frames: int = 80, seed: int = 3) -> dict:
    det = detector_run(frames, seed)
    report_b = BUNDLES["synth_report"]()
    biased_b = BUNDLES["synth_biased"]()
    report = evaluate(report_b["id"], report_b["labels"], report_b["p_a"])
    # The probe is supposed to lose. Ask for the report instead of exiting 3; run_eval.py
    # still exits 3 when this bundle is invoked on its own.
    biased = evaluate(biased_b["id"], biased_b["labels"], biased_b["p_a"], allow_below_baseline=True)
    biased_beats = bool(biased["beats_best_trivial"])
    cal = bootstrap_ece(report_b["p_a"], _winners(report_b["labels"]), n_boot=n_boot, seed=seed)
    biased_cal = bootstrap_ece(biased_b["p_a"], _winners(biased_b["labels"]), n_boot=n_boot, seed=seed)
    return {
        "dataset": "synthetic-in-memory",
        "release": "work in progress",
        "far_player_recall": det["primary"]["far_player_recall"],
        "primary": {k: det["primary"][k] for k in PRIMARY},
        "calibration_report": {
            "ece": cal["ece"],
            "brier": cal["brier"],
            "lo": cal["lo"],
            "hi": cal["hi"],
            "half": cal["half"],
            "n": cal["n"],
            "beats_best_trivial": report["beats_best_trivial"],
            "verdict": report["model_minus_best_trivial"]["verdict"],
        },
        "calibration_biased_probe": {
            "ece": biased_cal["ece"],
            "brier": biased_cal["brier"],
            "half": biased_cal["half"],
            "n": biased_cal["n"],
            "beats_best_trivial": biased_beats,
        },
    }


def regressions(new: dict, baseline: dict) -> list[str]:
    bad = []
    for key in PRIMARY:
        if recall_regressed(new["primary"][key], baseline["primary"][key]):
            bad.append(f"{key} regressed beyond its CI")
    if not new["calibration_report"]["beats_best_trivial"]:
        bad.append("report bundle no longer beats the best trivial baseline")
    if new["calibration_biased_probe"]["beats_best_trivial"]:
        bad.append("biased p(A) probe now beats the trivial baseline; the calibration tripwire failed")
    if ece_regressed(new["calibration_report"]["ece"], baseline["calibration_report"]):
        bad.append("report-bundle ECE got worse by more than its CI half-width")
    return bad


def _fmt_recall(row: dict) -> str:
    return f"{row['p']:.3f}  95% CI [{row['lo']:.3f}, {row['hi']:.3f}]  ({row['k']}/{row['n']})"


def render(board: dict) -> str:
    cal = board["calibration_report"]
    trap = board["calibration_biased_probe"]
    lines = [
        "release: work in progress — point-winner scoring does not yet beat simple baselines on the held-out match",
        f"far-side player recall  {_fmt_recall(board['far_player_recall'])}",
        f"p(A) report ECE {cal['ece']:.4f}  bootstrap 95% CI [{cal['lo']:.4f}, {cal['hi']:.4f}]  "
        f"Brier {cal['brier']:.4f}  n={cal['n']}  beats trivial={cal['beats_best_trivial']} {cal['verdict']}",
        f"p(A) biased probe ECE {trap['ece']:.4f}  Brier {trap['brier']:.4f}  n={trap['n']}  "
        f"beats trivial={trap['beats_best_trivial']} (must stay false)",
        "other primary recalls:",
    ]
    for key, row in board["primary"].items():
        if key == "far_player_recall":
            continue
        lines.append(f"  {key:28s} {_fmt_recall(row)}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if this run regresses baseline.json")
    ap.add_argument("--baseline", default=str(BASELINE_PATH))
    ap.add_argument("--write-baseline", action="store_true", help="overwrite the frozen floor (do this only on purpose)")
    ap.add_argument("--json-out")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    board = score(n_boot=a.boot, frames=a.frames)
    text = render(board)
    print(text)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(board, indent=2) + "\n")
    if a.write_baseline:
        Path(a.baseline).write_text(json.dumps(board, indent=2) + "\n")
        print(f"wrote {a.baseline}")
    if a.check:
        baseline = json.loads(Path(a.baseline).read_text())
        bad = regressions(board, baseline)
        if bad:
            print("MEASURE CHECK FAIL", file=sys.stderr)
            for item in bad:
                print(" ", item, file=sys.stderr)
            return 1
        print("MEASURE CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
