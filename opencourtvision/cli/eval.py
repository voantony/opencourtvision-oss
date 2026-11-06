"""ocv-eval (also `ocv eval`): score predicted events against JSON ground truth in labels/, check the MVP edit's hard gates
and score the blind test against SwingVision.

Events: matching tolerance ±80 ms for contact and bounce events, ±150 ms for everything else. Reports precision, recall
and F1 per event type and per label value, a stroke confusion matrix and a confidence calibration table (ECE).
Ground-truth/prediction format: {"events": [{"type", "t", "player", "label": {...}, "confidence", ...}]}; see labels/README.md.

Hard gates (mvp_edit_style.md §9.2), per sport: --gates takes an edit-eval JSON (examples/edit_eval.example.json): labelled
points + warmup, the produced clips and measured metrics. Every gate prints pass/fail with its number; missing = not measured.
Blind test (§9.1): --blind takes a CSV of pairwise votes (examples/blind_votes.example.csv). Reel >= 70% with >= 30 raters,
full cut >= 60% over >= 80 comparisons, top 3 >= 65% over >= 40; each also needs the lower two-sided 90% Wilson bound > 50%.
Exit code 1 when any gate or blind check fails (2 for incomplete) with --strict.

Examples:
  ocv eval --pred work/events/ --labels labels/          # pair files by name
  ocv eval --pred work/events.json --labels labels/club_final.json --json report.json
  ocv eval --synthetic                                    # synthetic doubles labels vs corrupted predictions
  ocv eval --gates work/edit_eval_padel.json --sport padel
  ocv eval --blind ratings.csv --exclude-raters antony,team
  ocv eval --synthetic-gates --sport pickleball           # toy match that passes every gate + synthetic votes
"""
import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="predictions JSON file or directory")
    ap.add_argument("--labels", default="labels", help="ground-truth JSON file or directory (default labels/)")
    ap.add_argument("--json", help="write the full report JSON here")
    ap.add_argument("--synthetic", action="store_true", help="evaluate synthetic predictions against synthetic labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gates", help="edit-eval JSON (one match or a list) for the hard gates")
    ap.add_argument("--sport", choices=("tennis", "padel", "pickleball"), help="gate config (default: the file's sport)")
    ap.add_argument("--blind", help="CSV of blind pairwise votes vs SwingVision")
    ap.add_argument("--exclude-raters", default="", help="comma-separated rater ids to drop (e.g. Antony, the team)")
    ap.add_argument("--synthetic-gates", action="store_true", help="run the gates and blind scorer on synthetic data")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any FAIL, 2 on INCOMPLETE")
    A = ap.parse_args(argv)
    from ..eval.events import evaluate, evaluate_dirs, format_report
    report, out, status = {}, [], []
    if A.synthetic:
        from ..labels.synthetic import ground_truth, predictions
        gt = ground_truth(seed=A.seed); rep = evaluate(predictions(gt, seed=A.seed + 1), gt); rep["files"] = ["<synthetic>"]
        report["events"] = rep; out.append(format_report(rep))
    elif A.pred:
        rep = evaluate_dirs(A.pred, A.labels); report["events"] = rep; out.append(format_report(rep))
    if A.gates or A.synthetic_gates:
        from ..eval.gates import evaluate_edit, format_gates, load_edit_eval, synthetic_edit_eval
        doc = load_edit_eval(A.gates) if A.gates else synthetic_edit_eval(A.sport or "padel", A.seed)
        g = evaluate_edit(doc, A.sport); report["gates"] = g; out.append(format_gates(g)); status.append(g["overall"])
    if A.blind or A.synthetic_gates:
        from ..eval.blind import format_blind, read_votes, score_blind, synthetic_votes
        rows = read_votes(A.blind) if A.blind else synthetic_votes(A.seed)
        b = score_blind(rows, [x for x in A.exclude_raters.split(",") if x.strip()]); report["blind"] = b
        out.append(format_blind(b)); status.append(b["overall"])
    if not out:
        ap.error("give --pred (and --labels), --synthetic, --gates, --blind or --synthetic-gates")
    print("\n\n".join(out))
    if A.json:
        json.dump(report if len(report) > 1 or "events" not in report else report["events"], open(A.json, "w"), indent=1, default=str)
    if A.strict and status:
        raise SystemExit(1 if "FAIL" in status else 2 if "INCOMPLETE" in status else 0)


if __name__ == "__main__":
    main()
