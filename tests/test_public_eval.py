"""Public eval harness: validator, baseline gate, and the frozen split. Synthetic only."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "public_eval"))
from heldout import claim_label, roles  # noqa: E402
from measure import regressions  # noqa: E402
from run_eval import evaluate  # noqa: E402
from statsutil import ece_regressed, recall_regressed  # noqa: E402
from synthetic import BUNDLES  # noqa: E402
from validate_labels import validate  # noqa: E402


def test_split_roles_are_frozen():
    r = roles()
    assert r["tune"] == "synth_tune" and r["report"] == "synth_report"
    assert "HELD-OUT" in claim_label("synth_report")
    assert "IN-SAMPLE" in claim_label("synth_tune")


def test_validator_rejects_overlap_and_duplicate():
    lab = {
        "match": {"sport": "tennis", "preset": "advantage", "first_server": "A", "mine": "A"},
        "points": [
            {"idx": 1, "start": 0, "end": 10, "winner": "A"},
            {"idx": 1, "start": 1, "end": 12, "winner": "B"},
            {"idx": 3, "start": 2, "end": 9, "winner": "A"},
        ],
    }
    errors, _warnings, _pts = validate(lab)
    text = " ".join(errors)
    assert "duplicate" in text and "overlap" in text


def test_honest_bundle_beats_trivial_baseline():
    built = BUNDLES["synth_report"]()
    report = evaluate(built["id"], built["labels"], built["p_a"])
    assert report["beats_best_trivial"]
    assert report["model_minus_best_trivial"]["verdict"] == "DISTINGUISHABLE"
    assert report["n"] == 48


def test_biased_probe_fails_the_baseline_gate():
    built = BUNDLES["synth_biased"]()
    with pytest.raises(SystemExit) as ei:
        evaluate(built["id"], built["labels"], built["p_a"])
    assert ei.value.code == 3
    report = evaluate(built["id"], built["labels"], built["p_a"], allow_below_baseline=True)
    assert report["calibration"]["ece_10bin"] > 0.2
    assert report["calibration"]["brier"] > 0.25
    assert report["argmax_error_bias"]["pred_A"] == report["argmax_error_bias"]["n_wrong"]


def test_p_a_length_mismatch_is_an_error():
    built = BUNDLES["synth_tune"]()
    errors, _, _ = validate(built["labels"], p_a=[0.5])
    assert any("p_a length" in e for e in errors)


def _row(p, half, k, n):
    return {"p": p, "half": half, "k": k, "n": n, "lo": p - half, "hi": p + half}


def test_recall_regression_allows_an_improvement():
    base = _row(0.0, 0.023, 0, 80)
    assert not recall_regressed(_row(0.40, 0.05, 32, 80), base)
    assert recall_regressed(_row(0.0, 0.02, 0, 80), _row(0.80, 0.12, 32, 40))


def test_ece_regression_is_one_sided():
    base = {"ece": 0.18, "half": 0.05}
    assert not ece_regressed(0.10, base)
    assert ece_regressed(0.25, base)


def _board(far_p, ece, report_beats=True, biased_beats=False):
    primary = {k: _row(0.5, 0.1, 1, 2) for k in (
        "ball_recall_far_half", "ball_recall_fast_blurry", "far_player_recall", "bounce_recall", "point_end_recall")}
    primary["far_player_recall"] = _row(far_p, 0.02, 0, 80)
    return {
        "primary": primary,
        "calibration_report": {"ece": ece, "half": 0.04, "beats_best_trivial": report_beats},
        "calibration_biased_probe": {"beats_best_trivial": biased_beats},
    }


def test_measure_check_flags_a_worse_far_player_and_a_dead_tripwire():
    base = _board(0.40, 0.10)
    assert regressions(_board(0.40, 0.10), base) == []
    assert any("far_player" in item for item in regressions(_board(0.10, 0.10), base))
    assert any("tripwire" in item for item in regressions(_board(0.40, 0.10, biased_beats=True), base))
    assert any("ECE" in item for item in regressions(_board(0.40, 0.20), base))


def test_detector_bench_reports_intervals(tmp_path):
    import subprocess
    out = tmp_path / "det.json"
    subprocess.run([sys.executable, os.path.join(HERE, "..", "benchmarks", "detector_bench.py"),
                    "--frames", "40", "--json-out", str(out)], check=True)
    report = json.loads(out.read_text())
    far = report["primary"]["far_player_recall"]
    fast = report["primary"]["ball_recall_fast_blurry"]
    assert far["n"] == 40 and "half" in far and far["lo"] <= far["p"] <= far["hi"]
    assert fast["n"] > 0
    assert report["primary"]["bounce_recall"]["n"] > 0
