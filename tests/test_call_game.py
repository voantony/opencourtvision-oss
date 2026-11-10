"""Scoring grades, accuracy leaderboard, and the whole-match practice rule."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from opencourtvision.labels.call_game import GRADES, grade_for, leaderboard, playable_clips, to_team_call, training_rows
from opencourtvision.labels.leak_guard import LeakError, assert_warmup_not_counted, assert_whole_match_disjoint, discover_holdouts

ROOT = Path(__file__).resolve().parents[1]


def test_grades_follow_points_called():
    assert [need for _name, need in GRADES] == [1, 10, 50, 200]
    assert grade_for(0) is None
    assert grade_for(1) == "Iniciación"
    assert grade_for(9) == "Iniciación"
    assert grade_for(10) == "Intermedio"
    assert grade_for(50) == "Avanzado"
    assert grade_for(200) == "Pro"


def test_leaderboard_is_accuracy_on_checks_only():
    rows = leaderboard([
        {"nickname": "tapper", "points": 9000, "accuracy": 0.4, "check_count": 3, "opt_in": True},
        {"nickname": "steady", "points": 11, "accuracy": 0.81, "check_count": 10, "opt_in": True, "accuracy_band": "80-89%"},
        {"nickname": "higher", "points": 4, "accuracy": 0.93, "check_count": 12, "opt_in": True, "accuracy_band": "90-94%"},
        {"nickname": "quiet", "points": 100, "accuracy": 0.99, "check_count": 40, "opt_in": False, "accuracy_band": "95%+"},
    ])
    assert [row["nickname"] for row in rows] == ["higher", "steady"]
    assert "points" not in rows[0]


def test_practice_whole_match_rule():
    practice = json.loads((ROOT / "labeler-pages" / "practice.json").read_text(encoding="utf-8"))["points"]
    holdouts = discover_holdouts(ROOT)
    assert holdouts, "expected the synthetic holdout fixture"
    assert_whole_match_disjoint(practice, holdouts)
    overlap = [{"id": "other", "video_id": practice[0]["video_id"]}]
    try:
        assert_whole_match_disjoint(practice, overlap)
    except LeakError as exc:
        assert "same match" in str(exc)
    else:
        raise AssertionError("shared source video must fail")


def test_warmup_clips_stay_out_of_rounds_and_scoring():
    practice = json.loads((ROOT / "labeler-pages" / "practice.json").read_text(encoding="utf-8"))["points"]
    clips = json.loads((ROOT / "labeler-pages" / "clips.json").read_text(encoding="utf-8"))["clips"]
    warm = [p["id"] for p in practice]
    assert warm == ["2p6PcKP0720_p4", "2p6PcKP0720_p3", "2p6PcKP0720_p5"]
    assert [p["answer"] for p in practice] == ["A", "F", "B"]
    assert practice[1]["reason_en"].startswith("Let: the serve clipped the net")
    assert practice[1]["reason_es"] == "Let: el saque toca la red y entra, se repite y no hay punto."
    playable = playable_clips(clips, warm)
    playable_ids = [c["id"] for c in playable]
    assert playable_ids[:2] == ["2p6PcKP0720_p1", "2p6PcKP0720_p2"]
    assert len(clips) == 46 and len(playable) == 43
    assert not set(warm).intersection(playable_ids)
    assert to_team_call("A", "A") == "A"
    assert to_team_call("A", "B") == "B"
    assert to_team_call("F", "A") == "F"
    calls = [
        {"point_id": "2p6PcKP0720_p4", "call": "A"},
        {"point_id": "2p6PcKP0720_p1", "call": "B"},
        {"point_id": "632k3fKraeA_p1", "call": "A"},
    ]
    rows = training_rows(calls, clips, warm)
    by_row = {r["point_id"]: r for r in rows}
    assert "2p6PcKP0720_p4" not in by_row
    assert by_row["2p6PcKP0720_p1"]["team_call"] == "B"
    assert by_row["632k3fKraeA_p1"]["team_call"] == "A"
    assert_warmup_not_counted(warm, [c["id"] for c in playable], [r["point_id"] for r in rows])
    try:
        assert_warmup_not_counted(warm, ["2p6PcKP0720_p4"], [])
    except LeakError:
        pass
    else:
        raise AssertionError("a warm-up id in a round must fail")


def test_export_script_rejects_shared_match(tmp_path):
    practice = {
        "points": [{"id": "p1", "video_id": "match-a"}],
    }
    hold = {"check": [{"id": "c1", "video_id": "match-a", "answer_token": "HOLD_ANSWER_CHECK_7c1e"}]}
    points = {"points": [{"id": "e1", "video_id": "match-b"}]}
    practice_path = tmp_path / "practice.json"
    hold_path = tmp_path / "hold.json"
    points_path = tmp_path / "points.json"
    out = tmp_path / "out.json"
    practice_path.write_text(json.dumps(practice), encoding="utf-8")
    hold_path.write_text(json.dumps(hold), encoding="utf-8")
    points_path.write_text(json.dumps(points), encoding="utf-8")
    blocked = subprocess.run(
        ["python3", str(ROOT / "scripts" / "export_open_dataset.py"),
         "--points", str(points_path), "--practice", str(practice_path),
         "--holdout", str(hold_path), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert blocked.returncode == 2
    assert not out.exists()
    safe_practice = tmp_path / "safe-practice.json"
    safe_practice.write_text(json.dumps({"points": [{"id": "w1", "video_id": "synthetic-practice-match"}]}), encoding="utf-8")
    ok = subprocess.run(
        ["python3", str(ROOT / "scripts" / "export_open_dataset.py"),
         "--points", str(points_path), "--practice", str(safe_practice),
         "--holdout", str(hold_path), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert ok.returncode == 0
    assert json.loads(out.read_text(encoding="utf-8"))["points"][0]["id"] == "e1"
    source = (ROOT / "scripts" / "export_open_dataset.py").read_text(encoding="utf-8")
    assert "assert_whole_match_disjoint" in source
