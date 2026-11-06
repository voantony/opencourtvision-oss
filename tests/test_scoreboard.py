"""benchmarks/scoreboard.py + point_features.py on the synthetic match (no recorded match)."""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
from point_features import build  # noqa: E402
from scoreboard import match, played_rally, stage_metrics, sv_winner_and_overlay, table  # noqa: E402

from opencourtvision.labels.points import load_point_labels  # noqa: E402

sys.path.insert(0, HERE)
from synth_labels import write_synthetic_point_labels  # noqa: E402

sys.path.insert(0, HERE)
from synthetic_match import MATCH  # noqa: E402


def _segs():
    return [(float(r["start_s"]), float(r["end_s"])) for r in MATCH["segments"] if int(r["idx"]) > 0]


def test_match_and_rally():
    assert match([(0, 10), (20, 30)], [(25, 31), (1, 9), (40, 50)]) == {0: 1, 1: 0}
    assert played_rally([1.0, 2.0, 6.0, 7.0, 8.0, 20.0]) == (6.0, 8.0)       # fault + return skipped, ball pickup skipped
    assert played_rally([1.0, 9.0]) == (9.0, 9.0)                            # double fault -> the second serve


def test_scoreboard_on_fixture(tmp_path):
    L = load_point_labels(write_synthetic_point_labels(tmp_path / "labels.json"))
    sv = MATCH["points"]
    hits = sorted(float(q["t_last"]) + 2.0 for q in sv)                      # one 'hit' per point end is enough here
    m = stage_metrics(L, _segs(), hits)
    assert m["n_points"] == MATCH["n_points"] and 0.8 < m["recall"] <= 1.0 and m["non_points"] == m["n_pred"] - m["found"]
    w = sv_winner_and_overlay(L, sv)
    assert w["found"] == MATCH["n_points"] and w["winner_agree"] == MATCH["n_points"] and w["overlay_ok"] == MATCH["n_points"]
    md = table(L, {"sv": m, "ours": m}, w, {"E": "wk 5", "S": "wk 12"})
    assert md.count("\n") == 9 and "Taps per 100 points" in md


def test_point_features_rows(tmp_path):
    L = load_point_labels(write_synthetic_point_labels(tmp_path / "labels.json"))
    sv = MATCH["points"]
    rows = build(L, sv, None, [], _segs(), sorted(float(q["t_last"]) + 2.0 for q in sv))
    assert len(rows) == MATCH["n_points"] and rows[0]["score_before"]["call"] == "0-0" and rows[-1]["score_after"]["done"]
    assert rows[1]["streak_before"] == {"A": 1, "B": 0}
    assert all(r["score_after"]["pts"] != r["score_before"]["pts"] or r["score_after"]["games"] != r["score_before"]["games"]
               or r["score_after"]["sets"] != r["score_before"]["sets"] for r in rows)


def test_window_matcher_does_not_count_one_miss_twice():
    from types import SimpleNamespace as NS
    L = NS(segments=[NS(kind="point", start=0.0, end=40.0)])                  # label wraps a fault at 2 s, rally at 30 s
    preds = [(1.0, 20.0), (29.0, 36.0)]                                         # fault clip (more span overlap), rally clip
    ct = {0: (30.0, 33.0)}
    old = stage_metrics(L, preds, [], ct, matcher="label")
    new = stage_metrics(L, preds, [], ct)
    assert old["cut_celebration"] == 1 and old["non_points"] == 1               # label matcher: fault clip "cuts" the point
    assert new["cut_serve"] == 0 and new["cut_celebration"] == 0 and new["found"] == 1
