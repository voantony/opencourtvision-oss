"""ocv eval: MVP edit hard gates (mvp_edit_style.md §9.2) and the blind test vs SwingVision (§9.1)."""
import copy
import json
import math
import os

import pytest

from opencourtvision.eval.blind import TARGETS, Z90, read_votes, score_blind, synthetic_votes, wilson_lower, wilson_upper
from opencourtvision.eval.gates import check_gates, evaluate_edit, sport_config, synthetic_edit_eval

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _wilson_ref(p, n, z):
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


# ------------------------------------------------------------------ Wilson bound

def test_wilson_doc_example_and_reference():
    assert wilson_lower(21, 30) == pytest.approx(0.5506, abs=1e-4)          # doc: n = 30 at 70% -> ~0.55
    assert wilson_lower(21, 30) == pytest.approx(_wilson_ref(0.7, 30, Z90))
    assert Z90 == pytest.approx(1.6449, abs=1e-4)


def test_wilson_edges():
    assert wilson_lower(0, 0) == 0.0 and wilson_upper(0, 0) == 1.0           # no data: no claim
    assert wilson_lower(0, 30) == pytest.approx(0.0, abs=1e-12)
    assert wilson_lower(30, 30) == pytest.approx(30 / (30 + Z90 ** 2))       # all wins: n / (n + z^2), not 1.0
    assert wilson_lower(30, 30) < 1.0
    assert wilson_lower(12, 24) < 0.5 < wilson_upper(12, 24)                 # 50% never passes
    assert wilson_lower(7, 10) + wilson_upper(3, 10) == pytest.approx(1.0)   # symmetry
    assert wilson_lower(10.5, 15) == pytest.approx(_wilson_ref(0.7, 15, Z90))  # ties = half wins, fractional ok
    assert wilson_lower(18, 30) < 0.5                                         # doc: 60% at n = 30 isn't distinguishable
    assert wilson_lower(48, 80) > 0.5                                         # ... but is at n = 80
    assert wilson_lower(0.65 * 30, 30) < 0.5 < wilson_lower(0.65 * 31, 31)   # top 3 at exactly 65% needs n >= 31


def _rows(out, wins, ties, losses, raters=None, matches=3):
    rows, i = [], 0
    for score, k in ((1.0, wins), (0.5, ties), (0.0, losses)):
        for _ in range(k):
            r = f"r{(i % raters) if raters else i:03d}"
            rows.append(dict(rater=r, output=out, match=f"m{i % matches}", sport=("tennis", "padel", "pickleball")[i % 3], score=score, killer=""))
            i += 1
    return rows


def test_reel_boundary_70pct_at_30_raters_passes():
    d = score_blind(_rows("reel", 21, 0, 9))["outputs"]["reel"]
    assert d["win_rate"] == pytest.approx(0.70) and d["n"] == 30 and d["raters"] == 30
    assert d["checks"] == dict(win_rate=True, comparisons=True, raters=True, wilson=True) and d["passed"]


def test_reel_fails_below_70_or_too_few_raters():
    assert not score_blind(_rows("reel", 20, 0, 10))["outputs"]["reel"]["checks"]["win_rate"]
    d = score_blind(_rows("reel", 21, 0, 9, raters=29))["outputs"]["reel"]          # 30 votes, 29 raters
    assert d["checks"]["comparisons"] and not d["checks"]["raters"] and not d["passed"]
    d = score_blind(_rows("reel", 20, 1, 8))["outputs"]["reel"]                       # tie = half: 20.5 / 29
    assert d["wins"] == 20.5 and d["n"] == 29 and not d["checks"]["comparisons"]


def test_high_rate_small_n_fails_wilson():
    d = score_blind(_rows("reel", 9, 0, 1))["outputs"]["reel"]                        # 90% of 10
    assert d["checks"]["win_rate"] and not d["checks"]["comparisons"]
    d = score_blind(_rows("reel", 2, 0, 0))["outputs"]["reel"]                        # 2/2: lower = 2 / (2 + z^2) = 0.43
    assert d["win_rate"] == 1.0 and not d["checks"]["wilson"]
    assert score_blind(_rows("reel", 3, 0, 0))["outputs"]["reel"]["checks"]["wilson"]   # 3/3 already clears 0.5


def test_full_cut_needs_80_comparisons():
    d = score_blind(_rows("full_cut", 48, 0, 32, raters=40))["outputs"]["full_cut"]
    assert d["win_rate"] == pytest.approx(0.6) and d["passed"] and d["wilson_lower_90"] > 0.5
    d = score_blind(_rows("full_cut", 47, 2, 30, raters=40))["outputs"]["full_cut"]  # 79 comparisons
    assert not d["checks"]["comparisons"] and not d["passed"]
    d = score_blind(_rows("full_cut", 18, 0, 12))["outputs"]["full_cut"]
    assert not d["checks"]["wilson"] and not d["checks"]["comparisons"]


def test_overall_and_exclusions():
    rows = _rows("reel", 25, 0, 5) + _rows("full_cut", 60, 0, 30, raters=45) + _rows("top3", 30, 0, 10)
    rep = score_blind(rows)
    assert rep["overall"] == "PASS"
    assert score_blind(_rows("reel", 25, 0, 5))["overall"] == "INCOMPLETE"           # full cut / top 3 not run
    rep = score_blind(rows + [dict(rater="antony", output="reel", match="m0", sport="tennis", score=0.0, killer="")] * 5,
                      exclude_raters=["Antony"])
    assert rep["overall"] == "PASS" and rep["outputs"]["reel"]["n"] == 30
    assert score_blind(_rows("reel", 15, 0, 15) + rows[30:])["overall"] == "FAIL"


def test_top3_needs_40():
    assert TARGETS["top3"]["min_comparisons"] == 40 and TARGETS["top3"]["win_rate"] == 0.65
    d = score_blind(_rows("top3", 26, 0, 13))["outputs"]["top3"]                      # 39 comparisons
    assert not d["checks"]["comparisons"] and not d["passed"]
    d = score_blind(_rows("top3", 26, 0, 14))["outputs"]["top3"]                      # 65% of 40, lower bound 0.52
    assert d["passed"] and d["wilson_lower_90"] > 0.5


def test_cutoff_buffer_per_clip_type():
    d = synthetic_edit_eval("padel")
    p0 = d["labels"]["points"][0]
    d["outputs"]["full_cut"][0]["end"] = p0["t_dec"] + 0.6           # point clip needs 1.0 s -> cut off
    d["outputs"]["reel_beats"] = [dict(start=p0["serve_contact"] - 1, end=p0["t_dec"] + 0.6, point_id=p0["id"])]   # beat needs 0.4 -> ok
    d["outputs"]["other_clips"] = [dict(start=p0["serve_contact"] - 3, end=p0["t_dec"] + 0.9, point_id=p0["id"], type="highlight")]
    m = evaluate_edit(d)
    assert _g(m, "reel_t1_cutoffs")["value"] == 0
    assert _g(m, "cutoffs")["value"] == pytest.approx(2 / 122)        # the point clip and the highlight; 1.6% <= 2%: pass
    assert _g(m, "cutoffs")["status"] == "pass"
    assert sport_config("tennis")["cutoff_buffer_s"] == dict(reel_beat=0.4, point=1.0, highlight=1.0)


def test_read_votes_csv(tmp_path):
    rows = read_votes(os.path.join(ROOT, "examples", "blind_votes.example.csv"))
    assert {r["output"] for r in rows} == {"reel", "full_cut", "top3"} and rows[2]["score"] == 0.5
    bad = tmp_path / "bad.csv"; bad.write_text("rater_id,output,match_id,sport,vote\nr1,reel,m1,tennis,maybe\n")
    with pytest.raises(ValueError):
        read_votes(str(bad))


def test_synthetic_votes_shape():
    rows = synthetic_votes(0)
    assert len(rows) == 192 and len({r["rater"] for r in rows}) == 32
    rep = score_blind(rows)
    assert rep["outputs"]["reel"]["raters_under_3_matches"] == []
    assert set(TARGETS) == set(rep["outputs"])


# ------------------------------------------------------------------ hard gates

@pytest.mark.parametrize("sport", ["tennis", "padel", "pickleball"])
def test_synthetic_match_passes_all_gates(sport):
    rep = evaluate_edit(synthetic_edit_eval(sport), sport)
    assert rep["overall"] == "PASS" and rep["n_fail"] == 0 and rep["n_missing"] == 0
    assert all(g["shown"] is not None for g in rep["gates"])


def test_sport_params():
    assert sport_config("tennis")["pre_roll_s"] == 4.0 and sport_config("padel")["pre_roll_s"] == 3.0
    assert sport_config("padel")["in_play_gap_s"] == 4.0
    with pytest.raises(ValueError):
        sport_config("squash")


def _g(rep, gid):
    return next(g for g in rep["gates"] if g["id"] == gid)


def test_missed_and_cutoff_and_warmup_detected():
    d = synthetic_edit_eval("padel")
    d2 = copy.deepcopy(d); del d2["outputs"]["full_cut"][5:7]                   # 2 of 120 missed = 1.7% > 1%
    rep = evaluate_edit(d2); assert _g(rep, "missed_points")["status"] == "fail" and rep["overall"] == "FAIL"
    d3 = copy.deepcopy(d); del d3["outputs"]["full_cut"][5]                     # 1 of 120 = 0.83% <= 1%
    assert _g(evaluate_edit(d3), "missed_points")["status"] == "pass"
    d4 = copy.deepcopy(d)
    for i in range(3):                                                          # 3 of 128 clips end before t_dec + 0.4
        d4["outputs"]["full_cut"][i]["end"] = d4["labels"]["points"][i]["t_dec"] + 0.1
    assert _g(evaluate_edit(d4), "cutoffs")["status"] == "fail"
    d5 = copy.deepcopy(d); b = d5["outputs"]["reel_beats"][0]
    b["end"] = next(p for p in d5["labels"]["points"] if p["id"] == b["point_id"])["t_dec"]   # T1 reel beat cut off
    assert _g(evaluate_edit(d5), "reel_t1_cutoffs")["value"] == 1
    d6 = copy.deepcopy(d); d6["outputs"]["full_cut"].insert(0, dict(start=550.0, end=556.0))   # warmup clip
    rep = evaluate_edit(d6)
    assert _g(rep, "warmup")["value"] == pytest.approx(6.0) and _g(rep, "warmup")["status"] == "fail"
    assert _g(rep, "spurious_points")["value"] == pytest.approx(1 / 121)


def test_point_start_and_clipped_serve():
    d = synthetic_edit_eval("tennis")
    p0 = d["labels"]["points"][0]; d["outputs"]["full_cut"][0]["start"] = p0["serve_contact"] - 0.5   # after the motion start
    rep = evaluate_edit(d)
    assert _g(rep, "clipped_serves")["value"] == 1 and _g(rep, "clipped_serves")["status"] == "fail"
    assert _g(rep, "point_start")["value"] == pytest.approx(119 / 120)          # 99.2% within 0.5 s: still >= 98%


def test_jerk_gate_float_path_and_reel_stats():
    base = dict(jerk_p95_px=1.0, jerk_baseline_px=2.0)
    assert _g(check_gates("tennis", base), "jerk")["status"] == "pass"           # exactly 0.5x passes
    assert _g(check_gates("tennis", dict(jerk_p95_px=1.01, jerk_baseline_px=2.0)), "jerk")["status"] == "fail"
    rs = {"reel_stats": {"jerk_p95": {"px_per_frame3": 0.6}, "jerk_p95_rounded": {"px_per_frame3": 2.0}, "jerk_qa": {"baseline": 2.0}}}
    g = _g(check_gates("padel", rs), "jerk")
    assert g["value"] == pytest.approx(0.3) and g["status"] == "pass"            # uses the float path, not the rounded one
    assert _g(check_gates("padel", dict(jerk_p95_px=0.6)), "jerk")["status"] == "missing"   # no baseline -> not measured


def test_missing_metrics_incomplete_and_zero_gates():
    rep = check_gates("pickleball", {})
    assert rep["overall"] == "INCOMPLETE" and rep["n_missing"] == len(rep["gates"])
    m = evaluate_edit(synthetic_edit_eval("pickleball"))
    d = synthetic_edit_eval("pickleball"); d["metrics"].update(captions_wrong=1, consent_violations=1, best_point_today_s=601)
    rep = evaluate_edit(d)
    assert {g["id"] for g in rep["gates"] if g["status"] == "fail"} == {"wrong_captions", "consent", "best_point_today"}
    assert m["overall"] == "PASS"


def test_multi_match_pooling():
    a, b = synthetic_edit_eval("padel", 0), synthetic_edit_eval("padel", 1)
    b["metrics"]["reel_dead_time_max_s"] = 1.4
    rep = evaluate_edit([a, b])
    assert rep["matches"] == 2 and _g(rep, "reel_dead_time")["status"] == "fail"


def test_cli_sample_output(capsys, tmp_path):
    from opencourtvision.cli.ocv import main
    main(["eval", "--gates", os.path.join(ROOT, "examples", "edit_eval.example.json"), "--blind",
          os.path.join(ROOT, "examples", "blind_votes.example.csv"), "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "hard gates · padel" in out and "PASS" in out and "Wilson90 lower" in out
    rep = json.load(open(tmp_path / "r.json"))
    assert rep["gates"]["overall"] == "PASS" and rep["blind"]["overall"] == "FAIL"
    with pytest.raises(SystemExit) as e:
        main(["eval", "--synthetic-gates", "--sport", "tennis", "--strict"])
    assert e.value.code in (1, 2)
