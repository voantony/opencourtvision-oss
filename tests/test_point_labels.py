"""tools/labeler output (ocv.point_labels.v1) -> loader, rules re-check, decoder evidence, training rows."""
import copy
import json
import os
import sys

import pytest

from opencourtvision.labels.points import load_point_labels, parse_point_labels
from opencourtvision.scoring.decoder import decode

sys.path.insert(0, os.path.dirname(__file__))
from synth_labels import synthetic_point_labels  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from synthetic_match import MATCH  # noqa: E402


@pytest.fixture
def raw():
    return synthetic_point_labels()


def test_loads_and_reproduces_the_synthetic_score(raw):
    L = parse_point_labels(raw)
    assert L.complete and L.verify() == [] and L.final_score() == MATCH["final_score"]
    sv = MATCH["points"]
    assert [s.winner for s in L.segments if s.kind == "point"] == ["A" if q["winner"] == "host" else "B" for q in sv]


def test_labels_are_hard_evidence_for_the_decoder(raw):
    L = parse_point_labels(raw)
    d = decode(L.to_evidence(), L.fmt, first_server=L.match["first_server"], final=tuple(MATCH["final_sets"]))
    assert d.winners == L.winners() and min(d.confidence) > 0.999


def test_verify_catches_a_tampered_score(raw):
    bad = copy.deepcopy(raw)
    i = next(i for i, p in enumerate(bad["points"]) if p["kind"] == "point" and p["score_before"]["pts"] == [0, 0])
    bad["points"][i]["score_before"]["pts"] = [1, 0]
    assert [m[0] for m in parse_point_labels(bad).verify()] == [bad["points"][i]["idx"]]


def test_server_override_resets_rotation(raw):
    o = copy.deepcopy(raw)
    first = next(p for p in o["points"] if p["kind"] == "point")
    first["server_override"] = "A" if first["server"] == "B" else "B"
    L = parse_point_labels(o)
    assert L.states()[first["idx"] - 1].server == first["server_override"]
    assert L.verify()          # stored derived fields no longer match: the labeler recomputes them on save


def test_training_rows(raw):
    rows = parse_point_labels(raw).training_rows()
    assert sum(r["is_point"] for r in rows) == MATCH["n_points"]
    r = next(r for r in rows if r["is_point"])
    assert {"start", "end", "winner", "winner_is_mine", "server", "side", "call", "games", "sets", "label_src"} <= set(r)


def test_partial_and_invalid_files(raw, tmp_path):
    part = copy.deepcopy(raw)
    mid = len(part["points"]) // 2
    for p in part["points"][mid:]:
        p["kind"] = p["winner"] = None
    L = parse_point_labels(part)
    assert not L.complete and L.winners()[mid:] == [None] * (len(part["points"]) - mid)
    with pytest.raises(ValueError):
        parse_point_labels({"schema": "nope", "points": []})
    bad = copy.deepcopy(raw); bad["points"][0]["kind"] = "point"; bad["points"][0]["winner"] = None
    with pytest.raises(ValueError):
        parse_point_labels(bad)
    f = tmp_path / "x.labels.json"; f.write_text(json.dumps(raw))
    assert load_point_labels(str(f)).final_score() == MATCH["final_score"]


def test_make_seed_covers_every_sv_point():
    sys.path.insert(0, os.path.join(HERE, "..", "tools", "labeler"))
    from make_seed import build_seed
    seed = build_seed(MATCH["segments"], MATCH["points"])
    P = seed["points"]
    assert sorted(p["sv_point"] for p in P if "sv_point" in p) == list(range(1, MATCH["n_points"] + 1))
    assert all(a["end"] <= b["start"] for a, b in zip(P, P[1:])) and seed["first_server"] == MATCH["first_server"]


def test_highlight_rows_and_ranker_vs_favorites(raw):
    from opencourtvision.labels.points import ranker_vs_favorites
    sv = MATCH["points"]
    o = copy.deepcopy(raw)
    L0 = parse_point_labels(o)
    rows = L0.highlight_rows(sv)
    pts = [r for r in rows if r["is_point"]]
    assert len(pts) == MATCH["n_points"] and all(r["pq"] is not None for r in pts) and not any(r["favorite"] for r in rows)
    assert ranker_vs_favorites(rows)["n_fav"] == 0
    top = sorted(pts, key=lambda r: -r["pq"])[:3]          # star the ranker's own top 3 + one low point; all points seen
    low = min(pts, key=lambda r: r["pq"])
    star = {r["idx"] for r in top} | {low["idx"]}
    for q in o["points"]:
        q["fav_seen"] = q["kind"] == "point"
        q["favorite"] = q["idx"] in star
    R = ranker_vs_favorites(parse_point_labels(o).highlight_rows(sv))
    assert R["n_fav"] == 4 and R["precision@5"] == 0.6 and R["recall@10"] == 0.75 and R["favorite_ranks"][:3] == [1, 2, 3]
    assert R["favorite_ranks"][-1] > len(pts) // 2 and R["average_precision"] > R["random_ap"]
