from opencourtvision.matchstate.classify import Seg, features_for_segment, rule_prior, score_signals


def test_warmup_vs_point_rules():
    # near-continuous hitting in the opening window, no score yet
    warm_hits = [i * 1.2 for i in range(45)]  # cover ~0.88 over 60 s
    warm = Seg(0, 0, 60, kind="non_point", hits=warm_hits, file_dur=600)
    pt = Seg(1, 200, 220, kind="point", server="A", score_before={"games": [1, 0]}, hits=[201, 203, 206, 210], file_dur=600)
    fw, fp = features_for_segment(warm), features_for_segment(pt)
    assert rule_prior(fw) == "warm_up"
    assert rule_prior(fp) == "match_point"

def test_score_signals_runs():
    segs = [
        Seg(i, i * 20, i * 20 + 15, kind=("point" if i % 2 else "non_point"),
            hits=[i * 20 + 1, i * 20 + 3, i * 20 + 6], file_dur=400, server="A",
            score_before={"games": [0, 0]})
        for i in range(10)
    ]
    m = score_signals(segs)
    assert "rule_prior_keep" in m
    assert "point_recall" in m["always_keep"]
