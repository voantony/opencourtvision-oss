from opencourtvision.scoring.callouts import parse_callout, score_timeline
from opencourtvision.scoring.games import changeover_games, parse_final


def test_parse_callout_english_and_spanish():
    assert parse_callout("Fifteen love") == ("pts", (15, 0))
    assert parse_callout("thirty all") == ("pts", (30, 30))
    assert parse_callout("quince iguales") == ("pts", (15, 15))   # "<n> iguales" = n-all; "iguales" alone = deuce
    assert parse_callout("cuarenta quince") == ("pts", (40, 15))
    assert parse_callout("Deuce!") == ("deuce", None)
    assert parse_callout("ad in") == ("ad", "in")
    assert parse_callout("ventaja resto") == ("ad", "resto")
    assert parse_callout("nice shot") is None


def test_final_and_changeovers():
    sets = parse_final("6-4 6-3")
    assert sets == [(6, 4), (6, 3)]
    assert changeover_games(sets) == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert changeover_games(parse_final("6-2 6-1")) == [1, 3, 5, 7, 9, 11, 13, 15]


def test_score_timeline_flags_unreachable_callout():
    segs = [dict(A=10.0 * i, B=10.0 * i + 8) for i in range(4)]
    C = {"a": dict(a=0, b=1, first_hit=0.0, text="love love"),
         "b": dict(a=0, b=1, first_hit=10.0, text="fifteen love"),
         "c": dict(a=0, b=1, first_hit=20.0, text="forty fifteen")}  # needs 3 points, only 1 played
    rows = score_timeline(segs, C, 0.0)
    assert [r["conf"] for r in rows] == ["callout", "callout", "callout", "none"]
    assert not rows[1]["flag"] and rows[2]["flag"].startswith("discrepancy")


def test_parse_padel_calls_english_and_spanish():
    from opencourtvision.scoring.callouts import describe_call, detect_deuce_rule
    assert parse_callout("deuce 2") == ("deuce_n", 2) and parse_callout("Deuce three") == ("deuce_n", 3)
    assert parse_callout("iguales dos") == ("deuce_n", 2) and parse_callout("tercer iguales") == ("deuce_n", 3)
    assert parse_callout("iguales") == ("deuce", None) and parse_callout("cuarenta iguales") == ("deuce", None)
    assert parse_callout("forty all") == ("deuce", None)
    assert parse_callout("ventaja 1") == ("ad_n", (1, None))
    assert parse_callout("ventaja dos resto") == ("ad_n", (2, "resto"))
    assert parse_callout("advantage two receiver") == ("ad_n", (2, "receiver"))
    assert parse_callout("Punto de oro!") == ("deciding", "golden") and parse_callout("golden point") == ("deciding", "golden")
    assert parse_callout("star point") == ("deciding", "star")
    assert parse_callout("punto estrella") == ("deciding", "star") and parse_callout("punto de estrella") == ("deciding", "star")
    assert parse_callout("treinta iguales") == ("pts", (30, 30))
    assert describe_call(("ad_n", (2, "resto"))) == "ad 2 resto" and describe_call(("deciding", "star")) == "star point"
    assert detect_deuce_rule(["punto de oro", "quince cero"]) == "golden_point"
    assert detect_deuce_rule(["iguales dos"]) == "star_point" and detect_deuce_rule(["deuce 1"]) is None
    assert detect_deuce_rule(["punto de oro", "star point", "punto estrella"]) == "star_point"     # majority
    assert detect_deuce_rule(["star point", "punto de oro"]) == "golden_point"                    # tie: latest call
    assert detect_deuce_rule(["thirty all"]) is None
