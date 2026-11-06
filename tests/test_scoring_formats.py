"""Scoring formats checked against FIP 2026 (padel) and USA Pickleball 2026 / UPA-A / MLP 2026 (pickleball)."""
import pytest

from opencourtvision.scoring.formats import (
    DEFAULT_PRESET,
    bug_point_cell,
    game_point_state,
    game_pressure,
    get_format,
    match_format,
    pickleball_game_point,
    play_game,
    play_pickleball_game,
    pressure_multiplier,
    stakes_caption,
)


def test_defaults():
    assert get_format("padel").deuce_rule == "golden_point" and DEFAULT_PRESET["padel"] == "golden_point"   # club default
    assert get_format("padel", "fip_star_point").deuce_rule == "star_point"
    assert get_format("padel").deciding_point_chooser == "receiver"
    pb = get_format("pickleball")
    assert (pb.system, pb.points_to_win, pb.win_by, pb.first_server_exception, pb.score_call_numbers) == ("side_out", 11, 2, True, 3)
    assert get_format("pickleball", "usap_rally_2026").game_point_only_when_serving is False
    assert get_format("pickleball", "mlp_dreambreaker_2026").game_point_only_when_serving is True
    with pytest.raises(ValueError):
        get_format("padel", "wpt_2019")


def test_padel_golden_point_one_deciding_point():
    st = game_point_state(3, 3, "golden_point")
    assert st["deciding"] and st["break_point"]
    assert play_game("SSSRRRR", "golden_point")["winner"] == "R"       # 40-40 then receiver wins the golden point
    assert play_game("SSSRRRR", "golden_point")["points_played"] == 7


def test_padel_star_point_two_advantages_then_deciding():
    # deuce 1 -> adv S -> deuce 2 -> adv R -> deuce 3 = Star Point
    pts = "SSSRRR" + "SR" + "RS"
    g = play_game(pts + "R", "star_point")
    assert g["winner"] == "R" and g["points_played"] == 11
    st = g["states"][-1]
    assert st["deciding"] and st["call"] == "deuce 3"
    assert not game_point_state(3, 3, "star_point")["deciding"] and not game_point_state(4, 4, "star_point")["deciding"]
    assert game_point_state(4, 5, "star_point")["break_point"]           # advantage 2 to the receiver = break point
    assert play_game("SSSRRR" + "SS", "star_point")["winner"] == "S"      # advantage 1 converted
    assert play_game("SSSRRR" + "SRSRSRSR", "advantage")["winner"] is None   # advantage: unbounded


def test_stakes_caption_and_bug_only_at_verified_deciding_point():
    gp, sp = "golden_point", "star_point"
    assert stakes_caption(game_point_state(3, 3, gp), "padel", gp, verified=True) == "PUNTO DE ORO"
    assert stakes_caption(game_point_state(3, 3, gp), "padel", gp) is None                         # unverified: nothing
    assert stakes_caption(game_point_state(5, 5, sp), "padel", sp, verified=True) == "STAR POINT"
    assert stakes_caption(game_point_state(3, 3, sp), "padel", sp, verified=True) is None          # deuce 1 isn't decisive
    assert stakes_caption(game_point_state(4, 4, sp), "padel", sp, verified=True) is None          # nor deuce 2
    assert stakes_caption(game_point_state(1, 3, gp), "padel", gp, verified=True) == "BREAK POINT"
    assert bug_point_cell(game_point_state(3, 3, gp), gp, verified=True) == "ORO"
    assert bug_point_cell(game_point_state(5, 5, sp), sp, verified=True) == "STAR"
    assert bug_point_cell(game_point_state(4, 4, sp), sp, verified=True) is None
    assert bug_point_cell(game_point_state(5, 5, sp), sp, verified=False) is None


def test_match_format_setting_callouts_default():
    f, how = match_format("padel")
    assert (f.deuce_rule, how) == ("golden_point", "default")
    f, how = match_format("padel", callouts=["quince cero", "Punto de oro", "treinta iguales"])
    assert (f.deuce_rule, how) == ("golden_point", "callouts")
    f, how = match_format("padel", callouts=["deuce one", "ventaja 1", "iguales dos", "punto estrella"])
    assert (f.deuce_rule, how) == ("star_point", "callouts")
    f, how = match_format("padel", callouts=["Star point!"])
    assert f.preset == "fip_star_point"
    f, how = match_format("padel", "advantage", callouts=["punto de oro"])                       # the setting wins
    assert (f.deuce_rule, how) == ("advantage", "setting")
    assert match_format("padel", callouts=["thirty fifteen", "deuce"])[1] == "default"           # no evidence


def test_o1_pressure_star_point():
    sp, gp = "star_point", "golden_point"
    assert game_pressure(4, 5, sp) == "break_point"                     # advantage 2 to the receiver
    assert game_pressure(3, 4, sp) == "break_point"                     # advantage 1 to the receiver
    assert game_pressure(4, 3, sp) is None                              # advantage to the server
    assert game_pressure(3, 3, sp) == "deuce" and game_pressure(4, 4, sp) == "deuce"
    # Coach 19 Oct: deciding point ALWAYS gets BP ×1.3, whoever wins it / is receiving
    assert game_pressure(5, 5, sp, winner="R") == "break_point"
    assert game_pressure(5, 5, sp, winner="S") == "break_point"
    assert game_pressure(5, 5, sp) == "break_point"
    assert game_pressure(3, 3, gp, winner="R") == "break_point"
    assert game_pressure(3, 3, gp, winner="S") == "break_point"
    assert game_pressure(2, 2, sp) == "deuce"                           # 30-30
    assert pressure_multiplier(game_state="break_point") == pytest.approx(1.3)
    assert pressure_multiplier(game_state="deuce") == pytest.approx(1.15)
    assert pressure_multiplier(dict(set_point=True), "break_point") == pytest.approx(1.45)        # nested max, no stacking
    assert pressure_multiplier(dict(match_point=True), "deuce") == pytest.approx(1.75)            # 1.6 × 1.15 = 1.84 -> cap
    assert pressure_multiplier() == 1.0
    for fl in ({}, dict(match_point=True), dict(set_point=True, break_point=True)):
        for gs in (None, "deuce", "break_point"):
            assert pressure_multiplier(fl, gs) <= 1.75


def test_coach_star_point_deciding_always_break_point():
    """Coach 19 Oct: deuce-3 (Star Point) / golden deciding point gets BP ×1.3 whoever is receiving; nested max."""
    sp, gp = "star_point", "golden_point"
    for winner in (None, "R", "S"):
        assert game_pressure(5, 5, sp, winner=winner) == "break_point"
        assert game_pressure(3, 3, gp, winner=winner) == "break_point"
        assert pressure_multiplier(game_state=game_pressure(5, 5, sp, winner=winner)) == pytest.approx(1.3)
    # also a set / match point: take the max (no stacking)
    assert pressure_multiplier(dict(set_point=True), "break_point") == pytest.approx(1.45)
    assert pressure_multiplier(dict(match_point=True), "break_point") == pytest.approx(1.6)
    # advantage 1 / 2 to the receivers stays an ordinary break point (not the deciding point)
    assert game_pressure(3, 4, sp) == "break_point" and not game_point_state(3, 4, sp)["deciding"]
    assert game_pressure(4, 5, sp) == "break_point" and not game_point_state(4, 5, sp)["deciding"]
    assert game_pressure(4, 3, sp) is None and game_pressure(5, 4, sp) is None   # adv to server: no BP


def test_pickleball_side_out_first_server_exception():
    f = get_format("pickleball")
    g = play_pickleball_game("BAAB", f)
    assert g["calls"] == ["0-0-2", "0-0-1", "0-0-2", "0-0-1"]   # A starts as server 2: one lost rally = side out; B gets 2 servers
    g = play_pickleball_game("ABBAAB", f)
    # A: 0-0-2 win -> 1-0-2; lose -> side out B 0-1-1; B wins (1-1-1); B loses -> 1-1-2; B loses -> side out A 1-1-1; ...
    assert g["calls"] == ["0-0-2", "1-0-2", "0-1-1", "1-1-1", "1-1-2", "1-1-1"]
    assert play_pickleball_game("A" * 11, f)["winner"] == "A"
    g = play_pickleball_game("A" * 10 + "B" + "BB" + "B" * 10, f)   # 10-0, then B claws back; must win by 2
    assert g["score"]["B"] >= 10


def test_pickleball_win_by_two_and_receiver_cannot_score():
    f = get_format("pickleball")
    assert pickleball_game_point(10, 9, f) == "serving" and pickleball_game_point(9, 10, f) is None
    assert pickleball_game_point(10, 10, f) is None
    r = get_format("pickleball", "usap_rally_2026")
    assert pickleball_game_point(9, 10, r) == "receiving"        # 2026: receiver can win on game point
    assert pickleball_game_point(9, 10, get_format("pickleball", "mlp_dreambreaker_2026")) is None


def test_rally_scoring_freeze_vs_2026():
    r = get_format("pickleball", "usap_rally_2026")
    g = play_pickleball_game("A" * 10 + "B" * 9 + "BB" + "B", r)
    assert g["winner"] == "B"
    assert play_pickleball_game("AAAAAAAAAA" + "B", r)["calls"][-1] == "10-0"
    db = get_format("pickleball", "mlp_dreambreaker_2026")
    g = play_pickleball_game("A" * 20 + "B" + "A", db, first="A")     # 20-0 A; B wins a rally on A's serve: side out
    assert g["winner"] is None or g["winner"] == "A"
    g = play_pickleball_game("B" * 20 + "A" + "B", db, first="A")      # B at 20 receiving, then A serves..., B wins on A's serve: frozen
    assert g["score"]["B"] == 20 and g["winner"] is None
