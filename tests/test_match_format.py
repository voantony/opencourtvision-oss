"""Per-match format setting (deuce rule x tiebreak option) read by the labels loader and both scorers."""
import pytest

from opencourtvision.scoring.formats import TIEBREAK_OPTIONS, format_from_settings, format_options, get_format, match_format, with_options
from opencourtvision.scoring.rules import advance, initial_state, replay


def play(fmt, seq, first="A"):
    st = initial_state(fmt, first)
    for w in seq:
        st = advance(st, w, fmt)
    return st


def game(w):          # one game won to love by w
    return w * 4


def test_options_round_trip():
    for rule in ("advantage", "golden_point", "star_point"):
        for tb in TIEBREAK_OPTIONS:
            f = with_options(get_format("padel"), rule, tb)
            assert format_options(f) == {"deuce_rule": rule, "tiebreak": tb}
            assert format_from_settings({"sport": "padel", "format": {"deuce_rule": rule, "tiebreak": tb}}) == f
    assert format_options(get_format("tennis", "no_ad_match_tb")) == {"deuce_rule": "golden_point", "tiebreak": "super_tiebreak"}
    assert format_options(get_format("pickleball")) is None
    with pytest.raises(ValueError):
        with_options(get_format("padel"), "sudden_death")
    with pytest.raises(ValueError):
        with_options(get_format("padel"), tiebreak="12_point")
    with pytest.raises(ValueError):
        with_options(get_format("pickleball"), "golden_point")


def test_settings_defaults_and_legacy_presets():
    assert format_from_settings({"sport": "padel"}).deuce_rule == "golden_point"              # club default
    assert format_from_settings({"sport": "padel", "preset": "advantage"}).deuce_rule == "advantage"
    assert format_from_settings({"sport": "tennis", "preset": "no_ad"}).deuce_rule == "golden_point"
    assert format_from_settings({"sport": "tennis", "preset": "golden_point"}).deuce_rule == "golden_point"
    assert format_from_settings({"sport": "tennis", "preset": "advantage"}).deuce_rule == "advantage"
    f, how = match_format("padel", deuce_rule="advantage", tiebreak="none")
    assert how == "setting" and format_options(f) == {"deuce_rule": "advantage", "tiebreak": "none"}
    assert with_options(get_format("tennis")) is get_format("tennis")


def test_golden_vs_advantage_game():
    seq = "AAABBB" + "A"                     # 40-40, then A wins the next point
    g = play(with_options(get_format("padel"), "golden_point"), seq)
    a = play(with_options(get_format("padel"), "advantage"), seq)
    assert g.games == (1, 0) and g.pts == (0, 0)
    assert a.games == (0, 0)                 # advantage A, game still on
    a = play(with_options(get_format("padel"), "advantage"), seq + "A")
    assert a.games == (1, 0)


def _to_six_all(fmt):
    seq = ""
    for i in range(12):                      # alternate holds: 6-6
        seq += game("A" if i % 2 == 0 else "B")
    return seq


def test_standard_tiebreak_every_set():
    f = with_options(get_format("padel"), tiebreak="standard")
    st = play(f, _to_six_all(f))
    assert st.games == (6, 6) and st.tb == 7
    st = play(f, _to_six_all(f) + "A" * 7)
    assert st.sets == (1, 0) and st.games == (0, 0)


def test_no_tiebreak_sets_won_by_two():
    f = with_options(get_format("padel"), tiebreak="none")
    st = play(f, _to_six_all(f))
    assert st.games == (6, 6) and st.tb == 0
    st = play(f, _to_six_all(f) + game("A") + game("B"))      # 7-7, still no tiebreak
    assert st.games == (7, 7) and st.tb == 0 and st.sets == (0, 0)
    st = play(f, _to_six_all(f) + game("A") + game("A"))      # 8-6
    assert st.sets == (1, 0)


def test_super_tiebreak_replaces_deciding_set():
    f = with_options(get_format("padel"), "golden_point", "super_tiebreak")
    one_set = lambda w: game(w) * 6                           # noqa: E731
    st = play(f, one_set("A") + one_set("B"))
    assert st.sets == (1, 1) and st.tb == 10
    st = play(f, one_set("A") + one_set("B") + "B" * 9)
    assert not st.done
    st = advance(st, "B", f)
    assert st.done and st.sets == (1, 2)
    # standard tiebreaks still apply inside the first two sets
    st = play(f, _to_six_all(f))
    assert st.tb == 7


def test_9oct_padel_shape_golden_point():
    """6-1 6-7(5) 6-1 replays with golden point + standard tiebreak (the 9 Oct labelling setting)."""
    f = with_options(get_format("padel"), "golden_point", "standard")
    s1 = (game("A") * 2 + game("B") + game("A") * 4)               # 6-1 (A broke on every B service game? order irrelevant)
    s2 = _to_six_all(f) + "B" * 6 + "A" * 5 + "B"                  # tiebreak 5-7
    s3 = (game("A") * 2 + game("B") + game("A") * 4)
    st = play(f, s1 + s2 + s3)
    assert st.done and st.sets == (2, 1)


def test_scorers_and_labels_read_the_setting():
    """The labels loader and both scorers take the ScoringFormat built from the match setting."""
    from opencourtvision.labels.points import PointLabels
    from opencourtvision.scoring import decoder, forward  # noqa: F401  (decoder imports with custom formats)
    f = format_from_settings({"sport": "padel", "format": {"deuce_rule": "golden_point", "tiebreak": "standard"}})
    assert replay(list("AAABBBA"), f)[-1].games == (1, 0)
    assert forward.start_state(f, None, first_server="A").server == "A"
    L = PointLabels(video={}, match={"sport": "padel", "preset": "advantage", "format": {"deuce_rule": "golden_point"}},
                    segments=[], decoded={}, labeler={})
    assert L.fmt.deuce_rule == "golden_point"
    L = PointLabels(video={}, match={"sport": "tennis", "preset": "advantage"}, segments=[], decoded={}, labeler={})
    assert L.fmt == get_format("tennis", "advantage")
