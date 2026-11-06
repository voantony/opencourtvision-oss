"""Overlays clear on every cut; speed labels pass a per-sport/stroke plausibility filter."""
import pytest

from opencourtvision.render.overlays import clear_on_cuts, cuts_from_clips, spans_cut
from opencourtvision.speeds import check_speed, speed_label


def test_every_overlay_clears_on_a_cut():
    cuts = cuts_from_clips([dict(start=10, end=14), dict(start=30, end=33.5), dict(start=50, end=58)])
    assert cuts == [4.0, 7.5]
    ov = [dict(kind="shot_card", start=2.0, end=6.0), dict(kind="minimap", start=0.5, end=12.0), dict(kind="tag", start=3.9, end=5.0),
          dict(kind="speed", start=4.0, end=5.5), dict(kind="score_bug", start=0.0, end=15.5), dict(kind="caption", start=7.0, end=9.0)]
    out = clear_on_cuts(ov, cuts)
    kinds = {o["kind"]: o for o in out}
    assert kinds["shot_card"]["end"] == 4.0 and kinds["shot_card"]["clipped_at_cut"]
    assert kinds["minimap"]["end"] == 4.0                       # clears at the FIRST cut, doesn't come back
    assert "tag" not in kinds                                   # 0.1 s left after clipping: dropped, never flashed
    assert kinds["speed"]["start"] == 4.0 and kinds["speed"]["end"] == 5.5   # starts on the cut = belongs to the next shot
    assert kinds["caption"]["end"] == 7.5
    assert kinds["score_bug"]["end"] == 15.5                    # the bug is the one persistent overlay
    for o in out:
        if o["kind"] != "score_bug":
            assert not spans_cut(o, cuts) and o["fade_out_s"] <= 0.15


def test_overlays_without_cuts_unchanged():
    out = clear_on_cuts([dict(kind="tag", start=1.0, end=2.0)], [])
    assert out[0]["end"] == 2.0 and "clipped_at_cut" not in out[0]


@pytest.mark.parametrize("sport,stroke,mph,ok", [
    ("tennis", "serve", 45, False), ("tennis", "serve", 118, True), ("tennis", "serve", 170, False),
    ("tennis", "forehand", 82, True), ("tennis", "volley", 5, False),
    ("padel", "smash", 95, True), ("padel", "bandeja", 120, False), ("padel", "serve", 90, False),
    ("pickleball", "dink", 30, False), ("pickleball", "drive", 52, True), ("pickleball", "serve", 70, False),
    ("pickleball", "unknown_stroke", 40, True),
])
def test_speed_ranges(sport, stroke, mph, ok):
    assert check_speed(sport, stroke, mph)[0] is ok


def test_speed_physics_vs_bounce_distance():
    # tennis serve: contact to bounce ~ 19 m in 0.45 s -> mean ~94 mph; 120 mph peak plausible, 190 not, 70 slower than the mean
    assert check_speed("tennis", "serve", 120, 19.0, 0.45)[0]
    ok, why = check_speed("tennis", "serve", 160, 19.0, 0.75)        # mean ~57 mph: 160 needs < 1.8x
    assert not ok and "faster than" in why
    ok, why = check_speed("tennis", "serve", 70, 19.0, 0.45)
    assert not ok and "slower than" in why
    assert not check_speed("padel", "smash", 80, 0.0, 0.5)[0]


def test_speed_label_suppressed_not_corrected():
    assert speed_label("tennis", "serve", 118.4) == "118 MPH"
    assert speed_label("tennis", "serve", 118.4, units="kmh") == "191 KM/H"
    assert speed_label("tennis", "serve", 38) is None
    assert speed_label("tennis", "serve", 118, verified=False) is None
    assert speed_label("tennis", "serve", float("nan")) is None
