"""Editorial spec (Media Director v1): clip end per type, caption triggers, player-framing fallback, jerk_p95."""
import shutil

import numpy as np
import pytest

from opencourtvision.captions import moment_caption, rallies
from opencourtvision.events import Event
from opencourtvision.render.cuts import CLIP_TYPES, MIN_TAIL_S, point_end

T = 100.0


# ------------------------------------------------------------------ point_end per clip type
def test_point_default_and_decisive_bounce():
    assert point_end(T) == pytest.approx(T + 2.5)
    assert point_end(T, ball_dead=T + 0.5) == pytest.approx(T + 2.5)          # max(t_last + 2.5, t_dec + 1.5)
    assert point_end(T, ball_dead=T + 2.0) == pytest.approx(T + 3.5)


def test_point_reaction_and_cap():
    assert point_end(T, reaction=0.5) == pytest.approx(T + 2.5)               # not above 0.6
    assert point_end(T, reaction=0.9) == pytest.approx(T + 5.0)
    assert point_end(T, tail=9.0) == pytest.approx(T + 6.0)                    # hard cap t_last + 6
    assert point_end(T, ball_dead=T + 5.5, reaction=0.9) == pytest.approx(T + 6.5)   # cap never cuts < t_dec + 1.0


def test_rally_highlight_reel_beat():
    assert point_end(T, ball_dead=T + 1.0, clip_type="rally") == pytest.approx(T + 2.5)
    assert point_end(T, ball_dead=T + 1.5, clip_type="highlight_shot") == pytest.approx(T + 2.7)
    assert point_end(T, ball_dead=T + 2.0, clip_type="reel_beat") == pytest.approx(T + 2.8)    # t_dec + 0.8
    assert point_end(T, ball_dead=T + 2.0, clip_type="reel_beat", reaction=0.8, reaction_end=T + 4.2) == pytest.approx(T + 4.2)


def test_reaction_and_celebration():
    assert point_end(T, clip_type="reaction", reaction_onset=T + 1.0) == pytest.approx(T + 3.5)
    assert point_end(T, clip_type="reaction", reaction_onset=T + 1.0, reaction_end=T + 9.0) == pytest.approx(T + 4.5)   # max 3.5 after onset
    assert point_end(T, ball_dead=T + 1.0, clip_type="celebration", celebration_end=T + 3.0) == pytest.approx(T + 3.5)
    assert point_end(T, ball_dead=T + 1.0, clip_type="celebration", celebration_end=T + 20, output="reel") == pytest.approx(T + 4.5)
    assert point_end(T, ball_dead=T + 1.0, clip_type="celebration", celebration_end=T + 20) == pytest.approx(T + 8.5)


@pytest.mark.parametrize("ct", sorted(set(CLIP_TYPES) - {"reel_beat"}))
def test_invariant_every_type(ct):
    for dead in (None, T, T + 0.3):
        for tail in (None, 0.0, 0.5):
            assert point_end(T, ball_dead=dead, tail=tail, buffer=0.0, clip_type=ct, reaction_onset=T - 5,
                             celebration_end=T - 3) >= T + MIN_TAIL_S
    with pytest.raises(ValueError):
        point_end(T, clip_type="nope")


# ------------------------------------------------------------------ captions only from verified triggers
def _volley(t, player, verified=True, net=1.5):
    return Event("contact", t, player, {"shot_type": "volley", "side": "forehand"}, 0.9, ["audio: hit", "pose: volley"],
                 verified, attrs={"net_dist_m": net} if net is not None else {})


def test_unverified_moment_renders_no_caption():
    from opencourtvision.reel.reel import resolve_moments, text_filters
    ev = [_volley(10.0, "near_left", False), _volley(10.5, "far_right", False), _volley(10.9, "near_left", False)]
    cfg = {"moments": [{"name": "m", "start": 9.0, "end": 12.0, "caption_trigger": "net_exchange", "caption": "QUICK HANDS AT THE NET"},
                       {"name": "bare", "start": 9.0, "end": 12.0, "caption": "RALLY OF THE MATCH"}]}
    ms = resolve_moments(cfg, [], ev)
    for m in ms:
        assert m["caption"] == ""
        assert not any("QUICK" in f or "RALLY" in f for f in text_filters(m, [(0.0, m["b"] - m["a"], 1.0)]))
    assert "volley" in ms[0]["caption_why"] and "trigger" in ms[1]["caption_why"]


def test_verified_net_exchange_caption_after_event():
    from opencourtvision.reel.reel import resolve_moments, text_filters
    ev = [_volley(10.0, "near_left"), _volley(10.5, "far_right")]
    m = resolve_moments({"moments": [{"name": "m", "start": 9.0, "end": 14.0, "caption_trigger": "net_exchange"}]}, [], ev)[0]
    assert m["caption"] == "Quick hands" and m["caption_t"] == 10.5
    f = text_filters(m, [(0.0, 5.0, 1.0)])
    assert len(f) == 1 and "between(t,1.50,3.50)" in f[0]            # appears at the event, 2 s on screen
    # same team only, or no net distance (no player tracks), or a SwingVision import: no caption
    assert moment_caption("net_exchange", [_volley(10.0, "near_left"), _volley(10.5, "near_right")], 9, 14)[0] is None
    assert moment_caption("net_exchange", [_volley(10.0, "near_left", net=None), _volley(10.5, "far_right", net=None)], 9, 14)[0] is None
    sv = [_volley(10.0, "H1"), _volley(10.5, "G1")]
    for e in sv: e.source = "swingvision"
    assert moment_caption("net_exchange", sv, 9, 14)[0] is None


def test_rally_of_match_needs_longest_verified_rally():
    mk = lambda ts, v=True: [Event("contact", t, "near_left", None, 0.9, [], v) for t in ts]
    ev = mk([10, 11, 12, 13, 14]) + mk([40, 41, 42, 43, 44, 45]) + mk([70, 71, 72, 73, 74, 75, 76, 77, 78], v=False)
    assert [len(r) for r in rallies(ev)] == [5, 6]
    assert moment_caption("rally_of_match", ev, 39, 46)[0] == "Rally of the match"
    assert moment_caption("rally_of_match", ev, 9, 15)[0] is None
    assert moment_caption("rally_of_match", ev, 69, 79)[0] is None           # longest but unverified
    assert moment_caption(None, ev, 39, 46)[0] is None


def test_stakes_and_ace_triggers():
    ss = Event("score_state", 5.0, None, None, 1.0, [], True, attrs={"break_point": True})
    assert moment_caption("break_point", [ss], 4, 8)[0] == "Break point"
    ss.verified = False
    assert moment_caption("break_point", [ss], 4, 8)[0] is None
    serve = Event("contact", 1.0, "near_left", {"shot_type": "serve", "direction": "T"}, 0.9, [], True)
    bounce = Event("bounce", 1.6, None, {"outcome": "in"}, 0.9, [], True)
    assert moment_caption("ace", [serve, bounce, Event("point_end", 4.0, verified=True)], 0, 5)[0] == "Ace"
    ret = Event("contact", 2.0, "far_left", None, 0.9, [], True)
    assert moment_caption("ace", [serve, bounce, ret, Event("point_end", 4.0, verified=True)], 0, 5)[0] is None


def test_example_config_has_no_bare_captions():
    import json
    import os
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "examples", "reel_moments.example.json")))
    for m in cfg["moments"]:
        assert "caption" not in m or "caption_trigger" in m


# ------------------------------------------------------------------ camera: fallback + jerk
def test_jerk_p95_synthetic():
    from opencourtvision.reel.ball import jerk_p95
    t = np.arange(200, dtype=float)
    assert jerk_p95(3.0 * t + 50)["px_per_frame3"] == pytest.approx(0.0, abs=1e-9)
    j = jerk_p95(0.001 * t ** 3, 3840)                                    # d3 of a t^3 = 6a everywhere
    assert j["px_per_frame3"] == pytest.approx(0.006, abs=1e-6) and j["norm"] == pytest.approx(0.006 / 3840, rel=1e-3)
    x = 3.0 * t; x[100] += 10                                             # one 10 px glitch -> d3 spikes 10/30/30/10 on 4 of 197
    assert jerk_p95(x)["px_per_frame3"] < 30 and jerk_p95(x)["n"] == 197
    assert jerk_p95([1, 2, 3]) is None


def test_fallback_trigger():
    from opencourtvision.reel.ball import fallback_mask
    ok = np.ones(60, bool); ok[10:16] = False; ok[30:37] = False             # 6-frame gap: interpolate; 7-frame gap: players
    fb = fallback_mask(ok)
    assert not fb[10:16].any() and fb[30:37].all() and fb.sum() == 7
    conf = np.ones(60); conf[45:55] = 0.3                                  # low-confidence run counts as lost
    assert fallback_mask(np.ones(60, bool), conf)[45:55].all()


def test_path_frames_players_when_ball_lost():
    from opencourtvision.reel.ball import path, player_fit_x
    n = 240
    pos = np.column_stack([np.full(n, 300.0), np.full(n, 400.0)])           # ball at x=300 (analysis px)
    pos[60:200] = np.nan                                                    # lost for 140 frames
    boxes = [[(250.0, 330.0, 900), (340.0, 420.0, 800), (900.0, 980.0, 900)] for _ in range(n)]
    assert player_fit_x(boxes[0], 300.0) == pytest.approx((250 + 420) / 2)  # 2 nearest players: union 250..420
    info = {}
    cx, _ = path(pos, np.full(n, np.nan), 1920.0, 3.0, boxes=boxes, info=info)
    assert info["fallback"][60:200].all() and not info["fallback"][:60].any()
    assert info["target"][130] == pytest.approx(335.0 * 3.0, abs=1.0)       # player-fit centre, not the frame centre
    assert np.abs(np.diff(cx)).max() <= 20.0 + 1e-6
    # without player boxes and without activity, it falls back to the given default instead of holding the last ball x
    info2 = {}
    path(pos, np.full(n, np.nan), 1920.0, 3.0, info=info2)
    assert info2["target"][130] == pytest.approx(1920.0)


# ------------------------------------------------------------------ reel_beat floor (Media Director answer a)
def test_reel_beat_own_floor():
    assert point_end(T, ball_dead=T + 1.0, clip_type="reel_beat") == pytest.approx(T + 1.8)      # outcome + 0.8
    assert point_end(T, ball_dead=T + 0.1, clip_type="reel_beat") == pytest.approx(T + 0.9)      # below the 2 s floor: allowed
    assert point_end(T, ball_dead=T + 0.1, buffer=0.0, clip_type="reel_beat") == pytest.approx(T + 0.6)   # t_last + 0.6
    assert point_end(T, ball_dead=T + 1.0, buffer=0.0, clip_type="reel_beat") == pytest.approx(T + 1.4)   # outcome + 0.4
    assert point_end(T, clip_type="reel_beat") == pytest.approx(T + 2.5)                          # outcome unknown: point post-roll


def test_reel_final_clip_uses_point_postroll():
    from opencourtvision.reel.reel import resolve_moments
    rallies = [{"times": [10.0, 11.0, 12.0]}, {"times": [30.0, 31.0]}]
    ev = [Event("bounce", 12.5), Event("bounce", 31.4)]
    cfg = {"moments": [{"name": "a", "start": 9.9, "end": 12.0}, {"name": "b", "start": 29.9, "end": 31.0},
                       {"name": "end", "start": 31.0, "end": 34.0, "end_card": True}]}
    a, b, end = resolve_moments(cfg, rallies, ev)
    assert a["end_kind"] == "reel_beat" and a["b"] == pytest.approx(13.3)                      # bounce 12.5 + 0.8
    assert b["end_kind"] == "point" and b["b"] == pytest.approx(33.5)                           # last play clip: max(+2.5, bounce + 1.5)
    assert end["end_kind"] is None


# ------------------------------------------------------------------ camera caps, reversal, zoom range (answers b, c)
def _changes(v, eps=1e-9):
    s = np.sign(np.where(np.abs(v) > eps, v, 0)); last = 0; idx = []
    for i, x in enumerate(s):
        if x and x != last: idx.append(i); last = x
    return idx


def test_pan_cap_and_punch_exemption():
    from opencourtvision.reel.camera import CAMERA, pan_caps
    fps, n = 60, 600
    cw = np.full(n, 911.0)
    target = np.where(np.arange(n) < 100, 1000.0, 2800.0)                                 # a big jump
    cx = pan_caps(target, cw, fps)
    assert np.abs(np.diff(cx)).max() <= CAMERA["pan_max"] * 911 / fps + 1e-9                # <= 0.35 crop-widths/s
    assert np.abs(np.diff(cx, 2)).max() <= CAMERA["pan_acc"] * 911 / fps ** 2 + 1e-9
    assert cx[-1] == pytest.approx(2800.0, abs=1.0)
    fast = pan_caps(target, cw, fps, punch_in=np.ones(n, bool))                              # explicit punch-in: exempt
    assert np.abs(np.diff(fast)).max() > 100


def test_no_reversal_within_half_second():
    from opencourtvision.reel.camera import pan_caps, zoom_caps
    fps, n = 60, 400
    t = np.arange(n)
    target = 1900 + 300 * np.sign(np.sin(2 * np.pi * t / 24))                               # flips every 0.2 s
    cx = pan_caps(target, np.full(n, 911.0), fps)
    ch = _changes(np.diff(cx))
    assert len(ch) >= 1 and all(b - a >= 30 for a, b in zip(ch, ch[1:]))
    h = zoom_caps(np.where((t // 12) % 2 == 0, 0.6, 1.0), fps)
    zc = _changes(np.diff(np.log(h)))
    assert all(b - a >= 30 for a, b in zip(zc, zc[1:]))


def test_zoom_rate_and_range():
    from opencourtvision.reel.camera import CAMERA, crop_size, zoom_caps, zoom_target
    fps, n = 60, 900
    assert tuple(int(v) for v in crop_size(0.75)) == (912, 1620) and tuple(int(v) for v in crop_size(1.0)) == (1216, 2160)
    fb = np.zeros(n, bool); fb[600:800] = True
    tgt = zoom_target(n, fps, 0.0, fallback=fb, big_shots=[5.0])
    assert tgt[0] == CAMERA["h_default"] and tgt[int(5.0 * fps)] == CAMERA["h_tight"] and tgt[700] == CAMERA["h_ceiling"]
    assert zoom_target(10, fps, 0.0, big_shots=[])[0] == 0.75                               # no verified big shot: never tight
    h = zoom_caps(tgt, fps)
    assert h.min() >= CAMERA["h_tight"] - 1e-9 and h.max() <= CAMERA["h_wide"] + 1e-9
    assert np.abs(np.diff(np.log(h))).max() <= np.log1p(CAMERA["zoom_rate"]) / fps + 1e-12  # <= 22 %/s (v4.3)
    sp = np.full(n, 0.95 * 912)                                                              # players spread: widen
    assert zoom_target(n, fps, spread_px=sp)[0] == pytest.approx(0.95 * 912 / 0.9 * 16 / 9 / 2160)   # as much as needed


def test_wide_mode_trigger_and_filter():
    from opencourtvision.reel.camera import wide_filter, wide_trigger
    sp = np.full(100, 1400.0)                                                                # wider than the full-height crop
    assert wide_trigger(sp, [1.0, 1.5, 2.0, 2.6], 60)[0]                                     # fast net exchange
    assert not wide_trigger(sp, [1.0, 3.0, 5.0], 60)[0]                                      # slow rally: no
    assert not wide_trigger(np.full(100, 900.0), [1.0, 1.5, 2.0], 60)[0]                     # fits at full height: no
    f = wide_filter(60, 2)
    assert "boxblur" in f and "overlay" in f and f.endswith("split=2")


def test_jerk_qa_baseline():
    from opencourtvision.reel.camera import CAMERA, jerk_qa
    assert jerk_qa(1.0)["pass"] is None                                                      # no baseline configured
    cfg = dict(CAMERA, jerk_baseline=4.0)
    assert jerk_qa(1.9, cfg)["pass"] is True and jerk_qa(2.1, cfg)["pass"] is False


# ------------------------------------------------------------------ camera round 3 (Media Director answers)
def test_big_shot_tighten_65_with_lookahead():
    from opencourtvision.reel.camera import CAMERA, zoom_caps, zoom_rates, zoom_target
    fps, n, t = 60, 600, 5.0
    tgt = zoom_target(n, fps, 0.0, big_shots=[t])
    h = zoom_caps(tgt, fps, CAMERA, zoom_rates(n, fps, 0.0, [t]))
    i = int(t * fps)
    assert CAMERA["h_tight"] == 0.65
    assert h[i] == pytest.approx(0.65, abs=2e-3)                                           # arrives on the shot
    start = np.where(h < 0.75 - 1e-6)[0][0] / fps
    assert t - start == pytest.approx(1.3, abs=0.05)                                        # ease starts ~1.3 s before
    assert np.abs(np.diff(np.log(h))).max() <= np.log1p(CAMERA["zoom_rate"]) / fps + 1e-12  # 22 %/s (v4.3)
    assert h.min() >= 0.65 - 1e-9


def test_punch_in_allows_25_percent_only_on_verified_shot():
    from opencourtvision.reel.camera import CAMERA, zoom_caps, zoom_rates, zoom_target
    fps, n, t = 60, 600, 5.0
    r = zoom_rates(n, fps, 0.0, [t], punch_in=True)
    zr = CAMERA["zoom_rate"]
    assert r.max() == 0.25 and r[0] == zr and r[int(t * fps)] == 0.25
    assert (zoom_rates(n, fps, 0.0, [], punch_in=True) == zr).all()                        # no verified shot: no punch
    assert (zoom_rates(n, fps, 0.0, [t], punch_in=False) == zr).all()
    h = zoom_caps(zoom_target(n, fps, 0.0, big_shots=[t], punch_in=True), fps, CAMERA, r)
    d = np.abs(np.diff(np.log(h)))
    assert d.max() > np.log1p(zr) / fps and d.max() <= np.log1p(0.25) / fps + 1e-9
    assert h[int(t * fps)] == pytest.approx(0.65, abs=3e-3)
    assert t - np.where(h < 0.75 - 1e-6)[0][0] / fps < 0.8                                 # shorter, faster ramp


def _vboxes(n, top, bot):
    return [[(100.0, 200.0, 900, float(top[i]), float(bot[i]))] for i in range(n)]


def test_vertical_dead_zone_and_caps():
    from opencourtvision.reel.camera import CAMERA, crop_size, vertical_box, vertical_follow
    fps, n, SH = 60, 600, 2160
    ch = np.full(n, float(crop_size(0.75)[1]))
    # small wobble of the box centre (< 7.5 % of crop height): camera does not move
    wob = 30 * np.sin(np.arange(n) / 7.0)
    top, bot = vertical_box(_vboxes(n, (300 + wob) / 3, (1400 + wob) / 3), 3.0)
    y = vertical_follow(top, bot, ch, fps, SH)
    assert np.ptp(y) <= 1
    # big jump of the box: follows, under 0.35 crop-heights/s, no reversal within 0.5 s
    jump = np.where(np.arange(n) < 200, 300.0, 700.0)
    top, bot = vertical_box(_vboxes(n, jump / 3, (jump + 1100) / 3), 3.0)
    y = vertical_follow(top, bot, ch, fps, SH)
    assert y[-1] > y[0] + 100
    assert np.abs(np.diff(y)).max() <= CAMERA["pan_max"] * ch[0] / fps + 1.0               # +1 px rounding
    ch_ = _changes(np.diff(y.astype(float)))
    assert all(b - a >= 29 for a, b in zip(ch_, ch_[1:]))                                  # 0.5 s, -1 frame of whole-px rounding


def test_vertical_box_widens_instead_of_clipping():
    from opencourtvision.reel.camera import need_height, vertical_box, zoom_target
    fps, n = 60, 300
    top, bot = vertical_box(_vboxes(n, np.full(n, 50.0), np.full(n, 700.0)), 3.0)          # 150..2100 px: needs ~0.99
    need = need_height(top, bot, fps)
    assert need[0] == pytest.approx((2100 - 150 + 2 * 0.04 * 2160) / 2160)
    h = zoom_target(n, fps, need_h=need)
    assert (h >= need - 1e-9).all() and h[0] > 0.75


def test_vertical_fallback_without_boxes():
    from opencourtvision.reel.camera import crop_size, vertical_box, vertical_follow
    n = 50
    top, bot = vertical_box([[(1.0, 2.0, 300)] for _ in range(n)], 3.0)                   # old 3-tuple blobs: no y
    assert np.isnan(top).all()
    ch = np.full(n, float(crop_size(0.75)[1]))
    assert (vertical_follow(top, bot, ch, 60) == (2160 - 1620) // 2).all()


def test_jerk_float_path_vs_rounded_and_baseline_helpers(tmp_path):
    from opencourtvision.reel.ball import crop_cmd_centres, jerk_p95, pooled_jerk, smooth_mean
    from opencourtvision.reel.camera import CAMERA, QUANT_FLOOR_PX, jerk_qa
    x = 1000 + 5.3 * np.arange(600)                                                       # smooth pan
    assert jerk_p95(x)["px_per_frame3"] == pytest.approx(0.0, abs=1e-6)                   # float path: metric (a)
    assert jerk_p95(np.round(x))["px_per_frame3"] == QUANT_FLOOR_PX                       # rounded: debug only
    assert np.allclose(smooth_mean(np.full(20, 7.0)), 7.0) and len(smooth_mean(np.arange(20.0))) == 20
    # pooled over cuts: a jump between two shots never enters the third differences
    j = pooled_jerk([np.zeros(50), np.full(50, 900.0)], 3840)
    assert j["px_per_frame3"] == 0.0 and j["n"] == 94
    assert pooled_jerk([np.round(x)], 3840, smooth_k=9)["px_per_frame3"] < QUANT_FLOOR_PX
    f = tmp_path / "cmd_m1.txt"
    f.write_text("0.0000 crop w 1216, crop h 2160, crop x 1934, crop y 0;\n0.0167 crop w 1124, crop h 2000, crop x 957, crop y 80;\n")
    assert list(crop_cmd_centres(f)) == [2542.0, 1519.0]
    q = jerk_qa(0.1, dict(CAMERA, jerk_baseline=0.333333, jerk_baseline_source="cross-check"))
    assert q["pass"] is True and q["limit"] == pytest.approx(0.1666665, abs=1e-6) and q["source"] == "cross-check"
    assert jerk_qa(0.2, dict(CAMERA, jerk_baseline=0.333333))["pass"] is False


def test_example_config_baseline_is_marked_cross_check():
    import json
    import pathlib
    cam = json.loads((pathlib.Path(__file__).parents[1] / "examples/reel_moments.example.json").read_text())["camera"]
    assert cam["jerk_baseline"] == pytest.approx(0.333333) and cam["jerk_baseline_source"].startswith("cross-check")


def test_crop_render_maps_variable_crop_to_fixed_output():
    from opencourtvision.reel.camera import crop_render, crop_size
    rng = np.random.default_rng(1)
    SW, SH = 3840, 2160
    h = rng.uniform(0.65, 1.0, 200)
    cw, ch = crop_size(h, SH)
    x = np.array([rng.integers(0, SW - w + 1) for w in cw])
    y = np.array([rng.integers(0, SH - c + 1) for c in ch])
    p = crop_render(x, y, cw, ch, SW, SH)
    assert (p["win_x"] <= x).all() and (x + cw <= p["win_x"] + p["cw0"]).all()            # window holds the crop
    k = 1920 / ch
    assert np.abs(p["win_x"] + p["ox"] / k - x).max() <= 1.0                               # output starts at crop x
    assert np.abs(p["oy"] / k - y).max() <= 1.0
    assert (p["ox"] + 1080 <= p["sw"]).all() and (p["oy"] + 1920 <= p["sh"]).all()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_crop_render_ffmpeg_keeps_every_frame_through_zoom(tmp_path):
    import subprocess

    from opencourtvision.reel.camera import crop_render, crop_render_cmds, crop_render_head, crop_size
    n, SW, SH = 60, 1280, 720
    cw, ch = crop_size(np.linspace(0.75, 0.65, n), SH)                                     # size changes every frame
    x, y = np.full(n, 400), np.full(n, 60)
    p = crop_render(x, y, cw, ch, SW, SH)
    (tmp_path / "c.txt").write_text("\n".join(crop_render_cmds(p, 60)) + "\n")
    head = crop_render_head(p, "c.txt", SH, 60, 1)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={SW}x{SH}:r=60:d=1",
                    "-filter_complex", head + "[o]", "-map", "[o]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "o.mp4"],
                   check=True, cwd=tmp_path)
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames",
                          "-of", "csv=p=0", "o.mp4"], capture_output=True, text=True, cwd=tmp_path).stdout.strip()
    assert out == f"1080,1920,{n}"


# ------------------------------------------------------------------ hybrid pan rule (Media Director)
def _rally(fps=60, dur=12.0):
    """Ball x (4K px) flying between alternating contact points, like the synthetic render clip."""
    ct = [0.8, 1.9, 3.0, 4.0, 5.2, 6.3, 7.3, 8.5, 9.6, 10.6]
    cx = [1500, 2600, 1100, 2900, 1900, 900, 2700, 1600, 2400, 1200]
    ts = np.arange(int(dur * fps)) / fps
    return ts, np.interp(ts, ct, cx), ct, cx


def test_fast_pan_cap_only_between_verified_contacts():
    from opencourtvision.events import Event
    from opencourtvision.reel.camera import CAMERA, flight_mask, pan_caps
    from opencourtvision.reel.reel import verified_contacts
    ev = [Event("contact", 1.0, verified=True), Event("contact", 2.0, verified=True),
          Event("contact", 2.5, verified=False), Event("contact", 2.7, verified=True, source="swingvision"),
          Event("contact", 6.0, verified=True)]
    ct = verified_contacts(ev, 0, 10)
    assert ct == [1.0, 2.0, 6.0]
    fps, n = 60, 480
    m = flight_mask(n, fps, 0.0, ct)
    ts = np.arange(n) / fps
    assert m[(ts > 1.0) & (ts < 2.0)].all() and not m[(ts <= 1.0) | (ts >= 2.0)].any()     # 2.0 -> 6.0 is > 2.5 s: no
    cw = np.full(n, 912.0)
    x = pan_caps(np.where(ts < 0.5, 0.0, 6000.0), cw, fps, None, CAMERA, m)
    v = np.abs(np.diff(x)) * fps / 912
    assert v.max() <= CAMERA["pan_fast"] + 1e-9 and v[m[1:]].max() > CAMERA["pan_max"] + 0.05
    assert v[~m[1:]].max() <= CAMERA["pan_max"] + 1e-9


def test_lookahead_moves_before_the_contact():
    from opencourtvision.reel.camera import CAMERA, follow_pan, pan_caps, plan_pan
    fps, n = 60, 360
    ts = np.arange(n) / fps
    tgt = np.interp(ts, [2.0, 3.0], [1500.0, 2700.0])                                       # ball leaves at the 2.0 s hit
    pp = plan_pan(tgt, fps, 0.0, [2.0, 3.0], [1500.0, 2700.0], 912)
    cam = follow_pan(pp["planned"], np.full(n, 912.0), 912, fps, pp["fast"])
    chase = pan_caps(tgt, np.full(n, 912.0), fps, None, CAMERA)
    first = lambda c: ts[np.argmax(np.abs(c - c[0]) > 2.0)]                                # noqa: E731
    assert first(cam) < 2.0 <= first(chase)                                                  # moving before the hit
    assert abs(cam[int(3.0 * fps)] - 2700) < abs(chase[int(3.0 * fps)] - 2700)             # closer at the next contact


def test_widen_before_panning_when_exit_predicted():
    from opencourtvision.reel.camera import CAMERA as CAM43
    from opencourtvision.reel.camera import crop_size, follow_pan, plan_pan, zoom_caps, zoom_target
    CAMERA = dict(CAM43, pan_acc=0.6, zoom_rate=0.15)     # 303246d follower (v4.3 keeps it only as the no-contact fallback)
    fps, n = 60, 600
    ts = np.arange(n) / fps
    tgt = np.interp(ts, [5.0, 5.8, 6.6], [1500.0, 2400.0, 1500.0])                          # fast excursion and back
    cw_def = int(crop_size(0.75)[0])
    pp = plan_pan(tgt, fps, 0.0, [], [], cw_def, cfg=CAMERA)
    assert pp["exit_frames"] > 0
    first_exit = int(np.argmax(pp["exit"]))                                                # where the capped pan would lose it
    first_need = int(np.argmax(pp["need_h"] > 0))
    assert 0 < first_exit - first_need <= round(CAMERA["exit_lead_s"] * fps)              # only exits < 0.5 s ahead
    assert pp["need_h"].max() <= CAMERA["h_ceiling"] + 1e-12                               # no fast flight: soft ceiling
    h = zoom_caps(zoom_target(n, fps, need_h=pp["need_h"], cfg=CAMERA), fps, CAMERA)
    assert np.argmax(h > 0.76) < first_exit                                                # widening starts first
    assert np.abs(np.diff(np.log(h))).max() <= np.log1p(CAMERA["zoom_rate"]) / fps + 1e-12  # within the zoom cap
    cw, _ = crop_size(h)
    cam_w = follow_pan(pp["planned"], cw, cw_def, fps, pp["fast"], cfg=CAMERA)
    cam_n = follow_pan(pp["planned"], np.full(n, float(cw_def)), cw_def, fps, pp["fast"], cfg=CAMERA)
    assert (np.abs(np.diff(cam_w)) * fps / cw[1:]).max() <= CAMERA["pan_max"] + 1e-9
    assert np.abs(np.diff(cam_w)).sum() < np.abs(np.diff(cam_n)).sum()                     # pans only the remainder
    assert (np.abs(tgt - cam_w) <= cw / 2).mean() >= (np.abs(tgt - cam_n) <= cw_def / 2).mean()


def test_ceiling_override_player_spread():
    from opencourtvision.reel.camera import CAMERA, crop_size, zoom_target
    n, fps = 120, 60
    w_ceil = crop_size(CAMERA["h_ceiling"])[0]
    fb = np.ones(n, bool)
    assert (zoom_target(n, fps, fallback=fb) == CAMERA["h_ceiling"]).all()                 # player framing alone: 90 %
    sp = np.full(n, 1.0 * w_ceil)                                                          # players wider than a 90 % crop
    h = zoom_target(n, fps, fallback=fb, spread_px=sp)
    assert (h > CAMERA["h_ceiling"]).all() and (h <= 1.0).all()
    assert (zoom_target(n, fps, spread_px=np.full(n, 3000.0)) == 1.0).all()                # physically needs everything


def test_ceiling_override_fast_stretch_only():
    from opencourtvision.reel.camera import CAMERA, crop_size, plan_pan
    fps, n = 60, 480
    ts = np.arange(n) / fps
    tgt = np.interp(ts, [2.0, 2.6, 5.0, 5.6], [800.0, 3000.0, 3000.0, 800.0])              # two very fast swings
    cw_def = int(crop_size(0.75)[0])
    pp = plan_pan(tgt, fps, 0.0, [2.0, 2.6], [800.0, 3000.0], cw_def)                      # only the first is verified flight
    fast = pp["fast"]
    assert fast.any() and pp["need_h"][fast].max() > CAMERA["h_ceiling"]                   # verified flight may exceed 90 %
    assert pp["need_h"][~fast].max() <= CAMERA["h_ceiling"] + 1e-12                        # elsewhere capped at 90 %
    assert pp["need_h"][ts > 4.0].max() == pytest.approx(CAMERA["h_ceiling"])              # the unverified swing hits the cap


def test_crop_height_soft_check():
    from opencourtvision.reel.camera import crop_height_qa
    ok = crop_height_qa(np.r_[np.full(80, 0.75), np.full(20, 0.95)])
    assert ok["mean_crop_height"] == 0.79 and ok["pct_frames_ge95"] == 20.0 and ok["warnings"] == []
    bad = crop_height_qa(np.full(100, 0.99))
    assert len(bad["warnings"]) == 2 and "pass" not in bad                                  # warning only, never a failure


def test_hybrid_pan_jerk_under_limit_on_synthetic_rally():
    from opencourtvision.reel.ball import jerk_p95
    from opencourtvision.reel.camera import CAMERA, crop_size, flight_mask, follow_pan, plan_pan, zoom_caps, zoom_target
    fps = 60
    ts, bx, ct, cx = _rally(fps)
    n = len(ts); cw_def = int(crop_size(0.75)[0])
    pp = plan_pan(bx, fps, 0.0, ct, cx, cw_def)
    h = zoom_caps(zoom_target(n, fps, need_h=pp["need_h"]), fps)
    cw, _ = crop_size(h)
    cam = np.clip(follow_pan(np.clip(pp["planned"], cw / 2, 3840 - cw / 2), cw, cw_def, fps, pp["fast"]), cw / 2, 3840 - cw / 2)
    j = jerk_p95(cam, 3840)["px_per_frame3"]
    assert j < 0.5 * 0.333333                                                                # 0.1667 limit (float path)
    v = np.abs(np.diff(cam)) * fps / cw[1:]
    fast = flight_mask(n, fps, 0.0, ct)
    assert v.max() <= CAMERA["pan_fast"] + 1e-6 and v[~fast[1:]].max() <= CAMERA["pan_max"] + 1e-6
    a = np.abs(np.diff(cam, 2)) * fps ** 2 / np.median(cw)
    assert a.max() <= CAMERA["pan_acc"] + 1e-6
