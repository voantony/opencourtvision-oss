"""Highlights only (app/highlights.py): audio rallies, drill splitting, audio+rules rank, live gate, focus crop,
recruiting sections, and end-to-end social + recruiting exports on the synthetic match clip."""
import json
import os
import shutil

import numpy as np
import pytest

from opencourtvision.app import highlights as H
from opencourtvision.style.plan import clip_score


def _r(times, amp=20.0, pause=30.0):
    g = H.group_hits(np.array(times, float), np.full(len(times), amp), 4.0)
    assert len(g) == 1
    g[0]["pause_before"] = pause
    return g[0]


def test_group_hits_splits_on_gap():
    t = np.array([1, 2, 3, 10, 11, 12, 13.0]); a = np.ones(7) * 20
    g = H.group_hits(t, a, 4.0)
    assert [x["hits"] for x in g] == [3, 4] and g[1]["pause_before"] == pytest.approx(7.0)


def test_real_rally_never_split_drill_is():
    rally = _r(list(np.arange(0, 30, 1.2)))                         # 30 s live rally: kept whole
    assert H.split_long(rally) == [rally]
    drill = _r([t for k in range(20) for t in (k * 5.0, k * 5.0 + 1.1, k * 5.0 + 2.2)])   # 100 s feed drill
    parts = H.split_long(drill)
    assert len(parts) > 3 and all(p["t1"] - p["t0"] <= H.HL["piece_s"] for p in parts)
    assert sum(p["hits"] for p in parts) == drill["hits"] and all(p.get("split_from_drill") for p in parts)


def test_rank_prefers_long_fast_rallies_and_ignores_short():
    lb = np.full(100 * 400, -40.0)
    R = [_r([10, 11, 12]), _r(list(np.arange(50, 70, 0.9))), _r([100, 101]), _r([150, 152, 154, 156])]
    C = H.rank(R, lb)
    assert len(C) == 3 and C[0]["t0"] == 50 and all("rank_score" in r for r in C)


def test_plan_uses_rank_score():
    assert clip_score(dict(first_hit=0, last_hit=10, hits=3, react=0, rank_score=-1.5)) == -1.5
    assert clip_score(dict(first_hit=0, last_hit=10, hits=3, react=0)) == 8.0


def test_live_gate_drops_idle_only():
    C = [dict(motion=dict(activity=a)) for a in (0.1, 0.1, 0.12, 0.09, 0.01)] + [dict()]
    keep, drop = H.live_gate(C)
    assert drop == 1 and len(keep) == 5


def test_focus_cx():
    r = dict(motion=dict(cx_near=0.8, cx_far=0.3, cx_all=0.5))
    assert H.focus_cx(r, None) == 0.5 and H.focus_cx(r, "near") == pytest.approx(0.8)
    assert H.focus_cx(r, "far") == pytest.approx(0.3) and H.focus_cx(r, "0.2") == pytest.approx(0.2)
    assert H.focus_cx(dict(), "near") == 0.5


def test_recruiting_sections_audio_types_only():
    lb = np.full(100 * 2000, -40.0)
    R = []
    for k in range(12):
        t0 = 20 + k * 40.0
        times = [t0, t0 + 1.4] + list(t0 + 2.8 + np.cumsum([0.7, 0.7, 0.7, 0.8] if k % 3 == 0 else [1.5] * 6))
        R.append(_r(times, pause=25.0))
    C = H.rank(R, lb)
    secs = H.recruiting_sections(C, R)
    names = [s[0] for s in secs]
    assert names[0] == "SERVES" and "NET PLAY" in names and names[-1] in ("POINTS", "RALLIES")
    spans = [(r.get("clip") or H._bounds(r)) for _, rs in secs for r in rs]
    spans.sort()
    assert all(b[0] >= a[1] - 1e-6 for a, b in zip(spans, spans[1:]))   # no clip shown twice


@pytest.mark.skipif(shutil.which("ffmpeg") is None and not os.path.exists(os.path.expanduser("~/bin/ffmpeg")), reason="ffmpeg")
def test_end_to_end_social_and_recruiting(clips, tmp_path):
    v = clips["hits_clean"]["mp4"]
    r = H.run(v, out=str(tmp_path / "s"), aspects=("9:16", "16:9"), motion=False, title="Club final")
    for asp in ("9:16", "16:9"):
        e = r["exports"][asp]
        assert os.path.getsize(e["reel"]) > 0 and e["n_clips"] >= 3 and len(e["clips"]) == e["n_clips"]
        assert all(os.path.getsize(c) > 0 for c in e["clips"])
    assert r["minutes_per_hour_of_footage"] > 0 and json.load(open(tmp_path / "s" / "highlights.json"))["rallies"] >= 3
    q = H.run(v, out=str(tmp_path / "r"), preset="recruiting", motion=False, mute=True,
              profile=dict(name="Alex Example", class_year="2028", ranking="UTR 8.4", contact="alex@example.com"))
    e = q["exports"]["16:9"]
    assert os.path.getsize(e["reel"]) > 0 and e["muted"] and e["sections"]


def _motion(a, b, fps=3, x=0.3):
    t = np.arange(a, b, 1 / fps)
    maps = np.zeros((len(t), 72, 128), np.float32)
    for i, ti in enumerate(t):
        cx = int((x + 0.2 * np.sin(ti)) * 128); maps[i, 50:60, max(0, cx - 3):cx + 3] = 50.0
    return dict(t=t, x=np.array([x + 0.2 * np.sin(ti) for ti in t]), maps=maps)


def test_broadcast_grammar_169_locked_wide_then_reaction():
    from opencourtvision.reel.camera import broadcast_check, broadcast_plan
    c = dict(id=1, a=10.0, b=30.5, first_hit=13.5, last_hit=28.0, hits_t=list(np.arange(13.5, 28.1, 1.2)))
    segs = broadcast_plan(c, _motion(10, 31), "16:9", 3840, 2160, 30)
    assert [s["kind"] for s in segs] == ["wide", "reaction"]
    assert segs[0]["t0"] == 10.0 and segs[0]["t1"] == pytest.approx(29.0) and segs[1]["t1"] == 30.5
    x, y, w, h = segs[1]["window"]
    assert w == 1920 and h == 1080 and 0 <= x <= 3840 - w            # 2x zoom inside the frame
    segs = broadcast_plan(c, _motion(10, 31), "16:9", 3840, 2160, 30, starred=True)
    rep = segs[-1]
    assert rep["kind"] == "replay" and rep["speed"] >= 0.45 and 2.0 <= 3840 / rep["window"][2] <= 4.0
    with pytest.raises(ValueError):
        broadcast_check([dict(framing=segs, starred=False)])
    with pytest.raises(ValueError):
        broadcast_check([dict(framing=segs, starred=True)] * 3)
    assert broadcast_check([dict(framing=segs, starred=True)] * 2)


def test_broadcast_916_follow_respects_pan_caps():
    from opencourtvision.reel.camera import CAMERA, broadcast_plan
    c = dict(id=1, a=10.0, b=30.5, first_hit=13.5, last_hit=28.0, hits_t=list(np.arange(13.5, 28.1, 1.2)))
    f = broadcast_plan(c, _motion(10, 31), "9:16", 3840, 2160, 30)[0]
    assert f["kind"] == "follow" and f["long_shot"]["mode"] == "cut"
    v = np.abs(np.diff(f["path"])) * 30 / f["cw"]
    for t in f["long_shot"]["cuts"]: v[int(round((t - c["a"]) * 30)) - 1] = 0     # a hard cut is not a pan
    assert v.max() <= CAMERA["pan_fast"] + 1e-6 and f["path"].std() > 1.0  # it moves, within the caps
    for mode, cap in (("fast_pan", CAMERA["pan_fast_flight"]), ("off", CAMERA["pan_fast"])):
        g = broadcast_plan(c, _motion(10, 31), "9:16", 3840, 2160, 30, camera_cfg=dict(CAMERA, long_shot_mode=mode))[0]
        assert np.abs(np.diff(g["path"])).max() * 30 / g["cw"] <= cap + 1e-6 and not g["long_shot"].get("cuts")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_crop_chains_keep_colour_on_nv12_hw_frames(tmp_path, monkeypatch):
    """v4.9 regression (Antony's oct17 9:16 reel had a blue clay court on ~1/3 of frames): VideoToolbox decodes to
    semi-planar NV12; a crop at an odd x offset there swaps U and V. Every crop chain must go planar first."""
    import subprocess

    from opencourtvision.style import render as R
    from opencourtvision.style.preset import load
    seen = []; real_run = subprocess.run
    monkeypatch.setattr(R.subprocess, "run", lambda cmd, **k: seen.append(cmd))
    path = np.linspace(1001.0, 2203.0, 15)                          # odd crop offsets
    clip = dict(id=1, framing=[dict(kind="follow", t0=0.0, t1=0.5, path=path, cw=1215),
                               dict(kind="reaction", t0=0.5, t1=1.0, window=(1001, 675, 607, 1081))])
    R.render_broadcast_clip("in.mov", clip, load("vamos_clean_v1"), "9:16", str(tmp_path), fps=30)
    fcs = [c[c.index("-filter_complex") + 1] for c in seen if "-filter_complex" in c]
    assert len(fcs) == 2
    src = tmp_path / "orange.mov"
    real_run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=0xD9754C:s=3840x2160:r=30:d=1",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "1", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    for k, fc in enumerate(fcs):
        fc = fc.split(";[0:a]")[0].replace("[0:v]", "[0:v]format=nv12,", 1)            # what the hardware decoder hands the graph
        out = tmp_path / f"o{k}.rgb"
        real_run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-filter_complex", fc,
                  "-map", "[v]", "-frames:v", "1", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "8x8", str(out)],
                 check=True, cwd=str(tmp_path))
        rgb = np.frombuffer(out.read_bytes(), np.uint8).reshape(-1, 3).astype(int).mean(0)
        assert rgb[0] > rgb[2] + 40, (k, rgb)                       # still clay (red > blue), not swapped
