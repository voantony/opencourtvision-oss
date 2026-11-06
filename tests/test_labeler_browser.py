"""End-to-end test of tools/labeler in headless Chrome: serve.py + labeler.html label a synthetic match from the
keyboard; the saved labels must load in Python and reproduce the rules score at every point.

Needs playwright (python), a Chrome/Chromium with H.264, and ffmpeg; skipped otherwise. OCV_CHROME overrides the browser path.
No recorded match is used. The video is a generated test pattern in a temp directory.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

pw = pytest.importorskip("playwright.sync_api")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools", "labeler")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from synthetic_match import MATCH  # noqa: E402

CHROME = os.environ.get("OCV_CHROME") or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

pytestmark = pytest.mark.skipif(not (CHROME and shutil.which("ffmpeg")), reason="needs Chrome and ffmpeg")

sys.path.insert(0, TOOLS)
from make_seed import build_seed  # noqa: E402

from opencourtvision.labels.points import load_point_labels  # noqa: E402


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    d = tmp_path_factory.mktemp("lab")
    video = d / "synthetic.mp4"
    dur = f"{MATCH['duration_s']:.3f}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=2", "-t", dur,
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(video)], check=True)
    seed = build_seed(MATCH["segments"], MATCH["points"], video=video.name, date="2026-01-01", final=MATCH["final_score"])
    sp = d / "seed.json"; sp.write_text(json.dumps(seed))
    port = _port()
    proc = subprocess.Popen([sys.executable, os.path.join(TOOLS, "serve.py"), str(video), "--seed", str(sp), "--port", str(port), "--no-open"])
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close(); break
        except OSError:
            time.sleep(0.1)
    yield dict(url=f"http://127.0.0.1:{port}/", video=video, seed=seed, labels=d / "synthetic.labels.json")
    proc.terminate(); proc.wait()


def test_label_whole_match_from_keyboard(served):
    seed, sv = served["seed"], MATCH["points"]
    with pw.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--autoplay-policy=no-user-gesture-required"])
        pg = b.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(served["url"])
        pg.wait_for_function("window.__ocv.S && document.querySelector('video').readyState >= 1", timeout=20000)
        assert pg.evaluate("document.querySelector('video').duration") > MATCH["duration_s"] - 2
        assert pg.evaluate("window.__ocv.S.points.length") == len(seed["points"])
        pg.evaluate("document.getElementById('autoplay').checked = false")
        # editing keys: split, undo, merge-undo, server flip + revert, speed, delete + undo
        n0 = len(seed["points"])
        pg.evaluate("document.querySelector('video').currentTime = window.__ocv.S.points[0].start + 3")
        pg.wait_for_timeout(200)
        pg.keyboard.press("i"); assert pg.evaluate("window.__ocv.S.points.length") == n0 + 1
        pg.keyboard.press("z"); assert pg.evaluate("window.__ocv.S.points.length") == n0
        pg.keyboard.press("m"); assert pg.evaluate("window.__ocv.S.points.length") == n0 - 1
        pg.keyboard.press("Meta+z") if sys.platform == "darwin" else pg.keyboard.press("Control+z")
        assert pg.evaluate("window.__ocv.S.points.length") == n0
        pg.keyboard.press("Delete"); pg.keyboard.press("z")
        assert pg.evaluate("window.__ocv.S.points.length") == n0
        pg.keyboard.press("s"); assert pg.evaluate("window.__ocv.S.points[0].server_override") == "A"
        pg.keyboard.press("s"); assert pg.evaluate("window.__ocv.S.points[0].server_override") is None
        pg.keyboard.press("="); assert pg.evaluate("document.querySelector('video').playbackRate") == 1.5
        pg.keyboard.press("-")
        # label every segment: SwingVision points with A/B (and some via Enter = accept suggestion), the rest X
        truth = {q["point"]: ("A" if q["winner"] == "host" else "B") for q in sv}
        for i, s in enumerate(seed["points"]):
            assert pg.evaluate("window.__ocv.S.cur") == i
            if "sv_point" in s:
                if i % 5 == 0:
                    pg.keyboard.press("Enter")
                else:
                    pg.keyboard.press(truth[s["sv_point"]].lower())
            else:
                pg.keyboard.press("x")
        pg.wait_for_timeout(1500)                       # debounced autosave
        hud = pg.inner_text("#prog")
        assert MATCH["final_score"] in hud and f"{len(seed['points'])}/{len(seed['points'])}" in hud
        exp = pg.evaluate("window.__ocv.exportObj()")
        assert not errors, errors
        b.close()
    assert exp["decoded"]["score"] == MATCH["final_score"] and exp["decoded"]["done"] and not exp["decoded"]["provisional"]
    L = load_point_labels(served["labels"])            # written next to the video by serve.py
    assert L.complete and L.verify() == [] and L.final_score() == MATCH["final_score"]
    pts = [s for s in L.segments if s.kind == "point"]
    assert len(pts) == len(sv) == MATCH["n_points"]
    for s, q in zip(pts, sv):                          # JS rules == SwingVision score at every point
        assert s.server == ("A" if q["server"] == "host" else "B")
        assert s.score_before["games"] == q["games_before"] and s.score_before["sets"] == q["sets_before"]
        assert s.score_before["pts"] == [q["host_pts"], q["guest_pts"]]
    rows = L.training_rows()
    assert len(rows) == len(seed["points"]) and sum(r["is_point"] for r in rows) == MATCH["n_points"]
    assert {r["label_src"] for r in rows} == {"user", "user_accepted_suggestion"}


def test_file_mode_without_server(served, tmp_path):
    """Opened straight from disk: pick the video + seed with the file inputs, label one point, export downloads JSON."""
    with pw.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        pg = b.new_page(accept_downloads=True)
        pg.goto("file://" + os.path.join(TOOLS, "labeler.html"))
        pg.set_input_files("#fv", str(served["video"]))
        sp = tmp_path / "seed.json"; sp.write_text(json.dumps(served["seed"]))
        pg.set_input_files("#fs", str(sp))
        pg.wait_for_function("window.__ocv.S && document.querySelector('video').readyState >= 1", timeout=20000)
        pg.evaluate("document.getElementById('autoplay').checked = false")
        pg.keyboard.press("a")
        with pg.expect_download() as dl:
            pg.click("#bexp")
        out = tmp_path / "out.json"; dl.value.save_as(out)
        saved = pg.evaluate("localStorage.getItem('ocv-labeler:' + window.__ocv.S.video.name)")
        b.close()
    obj = json.load(open(out))
    assert obj["points"][0]["winner"] == "A" and obj["points"][0]["winner_src"] == "user" and saved


def test_favourites_pass(served):
    """F stars any segment (also ×); the favourites pass plays only labelled points, pauses 1.2 s at each end (F still
    applies to that point), N skips; favourites + 'seen' are saved and become highlight training rows."""
    lab = served["labels"]
    assert lab.exists()                                    # left by the full-match test
    with pw.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--autoplay-policy=no-user-gesture-required"])
        pg = b.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(served["url"] + "?pass=1")
        pg.wait_for_function("window.__ocv.S && window.__ocv.passMode && document.querySelector('video').readyState >= 1", timeout=20000)
        S = "window.__ocv.S"
        kinds = pg.evaluate(f"{S}.points.map(p => p.kind)")
        pts = [i for i, k in enumerate(kinds) if k == "point"]
        assert pg.evaluate(f"{S}.cur") == pts[0]
        pg.keyboard.press("f")                             # star point 1
        pg.keyboard.press("n"); assert pg.evaluate(f"{S}.cur") == pts[1]   # skips any × in between
        pg.keyboard.press("n"); assert pg.evaluate(f"{S}.cur") == pts[2]
        # let point 3 play to its end (shorten it), press F in the 1.2 s pause: it stars point 3, then the pass moves on
        pg.evaluate(f"{S}.points[{pts[2]}].end = {S}.points[{pts[2]}].start + 1.0")
        pg.keyboard.press("r")                             # replay the point: stops at the new end
        pg.wait_for_function("document.querySelector('video').paused", timeout=10000)
        pg.keyboard.press("f")
        pg.wait_for_function(f"{S}.cur === {pts[3]}", timeout=5000)
        pg.keyboard.press("v"); assert not pg.evaluate("window.__ocv.passMode")
        x = kinds.index("non_point")                       # a × segment (high five) can be starred outside the pass
        pg.evaluate(f"{S}.cur = {x}")
        pg.keyboard.press("f")
        pg.wait_for_timeout(1500)
        exp = pg.evaluate("window.__ocv.exportObj()")
        assert not errors, errors
        b.close()
    favs = [q["idx"] for q in exp["points"] if q["favorite"]]
    assert favs == sorted([pts[0] + 1, pts[2] + 1, x + 1]) and exp["decoded"]["n_favorites"] == 3
    L = load_point_labels(lab)
    assert [s.idx for s in L.favorites()] == favs and L.verify() == []
    rows = L.highlight_rows(MATCH["points"])
    fr = [r for r in rows if r["favorite"]]
    assert len(fr) == 3 and sum(not r["is_point"] for r in fr) == 1
    assert all(r["seen"] for r in fr) and sum(r["seen"] for r in rows if r["is_point"]) >= 3
    assert all("pq" in r for r in fr if r["is_point"])
