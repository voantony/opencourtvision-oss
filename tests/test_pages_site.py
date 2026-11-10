"""Built Pages tree: homepage at /, labeler at /label/, no old host."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from opencourtvision.labels.call_game import to_team_call

ROOT = Path(__file__).resolve().parents[1]


def build(public):
    spec = importlib.util.spec_from_file_location("build_pages", ROOT / "scripts" / "build_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(public)
LIVE_HOST = "https://opencourtvision-10a09b.gitlab.io"
UNAPPROVED_HOST = "antonyvo.gitlab.io"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        raw = dict(attrs)
        if tag == "script":
            self.scripts += 1
        for key in ("href", "src"):
            if raw.get(key):
                self.hrefs.append(raw[key])


def test_built_site_shape(tmp_path):
    public = build(tmp_path / "public")
    assert (public / "index.html").is_file()
    assert (public / "es.html").is_file()
    assert (public / "label" / "index.html").is_file()
    assert (public / "assets" / "c-bang-mark.svg").is_file()
    assert (public / "fonts" / "archivo" / "OFL.txt").is_file()
    home = (public / "index.html").read_text(encoding="utf-8")
    es = (public / "es.html").read_text(encoding="utf-8")
    label = (public / "label" / "index.html").read_text(encoding="utf-8")
    parser = Links()
    parser.feed(home)
    assert parser.scripts == 0
    assert "label/" in parser.hrefs
    assert "https://fonts.googleapis.com" not in home
    assert "fonts.googleapis.com" not in (public / "home.css").read_text(encoding="utf-8")
    assert "Graba un partido" in es
    assert "Beat the Engine" in es
    blob = "\n".join(path.read_text(encoding="utf-8", errors="ignore")
                     for path in public.rglob("*") if path.is_file() and path.suffix in {".html", ".js", ".css", ".json", ".txt"})
    assert UNAPPROVED_HOST not in blob
    assert LIVE_HOST in (public / "label" / "share.js").read_text(encoding="utf-8")
    assert (public / "CC_ATTRIBUTION.txt").is_file()
    for href in parser.hrefs:
        assert UNAPPROVED_HOST not in href
        assert not href.startswith("/label")
    assert 'href="label/"' in label or 'href="../index.html"' in label
    clips = (public / "label" / "clips.json").read_text(encoding="utf-8")
    assert '"answer"' not in clips
    practice = (public / "label" / "practice.json").read_text(encoding="utf-8")
    assert "2p6PcKP0720_p4" in practice and "2p6PcKP0720_p3" in practice and "2p6PcKP0720_p5" in practice


def test_delete_link_on_footer_and_every_game_screen():
    for name in ("index.html", "es.html"):
        text = (ROOT / "site" / name).read_text(encoding="utf-8")
        assert 'id="delete-data"' in text.split("<footer", 1)[1]
        assert "mailto:antonyvo@mail.grokbot.com?subject=Delete%20my%20labels" in text
    label = (ROOT / "labeler-pages" / "index.html").read_text(encoding="utf-8")
    main, footer = label.split('<main id="app">', 1)[1].split("</main>", 1)
    assert "delete-data" not in main
    assert 'id="delete-data"' in footer
    assert "mailto:antonyvo@mail.grokbot.com?subject=Delete%20my%20labels" in footer
    app = (ROOT / "labeler-pages" / "app.js").read_text(encoding="utf-8")
    assert "document.cookie" not in app
    for fn in ("showStart", "showWarmup", "showLive", "showSummary", "showBoard"):
        body = app.split("function " + fn, 1)[1].split("\n  function ", 1)[0]
        assert "refreshFooter()" in body


def test_ends_mapping_and_no_answers_on_clips():
    clips = json.loads((ROOT / "labeler-pages" / "clips.json").read_text(encoding="utf-8"))["clips"]
    by = {c["id"]: c for c in clips}
    far = {"GKWNVYeEAlE_p11", "GKWNVYeEAlE_p12"}
    others = [c for c in clips if c["video"] in ("GKWNVYeEAlE", "632k3fKraeA") and c["id"] not in far]
    assert len(others) == 39
    for clip in others:
        assert to_team_call("A", clip["a_side_team"]) == "A"
        assert to_team_call("B", clip["a_side_team"]) == "B"
    for cid in far:
        assert to_team_call("A", by[cid]["a_side_team"]) == "B"
        assert to_team_call("B", by[cid]["a_side_team"]) == "A"
    for clip in clips:
        assert "answer" not in clip
        assert clip["file"] == "clips/" + clip["id"].replace("_p", "_") + ".mp4"


def test_node_client_rules():
    proc = subprocess.run(["node", str(ROOT / "tests" / "call_client_logic.js")], check=True, capture_output=True, text=True)
    assert "entry.343297185=se-test" in proc.stdout
    body = proc.stdout.split("FORM_BODY ", 1)[1].strip()
    assert "docs.google.com" not in body
    assert "email" not in body
