"""Built Pages tree: homepage at /, labeler at /label/, no old host."""
from __future__ import annotations

import importlib.util
import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(public):
    spec = importlib.util.spec_from_file_location("build_pages", ROOT / "scripts" / "build_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(public)
OLD_HOST = "opencourtvision-10a09b.gitlab.io"
PAGES_HOST = "antonyvo.gitlab.io"


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
                     for path in public.rglob("*") if path.is_file() and path.suffix in {".html", ".js", ".css", ".json"})
    assert OLD_HOST not in blob
    for href in parser.hrefs:
        assert PAGES_HOST not in href
        assert not href.startswith("/label")
    assert 'href="label/"' in label or 'href="../index.html"' in label
    clips = (public / "label" / "clips.json").read_text(encoding="utf-8")
    assert '"answer"' not in clips
    practice = (public / "label" / "practice.json").read_text(encoding="utf-8")
    assert "synthetic-practice-match" in practice


def test_no_old_host_in_source():
    for path in (ROOT / "site").rglob("*"):
        if path.is_file() and path.suffix in {".html", ".css", ".js", ".svg"}:
            assert OLD_HOST not in path.read_text(encoding="utf-8", errors="ignore")
    text = (ROOT / "labeler-pages" / "app.js").read_text(encoding="utf-8")
    assert "document.cookie" not in text
    assert "issues/new" not in text


def test_node_client_rules():
    proc = subprocess.run(["node", str(ROOT / "tests" / "call_client_logic.js")], check=True, capture_output=True, text=True)
    assert "entry.343297185=se-test" in proc.stdout
    body = proc.stdout.split("FORM_BODY ", 1)[1].strip()
    assert "docs.google.com" not in body
    assert "email" not in body
