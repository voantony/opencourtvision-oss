#!/usr/bin/env python3
"""Headless screenshots. Serves the built Pages tree. Does not submit the form."""
from __future__ import annotations

import importlib.util
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def build(public: Path) -> Path:
    spec = importlib.util.spec_from_file_location("build_pages", ROOT / "scripts" / "build_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build(public)
OUT = ROOT / "docs" / "screens"


def serve(directory: Path):
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def block_remote(page):
    def handle(route):
        url = route.request.url
        if "docs.google.com" in url or url.endswith(".mp4") or "gitlab.io" in url:
            route.abort()
        else:
            route.continue_()
    page.route("**/*", handle)


def shot(page, name):
    page.screenshot(path=str(OUT / name), full_page=False)
    print("wrote", name, page.viewport_size)


def main():
    public = build(ROOT / "public")
    OUT.mkdir(parents=True, exist_ok=True)
    httpd, base = serve(public)
    loads = []
    clicks = {"n": 0}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", args=["--no-sandbox", "--disable-dev-shm-usage"])
            # Homepage, both widths, English and Spanish. Full page so every section is in the file.
            for width, suffix in ((390, "390"), (1440, "1440")):
                page = browser.new_page(viewport={"width": width, "height": 844})
                block_remote(page)
                page.goto(base + "/index.html", wait_until="networkidle")
                page.evaluate("document.fonts.ready")
                page.screenshot(path=str(OUT / f"home-{suffix}.png"), full_page=True)
                page.goto(base + "/es.html", wait_until="networkidle")
                page.evaluate("document.fonts.ready")
                page.screenshot(path=str(OUT / f"home-es-{suffix}.png"), full_page=True)
                page.close()

            # Clicks and loads from the homepage to the first call.
            measure = browser.new_page(viewport={"width": 390, "height": 844})
            block_remote(measure)
            measure.on("request", lambda req: loads.append(req.url))
            measure.goto(base + "/index.html", wait_until="networkidle")
            measure.locator("a.btn.volt").click()
            clicks["n"] += 1
            measure.wait_for_selector("#play")
            measure.locator("#adult").check()
            clicks["n"] += 1
            measure.locator("#play").click()
            clicks["n"] += 1
            measure.locator("[data-call='A']").click()
            clicks["n"] += 1
            measure.wait_for_selector("#feedback")
            doc_loads = [u for u in loads if u.startswith(base)]
            print(f"CLICKS {clicks['n']}")
            print(f"LOADS {len(doc_loads)}")
            for url in doc_loads:
                print(" load", url[len(base):])
            measure.close()

            page = browser.new_page(viewport={"width": 390, "height": 844})
            block_remote(page)
            page.goto(base + "/label/", wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            page.wait_for_selector("#play")
            shot(page, "game-start.png")
            page.locator("#adult").check()
            page.locator("#play").click()
            page.locator("[data-call='B']").click()
            page.wait_for_selector("#reason")
            shot(page, "game-warmup.png")
            # Finish the 3-point warm-up, then stop on live point 3 of 10.
            page.locator("#next").click()
            page.locator("[data-call='A']").click()
            page.locator("#next").click()
            page.locator("[data-call='A']").click()
            page.locator("#next").click()
            page.locator("[data-call='A']").click()
            page.locator("#next").click()
            page.locator("[data-call='A']").click()
            page.locator("#next").click()
            assert "Point 3 of 10" in page.locator(".mono").first.inner_text()
            shot(page, "game-point.png")
            # Points 3 through 10, then the summary.
            for i in range(8):
                page.locator("[data-call='A']").click()
                page.locator("#next").click()
            page.wait_for_selector("#sum-count")
            assert "10 points called" in page.locator("#sum-count").inner_text()
            shot(page, "game-summary.png")
            four = page.evaluate("""() => {
              document.getElementById('share').click();
              const canvases = [...document.querySelectorAll('canvas')].map(c => [c.width, c.height]);
              return canvases;
            }""")
            assert four == [[1080, 1350], [1080, 1920]], four
            page.locator("#open-board").click()
            page.wait_for_selector("#board")
            shot(page, "game-leaderboard.png")
            page.close()
            browser.close()
    finally:
        httpd.shutdown()
    print("screens in", OUT)


if __name__ == "__main__":
    main()
