#!/usr/bin/env python3
"""Serve the point labeler for one match video and save labels next to it (stdlib only, macOS/Linux).

  python3 serve.py "/path/to/match.MP4" [--seed seed.json] [--port 8765] [--no-open]

Opens http://127.0.0.1:8765/ (Safari on macOS: it plays 4K HEVC). Labels autosave to "<video>.labels.json" beside the video
(previous version kept as .labels.bak.json); the browser also keeps a copy in localStorage. Ctrl-C to stop.
Only listens on 127.0.0.1; only the labeler page, the one video, the seed and the labels file are served.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 1 << 20


def labels_path_for(video):
    stem, _ = os.path.splitext(video)
    return stem + ".labels.json"


def make_handler(video, seed, html):
    lab = labels_path_for(video)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, open(html, "rb").read(), "text/html; charset=utf-8")
            if path == "/config.json":
                return self._send(200, dict(video_name=os.path.basename(video), has_video=os.path.exists(video),
                                            has_seed=bool(seed and os.path.exists(seed)), labels_path=lab))
            if path == "/seed.json" and seed and os.path.exists(seed):
                return self._send(200, open(seed, "rb").read())
            if path == "/labels":
                if os.path.exists(lab):
                    return self._send(200, open(lab, "rb").read())
                return self._send(404, {"error": "no labels yet"})
            if path == "/video":
                return self._video()
            return self._send(404, {"error": "not found"})

        def _video(self):
            size = os.path.getsize(video)
            ext = os.path.splitext(video)[1].lower()
            ctype = {".mov": "video/quicktime", ".m4v": "video/mp4"}.get(ext, "video/mp4")
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                with open(video, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        b = f.read(min(CHUNK, left))
                        if not b:
                            break
                        self.wfile.write(b)
                        left -= len(b)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path.split("?")[0] != "/labels":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 50 * CHUNK:
                return self._send(400, {"error": "bad size"})
            try:
                obj = json.loads(self.rfile.read(n))
                assert obj.get("schema") == "ocv.point_labels.v1" and isinstance(obj.get("points"), list)
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": f"not a labels file: {e}"})
            if os.path.exists(lab):
                shutil.copyfile(lab, lab[:-len(".labels.json")] + ".labels.bak.json")
            tmp = lab + ".tmp"
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=1)
            os.replace(tmp, lab)
            return self._send(200, {"ok": True, "path": os.path.basename(lab)})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seed", help="seed JSON (make_seed.py); default: seed*.json next to this script if exactly one")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--pass", dest="fav_pass", action="store_true", help="start in the favourites pass (labelled points only)")
    a = ap.parse_args()
    video = os.path.abspath(os.path.expanduser(a.video))
    if not os.path.exists(video):
        sys.exit(f"video not found: {video}")
    seed = a.seed
    if not seed:
        c = sorted(f for f in os.listdir(HERE) if f.startswith("seed") and f.endswith(".json"))
        seed = os.path.join(HERE, c[0]) if len(c) == 1 else None
    html = os.path.join(HERE, "labeler.html")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(video, seed, html))
    url = f"http://127.0.0.1:{a.port}/" + ("?pass=1" if a.fav_pass else "")
    print(f"Labeler: {url}\nVideo:   {video}\nSeed:    {seed}\nLabels:  {labels_path_for(video)}\nCtrl-C to stop.")
    if not a.no_open:
        if sys.platform == "darwin" and os.path.exists("/Applications/Safari.app"):
            subprocess.Popen(["open", "-a", "Safari", url])
        else:
            webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; labels are in", labels_path_for(video))


if __name__ == "__main__":
    main()
