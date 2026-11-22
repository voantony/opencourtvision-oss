#!/usr/bin/env python3
"""Serve the offline travel labeler for one exported session. Stdlib only.

  python serve.py --dir ~/label-session

Defaults to queue.v2.json + clips_v2/ when those exist. Listens on 127.0.0.1.
Autosaves labels.json beside the queue. Model p(A) stays hidden until the point
is confirmed (not merely prefilled).
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from labels import build_labels  # noqa: E402

PUBLIC_KEYS = ("id", "idx", "video_id", "clip", "start", "end")


def public_point(point: dict, reveal: bool) -> dict:
    out = {k: point[k] for k in PUBLIC_KEYS if k in point}
    if reveal:
        out["p_a"] = point.get("p_a")
        out["reason"] = point.get("reason")
        out["tier"] = point.get("tier")
        out["rules"] = list(point.get("rules") or [])
    return out


def answer_confirmed(ans: dict | None) -> bool:
    if not ans:
        return False
    return bool(ans.get("confirmed", True))


def make_handler(session: Path, html: str, queue_name: str = "queue.json"):
    queue_path = session / queue_name
    labels_path = session / "labels.json"
    answers_path = session / "answers.json"

    def load_answers() -> dict:
        if answers_path.is_file():
            return json.loads(answers_path.read_text())
        return {}

    def save(answers: dict) -> None:
        queue = json.loads(queue_path.read_text())
        answers_path.write_text(json.dumps(answers, indent=2) + "\n")
        doc = build_labels(queue, answers)
        labels_path.write_text(json.dumps(doc, indent=2) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _json(self, code: int, body) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _bytes(self, code: int, raw: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _under(self, rel: str) -> Path | None:
            rel = rel.split("?", 1)[0].lstrip("/")
            if not rel or ".." in rel.replace("\\", "/").split("/"):
                return None
            path = (session / rel).resolve()
            try:
                path.relative_to(session.resolve())
            except ValueError:
                return None
            return path

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/labeler.html"):
                self._bytes(200, html.encode(), "text/html; charset=utf-8")
                return
            if path == "/api/session":
                queue = json.loads(queue_path.read_text())
                answers = load_answers()
                points = []
                confirmed_n = 0
                for p in queue["points"]:
                    ans = answers.get(str(p["id"]))
                    conf = answer_confirmed(ans)
                    # Reveal model only after a confirmed answer.
                    row = public_point(p, conf)
                    if ans:
                        row["answer"] = {
                            "winner": ans.get("winner"),
                            "server": ans.get("server"),
                            "confirmed": conf,
                        }
                        if conf:
                            confirmed_n += 1
                    points.append(row)
                self._json(200, {
                    "video_id": queue.get("video_id"),
                    "points": points,
                    "answered": confirmed_n,
                    "total": len(points),
                })
                return
            file = self._under(path)
            if file is None or not file.is_file():
                self._json(404, {"error": "not found"})
                return
            ctype = "video/mp4" if file.suffix.lower() == ".mp4" else "application/octet-stream"
            self._bytes(200, file.read_bytes(), ctype)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/answer":
                self._json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                self._json(413, {"error": "too large"})
                return
            body = json.loads(self.rfile.read(n) or b"{}")
            queue = json.loads(queue_path.read_text())
            by_id = {str(p["id"]): p for p in queue["points"]}
            pid = str(body.get("id"))
            if pid not in by_id:
                self._json(400, {"error": "unknown point"})
                return
            winner = body.get("winner")
            if winner not in ("A", "B", "skip"):
                self._json(400, {"error": "winner must be A, B, or skip"})
                return
            server = body.get("server")
            if server not in ("A", "B", None):
                self._json(400, {"error": "server must be A, B, or null"})
                return
            answers = load_answers()
            answers[pid] = {"winner": winner, "server": server, "confirmed": True}
            save(answers)
            conf_n = sum(1 for a in answers.values() if answer_confirmed(a))
            self._json(200, {
                "ok": True,
                "point": public_point(by_id[pid], True),
                "answer": answers[pid],
                "answered": conf_n,
                "total": len(queue["points"]),
            })

    return Handler


def resolve_defaults(session: Path, queue_arg: str | None, clips_arg: str | None):
    """Prefer queue.v2.json + clips_v2 when present."""
    queue_name = queue_arg
    if not queue_name:
        queue_name = "queue.v2.json" if (session / "queue.v2.json").is_file() else "queue.json"
    clips_subdir = clips_arg
    if clips_subdir is None and (session / "clips_v2").is_dir():
        clips_subdir = "clips_v2"
    return queue_name, clips_subdir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path, help="folder from export_clips.py")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--queue", default=None, help="queue file (default: queue.v2.json if present)")
    ap.add_argument("--clips-subdir", default=None, help="rewrite clips/ -> this subdir (default: clips_v2 if present)")
    a = ap.parse_args(argv)
    session = a.dir.resolve()
    queue_name, clips_subdir = resolve_defaults(session, a.queue, a.clips_subdir)
    qsrc = session / queue_name
    if not qsrc.is_file():
        print(f"no {queue_name} in {session}", file=sys.stderr)
        return 2
    if clips_subdir:
        q = json.loads(qsrc.read_text())
        sub = clips_subdir.strip("/").strip("\\")
        for pt in q.get("points") or []:
            name = Path(str(pt.get("clip") or "")).name
            pt["clip"] = f"{sub}/{name}"
        live = session / "queue.live.json"
        live.write_text(json.dumps(q, indent=2) + "\n")
        queue_name = "queue.live.json"
        print(f"serving clips from {sub}/")
    html = (HERE / "labeler.html").read_text()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(session, html, queue_name=queue_name))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"labeler {url}  session {session}  queue {queue_name}")
    if not a.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
