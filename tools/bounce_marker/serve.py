#!/usr/bin/env python3
"""Last-bounce marker for training points 1-50 (answers are PRIVATE: they never leave the Mac and are never committed).
stdlib only, 127.0.0.1.   python3 serve.py [--port 8769] [--no-open]

Items: items.json (prep_bounce.py, from the physics tracker's run: bounce_gt/pick.json). Answers autosave to
~/VamosFootage/ballbench/run1/physball/bounce_gt/bounce_labels.json (previous version kept as bounce_labels.bak.json).
When every point is answered the tracker's last-bounce eval against these labels is queued once
(~/VamosFootage/queue/jobs/bounce_gt_eval.sh, light, CPU only).
"""
import argparse
import json
import os
import shutil
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.expanduser("~/VamosFootage")
ITEMS = os.path.join(HERE, "items.json")
ANSWERS = os.environ.get("BM_ANSWERS") or os.path.join(F, "ballbench/run1/physball/bounce_gt/bounce_labels.json")
FLAG = ANSWERS.replace(".json", ".eval_requested.json")
DRY = os.environ.get("BM_DRY") == "1"
FRAMES = os.path.join(HERE, "frames")


def send(h, code, body, ctype="application/json"):
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    elif isinstance(body, str):
        body = body.encode()
    h.send_response(code); h.send_header("Content-Type", ctype); h.send_header("Content-Length", str(len(body)))
    h.send_header("Cache-Control", "no-store"); h.end_headers(); h.wfile.write(body)


def queue_eval(reason):
    if os.path.exists(FLAG):
        return "already queued"
    if DRY:
        json.dump(dict(reason=reason, dry=True), open(FLAG, "w")); return "queued (dry run)"
    Q = os.path.join(F, "queue")
    if "bounce_gt_eval|" not in open(os.path.join(Q, "jobs.txt")).read():
        open(os.path.join(Q, "jobs.txt"), "a").write("bounce_gt_eval|-|light|NICE=20 bash jobs/bounce_gt_eval.sh\n")
    for x in ("done", "failed", "blocked", "pid"):
        p = os.path.join(Q, "state", "bounce_gt_eval." + x)
        if os.path.exists(p):
            os.rename(p, p + ".old." + str(int(os.path.getmtime(p))))
    r = subprocess.run(["bash", os.path.join(Q, "queue"), "run"], capture_output=True, text=True)
    json.dump(dict(reason=reason, queued=True, runner=(r.stdout or r.stderr).strip()), open(FLAG, "w"), indent=1)
    return "queued"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return send(self, 200, open(os.path.join(HERE, "bounce.html"), "rb").read(), "text/html; charset=utf-8")
        if p == "/items.json":
            return send(self, 200, open(ITEMS, "rb").read())
        if p == "/answers":
            return send(self, 200, open(ANSWERS, "rb").read()) if os.path.exists(ANSWERS) else send(self, 200, {"items": {}})
        if p == "/status":
            return send(self, 200, dict(queued=os.path.exists(FLAG)))
        if p.startswith("/f/"):
            f = os.path.join(FRAMES, p[3:])
            if os.path.exists(f) and os.path.realpath(f).startswith(os.path.realpath(FRAMES)):
                return send(self, 200, open(f, "rb").read(), "image/jpeg")
        return send(self, 404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if p == "/save":
            os.makedirs(os.path.dirname(ANSWERS), exist_ok=True)
            if os.path.exists(ANSWERS):
                shutil.copyfile(ANSWERS, ANSWERS.replace(".json", ".bak.json"))
            json.dump(body, open(ANSWERS, "w"), indent=1)
            return send(self, 200, {"ok": True})
        if p == "/eval":
            return send(self, 200, {"status": queue_eval(body.get("reason", "button"))})
        return send(self, 404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8769); ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"bounce marker on http://127.0.0.1:{a.port}/  answers -> {ANSWERS}", flush=True)
    if not a.no_open:
        webbrowser.open(f"http://127.0.0.1:{a.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
