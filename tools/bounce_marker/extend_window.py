"""One-off fix for marker items prepared before the window fix (19 Oct 2026): the window was last contact -0.1 s to
+2 s but clipped at the old tracker clip end (frame 224), so e.g. point 24 cut off before its bounce. Extends each item to
min(last contact + 3.0 s, next point start - 0.3 s) (tools/ball_track_v2/extend_clips.window_end), decoding the extra
frames from the source video at the clip time base (t = t0 + k/fps, 768x432 like the clip frames). Only adds frames;
answers are untouched (the k of a marked bounce keeps its meaning). New items: use prep_bounce.py on extended clips.

  extend_window.py --video MATCH.mp4 --labels MATCH.labels.json [--items items.json] [--ffmpeg ffmpeg]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ball_track_v2"))
from extend_clips import window_end  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--items", default=os.path.join(HERE, "items.json")); ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--after", type=float, default=3.0); ap.add_argument("--gap", type=float, default=0.3); a = ap.parse_args()
    starts = sorted(p["start"] for p in json.load(open(a.labels))["points"])
    d = json.load(open(a.items)); added = 0; frames = os.path.join(os.path.dirname(os.path.abspath(a.items)), "frames")
    for it in d["items"]:
        t0, fps, tl = it["t0"], it["fps"], it["t_last"]
        k_end = int(np.floor((window_end(tl, starts, a.after, a.gap) - t0) * fps))
        have = set(it["ks"]); want = list(range(max(it["ks"]) + 1, k_end + 1))
        if not want:
            continue
        dst = os.path.join(frames, str(it["idx"]))
        cmd = [a.ffmpeg, "-nostdin", "-loglevel", "error", "-ss", f"{t0 + want[0] / fps:.3f}", "-i", a.video, "-frames:v", str(len(want)),
               "-vf", f"fps={fps},scale=768:432:flags=area", "-q:v", "3", "-start_number", str(want[0]), f"{dst}/%05d.jpg"]
        subprocess.run(cmd, check=True)
        got = [k for k in want if os.path.exists(f"{dst}/{k:05d}.jpg")]
        it["ks"] = sorted(have | set(got)); it["ext_to_k"] = max(it["ks"]); it["ext_note"] = "window to min(last+3 s, next serve -0.3 s)"
        added += len(got)
        print(it["idx"], it["point_no"], "k_last", round(it["k_last"]), "->", max(it["ks"]), f"(+{len(got)})", flush=True)
    tmp = a.items + ".tmp"; json.dump(d, open(tmp, "w")); os.replace(tmp, a.items)
    print("added", added, "frames")


if __name__ == "__main__":
    main()
