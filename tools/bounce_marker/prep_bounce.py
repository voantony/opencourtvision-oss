"""Frames for the last-bounce marker: for each picked train point (pick.json from bounce_pick.py), copy the clip frames
k0..k1 into the marker's frames/ dir and write items.json. Uses the clip's own frames, so frame k here is exactly
frame k of the tracker. Point it at the EXTENDED clip data (tools/ball_track_v2/extend_clips.py) so windows reach
min(last contact + 3 s, next point - 0.3 s). Outputs are PRIVATE: never commit.

  prep_bounce.py --pick PICK.json --data CLIP_DATA_DIR [--out MARKER_DIR]
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pick", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=HERE); a = ap.parse_args()
    d = json.load(open(a.pick)); items = []
    for r in d["items"]:
        src = os.path.join(a.data, "frames", r["name"]); dst = os.path.join(a.out, "frames", str(r["idx"])); os.makedirs(dst, exist_ok=True)
        ks = [k for k in range(r["k0"], r["k1"]) if os.path.exists(f"{src}/{k:05d}.jpg")]
        for k in ks:
            if not os.path.exists(f"{dst}/{k:05d}.jpg"):
                shutil.copyfile(f"{src}/{k:05d}.jpg", f"{dst}/{k:05d}.jpg")
        items.append(dict(key=str(r["idx"]), idx=r["idx"], point_no=r["point_no"], name=r["name"], t0=r["t0"], fps=r["fps"],
                          t_last=r["t_last"], last_src=r["last_src"], k_last=round((r["t_last"] - r["t0"]) * r["fps"], 1), ks=ks,
                          status=r["status"], sv_result=r["sv_result"], sv_uv=r["sv_uv"], guess=r["guess"], ball=r["ball"][:len(ks)], k0=r["k0"]))
    json.dump(dict(note="PRIVATE. 1280x720 analysis px; k = frame in the clip; t = t0 + k/fps (video time).", items=items),
              open(os.path.join(a.out, "items.json"), "w"))
    print(len(items), "items", sum(len(i["ks"]) for i in items), "frames")


if __name__ == "__main__":
    main()
