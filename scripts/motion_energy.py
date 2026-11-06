#!/usr/bin/env python3
"""One cheap decode pass -> regional motion energy npz for the (flagged) vision rally cues. No object tracking.

  python scripts/motion_energy.py MATCH.MP4 out.npz      # macOS: AVFoundation hardware decode; ~25 min per 86-min 4K60

Grabs every frame, keeps every STEP-th (6 Hz at 60 fps), gray 192x108, |diff| mean per 8x9-px cell (24x12 grid), plus
1 Hz thumbnails. Needs opencv-python.
"""
import sys
import time

import numpy as np


def main(video, out, step=10):
    import cv2
    be = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    c = cv2.VideoCapture(video, be)
    fps = c.get(cv2.CAP_PROP_FPS) or 60.0
    E, T, TH, THT = [], [], [], []; prev = None; i = 0; t0 = time.time()
    while c.grab():
        if i % step == 0:
            ok, f = c.retrieve()
            if not ok: break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (192, 108), interpolation=cv2.INTER_AREA)
            if prev is not None:
                d = cv2.absdiff(g, prev).astype(np.float32)
                E.append(d.reshape(12, 9, 24, 8).mean(axis=(1, 3)).astype(np.float16)); T.append(i / fps)
            if i % int(round(fps)) == 0: TH.append(g); THT.append(i / fps)
            prev = g
        i += 1
    np.savez_compressed(out, E=np.array(E), t=np.array(T), thumbs=np.array(TH), thumb_t=np.array(THT), fps=fps, step=step)
    print("done", i, "frames", round(time.time() - t0), "s")


if __name__ == "__main__":
    if len(sys.argv) != 3: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
