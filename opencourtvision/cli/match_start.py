"""ocv-match-start: detect the warmup -> match transition in a fixed-camera tennis video (or WAV).

Signals: (1) audio rhythm of ball hits (warmup = near-continuous hitting, match = discrete rallies with
10-40 s pauses); (2) optionally the first cluster of score callouts in a whisper transcript JSON;
(3) optionally a contact sheet of frames at rally starts around the estimate for a quick visual check.
"""
import argparse
import json

from ..detection.match_start import contact_sheet, estimate_match_start


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocv-match-start", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="video or .wav")
    ap.add_argument("--transcript", help="whisper transcript JSON with 'segments'")
    ap.add_argument("--sheet", help="write a 6x4 contact sheet JPEG here")
    ap.add_argument("--out", help="write the full result JSON here")
    ap.add_argument("--gap", type=float, default=6.0, help="max gap between hits in one rally (s)")
    ap.add_argument("--win", type=float, default=300.0, help="rhythm window length (s)")
    ap.add_argument("--font", help="font file for the contact sheet labels (env OCV_FONT)")
    A = ap.parse_args(argv)
    res = estimate_match_start(A.src, A.transcript, A.gap, A.win)
    groups = res.pop("_groups")
    if A.sheet and res.get("estimate_s"):
        contact_sheet(A.src, groups, res["estimate_s"], A.sheet, A.font)
    s = json.dumps(res, indent=1, default=float)
    if A.out:
        open(A.out, "w").write(s)
    print(json.dumps({k: v for k, v in res.items() if k != "rhythm_table"}, indent=1, default=float))
    for r in res["rhythm_table"][:36]:
        print(r)


if __name__ == "__main__":
    main()
