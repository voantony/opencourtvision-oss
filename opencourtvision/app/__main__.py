"""CLI entry: python -m opencourtvision.app <command> ..."""
import argparse
import json
import os
import sys

from .. import privacy

privacy.enable_beta()  # closed EU beta: on-device only unless VAMOS_BETA=0 (Ops V2/V7); before any model import

from . import analyze as analyze_mod  # noqa: E402
from . import quality, review  # noqa: E402
from . import retrain as retrain_mod  # noqa: E402
from . import session as sess  # noqa: E402
from . import share as share_mod  # noqa: E402
from . import taps as taps_mod  # noqa: E402
from .tiers import catalogue  # noqa: E402


def _out(obj):
    json.dump(obj, sys.stdout, indent=2, default=str); print(); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m opencourtvision.app")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze"); p.add_argument("video"); p.add_argument("--work", required=True)
    p.add_argument("--final"); p.add_argument("--mode", choices=("match", "rallies"), default="match")
    p.add_argument("--labels"); p.add_argument("--sport", default="tennis"); p.add_argument("--region")
    p.add_argument("--framing", help="record screen's vamos.framing_quality.v1 JSON (quality signal, never a refusal)")
    p.add_argument("--teams", nargs=2)

    p = sub.add_parser("review"); p.add_argument("work"); p.add_argument("--n", type=int, default=12)
    p = sub.add_parser("fix"); p.add_argument("work"); p.add_argument("--item", required=True); p.add_argument("--value", required=True)
    p = sub.add_parser("taps"); p.add_argument("work"); p.add_argument("--taps", required=True)
    p = sub.add_parser("render"); p.add_argument("work"); p.add_argument("--aspect", default="9:16", choices=("9:16", "16:9"))
    p.add_argument("--preset", default="vamos_clean_v1"); p.add_argument("--out", required=True)
    p.add_argument("--no-grade", action="store_true"); p.add_argument("--max-points", type=int, default=8)
    p = sub.add_parser("retrain"); p.add_argument("work")
    p = sub.add_parser("share"); p.add_argument("work"); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--on", action="store_true"); g.add_argument("--off", action="store_true"); g.add_argument("--delete", action="store_true")
    p = sub.add_parser("quality"); p.add_argument("video")
    p = sub.add_parser("tiers")
    p = sub.add_parser("highlights", help="one-click Highlights only: reel (9:16 + 16:9) + best clips, no scoring")
    p.add_argument("video"); p.add_argument("--out", help="output folder (default: <video>_highlights next to it)")
    p.add_argument("--sport", default="tennis", choices=("tennis", "padel", "pickleball"))
    p.add_argument("--preset", default="social", choices=("social", "recruiting"),
                   help="social: 9:16 + 16:9 reel and clips; recruiting: 3-5 min 16:9 coach cut with a title card")
    p.add_argument("--style", default="vamos_clean_v1"); p.add_argument("--title", help="intro line, e.g. 'Club final'")
    p.add_argument("--focus", help="9:16 crop: center (default), near, far, auto, or a 0-1 x position")
    p.add_argument("--aspect", action="append", choices=("9:16", "16:9"), help="social only; default both")
    p.add_argument("--no-motion", action="store_true", help="audio only (skip the low-res live-play check)")
    p.add_argument("--region"); p.add_argument("--work")
    p.add_argument("--name"); p.add_argument("--class-year"); p.add_argument("--ranking", help="e.g. 'UTR 9.5'")
    p.add_argument("--contact", help="email / phone shown on the recruiting title card")
    p.add_argument("--mute", action="store_true", help="recruiting: silent audio track")
    p.add_argument("--camera", default="broadcast", choices=("broadcast", "static"),
                   help="social: broadcast grammar (wide rally, reaction close-up, starred replays) or a static crop")
    p.add_argument("--score", choices=("on", "off"), default=None,
                   help="score overlay on clips and reels (default per preset: social on, recruiting off); when on, a "
                        "point's score shows only if it is confirmed or high-confidence")
    p.add_argument("--long-shot", default="cut", choices=("cut", "fast_pan", "off"),
                   help="9:16: long cross-court shot -> hard cut to the receiving side (default), fast in-flight pan, or off")
    p.add_argument("--star", default="", help="social: comma list of clip ranks you starred (slow-mo replay, max 2)")

    A = ap.parse_args(argv)
    if A.cmd == "tiers": return _out(catalogue())
    if A.cmd == "quality": return _out(quality.analyse(A.video))
    if A.cmd == "highlights":
        import contextlib

        from . import highlights as hl_mod
        with contextlib.redirect_stdout(sys.stderr):
            r = hl_mod.run(A.video, out=A.out, sport=A.sport, preset=A.preset, style=A.style, focus=A.focus, title=A.title,
                           aspects=tuple(A.aspect or ("9:16", "16:9")), motion=not A.no_motion, region=A.region, work=A.work,
                           profile=dict(name=A.name, class_year=A.class_year, ranking=A.ranking, contact=A.contact), mute=A.mute,
                           camera=A.camera, star=[int(k) for k in A.star.split(",") if k.strip()], long_shot=A.long_shot,
                           score=None if A.score is None else A.score == "on")
        return _out(r)
    if A.cmd == "analyze":
        doc = analyze_mod.analyze(A.video, A.work, final=A.final, mode=A.mode, teams=A.teams,
                                  sport=A.sport, labels=A.labels, region=A.region, framing=A.framing)
        return _out(dict(session=doc["id"], work=doc["paths"]["work"], mode=doc["mode"],
                         n_points=len(doc["points"]), n_review=len(doc["review"]),
                         quality=doc["quality"], share_opt_in=doc["share_opt_in"],
                         sign_in_required=False))
    doc = sess.load(A.work)
    if A.cmd == "review":
        q = review.build_queue(doc, n=A.n); sess.save(doc); return _out(dict(n=len(q), items=q))
    if A.cmd == "fix":
        # value may be JSON
        try: value = json.loads(A.value)
        except Exception: value = A.value
        ex = review.apply_fix(doc, A.item, value); sess.save(doc)
        return _out(dict(fixed=ex["id"], kind=ex["kind"], remaining=len(doc["review"])))
    if A.cmd == "taps":
        r = taps_mod.ingest(doc, A.taps); sess.save(doc); return _out(r)
    if A.cmd == "retrain":
        r = retrain_mod.train_adapter(doc); sess.save(doc); return _out(r)
    if A.cmd == "share":
        if A.delete: r = share_mod.delete_my_data(doc)
        else: r = share_mod.set_opt_in(doc, on=A.on)
        sess.save(doc); return _out(r)
    if A.cmd == "render":
        import json as _json

        # probe size
        import subprocess

        from ..ffmpeg import ffprobe_bin
        from ..scoring.formats import get_format
        from ..style.plan import plan_reel
        from ..style.preset import load as load_preset
        from ..style.render import render_reel
        meta = _json.loads(subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
                                           "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", doc["video"]],
                                          capture_output=True, text=True).stdout)
        st = (meta.get("streams") or [{}])[0]
        SW, SH = int(st.get("width") or 3840), int(st.get("height") or 2160)
        fps = 30
        try:
            num, den = st.get("avg_frame_rate", st.get("r_frame_rate", "30/1")).split("/")
            fps = max(24, min(60, int(round(float(num) / float(den)))))
        except Exception:
            pass
        preset = load_preset(A.preset, grade=not A.no_grade)
        fmt = get_format(doc.get("sport") or "tennis")
        match = dict(sport=doc.get("sport") or "tennis", format=fmt, format_name="doubles",
                     teams=doc.get("teams") or ["Team A", "Team B"], final=doc.get("final"),
                     final_replayed=doc.get("final"), winner=(doc.get("teams") or ["Team A"])[0],
                     date_label=None)
        plan = plan_reel(doc["points"], preset, aspect=A.aspect, match=match, max_points=A.max_points)
        r = render_reel(doc["video"], plan, preset, A.out, work=os.path.join(doc["paths"]["exports"], "reel"),
                        SW=SW, SH=SH, fps=min(fps, 30))
        return _out(r)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        json.dump(dict(error=str(e), type=type(e).__name__), sys.stdout); print(); raise SystemExit(2)
