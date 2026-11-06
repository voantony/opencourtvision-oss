"""Import a match video -> auto point cutting + score (free tier, on device, no account)."""
import contextlib
import json
import os
import sys

from ..pipeline import run as pipeline_run
from . import quality, review
from . import session as sess


class Args:
    """Thin namespace matching pipeline.add_arguments / pipeline.run expectations."""
    def __init__(self, **kw):
        self.video = kw["video"]; self.work = kw.get("work")
        self.final = kw.get("final"); self.mode = kw.get("mode", "match")
        self.reel_order = kw.get("reel_order", "length")
        self.teams = kw.get("teams") or ["Team A", "Team B"]
        self.stt = bool(kw.get("stt", False)); self.model = kw.get("model")
        self.render = kw.get("render"); self.name = kw.get("name", "Match")
        self.start = kw.get("start"); self.lead = float(kw.get("lead", 4.0))
        self.tail = float(kw.get("tail", 2.5)); self.merge_gap = float(kw.get("merge_gap", 1.0))
        self.story = bool(kw.get("story", True)); self.hl = int(kw.get("hl", 12))
        self.hl_max = float(kw.get("hl_max", 240)); self.segmenter = kw.get("segmenter", "smart")
        self.region = kw.get("region"); self.motion = kw.get("motion")


def analyze(video, work, final=None, mode="match", teams=None, sport="tennis", labels=None, region=None, framing=None):
    """Run the pipeline, write a session.json, seed the fix-screen queue. No sign-in required.
    framing: the record screen's vamos.framing_quality.v1 (dict or path). Never a reason to refuse: it is stored as
    doc["framing_quality"] and each point gets point["framing"] (court visible, per-side visibility, vision mode,
    uncertainty scale) so later stages widen their uncertainty on imperfect framing."""
    os.makedirs(work, exist_ok=True)
    q = quality.analyse(video)
    doc = sess.new_session(work, video, mode=mode, final=final, teams=teams, sport=sport)
    doc["quality"] = q
    # If the user passed labels (e.g. Antony's dev_match labels), prefer those as ground-truth points.
    if labels and os.path.exists(labels):
        doc["points"] = points_from_labels(labels, doc)
        doc["labels_path"] = os.path.abspath(labels)
        # still run audio segmentation for comparison / mode suggestion when possible
    args = Args(video=video, work=work, final=final or (doc.get("final")), mode=mode,
                teams=doc["teams"], story=True, region=region)
    if mode == "match" and not args.final:
        # free-tier friendly: fall back to rallies rather than refusing
        args.mode = "rallies"; doc["mode"] = "rallies"
        doc["mode_note"] = "no final score yet; ran rallies-only (free). Add --final anytime to score."
    try:
        # The pipeline prints progress + its rally list to stdout; the app CLI's stdout must be exactly one JSON doc.
        with contextlib.redirect_stdout(sys.stderr):
            rep = pipeline_run(args)
    except SystemExit as e:
        # pipeline refuses match without --final; already handled above, but be safe
        raise RuntimeError(str(e))
    doc["report"] = {k: rep.get(k) for k in ("mode", "match_start_s", "match_start_hms", "checks", "highlights",
                                             "cut_total_s", "stage_times_s", "mode_suggestion") if k in rep}
    if not doc.get("points"):
        doc["points"] = points_from_segments(work, doc)
    if framing:
        from ..framing.quality import attach_to_points, summarise_samples
        fq = json.load(open(framing)) if isinstance(framing, str) else framing
        doc["framing_quality"] = dict(schema=fq.get("schema"), summary=summarise_samples(fq.get("samples", [])))
        attach_to_points(doc["points"], fq)
    review.build_queue(doc, n=12)
    sess.save(doc)
    return doc


def points_from_segments(work, doc):
    """Build session points from segments.csv produced by the pipeline."""
    path = os.path.join(work, "segments.csv")
    if not os.path.exists(path):
        return points_from_rally_list(work)
    import csv
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(dict(id=int(row["idx"]), kind="point",
                            A=float(row["start_s"]), B=float(row["end_s"]),
                            first_hit=float(row["first_hit_s"]), last_hit=float(row["end_s"]) - 2.0,
                            hits=int(row["hits"]), react=float(row.get("react_db") or 0),
                            favorite=False, winner=None, winner_src=None, winner_confidence=0.5,
                            bounce=None, bounce_confidence=0.5, player_id=None, player_id_confidence=0.5,
                            sources=["audio"]))
    return out


def points_from_rally_list(work):
    """Rallies mode (no final score) writes rally_list.json instead of segments.csv; map those rallies to points."""
    path = os.path.join(work, "rally_list.json")
    if not os.path.exists(path):
        return []
    raw = json.load(open(path))
    out = []
    for r in raw.get("rallies") or []:
        a, b = float(r["start_s"]), float(r["end_s"])
        out.append(dict(id=int(r["idx"]), kind="point", A=a, B=b,
                        first_hit=round(b - float(r.get("rally_s") or (b - a)), 3), last_hit=max(a, b - 2.0),
                        hits=int(r.get("hits") or 0), react=float(r.get("reaction_db") or 0),
                        favorite=False, winner=None, winner_src=None, winner_confidence=0.5,
                        bounce=None, bounce_confidence=0.5, player_id=None, player_id_confidence=0.5,
                        sources=["audio"]))
    return out


def points_from_labels(path, doc):
    """Import ocv.point_labels.v1 (Antony's labeller). Private; never committed."""
    raw = json.load(open(path))
    out = []
    for p in raw.get("points") or []:
        if p.get("kind") not in ("point", None):
            continue
        if p.get("kind") == "non_point":
            continue
        sb = p.get("score_before") or {}
        out.append(dict(id=p.get("point_no") or p["idx"], kind="point",
                        A=float(p["start"]), B=float(p["end"]),
                        first_hit=float(p["start"]) + 2.0, last_hit=float(p["end"]) - 1.5,
                        hits=0, react=0.0, favorite=bool(p.get("favorite")),
                        winner=p.get("winner"), winner_src=p.get("winner_src") or "user",
                        winner_confidence=1.0 if p.get("winner_src") == "user" else 0.7,
                        bounce=None, bounce_confidence=0.5, player_id=None, player_id_confidence=0.5,
                        score_before=sb, server=p.get("server"), sources=["user", "audio"],
                        sv_detail=p.get("sv_detail")))
    if raw.get("match", {}).get("final_hint"):
        doc["final"] = raw["match"]["final_hint"]
    return out
