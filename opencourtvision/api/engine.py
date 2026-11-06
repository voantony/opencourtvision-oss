"""analyze / reinfer / render over the engine on main, with fixture-backed stubs where the engine is not ready."""
import copy
import os
import shutil
import tempfile

from . import features as feat
from . import fixtures
from .types import API_VERSION, CORRECTION_KINDS, Correction, InvalidCorrection, MatchResult, valid_value

ENGINES = ("real", "stub", "auto")
ITEM_PREFIX = {"pw": "point_winner", "bc": "bounce", "pid": "player_id", "co": "cut_off", "sc": "score_check"}
STUB_ALWAYS = ["players"]          # roles from court side only until the 4-player ID tracker lands (engine-v2 step 2)


# ---------------------------------------------------------------- derived fields (real code on main)

def _fmt(meta):
    from ..scoring.formats import get_format
    return get_format(meta.get("sport") or "tennis", meta.get("format_preset"))


def _state_dict(st, fmt):
    from ..scoring.rules import score_call
    return dict(sets=list(st.sets), games=list(st.games), pts=list(st.pts), server=st.server, tiebreak=st.tb,
                call=None if st.done else score_call(st, fmt), done=st.done)


def score_timeline(points, meta):
    """Rules-exact score before / after every point. A point with an unknown winner keeps the score and marks every later
    entry unverified; a point after match end is flagged rather than crashing."""
    from ..scoring.rules import advance, initial_state
    fmt = _fmt(meta)
    st = initial_state(fmt, meta.get("first_server") or "A")
    verified, out, states = True, [], {}
    for p in sorted(points, key=lambda p: p["first_hit"]):
        if p.get("kind", "point") != "point":
            continue
        before = st
        w = p.get("winner")
        note = None
        if w in ("A", "B") and not st.done:
            st = advance(st, w, fmt)
        else:
            verified = False
            note = "match_already_over" if st.done else "winner_unknown"
        states[p["id"]] = (before, verified and w in ("A", "B"))
        out.append(dict(point=p["id"], t=p["first_hit"], winner=w, before=_state_dict(before, fmt), after=_state_dict(st, fmt),
                        verified=verified, note=note))
    return out, states


def uncertain_queue(points, quality, n=12, fixed=()):
    from ..app import review
    q = review.build_queue(dict(points=points, quality=quality or {}), n=n + len(fixed))
    return [it for it in q if it["id"] not in set(fixed)][:n]


def clips_for(points):
    from ..style.plan import clip_score, whole_bounds
    out = []
    for p in points:
        if p.get("kind", "point") != "point" or p.get("cut_off"):
            continue
        a, b = whole_bounds(p)
        out.append(dict(point=p["id"], a=round(a, 3), b=round(b, 3), score=clip_score(p), favorite=bool(p.get("favorite"))))
    return out


def reel_plan_for(points, states, meta, preset="vamos_clean_v1", aspect="9:16", max_points=8):
    from ..style.plan import plan_reel
    from ..style.preset import load as load_preset
    pre = load_preset(preset)
    pts = []
    for p in points:
        if p.get("kind", "point") != "point" or p.get("cut_off"):
            continue
        st, ok = states.get(p["id"], (None, False))
        pts.append(dict(p, state=st, state_verified=ok, winner_verified=ok and p.get("winner_src") in ("user", "confirm", "watch")))
    teams = meta.get("teams") or ["Team A", "Team B"]
    match = dict(sport=meta.get("sport") or "tennis", format=_fmt(meta), format_name=meta.get("format_name") or "doubles",
                 teams=teams, final=meta.get("final"), final_replayed=None, winner=None, date_label=None)
    return plan_reel(pts, pre, aspect=aspect, match=match, max_points=max_points)


def _players(meta):
    teams = meta.get("teams") or ["Team A", "Team B"]
    return [dict(id="A", team=teams[0], side="near", identity="court_side_role"),
            dict(id="B", team=teams[1], side="far", identity="court_side_role")]


def _derive(points, meta, quality, fixed=()):
    tl, states = score_timeline(points, meta)
    try:
        plan = reel_plan_for(points, states, meta)
    except Exception as e:                       # too few points etc.: no reel yet, say why
        plan = None
        meta.setdefault("notes", []).append(f"reel_plan unavailable: {type(e).__name__}: {e}")
    q = uncertain_queue(points, quality, fixed=fixed)
    _attach_features(points, tl, q, quality, meta)
    return MatchResult(points=points, score_timeline=tl, uncertain_queue=q, clips=clips_for(points), reel_plan=plan,
                       players=_players(meta), meta=meta, features_version=feat.FEATURES_VERSION)


def _attach_features(points, tl, queue, quality, meta):
    """points[].features (+ features_src) and uncertain_queue[].features, vamos.feat.v1 (16 floats, see api/features.py)."""
    before = {e["point"]: e["before"] for e in tl}
    tags = (quality or {}).get("struggle_tags") or []
    seed = meta.get("stub_seed") if meta.get("engine") == "stub" else None
    by = {}
    for p in points:
        if p.get("kind", "point") != "point":
            continue
        p["features"], p["features_src"] = feat.point_features(p, before.get(p["id"]), tags, stub_seed=seed)
        by[p["id"]] = p
    for it in queue:
        p = by.get(it["point"])
        f, _ = feat.point_features(p or {}, before.get(it["point"]), it.get("struggle_tags") or tags,
                                   confidence=it.get("confidence"), stub_seed=seed)
        it["features"], it["features_version"] = f, feat.FEATURES_VERSION


# ---------------------------------------------------------------- public API

def _real_available(video):
    from ..ffmpeg import ffmpeg_bin
    if not (video and os.path.exists(video)):
        return False, "video not found"
    try:
        b = ffmpeg_bin()
    except Exception as e:
        return False, f"ffmpeg unavailable: {e}"
    if not (os.path.exists(b) or shutil.which(b)):
        return False, "ffmpeg unavailable"
    return True, None


def analyze(video, opts=None):
    """video path -> MatchResult. opts: engine (real|stub|auto, default real), work, final, mode (match|rallies), sport,
    teams, format_preset, first_server, n_points/seed (stub only)."""
    opts = dict(opts or {})
    engine = opts.get("engine", "real")
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    meta = dict(api_version=API_VERSION, sport=opts.get("sport") or "tennis", final=opts.get("final"),
                teams=list(opts.get("teams") or ["Team A", "Team B"]), format_preset=opts.get("format_preset"),
                first_server=opts.get("first_server") or "A", mode=opts.get("mode") or "match", fixed=[], n_corrections=0)
    if engine == "auto":
        ok, why = _real_available(video)
        engine = "real" if ok else "stub"
        if not ok: meta["fallback_reason"] = why
    if engine == "stub":
        pts = fixtures.fixture_points(int(opts.get("n_points", 24)), int(opts.get("seed", 7)))
        meta.update(engine="stub", stubbed=["points", "uncertain_queue_source", "features[1,2,3,7,8]", *STUB_ALWAYS], video=None,
                    stub_seed=int(opts.get("seed", 7)))
        quality = fixtures.fixture_quality()
    else:
        from ..app.analyze import analyze as app_analyze
        work = opts.get("work") or tempfile.mkdtemp(prefix="ocv-api-")
        doc = app_analyze(video, work, final=opts.get("final"), mode=meta["mode"], teams=meta["teams"], sport=meta["sport"],
                          labels=opts.get("labels"), region=opts.get("region"))
        pts = doc.get("points") or []
        quality = doc.get("quality") or {}
        meta.update(engine="real", stubbed=list(STUB_ALWAYS), video=doc.get("video"), work=doc["paths"]["work"],
                    session=doc.get("id"), mode=doc.get("mode"), final=doc.get("final") or meta["final"])
        if doc.get("mode_note"): meta["mode_note"] = doc["mode_note"]
    meta["quality"] = quality
    return _derive(pts, meta, quality)


def _resolve(c, match):
    kind, point = c.kind, c.point
    if c.item_id:
        it = next((x for x in match.uncertain_queue if x["id"] == c.item_id), None)
        if it is not None:
            kind, point = it["kind"], it["point"]
        else:
            pre, _, num = c.item_id.partition("-")
            if pre not in ITEM_PREFIX or not num.isdigit():
                raise InvalidCorrection(f"unknown item_id {c.item_id!r}")
            kind, point = ITEM_PREFIX[pre], int(num)
    if kind not in CORRECTION_KINDS:
        raise InvalidCorrection(f"kind must be one of {CORRECTION_KINDS}")
    if not any(p["id"] == point for p in match.points):
        raise InvalidCorrection(f"no point {point!r}")
    if not valid_value(kind, c.value):
        raise InvalidCorrection(f"bad value for {kind}: {c.value!r}")
    return kind, point


def reinfer(match, corrections):
    """Apply one-tap fixes and recompute everything downstream (score timeline, queue, clips, reel plan). Pure: returns a
    new MatchResult and never mutates `match`; all-or-nothing (one bad correction rejects the batch). Persisting the fix
    (session/fixes, opt-in collector upload) is the caller's job."""
    if isinstance(match, dict):
        match = MatchResult.from_dict(match)
    cs = [Correction.parse(c) for c in (corrections or [])]
    resolved = [(c, *_resolve(c, match)) for c in cs]
    pts = copy.deepcopy(match.points)
    meta = copy.deepcopy(match.meta)
    meta.pop("notes", None)
    fixed = list(meta.get("fixed") or [])
    by = {p["id"]: p for p in pts}
    prefix = {v: k for k, v in ITEM_PREFIX.items()}
    for c, kind, point in resolved:
        p = by[point]
        if kind == "point_winner":
            if c.value == "skip":
                p["winner"], p["winner_src"], p["winner_confidence"] = None, c.source, 1.0
            else:
                p["winner"], p["winner_src"], p["winner_confidence"] = c.value, c.source, 1.0
        elif kind == "bounce":
            p["bounce"], p["bounce_confidence"] = c.value, 1.0
        elif kind == "player_id":
            p["player_id"], p["player_id_confidence"] = c.value, 1.0
        elif kind == "cut_off":
            p["cut_off"] = c.value
        elif kind == "score_check":
            p["score_check"] = c.value
        fid = f"{prefix[kind]}-{point}"
        if fid not in fixed: fixed.append(fid)
    meta["fixed"] = fixed
    meta["n_corrections"] = int(meta.get("n_corrections") or 0) + len(resolved)
    return _derive(pts, meta, meta.get("quality") or {}, fixed=fixed)


def render(match, preset="vamos_clean_v1", opts=None):
    """Reel for a MatchResult. opts: score_overlay (default True; False drops the score bug and score captions),
    long_shot_mode (hitter | receiver | auto; recorded, camera not wired yet), aspect, max_points, out, dry_run.
    Returns dict(plan, out, rendered, stubbed). A stub match or dry_run returns the plan only (no encode)."""
    if isinstance(match, dict):
        match = MatchResult.from_dict(match)
    opts = dict(opts or {})
    lsm = opts.get("long_shot_mode") or "hitter"
    if lsm not in ("hitter", "receiver", "auto"):
        raise ValueError("long_shot_mode must be hitter | receiver | auto")
    aspect = opts.get("aspect", "9:16")
    _, states = score_timeline(match.points, match.meta)
    plan = reel_plan_for(match.points, states, match.meta, preset=preset, aspect=aspect, max_points=int(opts.get("max_points", 8)))
    overlay = bool(opts.get("score_overlay", True))
    if not overlay:
        plan["bugs"], plan["captions"] = [], []
    plan["score_overlay"], plan["long_shot_mode"] = overlay, lsm
    stubbed = ["long_shot_mode"]
    video = match.meta.get("video")
    if opts.get("dry_run") or match.meta.get("engine") != "real" or not video or not os.path.exists(video):
        return dict(plan=plan, out=None, rendered=False, stubbed=stubbed + ["encode"])
    out = opts.get("out") or os.path.join(match.meta.get("work") or tempfile.mkdtemp(prefix="ocv-api-"), "reel.mp4")
    r = _encode(video, plan, preset, out, match.meta)
    return dict(plan=plan, out=r.get("out", out), rendered=True, stubbed=stubbed, result=r)


def _encode(video, plan, preset, out, meta):
    import json
    import subprocess

    from ..ffmpeg import ffprobe_bin
    from ..style.preset import load as load_preset
    from ..style.render import render_reel
    probe = json.loads(subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height,avg_frame_rate", "-of", "json", video], capture_output=True, text=True).stdout or "{}")
    st = (probe.get("streams") or [{}])[0]
    fps = 30
    try:
        num, den = (st.get("avg_frame_rate") or "30/1").split("/")
        fps = max(24, min(60, round(float(num) / float(den))))
    except Exception:
        pass
    work = os.path.join(meta.get("work") or os.path.dirname(os.path.abspath(out)), "exports", "reel")
    os.makedirs(work, exist_ok=True)
    return render_reel(video, plan, load_preset(preset), out, work=work, SW=int(st.get("width") or 3840),
                       SH=int(st.get("height") or 2160), fps=min(fps, 30))
