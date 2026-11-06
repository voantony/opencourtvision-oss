"""Common (US-safe) clip selection: E2, always. Selection and ranking read only audio/segment features and user inputs.

Hard rule: no tracked-ball event reaches selection (provenance.check on every segment; ball feature names are rejected).
The full app's score-aware / vision-ranked selection and tracking-typed playback live in the best-variant package,
which is not packaged into Vamos US.
"""
from .provenance import ProvenanceError, check

AUDIO_FEATURES = frozenset({"hits", "rally_hits", "fast", "tempo", "amax", "react", "react_dur", "dur", "clean",
                            "fault_like", "serve_only", "favourite", "user_rating"})
BALL_FEATURES = frozenset({"t_dead", "t_dec", "bounce", "bounce_xy", "ball_speed", "incoming_speed", "ball_track", "in_out",
                           "shot_count_ball", "net_cord"})
SCORE_FEATURES = frozenset({"break_point", "set_point", "match_point", "deuce", "winner", "pressure"})


def selection_features(seg):
    feats = dict(seg.get("features") or {})
    bad = set(feats) & BALL_FEATURES
    if bad:
        raise ProvenanceError(f"tracked-ball features {sorted(bad)} must never feed clip selection")
    return {k: v for k, v in feats.items() if k in AUDIO_FEATURES}


def select_clips(segs, n, key=None, allowed_sources=None):
    """Top-n segments by `key(features)` (default: audio excitement), in time order. Raises on forbidden provenance."""
    check(segs, allowed_sources) if allowed_sources else check(segs)
    key = key or (lambda f: 1.2 * f.get("hits", 0) + 0.9 * f.get("react", 0) + 2.0 * f.get("favourite", 0))
    scored = [(key(selection_features(s)), i) for i, s in enumerate(segs)
              if not s.get("fault_like") and (s["B"] - s["A"]) <= 40]
    return sorted(i for _, i in sorted(scored, reverse=True)[:n])


PRESSURE_KEYS = frozenset({"break_point", "set_point", "match_point", "deuce", "tb_game_point", "user_set_point",
                           "user_match_point", "pressure"})


def story_score_arg(score, profile):
    """What `story.build(..., score=...)` may receive under `profile`: None unless the decoded score may rank clips; and
    without the score-pressure keys unless `profile.score_pressure` (Sony EP 4236307 A1 / GB 2616012 A pending: switch)."""
    if score is None or not profile.score_in_selection:
        return None
    if getattr(profile, "score_pressure", False):
        return score
    return [{k: v for k, v in (d or {}).items() if k not in PRESSURE_KEYS} for d in score]


def strip_pressure(points, profile):
    """story.highlight PointInput list -> the same with pressure off (flags all False, no game score) unless
    profile.score_pressure. Outcome (winner / ending from the decoded score) is kept: it is a rule feature."""
    from dataclasses import replace
    if getattr(profile, "score_pressure", False):
        return list(points)
    off = {k: False for k in ("match_point", "set_point", "break_point", "deuce", "tb_game_point", "user_set_point",
                              "user_match_point")}
    return [replace(p, pressure_flags=dict(off)) for p in points]


# ---- v3.9 item 3: filter types and per-player reels (both builds) ----
# A filter / moment-search type must come from the decoded score state, from audio, or from a user tag; never from a
# tracking label (ball or player track). Every typed item carries the source of its type.
TYPE_SOURCES = frozenset({"score", "audio", "user"})
FORBIDDEN_TYPE_SOURCES = frozenset({"track", "ball_track", "player_track", "tracking", "pose", "vision"})


def typed_filter(segs, kind):
    """Segments whose type `kind` was set by the score state, audio or a user tag (seg["types"] = {kind: source})."""
    out = []
    for i, s in enumerate(segs):
        src = (s.get("types") or {}).get(kind)
        if src is None:
            continue
        if src in FORBIDDEN_TYPE_SOURCES or src not in TYPE_SOURCES:
            raise ProvenanceError(f"type {kind!r} from {src!r}: filter types come from score, audio or user tags only")
        out.append(i)
    return out


def player_reel(points, player, setup, states):
    """Per-player reel from setup tags + the decoded score, never from tracked presence.
    setup: {"teams": {player: 'A'|'B'}, "serve_order": {'A': [p1, p2], 'B': [p3, p4]}} (doubles; singles: one each).
    states: committed score state before each point (scoring.forward). A point is in the player's reel when the player
    served it (team server from the score + setup serve order and games served so far) or the player's team won it."""
    team = setup["teams"][player]
    order = setup.get("serve_order", {}).get(team, [player])
    served = {"A": 0, "B": 0}
    out, last = [], None
    for i, (pt, st) in enumerate(zip(points, states)):
        key = (st.sets, st.games, st.server)
        if key != last and last is not None and last[2] != st.server and not st.tb:
            served[last[2]] += 1
        last = key
        server_player = order[served[team] % len(order)] if st.server == team else None
        if server_player == player or pt.get("winner") == team:
            out.append(i)
    return out
