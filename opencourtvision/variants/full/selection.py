"""Full-app clip selection: may use decoded-score context (score-pressure weights only when the profile switch
`score_pressure` is on; Sony EP 4236307 A1 / GB 2616012 A pending). Vision rank features are behind
`profile.vision_rank_features`, off in every v3.9 profile (highlights rank from audio + rule features).
v3.9: typed playback by tracking labels is REMOVED in both builds (filter types come from score state, audio or user
tags: editloop.select.typed_filter). Hard rule kept: tracked ball events never create or select clips."""
from ...editloop.provenance import ProvenanceError, check
from ...editloop.select import AUDIO_FEATURES, BALL_FEATURES, PRESSURE_KEYS, SCORE_FEATURES

VISION_RANK_FEATURES = frozenset({"shot_difficulty", "defensive_get", "erne", "tweener", "padel_finish", "pushing_rally", "hitter"})


def score_aware_select(segs, profile, n, key=None):
    """Top-n using audio + decoded score (+ vision rank features unless the region profile disables them)."""
    check(segs)
    allowed = set(AUDIO_FEATURES) | set(SCORE_FEATURES) | (set(VISION_RANK_FEATURES) if profile.vision_rank_features else set())
    if not getattr(profile, "score_pressure", False):
        allowed -= PRESSURE_KEYS
    def feats(s):
        f = dict(s.get("features") or {})
        if set(f) & BALL_FEATURES: raise ProvenanceError("tracked-ball features never feed clip selection")
        return {k: v for k, v in f.items() if k in allowed}
    key = key or (lambda f: 1.2 * f.get("hits", 0) + 0.9 * f.get("react", 0) + 1.5 * f.get("match_point", 0)
                  + 0.8 * f.get("break_point", 0) + 1.0 * f.get("shot_difficulty", 0) + 2.0 * f.get("favourite", 0))
    sc = [(key(feats(s)), i) for i, s in enumerate(segs) if not s.get("fault_like")]
    return sorted(i for _, i in sorted(sc, reverse=True)[:n])

