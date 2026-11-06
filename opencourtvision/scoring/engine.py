"""Scorer selection by region profile (v3.9, D1). Default everywhere: forward-only (`scoring.forward`). The whole-match
decoder exists only in the full app's best-variant package and only runs when a full-app profile sets scorer="global"."""
from .forward import ForwardScorer


def make_scorer(profile, fmt, match_scoring=None, **kw):
    """-> a ForwardScorer, or the full-app global decode function. The decoder runs only in a full build, when the match
    was started with MatchScoring.kind == "global" (country gate, scoring/region_config.py) or a developer profile sets
    scorer="global". Vamos US has no decoder at all, so it always gets the ForwardScorer."""
    want = (match_scoring is not None and match_scoring.kind == "global") or getattr(profile, "scorer", "forward") == "global"
    if want and profile.variant == "full":
        from ..variants import SCORERS, discover
        discover()
        if "global" in SCORERS:
            return SCORERS["global"]
    # Q1 (hold until next serve) is disabled by Legal ruling 05:41: the policy is always point_end here.
    return ForwardScorer(fmt, policy="point_end", score_checks=getattr(profile, "score_checks", None),
                         voice_edits=getattr(profile, "voice_edits", False),
                         suggestions=getattr(profile, "suggested_fixes", False), **kw)
