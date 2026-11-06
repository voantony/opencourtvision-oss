"""Stable engine interface for every Vamos client (web beta, Mac runner, iPhone, Watch): one call shape, many engines.

    from opencourtvision import api
    m = api.analyze("match.mp4", {"work": "/tmp/w", "final": "6-4 6-3"})       # -> MatchResult
    m2 = api.reinfer(m, [{"item_id": "pw-3", "value": "B"}])                    # one-tap fixes -> new MatchResult
    r = api.render(m2, "vamos_clean_v1", {"score_overlay": True, "long_shot_mode": "hitter", "out": "reel.mp4"})

MatchResult (schema vamos.match_result.v1) carries points, score_timeline, uncertain_queue, clips, reel_plan, players and
meta. meta.engine says which engine produced it ("real" | "stub") and meta.stubbed lists every field that came from a
fixture rather than the engine, so a client can never mistake a fixture for a real call.

Engines (opts["engine"]):
  "real"  the on-device pipeline on main (opencourtvision.app.analyze: audio segmentation, rallies/match mode, fix queue)
  "stub"  deterministic synthetic fixture (no footage, no labels, no faces); for UI work, web beta and tests
  "auto"  real when the video exists and ffmpeg is available, else stub (meta.fallback_reason says why)

Real today: analyze (pipeline), score_timeline (rules-exact replay), uncertain_queue (app.review), clips (style.plan
whole-point bounds), reel_plan (style.plan.plan_reel), reinfer (fix application + recompute), render (style.render).
Stubbed today: players (court-side roles only), render long_shot_mode (recorded in the plan, camera not wired yet),
browser/ONNX inference (lives in the client). Never faces or biometrics: player identity is court side / role only.
"""
from .engine import ENGINES, analyze, reinfer, render
from .features import FEATURE_DIM, FEATURE_NAMES, FEATURES_VERSION, fix_example
from .types import API_VERSION, CORRECTION_KINDS, SCHEMA, Correction, InvalidCorrection, MatchResult

__all__ = ["analyze", "reinfer", "render", "MatchResult", "Correction", "InvalidCorrection", "SCHEMA", "API_VERSION", "FEATURES_VERSION", "FEATURE_NAMES", "FEATURE_DIM", "fix_example",
           "CORRECTION_KINDS", "ENGINES"]
