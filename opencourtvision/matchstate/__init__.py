"""Match-state classifier for one-tap recording (keep real points, reject casual play).

States: warm_up | match_point | between_points | changeover | break_casual | post_match.
US build: rules + ridge/GBDT-style linear model over hand features (N1–N5). Unconstrained may add more.
"""
from .classify import STATES, features_for_segment, predict_batch, rule_prior, score_signals

__all__ = ["STATES", "features_for_segment", "predict_batch", "rule_prior", "score_signals"]
