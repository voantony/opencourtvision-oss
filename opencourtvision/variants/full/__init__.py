"""Best variant (non-US "Vamos" app only; not packaged into "Vamos US"). Region rules inside the full app come from
`opencourtvision.editloop.build` profiles (UK: D2 stays; ES: auto-refereeing off until Spanish counsel)."""
from .. import REGISTRY, SCORERS


def _global_decode(*a, **k):
    """Lazy: the decoder module loads only when a full-app profile switches scorer="global"."""
    from .scoring_global import decode
    return decode(*a, **k)


REGISTRY["full"] = __name__
SCORERS["global"] = _global_decode      # switchable; Profile.scorer defaults to "forward"
