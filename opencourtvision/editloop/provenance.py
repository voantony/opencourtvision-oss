"""Provenance guard: tracked-ball (and other tracking-typed) evidence must never reach clip selection."""

ALLOWED_SOURCES = frozenset({"audio", "structure", "motion_energy", "pose", "user"})
FORBIDDEN_SOURCES = frozenset({"ball", "ball_track", "bounce", "tracknet", "swingvision", "events/vision", "track/ball"})


class ProvenanceError(ValueError):
    pass


def check(items, allowed=ALLOWED_SOURCES):
    """Raise unless every item's `sources` is a non-empty subset of `allowed` (and never a forbidden source)."""
    for i, it in enumerate(items):
        src = set(it.get("sources") or ())
        if not src:
            raise ProvenanceError(f"segment {i} has no provenance")
        bad = src & FORBIDDEN_SOURCES
        if bad:
            raise ProvenanceError(f"segment {i} carries tracked-ball/tracking-typed provenance {sorted(bad)}: never allowed in clip selection")
        if not src <= set(allowed):
            raise ProvenanceError(f"segment {i} sources {sorted(src - set(allowed))} not allowed in this build")
    return items
