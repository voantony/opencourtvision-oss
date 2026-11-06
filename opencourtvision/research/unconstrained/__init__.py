"""Unconstrained research track (Antony / CEO, 19 Oct 2026).

Best-possible engine with no patent constraints. Lives only here. Legal must clear each
feature per region before anything ships. The US build excludes this package and the
3D physics modules via scripts/build_variant.py + ci/us_build_denylist.txt.

Canonical first entry: the physics3d tracker from draft PR #18 (3D ODE + bounce joint
fit + shot-speed outputs). Import it through this package so call sites are obviously
unconstrained:

    from opencourtvision.research.unconstrained import physics3d
"""

from opencourtvision.ball import physics3d as physics3d  # noqa: F401

TRACK = "unconstrained"
CONSTRAINTS = ()  # none: patents ignored; AGPL still forbidden
