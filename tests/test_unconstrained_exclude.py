"""Unconstrained research track is importable and stripped from the US artefact."""
import os
import shutil
import tempfile

from scripts.build_variant import EXCLUDE_US, build
from scripts.scan_us_build import scan


def test_exclude_lists_research_and_physics3d():
    assert "opencourtvision/research" in EXCLUDE_US
    assert "opencourtvision/ball/physics3d.py" in EXCLUDE_US


def test_us_build_strips_unconstrained():
    out = tempfile.mkdtemp()
    try:
        build("us", out)
        assert not os.path.exists(os.path.join(out, "opencourtvision", "research"))
        assert not os.path.exists(os.path.join(out, "opencourtvision", "ball", "physics3d.py"))
        hits = scan(out)
        # denylist may still match strings inside remaining files; physics3d path must be gone
        phys_hits = [h for h in hits if h[1] == "physics3d"]
        # If any physics3d string remains in US tree, fail
        assert not phys_hits, phys_hits
    finally:
        shutil.rmtree(out)


def test_unconstrained_reexport():
    from opencourtvision.research.unconstrained import TRACK, physics3d
    assert TRACK == "unconstrained"
    assert hasattr(physics3d, "Camera")
