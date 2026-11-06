"""Closed EU beta (Vamos Ops V2, 21 Oct 2026): cloud processing is off in every profile, beta overrides included."""
import pytest

from opencourtvision.editloop import build

# ---- V2: cloud processing off in every beta profile ----

@pytest.mark.parametrize("p", list(build.PROFILES.values()) + [build.RESTRICTED], ids=lambda p: p.name)
def test_every_profile_has_cloud_processing_off(p):
    assert p.cloud_processing is False


@pytest.mark.parametrize("region", ["ROW", "UK", "ES", "US", None])
def test_beta_flags_cannot_switch_cloud_processing_on(region, monkeypatch):
    monkeypatch.setenv("VAMOS_BETA", "1")
    assert build.flags(region).cloud_processing is False
    assert build.flags(region, cloud_processing=True).cloud_processing is False
