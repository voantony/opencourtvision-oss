"""Two app records / two binaries (eng notes §2.8.8): the US artefact carries no best-variant code, and CI proves it."""
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_variant  # noqa: E402
import scan_us_build  # noqa: E402

from opencourtvision.editloop.build import ES, ROW, UK, US, flags, session_profile  # noqa: E402


def test_us_artefact_has_no_best_variant_symbols(tmp_path):
    out = build_variant.build("us", str(tmp_path / "us"))
    assert not os.path.exists(os.path.join(out, "opencourtvision", "variants", "full"))
    assert scan_us_build.scan(out) == []
    # the US artefact runs and reports itself as US whatever region is requested
    code = ("from opencourtvision.editloop.build import flags, built_variant\n"
            "f = flags('ROW', typed_filters_tracking=True)\n"
            "assert built_variant() == 'us' and f.name == 'US' and not f.typed_filters_tracking and not f.score_in_selection\n")
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": out}, cwd=str(tmp_path))


def test_scanner_catches_full_code(tmp_path):
    out = build_variant.build("full", str(tmp_path / "full"))
    hits = scan_us_build.scan(out)
    assert any(s == "score_aware_select" for _, s in hits)


def test_full_build_refused_on_us_ci(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_variant.py"), "--variant", "full", "--out", str(tmp_path / "x")],
                       env={**os.environ, "OCV_CI_REGION": "US"}, capture_output=True, text=True)
    assert r.returncode != 0


def test_region_profiles():
    assert UK.no_3d_trajectory and not UK.multi_phone                 # UK: full app, D2 stays (Maiden GB); multi-phone held
    assert not ES.auto_refereeing and not ES.multi_phone               # ES: compliant fallback until Spanish counsel
    assert ROW.score_in_selection and not US.score_in_selection
    assert flags("ES").name == "ES"


def test_session_gate_is_location_based_and_privacy_minimal():
    p, why = session_profile("ES", "ES"); assert p is ES and why == ""
    p, why = session_profile("ES", "US"); assert p.name == "RESTRICTED" and not p.typed_filters_tracking   # traveller in the US
    p, why = session_profile("GB", "denied"); assert p.name == "RESTRICTED" and "approximate location" in why
    p, why = session_profile("FR", "unknown", ("sim_us",)); assert p.name == "RESTRICTED"
    p, _ = session_profile("US", "ES"); assert p is US                                                    # US storefront
