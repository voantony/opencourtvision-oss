"""Per-country gate for the full app's whole-match decoder (Antony via Legal, 05:46)."""
import hashlib
import hmac
import json
import pickle

import pytest

from opencourtvision.editloop.build import flags
from opencourtvision.scoring.engine import make_scorer
from opencourtvision.scoring.formats import get_format
from opencourtvision.scoring.forward import ForwardScorer
from opencourtvision.scoring.region_config import DEFAULT_OFF, UPC_UNITARY_EP3834171, ConfigCache, MatchScoring, decoder_allowed, parse_config
from opencourtvision.scoring.rules import PointEvidence

FMT = get_format("tennis", "advantage")
KEY = b"test-only-key"                      # stands in for the platform public-key signature check


def sign(p):
    return hmac.new(KEY, p, hashlib.sha256).digest()


def verify(p, sig):
    return hmac.compare_digest(sign(p), sig)


def cfg(off=DEFAULT_OFF, issued=1_000_000.0, version=1):
    p = json.dumps(dict(version=version, issued_at=issued, off=sorted(off))).encode()
    return p, sign(p)


NOW = 1_000_000.0 + 3600


def test_legal_0559_country_list():
    c = parse_config(*cfg(), verify, now=NOW)
    off = "AT BE BG DE DK EE FI FR IT LT LU LV MT NL PT RO SE SI GB IE CH LI MC CZ SK HU HR GR CY TR AL MK SM US".split()
    on = "ES NO IS PL RS ME BA MA MD TN KH GE LA AU CA MX BR JP KR IN ZA NZ AE".split()
    for cc in off:
        assert not decoder_allowed(cc, cc, c), cc
    for cc in on:
        assert decoder_allowed(cc, cc, c), cc
    assert set(off) == set(DEFAULT_OFF)


def test_legal_laos_on():
    """Legal 18 Oct: LA ON (EPO validation in force 2024, after the 2019 PCT), like GE."""
    from opencourtvision.scoring.region_config import NATIONAL_ON
    c = parse_config(*cfg(), verify, now=NOW)
    assert "LA" in NATIONAL_ON and "LA" not in DEFAULT_OFF
    assert decoder_allowed("LA", "LA", c) and decoder_allowed("la", "LA", c)
    assert not decoder_allowed("LA", "LA", c, variant="us")
    assert not decoder_allowed("LA", "TH", parse_config(*cfg(off=DEFAULT_OFF | {"TH"}), verify, now=NOW))
    assert not decoder_allowed("LA", "DE", c) and not decoder_allowed("LA", "LA", None)


def test_upc_list_matches_clearance_file():
    assert UPC_UNITARY_EP3834171 == set("AT BE BG DE DK EE FI FR IT LT LU LV MT NL PT RO SE SI".split())
    assert {"GB", "IE", "US"} <= DEFAULT_OFF and "ES" not in DEFAULT_OFF


@pytest.mark.parametrize("st,loc,expect", [
    ("AU", "AU", True), ("CH", "NO", False), ("PL", "PL", True), ("AU", "TR", False), ("MA", "MA", True), ("AU", "DE", False), ("DE", "AU", False), ("ES", "ES", True), ("ES", "FR", False),
    ("GB", "GB", False), ("AU", "GB", False), ("FR", "FR", False), ("AU", "denied", False), ("AU", "unknown", False),
    ("US", "AU", False), ("AU", "US", False), ("IE", "IE", False), ("MX", "BR", True), ("", "AU", False),
])
def test_gating_truth_table(st, loc, expect):
    c = parse_config(*cfg(), verify, now=NOW)
    assert decoder_allowed(st, loc, c) is expect
    assert decoder_allowed(st, loc, c, variant="us") is False


def test_missing_bad_or_stale_config_is_off():
    p, s = cfg()
    assert parse_config(None, None, verify, now=NOW) is None
    assert parse_config(p, b"bad", verify, now=NOW) is None
    assert parse_config(p, s, None, now=NOW) is None                         # no verifier: nothing trusted
    assert parse_config(p, s, verify, now=NOW + 8 * 24 * 3600) is None        # stale
    assert parse_config(b"{not json", sign(b"{not json"), verify, now=NOW) is None
    for c in (None,):
        assert not decoder_allowed("AU", "AU", c)


def test_config_can_switch_countries_on_and_us_is_always_off():
    c = parse_config(*cfg(off={"GB"}), verify, now=NOW)                       # e.g. opposition succeeded
    assert decoder_allowed("DE", "DE", c) and not decoder_allowed("GB", "GB", c)
    c2 = parse_config(*cfg(off=set()), verify, now=NOW)
    assert "US" in c2.off and not decoder_allowed("US", "US", c2)


def test_cache_keeps_last_good_and_refuses_rollback():
    cache = ConfigCache()
    assert cache.current(verify, now=NOW) is None
    assert cache.update(*cfg(version=2), verify, now=NOW)
    assert not cache.update(*cfg(version=1), verify, now=NOW)
    assert not cache.update(b"x", b"y", verify, now=NOW)
    assert cache.current(verify, now=NOW).version == 2
    assert cache.current(verify, now=NOW + 30 * 24 * 3600) is None            # stale cache -> OFF


def test_match_keeps_its_scorer_and_never_rewrites_on_region_or_config_change():
    off_cfg = parse_config(*cfg(), verify, now=NOW)
    on_cfg = parse_config(*cfg(off=set()), verify, now=NOW)
    m = MatchScoring.start("FR", "FR", off_cfg)
    assert m.kind == "forward"
    sc = make_scorer(flags("ES"), FMT, match_scoring=m)
    assert isinstance(sc, ForwardScorer)
    for p in (0.9, 0.6, 0.55, 0.8):
        sc.point_ended(PointEvidence(p_a=p))
    before = pickle.dumps((sc.commits[:4], sc.states[:5]))
    assert m.region_or_config_changed("AU", "AU", on_cfg) == "forward"        # moved + config flipped mid-match
    sc.point_ended(PointEvidence(p_a=0.7))
    assert pickle.dumps((sc.commits[:4], sc.states[:5])) == before           # committed points untouched
    assert m.kind == "forward" and m.events[-1]["kind"] == "ignored_mid_match"
    assert m.user_rescore("AU", "AU", on_cfg) == "global"                    # explicit user action only


def test_global_kind_never_reaches_a_us_artefact_or_profile():
    m = MatchScoring("global")
    assert isinstance(make_scorer(flags("US"), FMT, match_scoring=m), ForwardScorer)
    assert not isinstance(make_scorer(flags("ROW"), FMT, match_scoring=m), ForwardScorer)


def test_us_artefact_has_zero_decoder_hits(tmp_path):
    import os
    import subprocess
    import sys
    root = os.path.join(os.path.dirname(__file__), "..")
    out = tmp_path / "us"
    subprocess.run([sys.executable, os.path.join(root, "scripts", "build_variant.py"), "--variant", "us", "--out", str(out)], check=True)
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "scan_us_build.py"), str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and "0 hit" in r.stdout
    assert not (out / "opencourtvision" / "scoring" / "decoder.py").exists()
    assert not (out / "opencourtvision" / "pose" / "syllables.py").exists()          # Legal 05:55
    r2 = subprocess.run(
        [sys.executable, "-c",
         "from opencourtvision.pose import analyze; assert analyze._syllables() is None"],
        cwd=str(out), capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(out)},
    )
    assert r2.returncode == 0, r2.stderr
    assert "viterbi" not in "".join(p.read_text().lower() for p in out.rglob("*.py"))
    assert (out / "opencourtvision" / "scoring" / "region_config.py").exists()
    src = "".join(p.read_text() for p in out.rglob("*.py"))
    for sym in ("forward_backward", "confirm_loop", "next_best_check", "anchor_conflicts", "ScoreConflict"):
        assert sym not in src, sym
    scoring = "".join(p.read_text().lower() for p in (out / "opencourtvision" / "scoring").rglob("*.py"))
    for sym in ("viterbi", "beam", "def decode("):          # no score lattice search anywhere in the US scoring package
        assert sym not in scoring, sym
