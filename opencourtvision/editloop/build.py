"""Region profiles (eng notes §2.8.8; CEO decision 18 Oct 04:43, based on Legal).

Two app records, two binaries:
- "Vamos US" (US storefront only): variant "us". The best-variant package is not packaged at all (excluded by
  scripts/build_variant.py; CI runs scripts/scan_us_build.py). Nothing here can switch it on.
- "Vamos" (every non-US storefront): variant "full", with per-region profiles inside it: UK keeps D2 (Maiden GB 2621269 B),
  ES runs the compliant fallback until Spanish counsel reads Foxtenn ES 2680393 B1, RoW is full.
Inside the full app the region is decided per session from the storefront plus an on-device coarse location check
(`session_profile`), never from IP alone.
"""
import os
from dataclasses import dataclass, replace

from .. import privacy
from .provenance import ALLOWED_SOURCES


@dataclass(frozen=True)
class Profile:
    name: str
    variant: str                      # "us" (Vamos US binary) or "full" (Vamos binary)
    score_in_selection: bool          # decoded score may rank/select clips (E2 off)
    vision_rank_features: bool        # pose/player-track signals may rank clips
    typed_filters_tracking: bool      # playback/filters by tracking-derived event type ('378)
    multi_phone: bool                 # '825
    cloud_processing: bool            # '478 (N3)
    no_3d_trajectory: bool            # D2 ('540 US, Maiden GB 2621269 B in the UK)
    auto_refereeing: bool             # automatic in/out calls shown as refereeing (Foxtenn ES 2680393 B1 in Spain)
    vision_rally_cues: bool = False   # motion-energy fallback for segmentation, off by default everywhere
    gap_fill: bool = True             # E3
    allowed_sources: frozenset = ALLOWED_SOURCES
    # v3.9 (CEO 05:03 + Legal 05:06/05:07)
    motion_boundaries: bool = False   # motion cues may admit/drop clips (EU); False = motion only gates live play ('807/'160)
    score_pressure: bool = False      # score-pressure weight in highlight ranking (Sony EP 4236307 A1 / GB 2616012 A pending)
    scorer: str = "forward"           # "forward" (D1, both builds) or "global" (full app only; Wingfield EP opposition switch)
    pending_commit: bool = False      # Legal Q1: hold until next serve - kept in code, DISABLED (Legal 05:41)
    voice_edits: bool = True          # Legal Q2: players' spoken score calls are user input (shipped 05:41)
    score_checks: float = 0.1         # Legal Q3: score-check prompts on the one shown state (shipped 05:41)
    suggested_fixes: bool = True      # ONE proposed backward fix, applied only on user approval (Antony via Legal, 05:28)
    # 18 Oct (MD for Antony): with a motion file, the live-play gate is on by default. Keep/drop whole clips only (an idle
    # court drops a clip); it never moves a boundary, so US boundaries stay audio-only (N rules, '807/'160).
    live_gate: bool = True


# Positional fields: name, variant, score_in_selection, vision_rank_features, typed_filters_tracking, multi_phone,
# cloud_processing, no_3d_trajectory, auto_refereeing. v3.9: vision rank features and tracking-typed filters are off in
# every build (highlights rank from audio + rules; filter types from score state, audio or user tags); multi-phone is held
# in every full-app profile pending a PlayReplay EP 4323076 A1 check.
US = Profile("US", "us", False, False, False, False, False, True, True)
# cloud_processing is False everywhere for the closed EU beta (Ops V2, 21 Oct 2026: Legal's position is on-device
# processing). UK and ES inherit it from ROW. Turning it back on needs a Legal sign-off and a notice change.
ROW = Profile("ROW", "full", True, False, False, False, False, False, True, motion_boundaries=True, score_pressure=True)
UK = replace(ROW, name="UK", no_3d_trajectory=True)
ES = replace(ROW, name="ES", auto_refereeing=False)          # compliant fallback, ready for wk 12; full once counsel clears
RESTRICTED = replace(US, name="RESTRICTED", variant="full")   # full app in the US, or location unknown with US signals
PROFILES = {p.name: p for p in (US, ROW, UK, ES)}


def built_variant():
    """'us' when the best-variant package is absent from this artefact, else 'full'."""
    from ..variants import has_full
    return "full" if has_full() else "us"


def flags(region=None, **over):
    """Profile for `region` (env OCV_BUILD_REGION, default ROW). OCV_VISION_RALLY_CUES=1 turns the vision fallback on.
    A US-variant artefact always returns the US profile, whatever is asked."""
    region = (region or os.environ.get("OCV_BUILD_REGION", "ROW")).upper()
    p = PROFILES.get(region, ROW)
    if built_variant() == "us" or p.variant == "us":
        p = US
        over = {k: v for k, v in over.items() if k in ("vision_rally_cues", "gap_fill", "voice_edits", "suggested_fixes", "score_checks", "live_gate")}
    if built_variant() != "full":
        over.pop("scorer", None)
    if os.environ.get("OCV_VISION_RALLY_CUES") == "1": over.setdefault("vision_rally_cues", True)
    if privacy.beta_mode(): over["cloud_processing"] = False   # Ops V2: no override can switch it on in a beta build
    return replace(p, **over)


US_SIGNALS = ("sim_us", "locale_us", "timezone_us")


def session_profile(storefront, location, signals=()):
    """Per-session region decision inside the full app (design: eng notes §2.8.8).

    storefront: App Store / Play country (ISO-2). location: on-device coarse result, one of an ISO-2 country code,
    "denied" or "unknown". Precise location is never requested or stored; only (decision, reason) is logged.
    Returns (profile, reason_for_user)."""
    st = (storefront or "").upper()
    if st == "US":
        return US, "This app is for use outside the US. Install Vamos US."
    loc = (location or "unknown")
    if loc.upper() == "US":
        return RESTRICTED, "Some features aren't available in your current region."
    if loc in ("denied", "unknown"):
        if any(s in signals for s in US_SIGNALS):
            return RESTRICTED, "Location is off and this device looks US-based, so some features are limited for this session."
        return RESTRICTED, ("Location is off, so features that depend on your region are limited for this session. "
                            "Allow approximate location to use them; it is checked on this phone only.")
    return PROFILES.get(loc.upper(), PROFILES.get(st, ROW)), ""
