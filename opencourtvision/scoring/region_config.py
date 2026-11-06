"""Per-country switch for the full app's whole-match decoder (Antony via Legal, 18 Oct 05:46; design, not legal advice).

The whole-match decoder (automatic forward and backward correction) runs only where Wingfield EP 3834171 B1 is not in
force, and only in the full ("Vamos") app; Vamos US never contains it. Everywhere else: forward-only + taps + Q3 score
checks + Q2 voice anchors.

Rule: decoder only if BOTH the storefront and the per-session on-device coarse location are allowed countries under a
valid config; anything else (missing / unverifiable / stale config, location denied or unknown, either country OFF) is
forward-only. A match keeps the scorer it started with (`MatchScoring`); only an explicit user re-score changes it.

Country list source: rallycut/eu_uk_clearance.md §3.1 (unitary effect from 18 May 2026 in 18 UPC states; GB and ES
validations unconfirmed; IE national record) Legal 05:59 (§9): OFF = 18 UPC states + GB IE CH LI MC CZ SK HU HR GR CY TR AL MK SM + US;
ON = ES NO IS PL RS ME BA MA MD TN KH GE LA + every non-EPC full-app storefront. Legal 18 Oct (batched memo): LA ON, since
Laos's EPO validation took effect in 2024, after Wingfield's 2019 PCT filing (same reasoning as GE). Countries are flipped
later through the signed server config.
"""
import json
import time
from dataclasses import dataclass, field

UPC_UNITARY_EP3834171 = frozenset("AT BE BG DE DK EE FI FR IT LT LU LV MT NL PT RO SE SI".split())
# Legal 05:59 (eu_uk_clearance.md §9): national status of EP 3834171 B1 outside the UPC.
NATIONAL_OFF = frozenset("GB IE CH LI MC CZ SK HU HR GR CY TR AL MK SM".split())
# lapsed / never validated / not covered. GE and LA: validation agreements in force after the 2019 PCT filing (Legal 18 Oct).
NATIONAL_ON = frozenset("ES NO IS PL RS ME BA MA MD TN KH GE LA".split())
DEFAULT_OFF = UPC_UNITARY_EP3834171 | NATIONAL_OFF | {"US"}

MAX_AGE_S = 7 * 24 * 3600                              # a cached config older than this is stale -> OFF


@dataclass(frozen=True)
class DecoderConfig:
    version: int
    issued_at: float                 # unix seconds
    off: frozenset                   # countries where the decoder is OFF (US is always OFF regardless)
    max_age_s: float = MAX_AGE_S


def parse_config(payload: bytes, signature: bytes, verify, now=None):
    """-> DecoderConfig, or None (= OFF everywhere) when the payload is missing, the signature does not verify, it is
    malformed, or it is stale. `verify(payload, signature) -> bool` is the platform public-key check (Ed25519 via
    CryptoKit / Tink on device); without a verifier nothing is trusted."""
    if not payload or not signature or verify is None:
        return None
    try:
        if not verify(payload, signature):
            return None
        d = json.loads(payload.decode("utf-8"))
        cfg = DecoderConfig(int(d["version"]), float(d["issued_at"]),
                            frozenset(str(c).upper() for c in d["off"]) | {"US"},
                            float(d.get("max_age_s", MAX_AGE_S)))
    except Exception:
        return None
    now = time.time() if now is None else now
    if now - cfg.issued_at > cfg.max_age_s or cfg.issued_at > now + 300:
        return None
    return cfg


@dataclass
class ConfigCache:
    """Last good signed config (payload + signature), re-verified and age-checked on every read."""
    payload: bytes = None
    signature: bytes = None

    def update(self, payload, signature, verify, now=None):
        new = parse_config(payload, signature, verify, now)
        if new is None:
            return False
        old = parse_config(self.payload, self.signature, verify, now)
        if old is not None and new.version < old.version:
            return False                                  # no rollback to an older list
        self.payload, self.signature = payload, signature
        return True

    def current(self, verify, now=None):
        return parse_config(self.payload, self.signature, verify, now)


def _country(x):
    x = (x or "").upper()
    return x if len(x) == 2 and x.isalpha() else None


def decoder_allowed(storefront, location, config, variant="full"):
    """Truth table: True only for the full app, a valid config, and BOTH storefront and location allowed countries."""
    if variant != "full" or config is None:
        return False
    st, loc = _country(storefront), _country(location)
    if st is None or loc is None:                         # denied / unknown location -> forward-only
        return False
    return st not in config.off and loc not in config.off


@dataclass
class MatchScoring:
    """The scorer a match uses, fixed at match start. Region moves or config changes mid-match are logged and ignored;
    only `user_rescore` (an explicit user action) changes it."""
    kind: str                                           # "global" or "forward"
    reason: str = ""
    events: list = field(default_factory=list)

    @classmethod
    def start(cls, storefront, location, config, variant="full"):
        ok = decoder_allowed(storefront, location, config, variant)
        return cls("global" if ok else "forward", f"storefront={storefront} location={location} config={'ok' if config else 'none'}")

    def region_or_config_changed(self, storefront, location, config, variant="full"):
        would = "global" if decoder_allowed(storefront, location, config, variant) else "forward"
        self.events.append(dict(kind="ignored_mid_match", would=would, storefront=storefront, location=location))
        return self.kind

    def user_rescore(self, storefront, location, config, variant="full"):
        """Explicit user re-score: re-evaluated under the CURRENT storefront / location / config."""
        self.kind = "global" if decoder_allowed(storefront, location, config, variant) else "forward"
        self.events.append(dict(kind="user_rescore", to=self.kind))
        return self.kind
