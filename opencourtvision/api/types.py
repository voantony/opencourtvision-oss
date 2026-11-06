"""MatchResult + Correction types (plain dataclasses, JSON round-trip, no third-party deps)."""
import copy
import re
from dataclasses import asdict, dataclass, field, is_dataclass

SCHEMA = "vamos.match_result.v1.1"
SCHEMAS_READ = ("vamos.match_result.v1", SCHEMA)   # v1 documents load; features are filled on the next reinfer
API_VERSION = "0.2.0"
CORRECTION_KINDS = ("point_winner", "bounce", "player_id", "cut_off", "score_check")
# never accepted anywhere in a correction (GDPR Art. 9: no faces / biometrics / names / raw media)
FORBIDDEN_KEY = re.compile(r"face|embedding|biometric|name|email|phone|gps|location|video|image|audio|voice|crop|device_id", re.I)


class InvalidCorrection(ValueError):
    pass


def _jsonable(x):
    if is_dataclass(x) and not isinstance(x, type):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


@dataclass
class MatchResult:
    points: list = field(default_factory=list)          # [{id, A, B, first_hit, last_hit, hits, winner, winner_confidence, ...}]
    score_timeline: list = field(default_factory=list)  # [{point, t, before:{sets,games,call,server}, after:{...}, verified}]
    uncertain_queue: list = field(default_factory=list)  # fix-screen items [{id, kind, point, t, confidence, prompt, options}]
    clips: list = field(default_factory=list)            # whole-point clips [{point, a, b, score}]
    reel_plan: dict = None                               # vamos.reel_plan.v1 or None
    players: list = field(default_factory=list)          # [{id: "A"|"B", team, side}] roles only, never faces
    meta: dict = field(default_factory=dict)             # engine, stubbed, sport, final, teams, video, work, api_version
    features_version: str = None                         # "vamos.feat.v1" (points[].features, uncertain_queue[].features)
    schema: str = SCHEMA

    def to_dict(self):
        return _jsonable(self)

    @classmethod
    def from_dict(cls, d):
        if d.get("schema") not in SCHEMAS_READ:
            raise ValueError(f"schema must be one of {SCHEMAS_READ}")
        d = copy.deepcopy(d)
        kw = {k: d.get(k) for k in ("points", "score_timeline", "uncertain_queue", "clips", "reel_plan", "players", "meta",
                                    "features_version")}
        return cls(**kw, schema=SCHEMA)

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class Correction:
    """One user fix. Address it by uncertain_queue item id, or by (kind, point)."""
    value: object
    item_id: str = None
    kind: str = None
    point: int = None
    source: str = "user"         # user | confirm | watch

    @classmethod
    def parse(cls, c):
        if isinstance(c, Correction):
            return c
        if not isinstance(c, dict):
            raise InvalidCorrection("correction must be an object")
        bad = [k for k in c if FORBIDDEN_KEY.search(str(k))]
        if bad:
            raise InvalidCorrection(f"forbidden field(s) {bad} (no faces / biometrics / names / raw media)")
        extra = set(c) - {"value", "item_id", "kind", "point", "source"}
        if extra:
            raise InvalidCorrection(f"unexpected field(s) {sorted(extra)}")
        if "value" not in c:
            raise InvalidCorrection("value is required")
        if c.get("source", "user") not in ("user", "confirm", "watch"):
            raise InvalidCorrection("source must be user | confirm | watch")
        return cls(**c)


def valid_value(kind, v):
    if kind == "point_winner":
        return v in ("A", "B", "skip")
    if kind == "bounce":
        return v == "none" or (isinstance(v, dict) and set(v) == {"x", "y"} and
                               all(isinstance(v[a], (int, float)) and not isinstance(v[a], bool) and -0.25 <= v[a] <= 1.25 for a in "xy"))
    if kind == "player_id":
        return isinstance(v, dict) and v and set(v) <= {"near", "far"} and all(x in ("A", "B") for x in v.values())
    if kind == "cut_off":
        return isinstance(v, bool)
    if kind == "score_check":
        return v in ("ok", "wrong")
    return False
