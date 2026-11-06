"""Per-fix engine feature vector (`vamos.feat.v1`, 16 floats) attached to MatchResult so web corrections can be posted
to the PR #21 collector (`POST /v1/fixes`, `vamos.share.v2`) directly. Must stay index-identical to
`opencourtvision/shared/features.py` on feature/shared-training. Engine numbers only: no pixels, faces, names.
Unknown values are 0.0 (same convention as the collector's `from_fix` fallback)."""
import hashlib
import math

FEATURES_VERSION = "vamos.feat.v1"
FEATURE_DIM = 16
TAGS = ("low_light", "shaky", "behind_fence", "portrait_phone", "narrow_fov", "low_resolution")
FEATURE_NAMES = ("rally_len", "last_contact_x", "last_contact_y", "ball_speed", "audio_reaction", "serve_side",
                 "server_team", "far_motion_share", "near_motion_share", "confidence",
                 *(f"tag_{t}" for t in TAGS))
MAX_ABS = 10.0
STUB_INDICES = (1, 2, 3, 7, 8)     # filled deterministically by the stub engine; real engine fills them only if measured


def _clip(x):
    x = float(x)
    return 0.0 if not math.isfinite(x) else max(-MAX_ABS, min(MAX_ABS, round(x, 4)))


def _stub_unit(seed, point, idx):
    h = hashlib.sha256(f"{seed}:{point}:{idx}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def point_features(p, before=None, tags=(), confidence=None, stub_seed=None):
    """-> (16 floats, sources list). `before` is the score_timeline 'before' state of the point (server, pts, tiebreak)."""
    v, src = [0.0] * FEATURE_DIM, ["none"] * FEATURE_DIM

    def put(i, x, s):
        v[i], src[i] = _clip(x), s
    if p.get("hits") is not None: put(0, p["hits"] / 20.0, "engine")
    lc = p.get("last_contact")
    if isinstance(lc, dict) and {"x", "y"} <= set(lc): put(1, lc["x"], "engine"); put(2, lc["y"], "engine")
    if p.get("ball_speed") is not None: put(3, p["ball_speed"] / 50.0, "engine")
    if p.get("react") is not None: put(4, p["react"] / 30.0, "engine")
    if before:
        put(5, 1.0 if sum(before.get("pts") or [0, 0]) % 2 == 0 else -1.0, "score_rules")
        if before.get("server") in ("A", "B"): put(6, 1.0 if before["server"] == "A" else -1.0, "score_rules")
    m = p.get("motion_share")
    if isinstance(m, dict):
        if m.get("far") is not None: put(7, m["far"], "engine")
        if m.get("near") is not None: put(8, m["near"], "engine")
    put(9, p.get("winner_confidence", 0.5) if confidence is None else confidence, "engine")
    tags = set(tags or ())
    for j, t in enumerate(TAGS):
        put(10 + j, 1.0 if t in tags else 0.0, "quality")
    if stub_seed is not None:
        for i in STUB_INDICES:
            if src[i] == "none":
                u = _stub_unit(stub_seed, p.get("id"), i)
                put(i, u * 2.0 if i == 3 else u, "stub")
    return v, src


def valid(vec):
    return (isinstance(vec, (list, tuple)) and len(vec) == FEATURE_DIM and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and abs(x) <= MAX_ABS for x in vec))


def fix_example(item, value, client_id, source="user", struggle_tags=None):
    """uncertain_queue item + user's answer -> one `vamos.share.v2` example for POST /v1/fixes."""
    return dict(client_id=client_id, kind=item["kind"], value=value, features=list(item["features"]),
                struggle_tags=list(struggle_tags if struggle_tags is not None else item.get("struggle_tags") or []), source=source)
