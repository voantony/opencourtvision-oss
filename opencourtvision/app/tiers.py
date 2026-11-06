"""Free core, paid extras (Antony, 19 Oct). Free = everything that runs on the device (near-zero marginal cost to us):
recording, auto-edit, scoring, basic stats, the fix screen and the personal on-device model; no account needed to try it.
Paid = anything that costs us servers or is a pro workflow. v1 has NO payment integration: pro features are shown as
clearly marked placeholders and `require` refuses them with a reason the UI shows."""
FREE = {
    "record": "Record a match (phone / camera)",
    "import": "Import a match video",
    "auto_edit": "Auto point cutting + dead-time trimming",
    "scoring": "Automatic score + in-order score check-ins",
    "basic_stats": "Basic stats (points, games, rally length)",
    "fix_screen": "Fix screen (confirm / correct the least-certain calls)",
    "personal_model": "Personal on-device model (retrains overnight while charging)",
    "highlights": "Highlight reels 9:16 + 16:9 (Vamos Clean style)",
    "watch_scoring": "Apple Watch score taps",
    "share_fixes": "Opt-in: share my fixes to improve the shared model",
}
PRO = {
    "cloud_storage": "Cloud storage + multi-device library",
    "advanced_analytics": "Advanced analytics (shot maps, serve +1, pressure points)",
    "coach_tools": "Coach tools (drawing, notes, shared sessions)",
    "club_licence": "Club licences (multi-court, members)",
    "premium_styles": "Premium highlight styles",
}
NEEDS_ACCOUNT = {"cloud_storage", "coach_tools", "club_licence"}


class ProFeature(PermissionError):
    pass


def tier(feature):
    if feature in FREE: return "free"
    if feature in PRO: return "pro"
    raise KeyError(feature)


def require(feature, entitlements=()):
    """Free features always pass, with no sign-in. Pro features pass only with an entitlement (none exist in v1)."""
    if tier(feature) == "free" or feature in set(entitlements):
        return True
    raise ProFeature(f"'{feature}' is a Vamos Pro feature (placeholder in v1; no payments wired)")


def catalogue():
    return dict(free=[dict(id=k, title=v, account=False) for k, v in FREE.items()],
                pro=[dict(id=k, title=v, account=k in NEEDS_ACCOUNT, placeholder=True) for k, v in PRO.items()],
                payments="none (v1 placeholder)", sign_in_required_for_free=False)
