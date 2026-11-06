"""Load and validate a style preset (Media Director's vamos.style_preset.v1 schema, e.g. presets/vamos_clean_v1).

A preset is style only. It can make the engine more restrained, never less: the HARD limits below are engine law and a
preset asking for more (3 slow-mos, 0.3x, letterboxed 9:16, a flash transition, baked music) is rejected."""
import json
import os

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
SCHEMA = "vamos.style_preset.v1"
HARD = dict(slow_mo_max_per_reel=2, slow_mo_min_factor=0.45, default_transition="hard_cut",
            forbidden_transitions=("whip", "flash", "zoom_blur", "glitch", "wipe", "shake"), pre_roll_s_min=3.0, post_roll_s_min=2.0)
REQUIRED = ("id", "palette", "typography", "grade", "score_bug", "captions", "transitions", "slow_mo", "audio", "safe_zones")


class PresetError(ValueError):
    pass


def preset_path(name_or_path):
    if os.path.isfile(name_or_path): return os.path.abspath(name_or_path)
    if os.path.isdir(name_or_path): return os.path.join(os.path.abspath(name_or_path), "preset.json")
    p = os.path.join(PRESET_DIR, name_or_path, "preset.json")
    if not os.path.exists(p):
        raise PresetError(f"unknown preset {name_or_path!r} (have: {', '.join(list_presets())})")
    return p


def list_presets():
    return sorted(d for d in os.listdir(PRESET_DIR) if os.path.exists(os.path.join(PRESET_DIR, d, "preset.json")))


def validate(p):
    """Raise PresetError on a missing block or anything that breaks a hard rule. Returns p."""
    miss = [k for k in REQUIRED if k not in p]
    if miss: raise PresetError(f"preset missing {miss}")
    sm = p["slow_mo"]
    if int(sm.get("max_per_reel", 0)) > HARD["slow_mo_max_per_reel"] or not 0 <= int(sm.get("max_per_reel", 0)):
        raise PresetError(f"slow_mo.max_per_reel must be 0-{HARD['slow_mo_max_per_reel']}")
    if float(sm.get("max_factor", 1.0)) < HARD["slow_mo_min_factor"] - 1e-9:
        raise PresetError(f"slow_mo.max_factor {sm.get('max_factor')} slower than {HARD['slow_mo_min_factor']}x")
    tr = p["transitions"]
    if tr.get("default") != HARD["default_transition"]:
        raise PresetError("transitions.default must be hard_cut")
    hc = p.get("hard_constraints", {})
    if hc.get("nine_sixteen_letterbox_forbidden") is False or hc.get("nine_sixteen_wide_mode_forbidden") is False:
        raise PresetError("9:16 letterbox / wide mode can never be enabled")
    if hc.get("reel_clips_whole_points_only") is False or hc.get("caption_requires_whole_event") is False:
        raise PresetError("whole points and whole-event captions can never be switched off")
    if float(hc.get("pre_roll_s_min", 3.0)) < HARD["pre_roll_s_min"] or float(hc.get("post_roll_s_min", 2.0)) < HARD["post_roll_s_min"]:
        raise PresetError("pre/post-roll below the whole-point floor")
    if p["audio"].get("bake_music"):
        raise PresetError("music is never baked into exports (platform sound picker at share)")
    if p["captions"].get("hype_forbidden") is False:
        raise PresetError("hype captions can never be enabled")
    return p


def load(name_or_path="vamos_clean_v1", grade=None):
    """-> preset dict with `_dir` (asset base dir). grade=False forces the grade off (preset grade.enabled flag)."""
    path = preset_path(name_or_path)
    p = json.load(open(path, encoding="utf-8"))
    validate(p)
    p["_dir"] = os.path.dirname(path)
    if grade is not None: p["grade"] = dict(p["grade"], enabled=bool(grade))
    p["grade"].setdefault("enabled", True)
    return p


def asset(p, rel):
    return rel if os.path.isabs(rel) else os.path.join(p["_dir"], rel)


def font(p, role):
    f = p.get("assets", {}).get("fonts", {}).get(role)
    return asset(p, f) if f and os.path.exists(asset(p, f)) else None


def hex_rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ff_color(p, token, alpha=1.0):
    """ffmpeg colour string 0xRRGGBB@a for a palette token."""
    return f"0x{p['palette'][token].lstrip('#').upper()}@{alpha:.2f}"


def grade_filter(p):
    """ffmpeg grade filter (after reframe, before overlays) or None when the grade is off."""
    g = p["grade"]
    if not g.get("enabled", True): return None
    lut = g.get("lut_file")
    if lut and os.path.exists(asset(p, lut)):
        path = asset(p, lut).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        return f"lut3d=file='{path}':interp={g.get('lut_interp', 'tetrahedral')}"
    return g.get("ffmpeg_approx")
