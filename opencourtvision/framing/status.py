"""App-facing framing status: five checks, three levels, one plain-language fix at a time.

Antony, 21 Oct: "sometimes the framing won't be perfect, just make do". The app works with imperfect framing and does
not nag. Every check is one of:
    ok    nothing to say
    hint  quiet and dismissable; never blocks recording. Once the user dismisses a check's hint it is not shown again
          this session (the check still reports level "hint" with dismissed=True, so the engine/UI can log it)
    warn  firm; only when scoring would really break:
            steady    the phone moved mid-recording (before recording the session just follows the court)
            baselines both baselines missing (each under BASELINE_MISSING in the picture)
            players   one whole team out of frame in >= 80% of the last TEAM_WARN_CHECKS checks (~20 s) while the other is seen
Everything else is a hint at most: a partly cut-off corner or baseline (dev_match: the near-right corner, all match), a
cut sideline, mild or strong tilt, a low phone, unclear lines, a team missing for a few seconds. Both teams missing is
a break / changeover / warm-up: no message. Nothing here refuses to record or analyse; the engine gets the framing
quality signal (framing/quality.py) and widens its uncertainty instead.

Fix order (the one shown): any warn first, then hints, in CHECKS order. A dismissed hint is skipped.
"""

CHECKS = ("steady", "baselines", "lines", "level", "players")
LEVELS = ("ok", "hint", "warn")

_ACTION = dict(top="Tilt the phone up", bottom="Tilt the phone down", left="Turn the phone left", right="Turn the phone right")
_BACK = "Move the phone back or switch to 0.5x"


def _edge(cut, corners):
    """The edge to fix first for these cut corners: both opposite edges -> "back" (zoom out), else the largest overshoot."""
    over = {}
    for c in corners:
        for e, px in cut.get("over_px", {}).get(c, {}).items(): over[e] = max(over.get(e, 0.0), px)
    if not over: return "back"
    if {"left", "right"} <= set(over) or {"top", "bottom"} <= set(over): return "back"
    return max(over, key=over.get)


def _baselines_hint(cut):
    out = cut.get("corners", [])
    near = [c for c in out if c.startswith("near")]; far = [c for c in out if c.startswith("far")]
    if near and far:
        e = _edge(cut, out)
        return f"{_BACK} to fit both baselines." if e == "back" else f"{_ACTION[e]} to fit both baselines."
    which, cs = ("far baseline", far) if far else ("near baseline", near)
    e = _edge(cut, cs)
    end = f" if you can; the {which} is partly cut off."
    return (_BACK if e == "back" else _ACTION[e]) + end


def _set(c, name, level, fix):
    c[name] = dict(level=level, fix=fix, dismissed=False)


def summarise(v, live=False, p=None, recording=False, sustained=None, dismissed=()):
    """Verdict -> (checks, fix, overall level, fix_check). checks = {name: {level, fix, dismissed}}."""
    from .check import CHECK
    p = {**CHECK, **(p or {})}
    sustained = sustained or {}
    active = set(v.shown if live else v.warnings)
    info, cut = v.info or {}, v.cut or {}
    lines = info.get("lines", {})
    out_c = cut.get("corners", []) if "court_cut_off" in active else []
    c = {}
    for k in CHECKS: _set(c, k, "ok", None)

    if "camera_moved" in active:
        if recording:
            _set(c, "steady", "warn", "The phone moved. Put it back where it was, or tap Re-lock if the new spot is good.")
        else:
            _set(c, "steady", "hint", "The phone moved since setup. Tap Re-lock when it's where you want it.")

    nb, fb = (lines.get(k, {}).get("in_frame", 1.0) for k in ("near_baseline", "far_baseline"))
    if out_c and nb < p["BASELINE_MISSING"] and fb < p["BASELINE_MISSING"]:
        _set(c, "baselines", "warn", f"{_BACK}: neither baseline is in the picture, so points can't be scored.")
    elif out_c:
        _set(c, "baselines", "hint", _baselines_hint(cut))

    # a cut sideline always comes with a cut corner, which is already the baselines hint: one cause, one hint
    if "lines_unclear" in active:
        _set(c, "lines", "hint", "The court lines are hard to see. Wiping the lens or avoiding glare will help.")

    if "too_low_or_tilted" in active:
        if info.get("tilted"):
            _set(c, "level", "hint", f"The phone is tilted about {abs(info.get('roll_deg', 0)):.0f}°. Level it if you can.")
        else:
            _set(c, "level", "hint", "A higher spot would help with far-court calls.")

    far, near = "far_team_missing" in active, "near_team_missing" in active
    if "lines_unclear" in active or "camera_moved" in active or (far and near):
        far = near = False                   # soft / moved picture explains it; both missing = break or changeover
    if far or near:
        side = "far" if far else "near"
        if live and sustained.get(side):
            _set(c, "players", "warn", f"The {side} players have been out of the picture for a while, so points can't be scored. "
                 + ("Tilt the phone up or move it back." if side == "far" else "Tilt the phone down or move it back."))
        else:
            _set(c, "players", "hint", f"Can't see the {side} players right now. Fine between points; "
                 + ("check the far side is in the picture." if side == "far" else "check the near side is in the picture."))

    for k in CHECKS:
        if c[k]["level"] == "hint" and k in dismissed: c[k]["dismissed"] = True
    warns = [k for k in CHECKS if c[k]["level"] == "warn"]
    hints = [k for k in CHECKS if c[k]["level"] == "hint" and not c[k]["dismissed"]]
    first = (warns or hints or [None])[0]
    level = "warn" if warns else "hint" if hints else "ok"
    return c, (c[first]["fix"] if first else None), level, first
