"""Scoring formats per sport (`SportProfile.scoring` in the MVP edit-style doc), checked against the 2026 rulebooks.

Padel: FIP Rules of Padel, review of application 01.01.2026, Rule 1 (three deuce options)
  https://www.padelfip.com/wp-content/uploads/2025/12/FIP_Rules-of-Padel-1.pdf
  - "advantage"    Option 1: classic deuce / advantage.
  - "star_point"   Option 2: deuce 1 -> advantage 1 -> deuce 2 -> advantage 2 -> deuce 3 = Star Point (one deciding point).
                   From 2026 on Premier Padel, CUPRA FIP Tour, FIP Promises and FIP Beyond
                   (https://www.padelfip.com/wp-content/uploads/2025/12/18-december_Star-Point_ENG.pdf); USPA uses it for
                   all sanctioned play, replacing the golden point (padelusa.org 2026 Competition Structure Guide).
  - "golden_point" Option 3: no advantage; at 40-40 one deciding point.
  For every deciding point (golden or star) the RECEIVING pair chooses to receive on the right or left; they can't switch
  positions for it; in mixed the receiver is the same sex as the server.
  Default for club footage: "golden_point" (Media Director decision, 18 Oct 2026). Per match: an explicit setting wins,
  else the rule is auto-detected from spoken calls ("punto de oro" -> golden point; "star point" / "punto estrella" or a
  numbered deuce >= 2 / numbered advantage -> Star Point), else the default. Pro-tour matches: preset "fip_star_point".

Pickleball: USA Pickleball 2026 Official Rulebook (https://www.usapickleball.org/docs/rules/USAP-Official-Rulebook.pdf)
  and UPA-A Rulebook v0.9, 21 Aug 2026 (PPA/MLP; https://upaa.unitedpickleball.com/official-rulebook/).
  - Side-out (standard): only the serving side scores (USAP 4.A; UPA-A 5.6). Doubles: both partners serve before a side
    out, except at the start of each game only the starting server serves (USAP 5.B, 5.B.2; UPA-A 5.5.2), so the
    opening call is "0-0-2" (USAP 6.B.2; UPA-A 5.2.1). Games to 11 win by 2 (USAP 15.C, 15.C.1; UPA-A TO 2.2); 15 or 21 allowed;
    win by 1 is allowed for USAP team play and UPA-A recommends round robins to 11 win by 1 (sanctioning page).
  - Rally (USAP 14.A, provisional for 2026): a point on every rally (14.A.2), including when the RECEIVING side is at game
    point (the 2025 "game point only when serving" freeze was removed for 2026); two-number call (14.A.3); one server per
    side out in doubles (14.A.5). Tournament options 11/15/21 (15.C.2), not for double-elimination doubles, Golden Ticket or
    Nationals events. USAP decides in 2027 whether to keep it.
  - MLP 2026 (MLPlay Rules Guide, Appendix C of UPA-A): gender and mixed doubles games are SIDE-OUT to 11 win by 2; the
    DreamBreaker is rally scoring to 21 win by 2, singles rotations of 4 rallies, and a team must win WHILE SERVING (freeze kept).
"""
from dataclasses import asdict, dataclass, field, replace

PADEL_DEUCE_RULES = ("advantage", "golden_point", "star_point")


@dataclass(frozen=True)
class ScoringFormat:
    sport: str
    preset: str
    system: str                      # "games_sets" (tennis/padel) | "side_out" | "rally" (pickleball)
    deuce_rule: str = None           # tennis/padel: advantage | golden_point | star_point (tennis no-ad = golden_point)
    deciding_point_chooser: str = None   # who picks the receiving side on a deciding point ("receiver")
    games_per_set: int = 6
    tiebreak_at: int = 6
    tiebreak_to: int = 7
    final_set: str = "set"           # "set" | "match_tiebreak_10" | "advantage_set"
    sets_to_win: int = 2
    points_to_win: int = None        # pickleball game target
    win_by: int = 2
    games_to_win: int = None         # pickleball: best of 3 -> 2
    first_server_exception: bool = None   # pickleball doubles side-out: only the starting server serves first rotation
    game_point_only_when_serving: bool = None   # rally-scoring freeze (MLP DreamBreaker) or side-out by definition
    servers_per_side_out: int = None # pickleball doubles: 2 side-out, 1 rally
    score_call_numbers: int = None   # pickleball: 3 (side-out doubles), 2 (rally / singles)
    source: str = ""
    notes: tuple = field(default_factory=tuple)

    def to_dict(self):
        return asdict(self)


FIP_URL = "https://www.padelfip.com/wp-content/uploads/2025/12/FIP_Rules-of-Padel-1.pdf"
USAP_URL = "https://www.usapickleball.org/docs/rules/USAP-Official-Rulebook.pdf"
UPAA_URL = "https://upaa.unitedpickleball.com/wp-content/uploads/2026/08/V0.9-8.24.26-UPA-A-Rulebook-.pdf"
MLP_URL = "https://upaa.unitedpickleball.com/wp-content/uploads/2026/05/2026-MLPlay-Rules-Guide-5.17.26.pdf"

_PADEL = ScoringFormat("padel", "fip_star_point", "games_sets", deuce_rule="star_point", deciding_point_chooser="receiver",
                       source=FIP_URL + " Rule 1 Option 2",
                       notes=("Premier Padel / CUPRA FIP Tour / FIP Promises / FIP Beyond from 2026",))
_PB_SIDEOUT = ScoringFormat("pickleball", "usap_side_out_doubles", "side_out", points_to_win=11, win_by=2, games_to_win=2,
                            first_server_exception=True, game_point_only_when_serving=True, servers_per_side_out=2,
                            score_call_numbers=3, source=USAP_URL + " 4.A, 5.B, 6.B.2, 15.C.1; UPA-A 5.2.1, 5.5.2, 5.6")

PRESETS = {
    "tennis": {
        "advantage": ScoringFormat("tennis", "advantage", "games_sets", deuce_rule="advantage"),
        "no_ad": ScoringFormat("tennis", "no_ad", "games_sets", deuce_rule="golden_point", deciding_point_chooser="receiver"),
        "no_ad_match_tb": ScoringFormat("tennis", "no_ad_match_tb", "games_sets", deuce_rule="golden_point",
                                        deciding_point_chooser="receiver", final_set="match_tiebreak_10"),
    },
    "padel": {
        "fip_star_point": _PADEL,
        "golden_point": replace(_PADEL, preset="golden_point", deuce_rule="golden_point", source=FIP_URL + " Rule 1 Option 3", notes=()),
        "advantage": replace(_PADEL, preset="advantage", deuce_rule="advantage", deciding_point_chooser=None,
                             source=FIP_URL + " Rule 1 Option 1", notes=()),
    },
    "pickleball": {
        "usap_side_out_doubles": _PB_SIDEOUT,
        "usap_side_out_15": replace(_PB_SIDEOUT, preset="usap_side_out_15", points_to_win=15, games_to_win=1),
        "usap_side_out_21": replace(_PB_SIDEOUT, preset="usap_side_out_21", points_to_win=21, games_to_win=1),
        "usap_rally_2026": ScoringFormat("pickleball", "usap_rally_2026", "rally", points_to_win=11, win_by=2, games_to_win=2,
                                         first_server_exception=False, game_point_only_when_serving=False,
                                         servers_per_side_out=1, score_call_numbers=2,
                                         source=USAP_URL + " 14.A (provisional), 15.C.2",
                                         notes=("11, 15 or 21 allowed; receiver can win on game point in 2026",)),
        "mlp_doubles_2026": replace(_PB_SIDEOUT, preset="mlp_doubles_2026", games_to_win=None, source=MLP_URL,
                                    notes=("MLP gender/mixed doubles games: side-out to 11 win by 2, ends change at 6",)),
        "mlp_dreambreaker_2026": ScoringFormat("pickleball", "mlp_dreambreaker_2026", "rally", points_to_win=21, win_by=2,
                                               games_to_win=1, first_server_exception=False, game_point_only_when_serving=True,
                                               servers_per_side_out=1, score_call_numbers=2, source=MLP_URL,
                                               notes=("singles rotation every 4 rallies; must win while serving",)),
    },
}
DEFAULT_PRESET = {"tennis": "advantage", "padel": "golden_point", "pickleball": "usap_side_out_doubles"}
PADEL_RULE_PRESET = {"golden_point": "golden_point", "star_point": "fip_star_point", "advantage": "advantage"}


def get_format(sport, preset=None):
    sport = sport.lower()
    if sport not in PRESETS:
        raise ValueError(f"unknown sport {sport!r}")
    p = preset or DEFAULT_PRESET[sport]
    if p not in PRESETS[sport]:
        raise ValueError(f"unknown {sport} preset {p!r}; choose from {', '.join(PRESETS[sport])}")
    return PRESETS[sport][p]


TIEBREAK_OPTIONS = ("standard", "super_tiebreak", "none")


def with_options(fmt, deuce_rule=None, tiebreak=None):
    """Per-match overrides on a games_sets format (the app's match setting, read by the labeller and by both scorers).
    deuce_rule: advantage | golden_point | star_point. tiebreak: "standard" (7-point tiebreak at 6-6 in every set),
    "super_tiebreak" (standard tiebreaks, and a 10-point match tiebreak instead of the deciding set) or "none" (no
    tiebreaks: every set, the deciding one included, is won by two games)."""
    if fmt.system != "games_sets":
        if deuce_rule or tiebreak:
            raise ValueError(f"deuce/tiebreak options apply to tennis and padel, not {fmt.sport} ({fmt.system})")
        return fmt
    kw = {}
    if deuce_rule:
        if deuce_rule not in PADEL_DEUCE_RULES:
            raise ValueError(f"unknown deuce rule {deuce_rule!r}; choose from {', '.join(PADEL_DEUCE_RULES)}")
        kw.update(deuce_rule=deuce_rule, deciding_point_chooser=None if deuce_rule == "advantage" else "receiver")
    if tiebreak:
        if tiebreak not in TIEBREAK_OPTIONS:
            raise ValueError(f"unknown tiebreak option {tiebreak!r}; choose from {', '.join(TIEBREAK_OPTIONS)}")
        kw.update({"standard": dict(tiebreak_at=fmt.games_per_set, tiebreak_to=7, final_set="set"),
                   "super_tiebreak": dict(tiebreak_at=fmt.games_per_set, tiebreak_to=7, final_set="match_tiebreak_10"),
                   "none": dict(tiebreak_at=None, final_set="advantage_set")}[tiebreak])
    if not kw:
        return fmt
    out = replace(fmt, **kw)
    return replace(out, preset=f"{fmt.preset}+{format_options(out)['deuce_rule']}/{format_options(out)['tiebreak']}")


def format_options(fmt):
    """The two per-match options of a games_sets format: {"deuce_rule", "tiebreak"} (None for pickleball)."""
    if fmt.system != "games_sets":
        return None
    tb = "none" if fmt.tiebreak_at is None else "super_tiebreak" if fmt.final_set == "match_tiebreak_10" else "standard"
    return {"deuce_rule": fmt.deuce_rule, "tiebreak": tb}


def format_from_settings(match):
    """ScoringFormat from a stored match-settings dict (labels file `match`, app match setting):
    {"sport", "preset", "format": {"deuce_rule", "tiebreak"}}. Older label files without "format" keep their preset;
    a tennis "no_ad" preset is golden point."""
    sport = (match.get("sport") or "tennis").lower()
    preset = match.get("preset")
    if preset and preset not in PRESETS.get(sport, {}):
        preset = {"golden_point": "no_ad" if sport == "tennis" else "golden_point", "no_ad": "golden_point",
                  "star_point": "fip_star_point"}.get(preset, preset)
        if preset not in PRESETS.get(sport, {}):
            preset = None
    opts = match.get("format") or {}
    return with_options(get_format(sport, preset), opts.get("deuce_rule"), opts.get("tiebreak"))


def match_format(sport, preset=None, callouts=None, deuce_rule=None, tiebreak=None):
    """Per-match scoring format. preset / explicit options (user setting) > padel rule detected from spoken calls >
    sport default. deuce_rule / tiebreak (see with_options) override the preset's. callouts: iterable of transcribed
    call texts (or dicts with "text"). -> (ScoringFormat, how: 'setting'|'callouts'|'default')."""
    if preset or deuce_rule or tiebreak:
        return with_options(get_format(sport, preset), deuce_rule, tiebreak), "setting"
    if sport.lower() == "padel" and callouts is not None:
        from .callouts import detect_deuce_rule
        rule = detect_deuce_rule(list(callouts))
        if rule:
            return get_format("padel", PADEL_RULE_PRESET[rule]), "callouts"
    return get_format(sport), "default"


# ---------------------------------------------------------------- tennis / padel game logic

def _deciding_after(deuce_rule):
    """Number of deuces after which the next point decides the game (None = never)."""
    return {"golden_point": 1, "star_point": 3, "advantage": None}[deuce_rule]


def game_point_state(s, r, deuce_rule):
    """State of a regular game before the next point. s, r = points won by server / receiver (0, 1, 2, ...).
    -> dict(call, deciding, deuce_n, game_point_for, break_point) where call is server-first ("30-15", "deuce 2", "ad in"...)."""
    names = ["0", "15", "30", "40"]
    dec = _deciding_after(deuce_rule)
    if s < 3 or r < 3:
        gp = "server" if s == 3 and r < 3 else "receiver" if r == 3 and s < 3 else None
        return dict(call=f"{names[min(s, 3)]}-{names[min(r, 3)]}", deciding=False, deuce_n=0, game_point_for=gp, break_point=gp == "receiver")
    # both >= 3: count deuces reached (3-3 = deuce 1, 4-4 = deuce 2, ...)
    lead = s - r
    deuce_n = min(s, r) - 2          # 3-3 = deuce 1, 4-3 = advantage 1, 4-4 = deuce 2, ...
    if dec is not None and deuce_n >= dec and lead == 0:
        return dict(call="deuce" if deuce_rule == "golden_point" else f"deuce {deuce_n}", deciding=True, deuce_n=deuce_n,
                    game_point_for="both", break_point=True)
    if lead == 0:
        return dict(call="deuce" if deuce_rule == "advantage" else f"deuce {deuce_n}", deciding=False, deuce_n=deuce_n,
                    game_point_for=None, break_point=False)
    who = "server" if lead > 0 else "receiver"
    return dict(call=f"ad {'in' if lead > 0 else 'out'}", deciding=False, deuce_n=deuce_n, game_point_for=who, break_point=who == "receiver")


def play_game(points, deuce_rule):
    """points: sequence of 'S'/'R' winners. -> dict(winner, states [before each point], points_played)."""
    s = r = 0; states = []
    for w in points:
        st = game_point_state(s, r, deuce_rule); states.append(st)
        s += w == "S"; r += w == "R"
        if st["deciding"] or (s >= 4 and s - r >= 2) or (r >= 4 and r - s >= 2):
            return dict(winner="S" if s > r else "R", states=states, points_played=len(states))
    return dict(winner=None, states=states, points_played=len(states))


def stakes_caption(state, sport, deuce_rule, verified=False):
    """Stakes caption for a padel/tennis game state (the set/match layer adds SET/MATCH POINT on top). Only from a verified
    score state (mvp_edit_style §4.3): unverified -> None. The deciding-point caption (PUNTO DE ORO / STAR POINT) appears
    only at the deciding point itself, never at an earlier Star Point deuce."""
    if not verified:
        return None
    if state["deciding"]:
        if sport == "padel":
            return "STAR POINT" if deuce_rule == "star_point" else "PUNTO DE ORO"
        return "DECIDING POINT"
    if state["break_point"]:
        return "BREAK POINT"
    return None


def bug_point_cell(state, deuce_rule, verified=False):
    """Score-bug point cell override at a verified deciding point: 'ORO' (golden point) or 'STAR' (Star Point), else None
    (the normal point score is shown, or hidden if unverified)."""
    if not (verified and state["deciding"]):
        return None
    return "STAR" if deuce_rule == "star_point" else "ORO"


# Coach O1 + O7 pressure factor (highlight_algorithm.md §3.3), used by the ranker in story/highlight.py (and story/ranking.py).
NESTED_M = dict(match_point=1.6, set_point=1.45, break_point=1.3, pickleball_game_point=1.3, tb_game_point=1.2)
DEUCE_M, M_CAP = 1.15, 1.75


def game_pressure(s, r, deuce_rule, winner=None):
    """Game-level pressure class before a point, from the points won by server s / receiver r.
    -> 'break_point' | 'deuce' | None. `winner` is accepted for API stability and ignored (Coach, 19 Oct): a deciding
    point gets the break-point factor whether the receiver or the server wins it.
    Coach (19 Oct):
      - Star Point / golden deciding point (sudden death at deuce 3 / 40-40) always gets break-point ×1.3, whoever is
        receiving. If it is also a set or match point, pressure_multiplier takes the max as usual.
      - Advantage 1 or 2 to the receivers stays an ordinary break point.
      - Deuce 1 / 2 and 30-30 stay 'deuce' (the ×1.15 factor)."""
    st = game_point_state(s, r, deuce_rule)
    if st["deciding"]:
        return "break_point"          # ×1.3 always; nested set/match point takes the max in pressure_multiplier
    if st["break_point"]:
        return "break_point"
    if (s == r == 2) or (s == r and s >= 3):
        return "deuce"
    return None


def pressure_multiplier(flags=None, game_state=None):
    """M = min(1.75, max(nested states present, default 1.0) × (1.15 if 30-30/deuce)). flags: dict of nested states
    (match_point, set_point, break_point, pickleball_game_point, tb_game_point) -> bool; game_state: game_pressure() output,
    which adds break_point or the deuce factor."""
    f = dict(flags or {})
    if game_state == "break_point":
        f["break_point"] = True
    nested = max([NESTED_M[k] for k, v in f.items() if v and k in NESTED_M] or [1.0])
    return min(M_CAP, nested * (DEUCE_M if game_state == "deuce" else 1.0))


# ---------------------------------------------------------------- pickleball

def play_pickleball_game(rallies, fmt, first="A"):
    """Doubles game. rallies: sequence of 'A'/'B' rally winners. -> dict(winner, calls [server-first call before each rally]).
    Side-out: only the serving team scores; two servers per side out except the first rotation of the game (starts '0-0-2').
    Rally: a point every rally; serve changes on a lost rally; game_point_only_when_serving freezes the receiver at game point."""
    score = {"A": 0, "B": 0}; serving = first
    server_no = 2 if fmt.first_server_exception else 1
    calls, other = [], {"A": "B", "B": "A"}
    target, by = fmt.points_to_win, fmt.win_by

    def won(t):
        return score[t] >= target and score[t] - score[other[t]] >= by

    for w in rallies:
        rcv = other[serving]
        calls.append(f"{score[serving]}-{score[rcv]}-{server_no}" if fmt.score_call_numbers == 3 else f"{score[serving]}-{score[rcv]}")
        if w == serving:
            score[w] += 1
        elif fmt.system == "side_out":
            if server_no == 1 and fmt.servers_per_side_out == 2:
                server_no = 2
            else:
                serving, server_no = rcv, 1
        else:   # rally scoring
            would_win = score[w] + 1 >= target and score[w] + 1 - score[serving] >= by
            if not (fmt.game_point_only_when_serving and would_win):
                score[w] += 1
            serving, server_no = w, 1
        for t in ("A", "B"):
            if won(t):
                return dict(winner=t, score=dict(score), calls=calls)
    return dict(winner=None, score=dict(score), calls=calls)


def pickleball_game_point(serving_score, receiving_score, fmt):
    """-> 'serving' | 'receiving' | 'both' | None: who can win the game on the next rally."""
    t, by = fmt.points_to_win, fmt.win_by
    s_gp = serving_score + 1 >= t and serving_score + 1 - receiving_score >= by
    r_gp = receiving_score + 1 >= t and receiving_score + 1 - serving_score >= by
    if fmt.game_point_only_when_serving:
        r_gp = False
    return "both" if s_gp and r_gp else "serving" if s_gp else "receiving" if r_gp else None
