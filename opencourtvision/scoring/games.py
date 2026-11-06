"""Set/game structure from the final score string (e.g. "6-4 6-3")."""


def parse_final(s):
    return [tuple(map(int, x.split("-"))) for x in s.split()]


def changeover_games(sets):
    """Overall game numbers after which players change ends (odd games of each set; set break counted)."""
    co, g = [], 0
    for a, b in sets:
        for k in range(1, a + b + 1):
            g += 1
            if k % 2 == 1:
                co.append(g)
    return co


def assign_changeovers(segs, sets, hms):
    """Pick the longest gaps between point segments as changeovers (one per expected changeover)."""
    co = changeover_games(sets)
    ngames = sum(a + b for a, b in sets)
    order = sorted(range(len(segs) - 1), key=lambda i: -segs[i]["gap_after"])
    co_after = sorted(order[:len(co)]) if len(segs) > len(co) else []
    return dict(n_games=ngames, changeover_after_games=co,
                changeover_gaps=[(i + 1, round(segs[i]["gap_after"]), hms(segs[i]["B"])) for i in co_after])
