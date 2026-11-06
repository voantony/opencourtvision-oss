"""Synthetic doubles ground truth + imperfect predictions for testing the eval (no real matches)."""
import random

from ..events import Event
from .schema import CourtPosition, Direction, Formation, Outcome, ShotLabel, allowed_directions

PLAYERS = {"P1": ("near", "right"), "P2": ("near", "left"), "P3": ("far", "right"), "P4": ("far", "right")}


def ground_truth(n_rallies=12, seed=0):
    """Rule-consistent rallies: contact events with full shot labels, plus a bounce after each contact."""
    rng = random.Random(seed); ev = []; t = 5.0
    for r in range(n_rallies):
        server = rng.choice(list(PLAYERS)); n = rng.randint(1, 8); end, _ = PLAYERS[server]
        half = rng.choice(["deuce", "ad"]); cur = server
        for i in range(n):
            e_end, hand = PLAYERS[cur]
            if i == 0:
                lab = ShotLabel(CourtPosition(f"{e_end}_{half}"), None, "serve", rng.choice(["T", "body", "wide"]),
                                rng.choice(["conventional", "i_formation", "australian"]), None, hand)
            else:
                pos = CourtPosition(f"{e_end}_{rng.choice(['deuce', 'ad'])}"); side = rng.choice(["forehand", "backhand"])
                st = "return" if i == 1 else rng.choice(["swing", "swing", "volley", "lob", "smash"])
                d = rng.choice(sorted(allowed_directions(pos, side, hand), key=str))
                lab = ShotLabel(pos, side, st, d, Formation.NON_SERVE, None, hand)
            lab.outcome = Outcome.IN if i < n - 1 else rng.choice([Outcome.WIN, Outcome.ERROR])
            ev.append(Event("contact", round(t, 3), cur, lab, 1.0, ["truth: synthetic"], True))
            ev.append(Event("bounce", round(t + rng.uniform(0.5, 0.9), 3), None, None, 1.0, ["truth: synthetic"], True))
            t += rng.uniform(1.0, 1.6)
            cur = rng.choice([p for p, (e, _) in PLAYERS.items() if e != e_end])
        ev.append(Event("point_end", round(t, 3), None, None, 1.0, ["truth: synthetic"], True))
        t += rng.uniform(15, 25)
    return ev


def predictions(gt, seed=1, miss=0.1, spurious=0.05, side_flip=0.15, late=0.05):
    """Corrupt ground truth like a real detector: misses, spurious events, timing jitter (some beyond tolerance),
    forehand/backhand flips, and confidences that are lower for corrupted events."""
    rng = random.Random(seed); out = []
    for g in gt:
        if rng.random() < miss: continue
        lab = ShotLabel.from_dict(g.label.to_dict()) if g.label else None; conf = rng.uniform(0.7, 0.98); bad = False
        jitter = rng.gauss(0, 0.02)
        if rng.random() < late: jitter = 0.2; bad = True
        if lab and lab.side and rng.random() < side_flip:
            lab.side = "backhand" if str(lab.side) == "forehand" else "forehand"; bad = True
        if lab and lab.direction and rng.random() < 0.3:
            lab.direction = rng.choice([d for d in Direction if (d in {Direction.T, Direction.BODY, Direction.WIDE}) == lab.is_serve]); bad = True
        if bad: conf = rng.uniform(0.3, 0.7)
        out.append(Event(g.type, round(g.t + jitter, 3), g.player, lab, round(conf, 3), ["audio: synthetic", "pose: synthetic"]))
    for g in rng.sample(gt, int(len(gt) * spurious)):
        out.append(Event(g.type, round(g.t + rng.uniform(0.4, 0.9), 3), g.player, None, round(rng.uniform(0.2, 0.6), 3), ["audio: synthetic"]))
    return sorted(out, key=lambda e: e.t)
