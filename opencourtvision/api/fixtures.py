"""Deterministic synthetic match fixture for the stub engine. No footage, no labels, no people: numbers only.

Shapes match the real engine's session points (opencourtvision.app.analyze) so a client built on the stub works unchanged
on the real engine.
"""
import random


def fixture_points(n_points=24, seed=7, start_s=30.0):
    rng = random.Random(seed)
    t, out = float(start_s), []
    for i in range(1, n_points + 1):
        rally = rng.uniform(4.0, 22.0)
        first = t + rng.uniform(2.0, 4.0)
        last = first + rally
        conf = round(rng.uniform(0.55, 0.99), 3)
        winner = "A" if rng.random() < 0.55 else "B"
        out.append(dict(id=i, kind="point", A=round(first - 3.5, 3), B=round(last + 3.0, 3),
                        first_hit=round(first, 3), last_hit=round(last, 3), hits=max(1, int(rally / 1.6)),
                        react=round(rng.uniform(0.0, 9.0), 2), favorite=rng.random() < 0.12,
                        winner=winner, winner_src="engine", winner_confidence=conf,
                        bounce=None, bounce_confidence=round(rng.uniform(0.5, 0.95), 3),
                        player_id=None, player_id_confidence=round(rng.uniform(0.7, 0.99), 3),
                        sources=["audio"]))
        t = last + rng.uniform(15.0, 35.0)
    return out


def fixture_quality():
    return dict(struggle_tags=[], note="synthetic fixture")
