"""Build synthetic label bundles in memory. Nothing here is a recorded match.

`honest` draws p(A) toward the true winner, so a calibrated decoder should beat a constant guess.
`biased` reports p(A)=0.95 on every point, which is the miscalibration probe the issue drafts quote.
"""
from __future__ import annotations

import random

from opencourtvision.scoring.decoder import advance, initial_state
from opencourtvision.scoring.formats import get_format

FMT = get_format("tennis", "advantage")


def _play(seed: int, p_host: float, n_cap: int = 48):
    rng = random.Random(seed)
    st = initial_state(FMT, "A")
    winners, servers = [], []
    while not st.done and len(winners) < n_cap:
        servers.append(st.server)
        w = "A" if rng.random() < p_host else "B"
        winners.append(w)
        st = advance(st, w, FMT)
    return winners, servers


def bundle(name: str, seed: int, kind: str = "honest"):
    winners, servers = _play(seed, 0.56)
    rng = random.Random(seed + 3)
    points, p_a = [], []
    t = 30.0
    for i, (w, srv) in enumerate(zip(winners, servers), 1):
        dur = 4.0 + (i % 5)
        points.append({
            "idx": i,
            "start": round(t, 3),
            "end": round(t + dur, 3),
            "kind": "point",
            "winner": w,
            "server": srv,
        })
        if kind == "biased":
            p = 0.95
        else:
            p = 0.82 if w == "A" else 0.18
            p = min(0.97, max(0.03, p + rng.uniform(-0.04, 0.04)))
        p_a.append(round(p, 4))
        t += dur + 6.0
    labels = {
        "schema": "ocv.public_labels.v1",
        "video": {"name": f"{name}.synthetic", "licence": "synthetic", "attribution": "generated, no footage"},
        "match": {
            "sport": "tennis",
            "preset": "advantage",
            "first_server": "A",
            "mine": "A",
            "source_licence": "synthetic",
        },
        "points": points,
    }
    return {"id": name, "kind": kind, "labels": labels, "p_a": p_a}


BUNDLES = {
    "synth_tune": lambda: bundle("synth_tune", 11, "honest"),
    "synth_report": lambda: bundle("synth_report", 29, "honest"),
    "synth_biased": lambda: bundle("synth_biased", 29, "biased"),
}
