"""Windows stage: restrict expensive vision work (pose, ball) to short windows around audio hits."""


def hit_times(rallies_or_times):
    """Accept rallies.json content (list of dicts with 'times'), a {'hits': [...]} dict, or a plain list of seconds."""
    x = rallies_or_times
    if isinstance(x, dict): x = x.get("hits", x.get("times", []))
    if x and isinstance(x[0], dict): return sorted(t for r in x for t in r.get("times", []))
    return sorted(float(t) for t in x)


def windows(times, pre=1.2, post=0.8, merge_gap=0.3, start=0.0, end=None):
    """Merged [a, b] intervals (s) covering each hit from t-pre to t+post."""
    out = []
    for t in sorted(times):
        a, b = max(start, t - pre), t + post if end is None else min(end, t + post)
        if out and a <= out[-1][1] + merge_gap: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [tuple(w) for w in out]


def frame_mask(n_frames, fps, wins):
    """Boolean list: which frames fall inside any window."""
    m = [False] * n_frames
    for a, b in wins:
        for i in range(max(0, int(a * fps)), min(n_frames, int(b * fps) + 1)): m[i] = True
    return m


def coverage(wins, duration):
    return sum(b - a for a, b in wins) / duration if duration else 0.0
