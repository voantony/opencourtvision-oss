"""Overlay lifetimes (shot card, minimap, tags, speed label, stakes caption): every overlay clears on a cut.

SwingVision takeaway (mvp_edit_style.md §2.1): an overlay that carries over a hard cut describes a shot the viewer can no
longer see. So an overlay lives inside one continuous shot of the OUTPUT timeline: it is clipped at the first cut after
it starts, gets a fade-out that ends exactly on the cut, and is dropped if what's left is too short to read. The score bug
is the one exception (it persists by design, §5) and is never passed through here.
"""
from bisect import bisect_right

PERSISTENT = {"score_bug"}
MIN_VISIBLE_S = 0.4          # shorter than this after clipping: drop rather than flash
FADE_S = 0.15


def clear_on_cuts(overlays, cuts, min_visible=MIN_VISIBLE_S, fade=FADE_S):
    """overlays: [{"kind", "start", "end", ...}] in output time; cuts: output times of every cut (clip boundaries).
    -> new list: each non-persistent overlay ends at or before the first cut after its start (never spans a cut), with
    fade_out_s <= fade; overlays starting exactly on a cut belong to the next shot. Persistent kinds pass through unchanged."""
    cuts = sorted(set(float(c) for c in cuts)); out = []
    for o in overlays:
        o = dict(o)
        if o.get("kind") in PERSISTENT:
            out.append(o); continue
        i = bisect_right(cuts, o["start"] + 1e-9)
        nxt = cuts[i] if i < len(cuts) else None
        if nxt is not None and o["end"] > nxt:
            o["end"] = nxt; o["clipped_at_cut"] = True
        if o["end"] - o["start"] < min_visible - 1e-9:
            continue
        o["fade_out_s"] = min(fade, o.get("fade_out_s", fade), (o["end"] - o["start"]) / 2)
        out.append(o)
    return out


def spans_cut(overlay, cuts):
    """True when the overlay is visible across a cut (start < cut < end)."""
    return any(overlay["start"] + 1e-9 < c < overlay["end"] - 1e-9 for c in cuts)


def cuts_from_clips(clips):
    """Output-time cut positions for clips laid end to end: [{"start","end"}] source times -> cumulative boundaries."""
    t, cuts = 0.0, []
    for c in clips[:-1]:
        t += c["end"] - c["start"]; cuts.append(round(t, 6))
    return cuts
