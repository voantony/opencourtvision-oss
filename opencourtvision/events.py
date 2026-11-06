"""Event record shared by every detector, and the verify gate.

Every detector (audio hits, pose strokes, ball bounces, callouts, ...) emits Events. Evidence items are strings
prefixed by their source, e.g. "audio: hit 0.42 amp" or "pose: wrist +1.3 torso right of hips". The verify gate marks
an event verified only when it is confident enough, backed by enough independent sources and its label passes the
schema rules. Captions and other user-facing claims are produced ONLY for verified events.
"""
import json
from dataclasses import dataclass, field

from .labels.schema import ShotLabel, validate

TIGHT_TYPES = frozenset({"contact", "bounce"})          # time-critical events (eval tolerance ±80 ms)
REFERENCE_SOURCES = frozenset({"swingvision"})          # imported third-party data: compared against, never verified


@dataclass
class Event:
    type: str                                   # contact, bounce, stroke, point_start, point_end, callout, ...
    t: float                                    # seconds in the source video
    player: str | None = None                   # e.g. "near", "near_left", "P1"
    label: ShotLabel | None = None
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    verified: bool = False
    source: str | None = None                   # who produced it: None (our detectors), "swingvision" (imported reference), ...
    attrs: dict = field(default_factory=dict)   # raw source fields kept for audits (speeds, coordinates, ids)

    def __post_init__(self):
        if isinstance(self.label, dict): self.label = ShotLabel.from_dict(self.label)
        self.t = float(self.t); self.confidence = float(self.confidence)

    @property
    def sources(self):
        return {e.split(":", 1)[0].strip() for e in self.evidence if ":" in e}

    def to_dict(self):
        d = dict(type=self.type, t=round(self.t, 4), player=self.player, confidence=round(self.confidence, 3),
                 evidence=list(self.evidence), verified=self.verified)
        if self.label is not None: d["label"] = self.label.to_dict()
        if self.source is not None: d["source"] = self.source
        if self.attrs: d["attrs"] = dict(self.attrs)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("type", "t", "player", "label", "confidence", "evidence", "verified", "source", "attrs") if k in d})


def verify(events, min_confidence=0.6, min_sources=2, hand=None):
    """Set .verified on each event in place and return the list. Reasons for rejection are appended to evidence
    as "verify: ..." so they show up in reports."""
    for e in events:
        why = []
        if e.source in REFERENCE_SOURCES: why.append(f"reference import ({e.source}), not ground truth")
        if e.confidence < min_confidence: why.append(f"confidence {e.confidence:.2f} < {min_confidence}")
        if len(e.sources) < min_sources: why.append(f"{len(e.sources)} evidence source(s) < {min_sources}")
        if e.label is not None:
            why += validate(e.label, hand)
        e.verified = not why
        e.evidence = [x for x in e.evidence if not x.startswith("verify:")] + [f"verify: {w}" for w in why]
    return events


def caption(e):
    """Short on-screen caption for a verified event, else None. Never captions unverified events."""
    if not e.verified: return None
    lab = e.label
    if lab is None: return None
    if lab.is_serve:
        return " ".join(x for x in ("Second serve" if str(lab.shot_type) == "second_serve" else "Serve",
                                    f"{lab.direction}".replace("_", " ") if lab.direction else "") if x)
    kind = str(lab.shot_type).replace("_", " ") if lab.shot_type and str(lab.shot_type) != "swing" else None
    parts = [str(lab.side).capitalize() if lab.side else None, kind, str(lab.direction).replace("_", " ") if lab.direction else None]
    return " ".join(p for p in parts if p) or None


def captions(events):
    return [(e.t, c) for e in events for c in [caption(e)] if c]


def save(path, events, **meta):
    json.dump(dict(meta, events=[e.to_dict() for e in events]), open(path, "w"), indent=1)


def load(path):
    d = json.load(open(path)); return [Event.from_dict(x) for x in d.get("events", [])], {k: v for k, v in d.items() if k != "events"}
