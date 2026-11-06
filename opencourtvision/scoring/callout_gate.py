"""Callout source gate (design §3.6): accept a heard score call as decoder evidence only if it is OURS.

A callout heard by the smart score listener becomes an `Obs` on the score state only when
  (a) timing: it falls inside our court's between-point window (ball dead), not during one of our live rallies;
  (b) source: it is attributed to one of our 4 players. Cues are combined as independent log-likelihood ratios
      (ours vs other court): in-match speaker clusters (ECAPA-style embeddings), mouth / head pose of a visible player,
      loudness and direct-to-reverberant ratio (distance), and stereo direction where the phone has 2+ mics;
  (c) consistency: the parsed call is a legal score for our match (soft: the decoder weighs it against the visual winner
      and the server/side parity; a call our best path rejects is FLAGGED, never silently fixed or silently dropped);
  (d) convention: calls are server-first, so a call by the receiving team is weakened and flagged.
Every rejection is returned with its reasons so it can be logged and shown in review. A legal call from an uncertain
source is weak evidence: its p_src goes into the Obs mixture (decoder._obs_ll).
"""
import math
from dataclasses import dataclass, field

from .rules import Obs, score_call


@dataclass(frozen=True)
class CalloutCandidate:
    t: float                         # seconds, centre of the speech segment (VAD)
    call: str                        # canonical call (decoder.canonical_call) or None if unparsable
    asr_conf: float = 0.9            # P(the transcript is the call actually said): constrained-decode posterior, fused with KWS
    speaker_llr: float = None        # log P(emb | our 4 in-match clusters) - log P(emb | background / other voices)
    visual_llr: float = None         # mouth or head pose of one of our players moving at t (None if nobody visible)
    distance_llr: float = None       # loudness + direct-to-reverberant ratio vs our players' typical values
    direction_llr: float = None      # stereo / multi-mic direction of arrival vs our court's bearing
    speaker_team: str = None         # 'A' / 'B' if the speaker cluster is tied to a team, else None


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    p_src: float
    reasons: tuple = field(default_factory=tuple)
    obs: Obs = None


def _sig(x):
    return 1 / (1 + math.exp(-x))


def source_posterior(c, prior_ours=0.6):
    """P(ours) from the independent source cues (missing cues contribute nothing)."""
    llr = math.log(prior_ours / (1 - prior_ours))
    for x in (c.speaker_llr, c.visual_llr, c.distance_llr, c.direction_llr):
        if x is not None:
            llr += x
    return _sig(llr)


def gate_callout(c, window, live=(), server_team=None, legal_calls=None, n_values=20, prior_ours=0.6, min_src=0.2,
                 receiver_penalty=0.5):
    """window: (t0, t1) of our between-point window; live: our live-rally intervals; server_team: our current server
    ('A'/'B', from the decoder's running state) or None; legal_calls: calls reachable from our plausible states, or None."""
    reasons = []
    if c.call is None:
        return GateResult(False, 0.0, ("unparsable",))
    if any(a <= c.t <= b for a, b in live):
        return GateResult(False, 0.0, ("during_our_live_rally",))
    if not (window[0] <= c.t <= window[1]):
        return GateResult(False, 0.0, ("outside_between_point_window",))
    p = source_posterior(c, prior_ours)
    if p < min_src:
        return GateResult(False, p, ("source_not_our_players",))
    if server_team and c.speaker_team and c.speaker_team != server_team:
        p *= receiver_penalty; reasons.append("called_by_receiving_team")
    if legal_calls is not None and c.call not in legal_calls:
        # kept, flagged, NOT used as evidence: either the players' scoring error (shown as played) or another court
        return GateResult(False, p, tuple(reasons) + ("not_a_legal_score_for_our_match",))
    if p < 0.5:
        reasons.append("uncertain_source_weak_evidence")
    return GateResult(True, p, tuple(reasons), Obs(c.call, c.asr_conf, n_values, p_src=p))


def flag_mismatches(decoded, evidence, fmt):
    """Accepted calls that disagree with the decoded best path: [(segment, heard, path_call)]. Shown in review with the
    audio snippet; the players' call is kept as played."""
    out = []
    for i, ev in enumerate(evidence):
        heard = [c for c in ((ev.call,) if ev.call else ()) + tuple(ev.calls) if c.p_src >= 0.5]
        path = score_call(decoded.states[i], fmt)
        out += [(i, c.value, path) for c in heard if c.value != path]
    return out
