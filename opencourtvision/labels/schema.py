"""Typed per-shot label schema for singles and doubles, with consistency rules.

The taxonomy (court position, side, shot type, direction, doubles formation, outcome) and the handedness rules follow
the annotation framework described in C. J. Wei, "Automated Video-Based Analytics Framework for Tennis Doubles"
(NUS B.Sc. dissertation, arXiv:2507.02906). Implemented from the paper's description; no code from its repository.

Direction rules for non-serves (geometry of the hitter's own half, relative to their hitting hand):
  right-handed: deuce side  forehand -> cross_court / down_the_line, backhand -> inside_in / inside_out
                ad side     backhand -> cross_court / down_the_line, forehand -> inside_in / inside_out
  left-handed:  mirrored (deuce: backhand CC/DL, forehand II/IO; ad: forehand CC/DL, backhand II/IO)
Serves take T / body / wide with a serving formation; non-serves take CC/DL/II/IO with formation non_serve.
"""
from dataclasses import asdict, dataclass, fields
from enum import Enum


class _E(str, Enum):
    def __str__(self):
        return self.value


class CourtPosition(_E):
    NEAR_DEUCE = "near_deuce"; NEAR_AD = "near_ad"; FAR_DEUCE = "far_deuce"; FAR_AD = "far_ad"

    @property
    def end(self): return self.value.split("_")[0]

    @property
    def half(self): return self.value.split("_")[1]


class Side(_E):
    FOREHAND = "forehand"; BACKHAND = "backhand"


class ShotType(_E):
    SERVE = "serve"; SECOND_SERVE = "second_serve"; RETURN = "return"; VOLLEY = "volley"; LOB = "lob"; SMASH = "smash"; SWING = "swing"


class Direction(_E):
    T = "T"; BODY = "body"; WIDE = "wide"
    CROSS_COURT = "cross_court"; DOWN_THE_LINE = "down_the_line"; INSIDE_IN = "inside_in"; INSIDE_OUT = "inside_out"


class Formation(_E):
    CONVENTIONAL = "conventional"; I_FORMATION = "i_formation"; AUSTRALIAN = "australian"; NON_SERVE = "non_serve"


class Outcome(_E):
    IN = "in"; WIN = "win"; ERROR = "error"


class Hand(_E):
    RIGHT = "right"; LEFT = "left"


SERVE_TYPES = frozenset({ShotType.SERVE, ShotType.SECOND_SERVE})
SERVE_DIRECTIONS = frozenset({Direction.T, Direction.BODY, Direction.WIDE})
RALLY_DIRECTIONS = frozenset({Direction.CROSS_COURT, Direction.DOWN_THE_LINE, Direction.INSIDE_IN, Direction.INSIDE_OUT})
SERVE_FORMATIONS = frozenset({Formation.CONVENTIONAL, Formation.I_FORMATION, Formation.AUSTRALIAN})
_OUTSIDE = frozenset({Direction.CROSS_COURT, Direction.DOWN_THE_LINE})
_INSIDE = frozenset({Direction.INSIDE_IN, Direction.INSIDE_OUT})
_ENUMS = {"position": CourtPosition, "side": Side, "shot_type": ShotType, "direction": Direction, "formation": Formation,
          "outcome": Outcome, "hand": Hand}


@dataclass
class ShotLabel:
    """One shot. Every field is optional so partial (e.g. model-predicted) labels are representable."""
    position: CourtPosition | None = None
    side: Side | None = None
    shot_type: ShotType | None = None
    direction: Direction | None = None
    formation: Formation | None = None
    outcome: Outcome | None = None
    hand: Hand | None = None              # the hitter's dominant hand (needed for the direction rules)

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None and not isinstance(v, _ENUMS[f.name]):
                setattr(self, f.name, _ENUMS[f.name](v))     # raises ValueError on unknown values

    @property
    def is_serve(self):
        return self.shot_type in SERVE_TYPES

    def to_dict(self):
        return {k: (str(v) if v is not None else None) for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in _ENUMS})


def allowed_directions(position, side, hand, serve=False):
    """Directions consistent with where the hitter stands, which side they hit and their dominant hand."""
    if serve: return set(SERVE_DIRECTIONS)
    if position is None or side is None or hand is None: return set(RALLY_DIRECTIONS)
    half = CourtPosition(position).half; side = Side(side); hand = Hand(hand)
    natural = Side.FOREHAND if (half == "deuce") == (hand == Hand.RIGHT) else Side.BACKHAND   # the side that hits "outside"
    return set(_OUTSIDE if side == natural else _INSIDE)


def validate(label, hand=None):
    """-> list of human-readable rule violations (empty = consistent). hand overrides label.hand."""
    lab = label if isinstance(label, ShotLabel) else ShotLabel.from_dict(label)
    hand = Hand(hand) if hand else lab.hand; err = []
    if lab.shot_type is not None and lab.direction is not None:
        if lab.is_serve and lab.direction not in SERVE_DIRECTIONS:
            err.append(f"serve direction must be T/body/wide, got {lab.direction}")
        if not lab.is_serve and lab.direction not in RALLY_DIRECTIONS:
            err.append(f"non-serve direction must be cross_court/down_the_line/inside_in/inside_out, got {lab.direction}")
    if lab.shot_type is not None and lab.formation is not None:
        if lab.is_serve and lab.formation not in SERVE_FORMATIONS:
            err.append(f"serve formation must be conventional/i_formation/australian, got {lab.formation}")
        if not lab.is_serve and lab.formation != Formation.NON_SERVE:
            err.append(f"non-serve formation must be non_serve, got {lab.formation}")
    if (not lab.is_serve and lab.direction in RALLY_DIRECTIONS and lab.position and lab.side and hand):
        ok = allowed_directions(lab.position, lab.side, hand)
        if lab.direction not in ok:
            err.append(f"{hand}-handed {lab.side} from the {lab.position.half} side can only be "
                       f"{'/'.join(sorted(str(d) for d in ok))}, got {lab.direction}")
    return err


def validate_rally(labels, hands=None):
    """Rally-level rules over an ordered list of ShotLabel/dict: starts with a serve, a return only right after a serve,
    outcome win/error only on the last shot. hands: optional per-shot hand list. -> list of (index, message)."""
    L = [x if isinstance(x, ShotLabel) else ShotLabel.from_dict(x) for x in labels]; err = []
    for i, lab in enumerate(L):
        for m in validate(lab, hands[i] if hands else None): err.append((i, m))
        if i == 0 and lab.shot_type is not None and not lab.is_serve: err.append((i, "a rally starts with a serve"))
        if i > 0 and lab.is_serve and lab.shot_type == ShotType.SERVE: err.append((i, "serve in the middle of a rally"))
        if lab.shot_type == ShotType.RETURN and not (i > 0 and L[i - 1].is_serve): err.append((i, "return must follow a serve"))
        if i < len(L) - 1 and lab.outcome in (Outcome.WIN, Outcome.ERROR): err.append((i, "only the last shot can end the point"))
    return err


def court_position(x, y, width=10.97, length=23.77):
    """Court metres (x from the near-left sideline, y from the near baseline, camera behind the near end) -> position.
    The deuce half is each player's right as they face the net: image-right for the near end, image-left for the far end."""
    near = y < length / 2
    right_half = x > width / 2
    deuce = right_half if near else not right_half
    return CourtPosition(f"{'near' if near else 'far'}_{'deuce' if deuce else 'ad'}")
