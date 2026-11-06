"""Per-user and per-match settings for highlight ranking (highlight_algorithm.md §5.6).

UserConfig (per user, persists across matches):
  loss_weight   0.0-1.0, default 0.5 (Coach O4). How much ORDINARY lost points count in the personal reel:
                0.0 = only wins, 1.0 = best points, win or lose. Great lost points (O8) and great points won by the
                opponents are always included and never scaled, at every value (Antony, O13).

MatchSettings (per match):
  user_team     "host" | "guest": which side of the score is the user's team. Default "host" = SwingVision's
                "Host Team" (the assumption in the spec header, O10). The app should ask once per match and set
                user_team_confirmed; a wrong mapping flips won and lost points.
  deuce_rule    None = the sport's default scoring format (scoring.formats); else advantage | golden_point | star_point.
  pose_confirmed_reactions  point ids whose reaction pose has confirmed (MD O5): only those may be standalone beats.

Storage: JSON at $OCV_USER_CONFIG, else ~/.config/opencourtvision/user.json, e.g. {"loss_weight": 0.3}.
"""
import json
import os
from dataclasses import asdict, dataclass, field

DEFAULT_LOSS_WEIGHT = 0.5
USER_TEAMS = ("host", "guest")


def _path(path=None):
    return path or os.environ.get("OCV_USER_CONFIG") or os.path.join(os.path.expanduser("~"), ".config", "opencourtvision", "user.json")


@dataclass
class UserConfig:
    loss_weight: float = DEFAULT_LOSS_WEIGHT

    def __post_init__(self):
        lw = float(self.loss_weight)
        if not 0.0 <= lw <= 1.0:
            raise ValueError(f"loss_weight must be between 0.0 and 1.0, got {self.loss_weight!r}")
        self.loss_weight = lw

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(**{k: d[k] for k in ("loss_weight",) if k in d and d[k] is not None})

    @classmethod
    def load(cls, path=None):
        """Read the user's config; a missing file gives the defaults. Unknown keys are ignored."""
        p = _path(path)
        if not os.path.exists(p):
            return cls()
        with open(p) as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path=None):
        p = _path(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        cur = {}
        if os.path.exists(p):
            with open(p) as fh:
                cur = json.load(fh)
        cur.update(self.to_dict())
        with open(p, "w") as fh:
            json.dump(cur, fh, indent=1)
        return p


@dataclass
class MatchSettings:
    user_team: str = "host"
    user_team_confirmed: bool = False
    sport: str = "tennis"
    deuce_rule: str = None
    pose_confirmed_reactions: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        if self.user_team not in USER_TEAMS:
            raise ValueError(f"user_team must be one of {USER_TEAMS}, got {self.user_team!r}")
        self.pose_confirmed_reactions = frozenset(self.pose_confirmed_reactions or ())

    @property
    def opponent_team(self):
        return "guest" if self.user_team == "host" else "host"
