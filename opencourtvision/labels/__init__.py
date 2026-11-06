"""Shot label schema (doubles-aware) and its consistency rules. See schema.py."""
from .schema import (  # noqa: F401
    SERVE_TYPES,
    CourtPosition,
    Direction,
    Formation,
    Hand,
    Outcome,
    ShotLabel,
    ShotType,
    Side,
    allowed_directions,
    court_position,
    validate,
    validate_rally,
)
