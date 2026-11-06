"""Vendored from https://github.com/qaz812345/TrackNetV3 (MIT, see LICENSE in this folder).

Only `model.py` is vendored, unmodified, at upstream commit 6eda442ada1740573f200f836d93edc9a541ee86 (2026-09-03).
Pre/post-processing is reimplemented in `opencourtvision.ball.tracknet3`.
"""
from .model import InpaintNet, TrackNet  # noqa: F401
