"""Highlight style engine: a swappable preset (presets/<id>/preset.json, schema vamos.style_preset.v1) drives the look
(grade, type, score bug, captions, intro / outro, audio); the engine enforces Antony's hard rules whatever the preset says.

Order per clip: reframe (reel/camera.py, unchanged) -> grade (LUT, optional) -> overlays. See plan.py and render.py."""
DEFAULT_PRESET = "vamos_clean_v1"
