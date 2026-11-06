# Stroke labelling (serve, forehand, backhand, volley, overhead)

Labels: `detection`, `research`, `enhancement`

## Problem
We know when each hit happens (audio onsets) but not what it was. Labelled strokes would enable better highlights
("best volleys"), richer story captions, and per-player stats.

## Current behavior
- Hits are timestamps plus a prominence value (`detection/audio.py`). There are no per-hit labels.
- `story/ranking.py` guesses "reflex volley" and "lob/overhead" purely from inter-hit gaps.

## Ideas
- Use the audio onset as the anchor and classify a short video window around it (pose keypoints from a lightweight pose model, then a small temporal classifier).
- Serve detection first (easiest: first hit after a pause, player pose overhead), then volley vs groundstroke (player court position), then forehand vs backhand.
- Annotation tool: export a contact sheet or short clips per hit and a CSV to label, building a small open dataset from contributors' own footage (with consent).

## Acceptance criteria
- [ ] A `detection/strokes.py` stage with a documented output schema (`hits.json`: time, label, confidence, player side).
- [ ] Serve vs non-serve labelling with ≥ 90% accuracy on a labelled evaluation set, reported by a benchmark script.
- [ ] Runs at ≥ real time on an M2 Pro for 1080p analysis frames, or is optional and off by default.
- [ ] Documented labelling format; no footage committed to the repo.
