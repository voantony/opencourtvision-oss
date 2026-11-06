# Better difficult-shot scoring for the story finder

Labels: `story`, `enhancement`

## Problem
"Difficult shot" candidates in `story/ranking.py` currently come from audio timing only (to be fused with ball and pose tracking) (reflex volleys < 0.6 s between hits, a
long mid-rally gap read as a lob or overhead, a fast finish). They miss gets after apparent winners, running shots,
and dives, and they over-score fast exchanges at the net.

## Current behavior
- Scores are additive heuristics per hit with fixed weights, and the top 15 go into `story.json` → `difficult_shots`.
- `note` in `story.json` states that distance covered and positional difficulty are not computed.
- With `--story`, up to three difficult shots go into the highlights.

## Ideas
- Add video signals: player displacement between hits (from the activity or pose tracker), ball speed from the tracker, and whether the shot was hit on the run.
- Use per-hit audio features (onset sharpness, loudness relative to the rally) to tell clean contact from frames and mishits.
- Learn weights from a small set of human "was this a great shot?" ratings (pairwise preferences).
- Synthetic unit cases for each pattern so scoring rules are explicit and testable.

## Acceptance criteria
- [ ] Each difficulty signal is a separate, unit-tested function with a documented range.
- [ ] A benchmark (synthetic or a rated set with timestamps only) where the top-5 difficult shots reach precision ≥ 0.6.
- [ ] `story.json` schema stays backward compatible (new fields only).
