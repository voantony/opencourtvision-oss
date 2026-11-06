# Padel support

Labels: `enhancement`, `help wanted`

## Problem
Padel recordings (glass walls, enclosed court, underhand serve, always doubles) break several tennis assumptions:
balls rebounding off the glass make extra loud transients, rallies are longer, the scoring includes golden point,
and the camera usually sits behind a wall.

## Current behavior
- Hit detection treats every 1.5–6 kHz transient as a hit, so wall and glass rebounds will be counted as hits.
- `warmup_cutoff` thresholds (pause ≥ 20 s, median inter-hit ≤ 2.5 s) and `match_like` were tuned on tennis.
- Changeover logic and `parse_callout` assume tennis rules. There is no golden point or "punto de oro".
- The reel's HSV range and size filters assume a tennis ball at tennis-camera distances.

## Ideas
- A `--sport padel` profile (a dataclass of thresholds) passed through the pipeline instead of module constants.
- Classify transients as racket hit vs glass/wall rebound (spectral shape, a second onset shortly after).
- Padel scoring: golden point and callout vocabulary.
- Synthetic padel scenario: hits plus rebound transients with known labels.

## Acceptance criteria
- [ ] `--sport {tennis,padel}` flag with tennis as the default, and tennis benchmark numbers unchanged.
- [ ] Synthetic padel benchmark with rebounds: hit precision ≥ 0.9 and recall ≥ 0.9.
- [ ] Golden-point scoring supported in the score timeline with tests.
- [ ] README documents padel limitations.
