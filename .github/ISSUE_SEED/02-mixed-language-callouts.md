# Callout-based score detection with mixed-language speech

Labels: `scoring`, `stt`, `enhancement`

## Problem
Players often call the score in more than one language within a match (e.g. English and Spanish, sometimes
mid-sentence: "thirty quince"), use local variants ("nada", "iguales", "cuarenta iguales", "ventaja resto"), or
mumble. Whisper's language auto-detection on 1–12 s snippets flips between languages and sometimes translates or
hallucinates, so callouts get missed or misparsed.

## Current behavior
- `scoring/callouts.py` transcribes the quiet window before each rally with mlx-whisper (no fixed language), then `parse_callout` matches a small English/Spanish keyword table.
- "X iguales" is parsed as deuce because `iguales` is checked before the number rule (`"quince iguales"` → `deuce`; see `tests/test_scoring.py`).
- Numbers written as digits ("15-30") are not parsed. Only the first two number words are used, and the server-first order is assumed.
- Discrepancies are flagged but not reconciled.

## Ideas
- Constrain decoding: an `initial_prompt` with the score vocabulary, or forcing per-snippet language from a match-level majority vote.
- Parse digits and mixed tokens. Fix the "X iguales" / "X all" rule order. Add French and Italian keyword tables.
- Score-state decoder (HMM/Viterbi over legal tennis scores) that combines noisy callouts with the point count between them.
- Synthetic test set: TTS-generated callouts in several languages mixed into the synthetic clips.

## Acceptance criteria
- [ ] Table-driven tests covering English, Spanish, and mixed callouts, digits, and "all/iguales" for each score.
- [ ] `"quince iguales"` → `('pts', (15, 15))`; `"cuarenta iguales"` and `"deuce"` → deuce.
- [ ] On a synthetic callout benchmark, ≥ 95% of callouts parse correctly and the score timeline has no false discrepancy flags.
- [ ] No new dependency outside the `stt` extra.
