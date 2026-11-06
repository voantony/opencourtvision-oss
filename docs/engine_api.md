# Engine API (`opencourtvision.api`)

One stable call shape for every Vamos client (web beta, Mac job runner, iPhone, Watch). Clients code against this, and
engines swap behind it one piece at a time (scoreboard gate).

```python
from opencourtvision import api
m  = api.analyze("match.mp4", {"engine": "real", "work": "/tmp/w", "final": "6-4 6-3"})   # MatchResult
m2 = api.reinfer(m, [{"item_id": "pw-3", "value": "B"}])                                  # one-tap fixes
r  = api.render(m2, "vamos_clean_v1", {"score_overlay": False, "long_shot_mode": "receiver", "out": "reel.mp4"})
```

## MatchResult (`vamos.match_result.v1.1`, JSON via `to_dict()` / `from_dict()`)

| field | contents | source today |
|---|---|---|
| `points` | session points (`id, A, B, first_hit, last_hit, hits, react, favorite, winner, winner_src, winner_confidence, bounce, player_id, features, features_src, ...`) | real: `app.analyze` pipeline; stub: synthetic fixture |
| `score_timeline` | per point `{point, t, winner, before, after, verified, note}`; `before/after` = sets, games, pts, server, tiebreak, call | real: `scoring.rules` exact replay |
| `uncertain_queue` | fix-screen items `{id, kind, point, t, confidence, prompt, options, current, struggle_tags, features, features_version}` | real: `app.review.build_queue` |
| `clips` | whole-point clips `{point, a, b, score, favorite}` | real: `style.plan.whole_bounds` |
| `reel_plan` | `vamos.reel_plan.v1` or `null` (reason in `meta.notes`) | real: `style.plan.plan_reel` |
| `players` | `{id: A/B, team, side, identity: court_side_role}`; never faces/biometrics | **stub** until the 4-player ID tracker |
| `features_version` | `"vamos.feat.v1"`: version of every `features` vector below (see "Per-fix features") | |
| `meta` | `engine` (real/stub), `stubbed` (list), `fallback_reason`, sport, teams, final, video, work, `fixed`, `n_corrections` | |

## Calls

- `analyze(video, opts)`: `opts.engine` = `real` (default) | `stub` | `auto` (real if the file exists and ffmpeg is present, else stub with `meta.fallback_reason`). Other opts: `work, final, mode (match|rallies), sport, teams, format_preset, first_server`; stub: `n_points, seed`.
- `reinfer(match, corrections)`: pure (never mutates the input), all-or-nothing. A correction is `{value, item_id}` or `{value, kind, point}`, plus optional `source` (`user|confirm|watch`). Kinds: `point_winner` (A/B/skip), `bounce` ({x,y} court-normalised, or "none"), `player_id` ({near/far: A/B}), `cut_off` (bool), `score_check` (ok/wrong). Fields that look like faces, biometrics, names or media are rejected. Recomputes score timeline, queue, clips and reel plan. Persisting fixes and the opt-in collector upload are the caller's job.
- `render(match, preset, opts)`: `score_overlay` (default true; false drops the score bug and the score captions), `long_shot_mode` (`hitter|receiver|auto`; recorded in the plan, camera not wired yet), `aspect, max_points, out, dry_run`. Stub matches and `dry_run` return the plan only (`rendered: false`).

No new dependencies. Imports only modules that ship in both the US and full builds (`scoring.rules`, not the whole-match decoder).

## Per-fix features (`vamos.feat.v1`) and posting to `/v1/fixes`

Every `uncertain_queue[]` item carries `features` (16 floats) and `features_version: "vamos.feat.v1"`; every point carries
`features` plus `features_src` (16 strings: `engine | score_rules | quality | stub | none`). The vector is index-identical to
`opencourtvision/shared/features.py` (PR #21 collector), so a web correction posts directly without recomputing anything.
Unknown values are `0.0` (the collector's own fallback convention). All values are finite and clamped to ±10. Numbers only:
no pixels, faces, biometrics or names. Names/order: `api.FEATURE_NAMES`; helper: `api.fix_example(item, value, client_id, source)`.

| idx | name | value | filled from today (real engine) | stub engine |
|---|---|---|---|---|
| 0 | `rally_len` | hits / 20 | `point.hits` (audio onsets) | fixture hits |
| 1 | `last_contact_x` | court x 0..1 | `point.last_contact.x` if present, else 0 | deterministic (seed, point) |
| 2 | `last_contact_y` | court y 0..1 | `point.last_contact.y` if present, else 0 | deterministic |
| 3 | `ball_speed` | speed proxy / 50 | `point.ball_speed` if present, else 0 | deterministic 0..2 |
| 4 | `audio_reaction` | dB / 30 | `point.react` | fixture react |
| 5 | `serve_side` | +1 deuce / -1 ad | score timeline `before.pts` parity | same |
| 6 | `server_team` | +1 A / -1 B | score timeline `before.server` | same |
| 7 | `far_motion_share` | 0..1 | `point.motion_share.far` if present, else 0 | deterministic |
| 8 | `near_motion_share` | 0..1 | `point.motion_share.near` if present, else 0 | deterministic |
| 9 | `confidence` | 0..1 | queue item `confidence` (points: `winner_confidence`; 1.0 after a user fix) | same |
| 10–15 | `tag_low_light, tag_shaky, tag_behind_fence, tag_portrait_phone, tag_narrow_fov, tag_low_resolution` | 0/1 | capture-quality `struggle_tags` | fixture tags (none) |

Mapping one fix to a `/v1/fixes` example (`vamos.share.v2`, Bearer token from `/v1/register`, consent required):

| `/v1/fixes` example field | from |
|---|---|
| `client_id` | new UUID of the fix (8–64 `[A-Za-z0-9-]`) |
| `kind` | `uncertain_queue[].kind` (`point_winner`, `bounce`, `player_id`; `cut_off` / `score_check` are not shareable) |
| `value` | the user's answer (`point_winner` A/B, not `skip`; `bounce` {x,y} or "none"; `player_id` {near/far}) |
| `features` | `uncertain_queue[].features` (exactly as returned; same vector that was on screen) |
| `struggle_tags` | `uncertain_queue[].struggle_tags` |
| `source` | `user` / `confirm` / `watch` |

```json
POST /v1/fixes
{"schema": "vamos.share.v2", "consent_version": "<accepted consent version>", "app_version": "web-beta-0.1",
 "examples": [{"client_id": "3f1c2a9e-7b1d-4c55-9a0e-1f2b3c4d5e6f", "kind": "point_winner", "value": "B",
               "features": [0.35, 0.4121, 0.8834, 1.2051, 0.1567, 1.0, -1.0, 0.2214, 0.7302, 0.61, 0, 0, 0, 0, 0, 0],
               "struggle_tags": [], "source": "user"}]}
```

Then `reinfer(m, [{"item_id": "pw-3", "value": "B"}])` locally; the reinfer refusal of face / biometric / name / media
fields (and any `features` key in a correction) is unchanged: corrections carry only `value, item_id, kind, point, source`.

**Compatibility.** `vamos.match_result.v1.1` is additive. `from_dict` reads both `v1` and `v1.1` and returns `v1.1`;
a v1 document has no `features` until the next `reinfer` (which recomputes them). v1 readers that ignore unknown keys keep
working. `API_VERSION` 0.2.0.
