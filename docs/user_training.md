# User training: personal + opt-in shared models

**Owner:** Engineer  
**Decision:** Antony, 19 Oct 2026 — **both** a per-user personal model and an opt-in shared model.  
**Product:** free core (on-device) / paid extras (cloud + pro). No account needed to try recording, auto-edit, scoring.  
**Status:** design + v1 engine API (`opencourtvision.app`). Nothing ships without Antony's go-ahead.

---

## 1. Goals

1. Any Vamos player can train the AI from their own matches, not just Antony.
2. Corrections feel like a 30-second review, not a labelling job.
3. Personal model improves that player's court / camera overnight, on the device, while charging.
4. Opt-in sharing improves the base model for everyone, under GDPR (explicit consent + delete-my-data).
5. Bad footage from many setups is expected: odd angles, behind-fence, shaky / handheld, low light. Never refuse to process; teach the model from the hard cases.

## 2. Free core vs paid extras

| Free (on device, no account) | Paid (cloud / pro; placeholder in v1) |
|---|---|
| Record + import | Cloud storage + multi-device library |
| Auto-edit (points, dead-time trim) | Advanced analytics |
| Scoring + in-order score check-ins | Coach tools |
| Basic stats | Club licences |
| Fix screen | Premium highlight styles |
| Personal on-device model | |
| Highlight reels (Vamos Clean) | |
| Apple Watch score taps | |
| Opt-in "share my fixes" toggle (export stays on device until cloud is live) | |

Paywall sits only on cloud and pro features. `opencourtvision.app.tiers` encodes this; `require()` refuses pro features with a clear reason. No payment integration in v1.

## 3. Fix screen

After a match (or mid-match via Watch taps), the app shows its **least-certain calls**:

- bounce location (final bounce of the point)
- point winner (A / B)
- player ID by court side / role (**never** a face)

**Interaction:** one tap confirms; a click / drag fixes. Same idea as Antony's checker (8768) and bounce marker (8769), built into the app.

**Queue priority** (`review.prioritise`):

1. Items tagged with capture-quality struggle tags (low light, shaky, behind-fence, portrait, narrow FOV, …).
2. Lowest model confidence.
3. Time order (score check-ins stay chronological and non-blocking).

## 4. Labelled examples (privacy)

Each fix is stored under `session/fixes/<uuid>.json`:

```json
{
  "schema": "vamos.fix.v1",
  "kind": "point_winner",
  "point": 12,
  "t": 954.2,
  "value": "A",
  "source": "user",
  "struggle_tags": ["behind_fence", "low_light"],
  "privacy": {"faces": false, "biometrics": false, "names": false}
}
```

No face embeddings, no biometric templates, no real names. Players are court side or role only.

## 5. Personal on-device adapter

- **When:** overnight while charging (or on demand via `python -m opencourtvision.app retrain WORK`).
- **What:** a small adapter over frozen embeddings of the user's fixes (v1: ridge-linear; production: Core ML / MLX LoRA-style head on RF-DETR / ball features).
- **Where:** `session/models/adapter_*.npz`. Never committed. No cloud GPU.
- **Eval gate:** held-out 20% of fixes. If accuracy drops vs the previous adapter → **auto-rollback**, keep the old weights.
- **Mac jobs:** `nice 20`, yield when the user is active (`yield_when_active.sh`).

## 6. Opt-in shared pool (GDPR)

- Default **OFF**.
- Toggle: "Share my fixes to improve Vamos for everyone."
- Consent timestamp stored on the session.
- Shared payload is redacted (kind, value, relative t, struggle tags only).
- **Delete my data:** wipes fixes, taps, personal model, share export (`share.delete_my_data`).
- Weekly base-model retrain (ops): pull opted-in redacted examples → retrain base → held-out eval gate → ship or roll back. Base weights are separate from any one user's private labels.

## 7. Watch taps as weak labels

Apple Watch companion (`VamosWatch`):

| Tap | Effect |
|---|---|
| Point to us / them | Weak point-winner label (confidence 0.6) on the nearest point |
| Undo | Removes the previous watch label |
| Mark highlight | Sets `favorite` on the nearest point (slow-mo / ranker eligibility) |
| Start / stop recording | Signals the phone / Mac recorder |

Watch labels never overwrite a confirmed user fix. Score check-ins remain in time order and non-blocking.

## 8. Capture-quality hints

`quality.analyse(video)` always returns `ok: true` plus hints, for example:

- "Raise the phone so the whole court is in frame."
- "The picture looks dark. More light helps."
- "Looks handheld — a tripod or leaning against the fence helps."

Struggle tags feed the fix-screen priority so the personal / shared models see hard conditions first.

## 9. Robustness (hard requirement)

The pipeline must degrade gracefully on:

- odd / behind-fence angles
- shaky or handheld video
- low light
- portrait phone recordings
- missing audio (vision-only fallback; rallies-only if score cannot be built)

Never refuse to process. Mode suggestion may recommend `rallies` over `match` when structure is weak; the user decides.

## 10. Data schema (summary)

| Artefact | Schema | Location |
|---|---|---|
| Session | `vamos.session.v1` | `WORK/session.json` |
| Fix | `vamos.fix.v1` | `WORK/fixes/*.json` |
| Watch taps | ad-hoc JSON | `WORK/taps/*.json` |
| Share export | `vamos.share.v1` | `WORK/exports/share/examples.json` |
| Style preset | `vamos.style_preset.v1` | `opencourtvision/style/presets/<id>/` |
| Reel plan | `vamos.reel_plan.v1` | `WORK/exports/reel/plan.json` |

## 11. Highlight style engine (related)

Preset-driven render (`style/`). Media Director's pack is `vamos_clean_v1`. Hard rules enforced in code + tests:

- whole points only
- captions only from verified score state, stakes allowlist
- 0–2 slow-mo per reel, never slower than 0.45×
- hard cuts by default
- no letterboxing in 9:16
- grade on/off flag (normalisation only)
- music = platform sound picker at share (never bake Epidemic / Artlist)

Pipeline order: **reframe → LUT → overlays**. Camera code stays Engineer's.

## 12. Rollout

1. Engine API + design (this doc) — draft PR.
2. Mac app shell (SwiftUI wrapping the engine) — draft PR.
3. watchOS companion (simulator) — draft PR.
4. Sample reel from a real match with `vamos_clean_v1`.
5. Antony dogfoods; then Media Director pass/fail on camera checks.
6. Ship only with Antony's go-ahead (no merge to main, no deploy without it).
