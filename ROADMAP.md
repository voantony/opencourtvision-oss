# Roadmap: beat SwingVision

## Pillars

1. **Doubles first.** Track all 4 players, detect formation, attribute shots to the right partner.
2. **Verified-only labels.** Nothing is captioned unless it passes the verify gate. Gate: caption precision ≥ 0.95 on the eval set. Accuracy numbers are published.
3. **Shareable highlights.** 9:16 ball-follow reels and story summaries.
4. **Private and on-device.** Free core, no upload required, batch processing across many matches.
5. **Padel support.**

## Milestones (each one ends on measurable exit criteria)

| | Milestone | Exit criteria |
|---|---|---|
| M1 | Mac engine solid on Antony's footage | ~200 hand-labelled hits from 2–3 of his matches in `labels/` (kept private, not committed). `ocv eval` reports per-event precision/recall: hit detection P ≥ 0.95, R ≥ 0.90 at ±80 ms. Runtime per match minute on the M2 Pro measured and in the README. |
| M2 | Doubles analytics | 4-player tracking with ID kept through a rally ≥ 95 % of frames (near pair), formation label accuracy ≥ 0.85, partner attribution of hits ≥ 0.90, shot taxonomy (`labels/schema.py`) predicted with verified-caption precision ≥ 0.95. |
| M3 | Ball tracking for direction, bounce, in/out | Ball visible-frame recall ≥ 0.85 on labelled clips, bounce F1 ≥ 0.85 at ±80 ms, in/out agreement ≥ 0.95 on clear calls (with "unsure" allowed), direction labels verified at ≥ 0.95 precision. |
| M4 | iPhone port, offline | Core ML models plus Apple Vision pose. A recorded 1-hour match is analysed on-device on a recent iPhone with M1-level eval scores within 2 points. No network needed. |
| M5 | Live tracking while filming | Real-time (≥ 30 fps) hit and ball tracking on device with ≤ 1 s latency for score and line feedback. This is SwingVision's current edge. |
| M6 | Padel | 10 × 20 m court geometry, wall-bounce events, padel label set and eval on ≥ 100 labelled padel hits. |

## Player identity and efficient ball tracking (Antony, 18 Oct 2026)

> "you should be able to identify in such a way where commonalities can rule out different players' outfits and physical features to identify who's who. and also have a way to follow the ball efficiently. it's a yellow ball that is the same size"

Licence limits are lifted for development; every non-MIT/Apache item is logged for Legal in engineering notes §8. **Status (Oct 2026):** part 2 (ball) is built as ball engine v2
(`opencourtvision/ball/`, branch `feature/engine-v2`): colour + motion candidates, size gate, ROI/Kalman search,
TrackNetV3 re-acquire, audio-contact fusion and a live-ball filter, with proxy metrics in `ocv-bench-ball`. The
acceptance metrics below still need labelled ball frames (`ocv-label-ball`). Part 1 (who's who / player-ID) and pose analysis are not part of this public snapshot.

**1. Who's who in doubles**

| Item | Approach | Acceptance metric |
|---|---|---|
| Player tagging at setup (v3.7, N5) | At setup the user taps two players (one per team); partner and the second opponent are deduced (`deduce_ids`), with a 1/0-tap fallback. No appearance auto-clustering into identities. After tagging, an appearance descriptor of each *tagged* box (clothing CIELAB histogram, bbox aspect) is used only for frame-to-frame continuity (flag for counsel) | Tagging ≤ 10 s for 4 players; continuity ID ≥ 0.95 of player-frames on 3 matches |
| Assignment | Hungarian algorithm per frame on cost = appearance distance + motion gate (Kalman-predicted court position; infeasible jumps get infinite cost) | ID accuracy ≥ 0.95 of player-frames over a rally (M2 target); ID switches ≤ 1 per 10 points |
| Rule constraints | Teammates stay on the same side between changeovers; sides swap after odd games; doubles serve rotation fixed for the set; receiver on deuce/ad side by score. Used as hard gates on assignment and to fix it after the fact | Server identity accuracy ≥ 0.98 per point; zero assignments that break a side or rotation rule |
| Re-ID after occlusion / net crossings | Hold the track as "lost", re-match on appearance + rules when it reappears; carry `id_uncertain` flag with a calibrated score. Uncertain IDs never feed captions (verify gate) | Re-ID correct ≥ 0.95 after occlusions; flagged-uncertain IDs are wrong ≥ 5× more often than unflagged (flag is informative) |

**2. Following the ball efficiently**

| Item | Approach | Acceptance metric |
|---|---|---|
| Colour prior | Yellow-green ball mask in HSV and Lab, with per-match auto-calibration from confirmed detections (dusk, clay, indoor light) | Candidate recall ≥ 0.95 of labelled ball frames inside the ROI |
| Known size (v3.7, N2) | Row-based pixel-diameter bands: expected ball diameter per image row, learned per mount from confirmed detections (no fitted 3D height); reject blobs outside the band (motion blur stretches one axis, so check the minor axis) | False candidates per frame cut ≥ 50 % with ≤ 1 % loss of true balls |
| ROI search (v3.7, N2) | Constant-acceleration Kalman in image pixels (gravity appears as image-y acceleration; no physics/drag model) predicts the next position; search only that window. Bounce handling: on a predicted ground crossing, branch to a reflected trajectory with restitution; racket hits re-seed from the audio onset | Ball visible-frame recall ≥ 0.85 (M3); bounce F1 ≥ 0.85 at ±80 ms |
| Re-acquire | Full-frame TrackNet (MIT, TrackNetV3, fine-tuned on our tennis labels) only when the track is lost for > N frames or confidence drops | TrackNet runs on ≤ 20 % of frames on a typical match |
| Audio fusion | Hit onsets from the audio stage mark contact times: expect a velocity change there, open a wider ROI around the hitter's racket, and confirm the event with two sources | Contact timing within ±80 ms on ≥ 0.90 of labelled hits |

**Expected speedup and how to measure it.** ROI search touches ~5–10 % of pixels and TrackNet runs on a minority of frames, so we expect roughly 3–8× faster ball tracking than full-frame TrackNet on every frame at the same resolution. This is an estimate, not a measurement. Measure on the M2 Pro at full resolution: fps of the ball stage (ROI+re-acquire vs TrackNet-every-frame) on the same clips, plus recall and position error against labelled ball frames via `ocv eval`. Ship only if recall is within 2 points of the full-frame baseline.

## Constraints

- **Licences.** The project is AGPL-3.0-only, so AGPL-compatible code can be combined with it (including Ultralytics YOLO, which stays for now). Non-commercial and no-licence code still cannot be copied in (keypoint-MoSeq, unlicensed TrackNet weights). Pose and player-ID weights are not release assets in v1. Ball, court and scoring weights, when hosted, are downloaded by `scripts/fetch_release_weights.py` and are not committed. See `NOTICE`.
- **Patents.** Get a patent check on ball tracking and line calling before any public launch (M3/M5).

## Help wanted

This public snapshot is a work in progress. The next measurements are far-side player recall and p(A) calibration (`python benchmarks/public_eval/measure.py`). Help-wanted issues are filed on GitLab; drafts also live under `docs/issue_drafts/`.

- **Replace YOLO11 with a stronger player/ball detector (e.g. RF-DETR or newer).** The measurement tool
  `tools/point_winner/pw_tracks.py` imports Ultralytics YOLO11-pose. A replacement must be AGPL-3.0-compatible
  (MIT, BSD, Apache-2.0 and AGPL-3.0 itself qualify; GPL-2.0-only, non-commercial and proprietary terms do not)
  and must not regress the primary metrics in `benchmarks/detector_bench.py`. Draft:
  `docs/issue_drafts/replace-yolo11-detector.md`.

## Known gaps (honest status)

- All accuracy numbers so far come from **synthetic** data. No real labelled footage has been evaluated yet (that is M1).
- Ball engine v2 exists but is only proxy-benchmarked (no labelled ball frames yet); its TrackNetV3 weights are the
  upstream badminton checkpoints (not fine-tuned on tennis); thresholds are first guesses; contacts are audio-only
  until player boxes exist. The reel still uses the old HSV tracker.
- Pose tracks one player per run, and forehand/backhand assumes the near-end player. 4-player ID tracking exists (`opencourtvision/players/`) but is not wired into pose or the reel, is measured only
  against proxy truth, and the 2-tap UI in the app is not built (backend `deduce_ids` only).
- Detection recall drops sharply in noisy audio, and the standalone match-start estimator fires early.
- Direction, formation, outcome and in/out are schema and eval only. No predictor exists for them yet.
- `pipeline.py` still uses its own per-file caches. Moving it onto `StageCache` is pending.
- Everything is Mac/Python today. There is no mobile code.
