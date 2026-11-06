# Third-party notices

The AGPL snapshot keeps attributions in [NOTICE](NOTICE). This file remains so older links still resolve.
OpenCourtVision is AGPL-3.0-only. Third-party source that is vendored stays under its own licence, named in NOTICE.

## Vendored or ported code

| Component | Upstream | Licence | Where | What |
|---|---|---|---|---|
| TrackNetV3 network definition | github.com/qaz812345/TrackNetV3 @ 6eda442 (2026-09-03), Copyright (c) 2024 qaz812345 | MIT | `opencourtvision/ball/_vendor/tracknetv3/` (`model.py` unmodified, `LICENSE`) | `TrackNet` / `InpaintNet` modules. Pre/post-processing (median background, 512×288 input, heatmap blobs) reimplemented in `ball/tracknet3.py`. |
| TrackNetV3 checkpoints (`TrackNet_best.pt`, `InpaintNet_best.pt`) | same repository (Google Drive link in its README) | MIT (repository licence; covers the released checkpoints) | **not in the repo**: downloaded at runtime by `ocv-fetch-weights tracknetv3` into `$OCV_CACHE/tracknetv3`, SHA-256 verified | Trained on badminton (shuttlecock) data; used for re-acquiring a lost ball. |
| Bounce clean-up (court locality + ~400 ms temporal dedup) | github.com/philippdubach/tennis-vision @ a501953 (2026-05-01), `events/player_tracker.py::mask_ball_track_to_broadcast`, `events/court_filter.py`, Copyright (c) 2026 Philipp Dubach | MIT | `opencourtvision/ball/bounce.py::clean_bounces` (ported and adapted to our `CourtGate`), licence in `opencourtvision/ball/LICENSE.tennis-vision` | Drop bounces > 2 m outside the court; keep the earliest of bounces closer than 400 ms. |

## Optional runtime dependencies

| Package / asset | Licence | Used by | Notes |
|---|---|---|---|
| numpy, scipy | BSD-3-Clause | core | |
| opencv-python | Apache-2.0 (OpenCV ≥ 4.5) | reel | |
| mlx-whisper | MIT | `--stt` | |
| torch | BSD-3-Clause | ball engine v2 (TrackNetV3), optional reel TrackNet | weights are downloaded at runtime, never committed |
| gdown | MIT | `ocv-fetch-weights` (optional) | Google Drive download of the TrackNetV3 checkpoints; falls back to urllib |

keypoint-MoSeq (Harvard non-commercial licence) is not used. Player-ID and pose code and weights are not part of this release; see `weights/MANIFEST.json`.

## Ideas reimplemented (no code or data copied)

- P. Wang & N. Christie, "Federer, Syllable by Syllable" (2026), https://pwang724.github.io/federer/. Code repository not publicly available, so no licence. Ideas only.
- C. J. Wei, "Automated Video-Based Analytics Framework for Tennis Doubles", arXiv:2507.02906. Repository github.com/jiaawe/tennis-prediction has no licence file. The taxonomy and labelling rules come from the paper text.
- iroblesrazzaq/RallyClip (github.com/iroblesrazzaq/RallyClip): no licence file, so **no code, weights or data are
  used**. Its pose + audio point segmentation is a benchmark target and an idea source for our planned point detector.
- yo-WASSUP/Good-Tennis and yastrebksv/TennisProject: run only outside this repository as benchmark baselines for
  `ocv-bench-ball` (Good-Tennis depends on Ultralytics, AGPL-3.0); nothing is copied or imported.
- C. Weinreb et al., "Keypoint-MoSeq", Nature Methods (2024), doi:10.1038/s41592-024-02318-2. Cited for the "syllable" concept only. Our segmenter is different (k-means + sticky Viterbi).

## OFL fonts (Vamos Clean highlight preset)

Bundled under `opencourtvision/style/presets/vamos_clean_v1/fonts/`:

- Barlow Condensed (Black Italic, Bold, SemiBold) — SIL Open Font Licence 1.1, Google Fonts
- Archivo Black — SIL Open Font Licence 1.1, Google Fonts
- JetBrains Mono (Bold, Medium) — SIL Open Font Licence 1.1, JetBrains

Licence texts: `OFL-*.txt` next to the font files. Used for burned-in stakes captions, score bug and cards only.
