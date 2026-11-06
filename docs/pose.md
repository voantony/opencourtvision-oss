# Pose stage, labels, verify gate and eval

## Where the ideas came from (no code, no data reused)

1. **"Federer, Syllable by Syllable"** (P. Wang & N. Christie, 2 Oct 2026, pwang724.github.io/federer). They tracked 17 joints of
   Federer in 45 US Open TV matches (2003–2019), split his movement into ~30 unsupervised "syllables" with keypoint-MoSeq
   (shown on a UMAP map with synced clips), labelled serve/forehand/backhand, and compared eras on joint-based measures
   (forehand stance angle, contact distance from the hips, finish height, serve lean) using per-match means, Welch's
   t-test, Holm correction and an effect-size gate, with camera/position covariates regressed out. Their code repository
   (github.com/pwang724/deep-sports-analysis) was not publicly reachable (HTTP 404), so no licence could be confirmed.
   Treated as unlicensed: **clean-room, ideas only**. keypoint-MoSeq itself is Harvard non-commercial licensed, so it is
   **not** used; `pose/syllables.py` is an independent k-means + sticky-Viterbi segmenter.
2. **C. J. Wei, "Automated Video-Based Analytics Framework for Tennis Doubles"** (NUS dissertation, arXiv:2507.02906).
   Its code (github.com/jiaawe/tennis-prediction) has **no licence file**, so nothing was copied. Taken from the paper:
   the doubles shot taxonomy, the serve/non-serve direction and formation combinations, and the handedness rules for
   cross-court / down-the-line vs inside-in / inside-out. Not taken: GroundingDINO, YOLO-Pose / YOLOv11 (Ultralytics, AGPL),
   the GCN/CNN classifiers.

## Modules

| Module | What it does |
|---|---|
| `pose/skeleton.py` | 17-joint layout (head, shoulders, elbows, wrists, hips, knees, ankles, heels, toes); MediaPipe-33 / COCO-17 mappings |
| `pose/estimate.py` | `PoseEstimator` interface; MediaPipe (Apache-2.0, model downloaded at runtime to `$OCV_CACHE`) and Apple Vision backends; near-player picking; `.npz` keypoint files |
| `pose/court.py` | 4-corner homography → court metres, local cm/px, feet-line angle to the baseline |
| `pose/strokes.py` | contact = audio hit with a hitting-wrist speed peak; serve / forehand / backhand from body geometry |
| `pose/metrics.py` | stance angle, contact distance, finish height, serve lean; footwork distance, speed, court coverage |
| `pose/compare.py` | per-session means, Welch t-test, Holm, 95 % CI, ≥0.4 SD effect gate, covariate residualisation |
| `pose/syllables.py` | unsupervised movement syllables + 2-D map (PCA, or UMAP if installed) |
| `labels/schema.py` | typed `ShotLabel` + `validate` / `validate_rally` rules |
| `events.py` | `Event` {type, t, player, label, confidence, evidence[], verified}; `verify()`; captions only for verified events |
| `eval/events.py` | `ocv eval`: matching (±80 ms contact/bounce, ±150 ms other), P/R/F1 per type and label, stroke confusion, calibration/ECE |
| `windows.py` | analyse only windows around audio hits |
| `stages.py` | stage layout + content-hash `StageCache` |

## Assumptions and limits

- One tracked player per run (the near-end player by default, or use `--region`). Tracking all four doubles players is not built yet.
- Forehand/backhand assumes the player has their back to the camera (near end). For the far end, mirror the image or extend `classify`.
- Body-plane distances use the court scale at the feet. Accuracy depends on the camera; treat them as relative (then vs now), not absolute biomechanics.
- Pose-only events have one evidence source, so with the default `min_sources=2` they are never verified or captioned.
