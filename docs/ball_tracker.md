# Own ball tracker (phase 1)

Status: research, draft. Code in `opencourtvision/ball/own/` (model, physics, fast paths) and `tools/ball_tracker/`
(Mac pipeline). Labels, footage and trained weights are private and never committed.

## Licence check (18 Oct 2026, GitHub licence metadata + LICENSE files)

| Project | Code licence | Weights | Ship / derive? |
|---|---|---|---|
| TrackNet v1 (NYCU, nol.cs.nctu.edu.tw GitLab) | none found (host unreachable 18 Oct; no licence in any mirror) | trained on the TrackNet tennis set (broadcast video, no licence) | No. Paper ideas only |
| TrackNetV2 (NYCU GitLab; ChgygLin/TrackNetV2-pytorch) | none stated | badminton set, no licence stated | No. Paper ideas only |
| TrackNetV3 (qaz812345/TrackNetV3) | MIT; the LICENSE (updated Aug 2026) explicitly covers the pretrained checkpoints | MIT, but trained on the Shuttlecock Trajectory Dataset (badminton; dataset rights not stated) | Code: yes (already vendored for engine-v2 re-acquire). Weights: legally MIT, data provenance for counsel; not used for the own tracker |
| WASB-SBDT (nttcom) | MIT | Google Drive checkpoints per sport; tennis ones trained on the TrackNet v1 tennis set (no licence) | Code: yes with notice. Tennis weights: no |
| yastrebksv/TrackNet (the S0 research ball) | none | none | No. Measurement only, as today |
| yastrebksv/TennisProject, TennisCourtDetector | none | none | No |
| ArtLabss/tennis-tracking | Unlicense | uses NYCU TrackNet weights + YOLOv3 | Code ideas only; weights no |
| abdullahtarek/tennis_analysis | none; Ultralytics (AGPL) | YOLO | No |
| Chang-Chia-Chi/TrackNet-Badminton (TF2) | MIT | badminton | Code yes; not needed |
| alenzenx/TracknetV3 (attention variant) | MIT | badminton | Code yes; not needed |
| HaydenFaulkner/Tennis | MIT | event models (not ball) | Not relevant |
| Ultralytics YOLO | AGPL-3.0 | AGPL | Never |

Decision: permissive code exists (TrackNetV3, WASB), but every available tennis weight traces back to data without a
licence. So the tracker is written from the papers' description (TrackNet: heatmap over stacked frames) as our own
code and trained from scratch on our own pseudo-labels only. No third-party code or weights in it.

## Design

* `heatnet.py`: 3 stacked RGB frames + 2 in-model frame differences -> 1 heatmap (middle frame), 4-level U-Net with
  additive skips. base=16: 354,641 parameters (~1.4 MB fp32 / 0.7 MB fp16), ~7.5 GMAC at 768x432.
* `physics.py`: arcs (quadratic/cubic in time), outlier rejection, in-arc filling, events (hit near player boxes or
  depth reversal; bounce = sharp drop of image-y velocity with horizontal direction kept).
* `fastpath.py`: court ROI (homography), motion-tile gating, physics-predicted crop tracking, fp16. Each is A/B tested
  against the plain pass by `eval_heldout.py` and kept only if faster without losing accuracy.
* Training (Mac only, M2 Pro 16 GB, nice 20): physics-filtered pseudo-labels from engine-v2 (own HSV code), crops
  384x224, fp16 autocast, hard-example subset, early stopping, then one self-training round (net + teacher, physics
  re-filter). A/B arm: same minutes on all labelled frames.
* Held-out: a recorded match points 51-101 (never used for training labels). Metrics are proxies (no hand ball labels); see
  `tools/ball_tracker/eval_heldout.py`.
