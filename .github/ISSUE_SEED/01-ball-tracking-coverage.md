# Improve ball-tracking coverage for the 9:16 reel

Labels: `reel`, `enhancement`, `help wanted`

## Problem
The ball-follow reel (`opencourtvision/reel/ball.py`) loses the ball often: at distance (far baseline), against
bright or yellow-ish backgrounds, under motion blur on hard hits, and when players occlude it. When the track is
lost, the crop drifts to the player-activity centre, which can cut the ball out of frame at the key moment.

## Current behavior
- Candidates come from frame differencing ANDed with a yellow HSV mask (fixed range `(22,80,130)-(45,255,255)`), then size and roundness filters, temporal support, and a greedy constant-velocity tracker (max 8 missed frames).
- Optional TrackNet (not shipped) is fused first, and HSV fills the gaps.
- `ball_stats.json` reports `pct_frames_with_ball_detected` and `pct_ball_frames_in_crop`, but there is no ground truth, so we can't tell a real improvement from noise.

## Ideas
- Synthetic tracking benchmark: render a moving ball (with blur, occluders, and varying size and colour) over `testsrc` or a court-like background, plus ground-truth positions.
- Adaptive HSV range per clip (sample the ball colour from confident detections).
- Kalman filter with gravity/bounce model instead of the greedy tracker, plus offline smoothing (RTS) since we process clips offline.
- Court-line homography to reject candidates outside the court volume.
- Make TrackNet easy to plug in (documented weights download, ONNX or Core ML export for M-series).

## Acceptance criteria
- [ ] Synthetic tracking benchmark added to `benchmarks/` with ground truth and a `run_eval.py` section (detection rate, mean px error, % frames ball inside crop).
- [ ] New tracker improves "ball inside crop" by ≥ 10 points on the benchmark without regressing crop smoothness (max per-frame velocity limit still respected).
- [ ] Unit tests for the new tracker; existing tests pass.
