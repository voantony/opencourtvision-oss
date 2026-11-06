# Physics-consistent ball tracker (`physics3d`)

Status: research prototype, draft. Code: `opencourtvision/ball/physics3d.py` (tracker), `physics3d_synth.py` (synthetic
truth), `tests/test_physics3d.py`, `tools/ball_tracker/{extract_cands,physball_eval,physball_synth_ab}.py`. Summary
numbers: `benchmarks/results/physics_ball_dev_match.json`. Labels, footage, SwingVision events and weights stay private.

## In plain words

The current tracker finds the ball in each frame with a heatmap network, then a 2D filter joins detections into
arcs. That filter only checks that the dots look smooth on screen, so it keeps a lot of junk and throws away real
ball frames. After filtering it keeps about a quarter of the SwingVision shots and finds no bounces after the last
contact on the held-out points.

The new tracker only reports a track when a real 3D ball flight explains it. That means a ball pulled down by gravity,
slowed by air drag, lifted or dipped by spin, and seen through the real camera. The camera comes from the court lines
we already tap, so it needs no extra calibration. A bounce only counts if ONE 3D flight can come down to the court,
bounce off it with a believable bounce, and fly on through the detections after it. The bounce is also what pins down
how far away the ball is, which one camera cannot measure from a single flight. Junk that no flight explains is
dropped, and on pure noise the tracker reports nothing at all.

What it gives you per frame is the ball in pixels and in court metres. For each bounce it gives the court position,
its uncertainty and P(in). Each hit gets a time and whether it was near a player.

Results in short:

* **Synthetic rallies (known truth):** it recovers 70-78% of ball frames against 20-63% for the 2D filter, even with
  50-70% false detections and 8-frame gaps. It found 42-59 of 120 bounces with zero false bounces (the 2D filter: 3-37
  found, 8-39 false). Median bounce position error is 6-7 cm against 19-50 cm for 2D.
* **Held-out real points 51-101:** it catches 79% of SwingVision shots, measured in 3D (the ball ray passes within
  1.5 m of SwingVision's hit point). The current tracker catches 18% and the raw detector 71%. Precision on that test
  is 82% against 84% and 75%. It finds 0.53 bounces per point, and 37% of them agree with SwingVision's bounce within
  1.5 m. The current tracker finds 0 bounces there.
* **Point winner (101 points, pooled out-of-fold):** the best variant feeds the new tracks plus their own bounces.
  First-call accuracy is EU S6 67.3% (baseline 67.3%) and US S6 64.4% (baseline 59.4%). Taps/100 are 87.1/88.1 EU and
  91.1/92.1 US, against 86.1/87.1 and 90.1/91.1, so one tap more. That is a tie in Europe and +5 points in the US, but
  with 101 points one standard error is about 5 points, so neither is a proven gain.
* **Speed:** 4 ms/frame on the box in numpy (6 ms on the synthetic stress test), against 0.8 ms for the 2D filter.
  Not yet "a few ms" on the Mac under load, and it needs a vectorised or compiled inner loop before shipping.

Recommendation: keep it as a research arm. It is a clear win on finding the ball (3x the current tracker at the same
precision) and on bounces (some against none). It is not yet a point-winner win. The next step is to feed its 3D
bounces and P(in) into the hard rules, which need precision >= 0.95; see "Next steps".

## Prior art and licences (checked 19 Oct 2026 from the repos' LICENSE files and the GitHub licence API)

| Work | Code / licence | Ship? | What we reuse |
|---|---|---|---|
| TT3D, table tennis 3D from one camera (cogsys-tuebingen/tt3d) | MIT. Its optimiser dependency CasADi is LGPL-3.0 | Code could ship. We don't use it | Idea: bounce-anchored reconstruction (the table contact fixes depth). Our joint bounce fit is the same principle, written in numpy |
| BlurBall (cogsys-tuebingen/blurball) | MIT | Yes | Idea for later: use motion blur length/direction as a velocity measurement |
| MonoTrack, badminton 3D (jhwang7628/monotrack) | Adobe Research licence, non-commercial only | No | Ideas only: physics-based 3D trajectory fit with hit segmentation |
| UpliftingTableTennis (KieDani) | GPL-3.0 | No (copyleft) | Ideas only: learned 2D-to-3D uplifting as a future prior |
| "Where Is The Ball" (CVPRW 2025, broadcast 3D) | No code released | n/a | Ideas only: height-over-plane-point representation |
| TrackNetV3 (qaz812345) | MIT (LICENSE covers code and checkpoints; the API says NOASSERTION) | Code yes | Already used for re-acquire; candidate detector A/B later |
| WASB-SBDT (nttcom) | MIT | Code yes (tennis weights no: unlicensed training data) | Candidate detector A/B later |
| filterpy (rlabbe), Kalman / IMM | MIT | Yes | Not needed: the exact batch fit plus DP replaces a recursive IMM. Kept as a fallback for live streaming |
| LorenzoAicardi/TennisBounceDetector | No licence | No | Nothing |
| Ultralytics YOLO | AGPL-3.0 | Never | Nothing |

Everything in `physics3d` is our own code, built on numpy only.

## Design

1. **Candidates:** the heatmap network's top 6 peaks per frame (threshold 0.10, 7 px NMS) instead of only the best
   one. Real balls are often the second or third peak. `extract_cands.py` runs on CPU, chunked and resumable.
2. **Camera** from the court homography (section 1 below).
3. **Seeds:** every plausible triple of candidates becomes a 3D flight in one linear solve (ballistic DLT). It then
   grows frame by frame with a predicted gate.
4. **Fit:** a robust Levenberg-Marquardt fit with drag and spin, plus a chi-square test and a moving-not-degenerate
   test.
5. **Selection:** the best chain of flights joined by bounces (one joint fit), hits or breaks, by exact dynamic
   programming on a likelihood-ratio score with an MDL charge.
6. **Outputs:** see section 6.

## The maths

### 1. Camera from the court (no extra calibration)

The court homography H (court plane z = 0 -> image) equals K [r1 r2 t] up to scale. Assume square pixels and the
principal point c at the image centre, so K = [[f,0,cx],[0,f,cy],[0,0,1]] and f is the only unknown. Shift H by -c
(H' = T H) and write its first two columns as (a1, b1, c1) and (a2, b2, c2). Then r1 . r2 = 0 and |r1| = |r2| are two
equations that are linear in w = 1/f^2:

    w (a1 a2 + b1 b2) + c1 c2 = 0
    w (a1^2 + b1^2 - a2^2 - b2^2) + (c1^2 - c2^2) = 0

These are solved for w by least squares; if w <= 0 the court is too frontal and we stop. Then [r1 r2 t] = K^-1 H /
|K^-1 h1|, with the sign flipped so the court is in front of the camera (t_z > 0). r3 = r1 x r2, R is snapped to the
nearest rotation (SVD), and z is flipped if needed so the camera centre C = -R^T t is above the court. Uncertainty
comes from Monte Carlo over the court taps (sigma 1.5 px, 200 draws). Real a recorded match view 0 (1280 px frame):
f = 828.7 +- 4.8 px, C = (5.32, -5.85, 2.73) m +- about 5 cm, horizontal field of view 75 deg.

### 2. Ball flight model

State x = (p, v) in court metres. Between contacts

    dv/dt = g - kd |v| v + L kd |v|^2 e(v),   dp/dt = v

with g = (0, 0, -9.81), kd = rho Cd A / (2 m) (tennis: 0.0204 /m; padel preset too), and a Magnus term of scalar
"lift" L along e(v), the unit vector perpendicular to v in the vertical plane through v (L > 0 backspin floats,
L < 0 topspin dips; |L| <= 0.6, Gaussian prior N(0, 0.25^2)). Sideways spin is not modelled (one camera cannot
separate it from depth). Integration: 4th-order Taylor series per step (jerk and snap by central differences of the
closed-form acceleration), steps <= 0.1 s, batched over hypotheses; within 4 mm of RK4 over a 1.7 s flight. RK4 is
used for the final chi-square.

### 3. Seeding: ballistic DLT (exact, linear)

With a known constant acceleration a, p(t) = p0 + v0 t + a t^2 / 2 is linear in theta = (p0, v0). For a pixel
(u, v) and camera rows P1..P3, each image point gives two equations linear in theta:

    (u P3 - P1) [p(t); 1] = 0,  (v P3 - P2) [p(t); 1] = 0.

Three detections give a square 6x6 system, solved in one batched call for every candidate triple
(frames f, f+s, f+2s, s in {2, 3}, all top-k peaks). Drag is handled by re-linearising: a = g - kd |v_mid| v_mid
at the mid-time velocity of the previous solve (3 rounds). An optional anchor row forces the flight through a known
3D point (used by the bounce fit, below). Triples are gated on plausibility (speed 3-75 m/s, height -0.3..10 m,
inside the court +- 8 m), then grown frame by frame with a predicted gate (6 px) and refitted.

### 4. Flight fit and score

Levenberg-Marquardt on reprojection error with Huber IRLS (k = 2.5 sigma, sigma = 2.5 px after tuning), parameters
(p0, v0[, L]) at a re-centred reference frame; covariance (J^T W J)^-1 sigma^2. A flight is accepted only if its
chi-square passes (p >= 1e-3), its rms <= 2 sigma, it moves (image extent >= 8 px, median step >= 1 px/frame:
flights aligned with a camera ray are degenerate) and it is physically possible. Its score is a likelihood ratio
against clutter,

    S = sum_inliers [ log(p_d W H / (2 pi sigma^2 lambda_c)) - r^2 / (2 sigma^2) + log peak ]
        + sum_missed log(1 - p_d) - (n_par / 2) log(n_inliers)          (MDL / BIC charge)

with lambda_c the clutter rate per frame (2 false peaks per frame) and p_d = 0.75.

### 5. Events and global selection

Two consecutive flights a, b are joined by one of:

* bounce: ONE joint 11-parameter fit (a: p, v, L; outgoing: v, L) in which flight b starts exactly at a's
  z = 0 root. The ground contact anchors depth, which a single monocular flight leaves loose (spin and gravity trade
  off against distance). Initialised at the pixel where the two image tracks meet, back-projected onto z = 0.
  Accepted only if chi-square passes, rms <= 2 sigma, the contact time is inside the gap, and restitution
  e = -v_z+/v_z- and horizontal ratio mu = |v_xy+|/|v_xy-| are in the ball's range (tennis e 0.55-0.9, mu
  0.45-1.0, +- 0.1). Bonus +4.
* hit: the two image tracks meet within 18 px (bonus 1.5, +1 near a player box).
* break: anything else (penalty 2).

A post-bounce flight may be short (4-5 inliers), since the receiver often hits it quickly. The chain is the maximum
of sum(flight scores) + sum(join terms) over non-overlapping flights, solved exactly by weighted-interval dynamic
programming. If the best chain scores <= 0 nothing is reported: on pure clutter the tracker says "no ball".

### 6. Outputs

Per frame: pixel and court xyz from the fitted flight (inlier frames conf 1, physics-filled frames 0.6; nothing
outside an accepted flight). Per bounce: frame time (sub-frame), court xy, its 2x2 covariance by Monte Carlo over the
joint-fit posterior, P(in) against the doubles court, e and mu. Per hit: time, near-player flag.

## Results

### Synthetic (30 rallies, 1.5 px noise; `physball_synth_ab.py`)

| Condition | Arm | Frame recall (5 px) | Precision | Bounces found | False bounces | Bounce error median / p90 | Timing error | ms/frame |
|---|---|---|---|---|---|---|---|---|
| clean | 2D filter | 0.63 | 0.997 | 37/120 | 39 | 0.37 / 0.61 m | 0.30 fr | 0.1 |
| clean | physics3d | **0.78** | 0.997 | **59/120** | **0** | **0.06 / 0.19 m** | 0.05 fr | 6.1 |
| 50% false + 8-frame gap | 2D filter | 0.38 | 0.998 | 14/120 | 26 | 0.50 / 0.90 m | 0.28 fr | 0.1 |
| 50% false + 8-frame gap | physics3d | **0.72** | 0.987 | **47/120** | **0** | **0.07 / 0.24 m** | 0.06 fr | 6.2 |
| 70% false + 8-frame gap | 2D filter | 0.20 | 0.978 | 3/120 | 8 | 0.19 / 0.21 m | 0.20 fr | 0.0 |
| 70% false + 8-frame gap | physics3d | **0.72** | 0.986 | **42/120** | **0** | **0.06 / 0.25 m** | 0.06 fr | 7.2 |

In debugging, missed bounces were mostly the last bounce of a rally, or bounces followed by a hit within 3-4 frames, where the
outgoing flight is too short to fit. Unit tests check exact DLT recovery, Taylor vs RK4 (< 4 mm), clean-rally bounces,
50% false detections plus a gap, clutter only (nothing reported), a non-physical track (rejected) and P(in).


Private-match evaluation numbers are not published.

### Tricks, each A/B tested against the plain version

| Trick | Plain version | Result | Kept? |
|---|---|---|---|
| top-6 candidates | top-1 only | held-out 3D recall 0.79 vs 0.45 | yes |
| joint bounce fit | flights joined by hits/breaks only | 0.53 bounces/point vs 0; 3D recall 0.79 vs 0.71 | yes |
| Magnus lift | gravity + drag only | held-out recall/precision within 0.01, bounce precision 0.37 vs 0.35 | yes (no measurable cost; revisit) |
| Taylor 4th-order integrator | RK4 | < 4 mm apart over 1.7 s (unit test); with batched Jacobians 50-90 -> 8-12 ms/frame | yes (RK4 for the final chi-square) |
| LM 8 iterations | 12 iterations | held-out identical; train bounces/point 0.76 vs 0.78; 3.8 vs 5.2 ms/frame | yes |
| sigma 2.5 px (tuned on 1-50) | sigma 1.5 px | train bounces 5 vs 2 at the same recall; held-out identical | yes |
| MDL charge + empty-chain rule + ray-degeneracy test | none | clutter-only input: nothing reported vs spurious flights (unit test) | yes |
| closed-form acceleration | numpy norm calls | identical to 1e-14; with the bounce-init early stop, synthetic 8.3 -> 6.5 ms/frame | yes |

## Caveats

* The precision numbers are proxies. SwingVision's own 3D points have errors of their own; 1.5 m is a generous
  tolerance. There are no hand ball labels.
* One match, one court, two halves with different camera views.
* Points 1-50 use the manual view-0 court; 51-101 use the stored per-clip court, which is correct for those views
  (checked visually).
* Candidates come only from our heatmap network. RF-DETR ball boxes are not yet used as candidates.
* Mac memory sat at 15-21% free during the run because of RF-DETR training (8.8 GB) and Safari. Candidate
  extraction (about 0.9 GB) was moved to the box once the Mac dropped below 20%. Box and Mac candidates were checked
  to be identical (max 2e-4 px).
* Speed: 4-7 ms/frame in numpy on the box; 15-19 ms/frame on the loaded Mac with the earlier, slower code. The inner loops (Taylor propagation,
  joint bounce LM) need vectorising or a compiled path before this can run live.
* A single flight's depth is only weakly fixed without a bounce (spin vs distance). The 3D of the last flight of a
  point is the least reliable part.

## Next steps

1. In/out rules need >= 10 decisive final bounces on 1-50 to pass the gate; more footage (or more found final bounces)
   is the way there, not a lower bar.
2. RF-DETR ball boxes once a checkpoint with a ball class exists; detector A/B with TrackNetV3/WASB code (MIT).
3. Speed: batch the bounce LM over all candidate joins, cache flight states, and try numba (BSD).
4. Focal self-calibration from bounces (refine f and the camera height using the fitted flights).
5. Rebuild the a recorded match clip data with `--court_override` so the pseudo-label teacher also uses the right view-0 court.

## How to run

    # candidates (CPU, resumable)
    python tools/ball_tracker/extract_cands.py --weights <heatnet.pt> --clips <data> --out <cands> --threads 3
    # A/B on held-out (tune with --split train --grid grid.json first)
    python tools/ball_tracker/physball_eval.py --data <data> --cands <cands> ... --split heldout --cfg '{"sigma": 2.5}' --out ab.json
    # hybrid tracks for the point-winner rerun
    python tools/ball_tracker/physball_eval.py ... --split all --write_v3 <out> --v3 <v3_own_ball> --write_arm p3
    # synthetic A/B
    python tools/ball_tracker/physball_synth_ab.py --n 30 --out synth_ab.json
