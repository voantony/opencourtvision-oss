# Record-screen framing check

Setup is the biggest adoption hurdle. Before recording, and live while recording, Vamos checks that every court line and
both teams are in frame, and warns when the phone moves, is tilted or is too low. Each check gives one verdict and one
short instruction.

Code: `opencourtvision/framing/` (`check.py` checker, `synth.py` failure synthesis). Tests: `tests/test_framing_check.py`.
Bench: `benchmarks/framing_check_dev_match.py` → `benchmarks/results/framing_check_dev_match*.json` (aggregate numbers only).

## v3: make do with imperfect framing (21 Oct, Antony: "sometimes the framing won't be perfect, just make do")

The app has to work with imperfect framing and not nag. Each of the five checks is at one of three levels
(`opencourtvision/framing/status.py`):
- **ok**: nothing to show.
- **hint**: quiet and dismissable, and it never blocks recording. Once the user dismisses a check's hint, that hint
  isn't shown again this session. The check still reports `level: "hint", dismissed: true`, so it can be logged.
- **warn**: firm, and only used when scoring would really break.

Nothing refuses to record or analyse: `blocks_recording` is always false.

| check | firm `warn` only when | `hint` (at most) when | example text |
|---|---|---|---|
| `steady` | the phone moved **while recording** (> 1% of the frame diagonal, or the court can't be re-found) | the phone moved after setup but before recording (the session normally just follows the court then) | warn: "The phone moved. Put it back where it was, or tap Re-lock if the new spot is good." |
| `baselines` | **both** baselines missing (each under half in the picture) | a corner or baseline partly cut off (a partly cut-off near corner) | hint: "Turn the phone right if you can; the near baseline is partly cut off." / warn: "Move the phone back or switch to 0.5x: neither baseline is in the picture, so points can't be scored." |
| `lines` | never | lines hard to see (smudge, glare, fog). A cut sideline always comes with a cut corner, which is already the baselines hint | "The court lines are hard to see. Wiping the lens or avoiding glare will help." |
| `level` | never | tilt (any amount) or a low phone | "The phone is tilted about 8°. Level it if you can." / "A higher spot would help with far-court calls." |
| `players` | **one** whole team out of frame for `TEAM_WARN_CHECKS` checks in a row (20, about 20 s at 1 Hz) while the other team is seen | a team unseen across the live window (5 s) | hint: "Can't see the far players right now. Fine between points; …" / warn: "The far players have been out of the picture for a while, so points can't be scored. …" |

Rules for what's shown:
- Both teams missing counts as a break, changeover or warm-up, so there's no message.
- Missing players aren't reported while the picture is soft or the phone moved (one cause, one message).
- One message is shown at a time: any warn first, then the first hint that hasn't been dismissed, in the table's order.
- `fix_check` names the check, so the UI can dismiss exactly that hint.

**Engine quality signal (`opencourtvision/framing/quality.py`).** Every check records numbers only:
- court visible fraction
- per-half visibility
- baseline in-frame fractions
- line support, roll and elevation
- moved, blurred and tracked flags

`FramingSession.quality()` returns `vamos.framing_quality.v1` with timestamped samples and a summary. The app saves it
with the video and passes it to `app.analyze(..., framing=...)` (CLI: `--framing`). The engine stores
`doc["framing_quality"]` and gives every point `point["framing"]`:

| field | how later stages use it |
|---|---|
| `court_visible`, `vis_near`, `vis_far` | `vis_near`/`vis_far` feed `robust.framing.point_evidence`: a side under 50% visible gives no vision evidence, and a partly visible side counts less |
| `reliable` | the share of samples that were neither moved nor blurred |
| `vision_mode` | full / partial / audio from `robust.framing.vision_mode` |
| `uncertainty_scale` | 1.0 for a full, steady court, growing to 4 as visible × reliable drops; later stages widen their spread by it |

A badly framed point is still analysed. It is just scored with less confidence, and it's never dropped.
`attach_to_points` also pulls each point's winner, bounce and player-ID confidence toward 0.5 by its
`uncertainty_scale` (`0.5 + (c - 0.5) / scale`), leaving user-confirmed calls alone, so weakly framed points rise in the
fix-screen queue instead of being refused.


Evaluation numbers from private matches are not published. Use the synthetic benches under `benchmarks/`.

## What it outputs (v1 verdicts, still in Verdict.warnings)

`FramingChecker(size, players_per_side).lock(ref_frame, H)` once the court fit exists, then `check(frame, person_boxes)` →

| verdict (priority order) | when | hint |
|---|---|---|
| `camera_moved` | the court points visible in frame moved > 1% of the frame diagonal (≈7 px at 640×360) from the fit locked at setup, or the court can't be re-found | "phone moved: put it back or tap re-lock" |
| `court_cut_off` | a doubles-court corner is outside the frame (`cut.corners`, `cut.sides`, `cut.edges`) | left+right or top+bottom out → "move phone back (or switch to 0.5x)"; top → "tilt up"; bottom → "tilt down"; one side → "turn phone left/right" |
| `too_low_or_tilted` | horizon roll > 4°, or elevation seen at the far baseline < 3.5° | "level the phone (rotate N° …)" / "raise phone" |
| `far_team_missing` / `near_team_missing` | nobody stands in that half's play zone (court + 4 m sides, 6.5 m behind the baseline). If the half itself is mostly out of frame, this is reported as `court_cut_off` instead | "far players not in view …" |
| `ok` | none of the above | "looks good" |

`warnings` lists every active verdict. In live mode, team presence is judged over a 2 s window and a warning has to hold
for 3 checks (`shown`) before the UI shows it, so a player hidden for one frame doesn't make the screen blink.

### How it works (no learned model in the checker)
- **Geometry from the court fit H (court → image):** project the four corners, test them against the frame with a 0.5% margin. The roll is the angle of the ground plane's horizon, `H⁻ᵀ·(0,0,1)`, which is independent of where the phone points. The elevation comes from foreshortening at the far baseline: a 1 m ground patch looks sin(elevation) times shorter along the line of sight, so the test doesn't depend on resolution or zoom. DLT fits can come out with w < 0, so `normalise_H` fixes the sign.
- **Camera movement:** ORB features of the locked frame are matched to the current frame, with player boxes masked out. An 8-dof RANSAC homography G gives `H_now = G·H_lock`, and drift is the mean displacement of the in-frame court grid between `H_lock` and `H_now`. The similarity transfer already on main (`ball.geometry.transfer_homography`) is available as `MOTION="similarity"`. It scored the same on drift but was 2.5× slower here, so the homography is the default.
- **Players:** the feet point (box bottom centre) of each person box is mapped to court metres through H⁻¹ and counted per half inside the play zone. Spectators on benches fall outside the zone. Any detector works: Apple Vision on iPhone, RF-DETR on the Mac.
- **Reuse from main:** `pose.court` (the court frame), `robust.framing.side_visible_fraction` (half visibility), `ball.geometry.transfer_homography` (the similarity option), and the robust-camera thresholds as the starting point.


## Evaluation

Numbers from private matches are not published here.

## iPhone feasibility (Core ML / Vision)

| piece | iPhone implementation | cost estimate |
|---|---|---|
| frame | `AVCaptureVideoDataOutput`, downscale to 640×360 luma (vImage); check at 2–5 Hz, not every frame | <1 ms |
| person boxes | `VNDetectHumanRectanglesRequest` (system framework, no model to ship). Full frame alone missed far players on the Mac bench (1% far recall), so also run a far-court ROI crop (92% far recall, 0 false players in the ensemble). Alternative: RF-DETR Nano in Core ML (Apache-2.0, 97/93% recall), conversion not done yet | Vision ≈ 31 ms full + ROI ≈ 88 ms per call on the M2 Pro bench (including request overhead). On A15+ at 2 Hz the ANE keeps this off the CPU |
| drift | `VNHomographicImageRegistrationRequest` (built in, returns the 3×3 warp to the locked frame) in place of ORB; or OpenCV ORB (Apache-2.0) at 640×360 | registration ~5–15 ms (estimate); ORB ~15–30 ms on A14+ CPU (estimate from 18–26 ms on the box) |
| geometry, roll, elevation, counts | ~40 lines of Swift with simd | µs |

Estimated budget: ~25–40 ms of work per check at 2 Hz. That's well under 10% of one core plus brief ANE use, so it fits
next to 4K/60 recording on iPhone 12 and newer. Older phones can check at 1 Hz.

### SwiftUI overlay sketch (UNCOMPILED: never built or run, for shape only)

```swift
// UNCOMPILED SKETCH. FramingModel calls the engine at ~1 Hz on the capture queue and publishes vamos.framing.v2.
struct FramingCheck: Decodable { let level: String; let fix: String?; let dismissed: Bool }
struct FramingResult: Decodable {
    let status: String                       // "ok" | "hint" | "warn"
    let fix: String?                         // the one message to show now
    let fix_check: String?                   // which check it belongs to (dismiss target)
    let checks: [String: FramingCheck]       // steady, baselines, lines, level, players
}

struct FramingOverlay: View {
    @ObservedObject var model: FramingModel  // .result, .courtPath, .isRecording, .dismiss(check:), .relock()
    var body: some View {
        ZStack(alignment: .top) {
            model.courtPath.stroke(model.result?.status == "warn" ? Color.red : Color.green.opacity(0.7), lineWidth: 2)
            if let r = model.result, let fix = r.fix {
                if r.status == "warn" {                       // firm: banner + action, stays until fixed
                    VStack(spacing: 6) {
                        Text(fix).font(.headline)
                        if r.fix_check == "steady" { Button("Re-lock") { model.relock() } }
                    }
                    .padding(12).background(.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 12)).padding(.top, 12)
                } else {                                      // hint: small, quiet, one tap to dismiss for this session
                    HStack {
                        Image(systemName: "lightbulb")
                        Text(fix).font(.footnote)
                        Button { model.dismiss(check: r.fix_check) } label: { Image(systemName: "xmark") }
                    }
                    .padding(8).background(.ultraThinMaterial, in: Capsule()).padding(.top, 12)
                }
            }
        }
        .onChange(of: model.result?.checks["steady"]?.level) { lv in
            if lv == "warn" && model.isRecording { UINotificationFeedbackGenerator().notificationOccurred(.warning) }
        }
    }
}
// The record button is never disabled by framing. After recording, session.quality() is saved next to the video and
// passed to the engine (analyze --framing), which widens its uncertainty instead of refusing.
```

## Licences
No new dependencies. Uses numpy (BSD-3) and opencv-python (Apache-2.0), both already in `dev` / `reel` extras. The bench
reads detector outputs the Mac bench produced earlier (RF-DETR: Apache-2.0; Apple Vision: system framework). No AGPL
code is imported.

v2 adds no runtime dependency. The v2 bench runs RF-DETR Nano (`rfdetr`, Apache-2.0) from a separate box-only venv; it
is not imported by the package, and the weights aren't bundled.
