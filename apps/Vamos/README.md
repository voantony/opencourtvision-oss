# Vamos Mac + Apple Watch (v1 shell)

SwiftUI shell around the on-device Python engine (`python -m opencourtvision.app`).

## Build (Mac, full Xcode 16+; tested with Xcode 27.1)

```
scripts/build_app.sh ~/VamosFootage/app-build    # -> ~/VamosFootage/app-build/Vamos.app (ad-hoc signed, "-")
open ~/VamosFootage/app-build/Vamos.app          # or: --args --session <session dir>
.build/release/VamosCLI [session dir]            # engine-bridge smoke test
```

The script also compile-checks `VamosWatch/` against the watchOS simulator SDK (no simulator runtime, device or signing
needed). Signing is ad-hoc only; nothing is signed with a personal Apple ID or team.

Engine discovery: `VAMOS_PYTHON` or `~/VamosFootage/venv/bin/python` (else `python3`); `VAMOS_ENGINE_ROOT` or
`~/VamosFootage/app-build/engine` is prepended to `PYTHONPATH` so the app can run this branch's `opencourtvision.app`
without touching the venv's own checkout. Sessions go to `~/VamosFootage/app-sessions/`.

## Screens (v1)

- Import: video picker -> `analyze` (optional final score), recording-quality tips. Open an existing session folder.
- Match: auto-cut points table (start, length, shots, winner, source, favourite).
- Fix: least-certain-first queue with one-click answers -> `fix`; refresh -> `review`.
- Highlights: Export 9:16 / 16:9 -> `render` (vamos_clean_v1), Show in Finder / Play.
- Your model: fix/model stats, share opt-in toggle -> `share --on/--off`, Retrain -> `retrain`, Delete -> `share --delete`.
- Pro (Cloud library, Coach tools): placeholders, no payments.

## What Antony needs for a real Watch install

1. Apple Developer account enrolled on the Mac (only if he wants it on his wrist; we don't sign without asking).
2. An Xcode project that wraps `VamosWatch/` (+ an iOS/Mac host) with WatchConnectivity and a signing team.
3. Pair the Apple Watch, pick it as the run destination, install.

## Free vs Pro

Free (no sign-in): import, auto-edit, scoring, fix screen, personal model, highlights, Watch taps.  
Pro (placeholders only in v1): cloud library, coach tools, club licences, premium styles. No payment wiring.
