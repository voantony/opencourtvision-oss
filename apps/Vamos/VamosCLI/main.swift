import Foundation
import VamosShared

// Smoke test for the engine bridge: `swift run VamosCLI` (or `VamosCLI <session-dir>`).
print("Vamos CLI — free core, no sign-in.")
print("Free features: \(FeatureGate.free.sorted().joined(separator: ", "))")
print("Pro placeholders: \(FeatureGate.proPlaceholders.sorted().joined(separator: ", "))")
let bridge = EngineBridge(workRoot: EngineBridge.defaultWorkRoot())
print("python: \(bridge.python)  engine root: \(bridge.engineRoot ?? "(venv default)")")
do {
    let t = try bridge.tiers()
    print("engine OK: \((t["free"] as? [Any])?.count ?? 0) free, \((t["pro"] as? [Any])?.count ?? 0) pro, payments=\(t["payments"] ?? "?")")
} catch {
    print("engine FAILED: \(error.localizedDescription)"); exit(1)
}
if CommandLine.arguments.count > 1 {
    let work = URL(fileURLWithPath: CommandLine.arguments[1])
    do {
        let q = try bridge.reviewQueue(work: work)
        let s = try EngineBridge.loadSession(work: work)
        print("session \(s.id): \(s.points.count) points, \(s.review.count) review items (engine says n=\(q["n"] ?? "?"))")
    } catch { print("session FAILED: \(error.localizedDescription)"); exit(1) }
}
