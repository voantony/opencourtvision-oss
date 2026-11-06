import Foundation

public struct EngineError: LocalizedError {
    public var message: String
    public var errorDescription: String? { message }
}

/// Talks to the on-device Python engine (`python -m opencourtvision.app ...`). Free-tier calls never need a network.
///
/// Discovery (first hit wins):
/// - python: `VAMOS_PYTHON`, then `~/VamosFootage/venv/bin/python`, then `python3` on PATH.
/// - engine root (prepended to PYTHONPATH): `VAMOS_ENGINE_ROOT`, then `~/VamosFootage/app-build/engine`, else whatever
///   `opencourtvision` the venv already imports.
public struct EngineBridge {
    public var python: String
    public var engineRoot: String?
    public var workRoot: URL

    public init(python: String? = nil, engineRoot: String? = nil, workRoot: URL) {
        let env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let fm = FileManager.default
        if let python { self.python = python }
        else if let p = env["VAMOS_PYTHON"], !p.isEmpty { self.python = p }
        else if fm.isExecutableFile(atPath: home + "/VamosFootage/venv/bin/python") { self.python = home + "/VamosFootage/venv/bin/python" }
        else { self.python = "python3" }
        if let engineRoot { self.engineRoot = engineRoot }
        else if let r = env["VAMOS_ENGINE_ROOT"], !r.isEmpty { self.engineRoot = r }
        else if fm.fileExists(atPath: home + "/VamosFootage/app-build/engine/opencourtvision/app/__main__.py") {
            self.engineRoot = home + "/VamosFootage/app-build/engine"
        } else { self.engineRoot = nil }
        self.workRoot = workRoot
    }

    public static func defaultWorkRoot() -> URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("VamosFootage/app-sessions", isDirectory: true)
    }

    public func tiers() throws -> [String: Any] { try run(["tiers"]) }

    public func analyze(video: URL, work: URL, final: String?) throws -> [String: Any] {
        var args = ["analyze", video.path, "--work", work.path]
        if let final, !final.isEmpty { args += ["--final", final] }
        return try run(args)
    }

    public func reviewQueue(work: URL) throws -> [String: Any] { try run(["review", work.path]) }

    public func applyFix(work: URL, item: String, value: String) throws -> [String: Any] {
        try run(["fix", work.path, "--item", item, "--value", value])
    }

    public func render(work: URL, aspect: String, out: URL) throws -> [String: Any] {
        try run(["render", work.path, "--aspect", aspect, "--out", out.path])
    }

    public func retrain(work: URL) throws -> [String: Any] { try run(["retrain", work.path]) }

    /// One-click Highlights only: no scoring, no player ID. preset "social" (9:16 + 16:9 reel + clips) or "recruiting".
    public func highlights(video: URL, out: URL, sport: String, preset: String, title: String? = nil,
                           recruit: [String: String] = [:], mute: Bool = false, score: Bool? = nil,
                           longShot: String = "cut") throws -> [String: Any] {
        var args = ["highlights", video.path, "--out", out.path, "--sport", sport, "--preset", preset]
        if let title, !title.isEmpty { args += ["--title", title] }
        for (k, flag) in [("name", "--name"), ("class_year", "--class-year"), ("ranking", "--ranking"), ("contact", "--contact")] {
            if let v = recruit[k], !v.isEmpty { args += [flag, v] }
        }
        if mute { args.append("--mute") }
        if let score { args += ["--score", score ? "on" : "off"] }      // nil = the preset's default
        args += ["--long-shot", longShot]
        return try run(args)
    }

    public func setShare(work: URL, on: Bool) throws -> [String: Any] {
        try run(["share", work.path, on ? "--on" : "--off"])
    }

    public func deleteMyData(work: URL) throws -> [String: Any] { try run(["share", work.path, "--delete"]) }

    /// Reads `session.json` directly (no subprocess).
    public static func loadSession(work: URL) throws -> VamosSession {
        let data = try Data(contentsOf: work.appendingPathComponent("session.json"))
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw EngineError(message: "session.json is not an object")
        }
        return try VamosSession(engineJSON: obj)
    }

    private func run(_ args: [String]) throws -> [String: Any] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = [python, "-m", "opencourtvision.app"] + args
        var env = ProcessInfo.processInfo.environment
        if let engineRoot {
            env["PYTHONPATH"] = [engineRoot, env["PYTHONPATH"]].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: ":")
        }
        // GUI apps get a minimal PATH; make Homebrew ffmpeg/ffprobe visible to the engine.
        env["PATH"] = ["/opt/homebrew/bin", "/usr/local/bin", env["PATH"] ?? "/usr/bin:/bin"].joined(separator: ":")
        p.environment = env
        try? FileManager.default.createDirectory(at: workRoot, withIntermediateDirectories: true)
        p.currentDirectoryURL = workRoot
        let out = Pipe(); let err = Pipe()
        p.standardOutput = out; p.standardError = err
        try p.run()
        // Drain stderr concurrently so a chatty engine can't fill the pipe and deadlock.
        var errData = Data()
        let errQueue = DispatchQueue(label: "vamos.engine.stderr")
        let group = DispatchGroup()
        group.enter()
        errQueue.async { errData = err.fileHandleForReading.readDataToEndOfFile(); group.leave() }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit(); group.wait()
        let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        if let obj, let e = obj["error"] as? String {
            throw EngineError(message: "engine: \(e)")
        }
        guard p.terminationStatus == 0, let obj else {
            let tail = String(decoding: errData.suffix(800), as: UTF8.self)
            throw EngineError(message: "engine exited \(p.terminationStatus): \(tail.isEmpty ? "non-JSON output" : tail)")
        }
        return obj
    }
}
