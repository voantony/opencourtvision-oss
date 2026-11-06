import Foundation

/// Free-core session mirror of `vamos.session.v1`. No account required to import / auto-edit.
public struct VamosSession: Codable, Identifiable {
    public var id: String
    public var videoPath: String
    public var mode: String
    public var teams: [String]
    public var finalScore: String?
    public var shareOptIn: Bool
    public var points: [VamosPoint]
    public var review: [ReviewItem]
    public var qualityHints: [String]
    public var nFixes: Int = 0
    public var modelExamples: Int = 0
    public var modelLastTrained: String?
    public var modelAccuracy: Double?
    public var modeNote: String?
    public init(id: String, videoPath: String, mode: String = "match", teams: [String] = ["Team A", "Team B"],
                finalScore: String? = nil, shareOptIn: Bool = false, points: [VamosPoint] = [],
                review: [ReviewItem] = [], qualityHints: [String] = []) {
        self.id = id; self.videoPath = videoPath; self.mode = mode; self.teams = teams
        self.finalScore = finalScore; self.shareOptIn = shareOptIn; self.points = points
        self.review = review; self.qualityHints = qualityHints
    }
}

public struct VamosPoint: Codable, Identifiable {
    public var id: Int
    public var start: Double
    public var end: Double
    public var winner: String?
    public var favorite: Bool
    public var winnerConfidence: Double?
    public var winnerSource: String?
    public var hits: Int?
    public var duration: Double { max(0, end - start) }
    public init(id: Int, start: Double, end: Double, winner: String? = nil, favorite: Bool = false,
                winnerConfidence: Double? = nil, winnerSource: String? = nil, hits: Int? = nil) {
        self.id = id; self.start = start; self.end = end; self.winner = winner; self.favorite = favorite
        self.winnerConfidence = winnerConfidence; self.winnerSource = winnerSource; self.hits = hits
    }
}

public struct ReviewItem: Codable, Identifiable {
    public var id: String
    public var kind: String
    public var prompt: String
    public var confidence: Double
    public var options: [String]
    public var point: Int?
    public var t: Double?
    public var current: String?
    public var hardCondition: Bool
    public init(id: String, kind: String, prompt: String, confidence: Double, options: [String] = [],
                point: Int? = nil, t: Double? = nil, current: String? = nil, hardCondition: Bool = false) {
        self.id = id; self.kind = kind; self.prompt = prompt; self.confidence = confidence; self.options = options
        self.point = point; self.t = t; self.current = current; self.hardCondition = hardCondition
    }
}

// MARK: - Mapping from the engine's session.json (Python field names)

private func dbl(_ v: Any?) -> Double? {
    if let d = v as? Double { return d }
    if let i = v as? Int { return Double(i) }
    if let n = v as? NSNumber { return n.doubleValue }
    return nil
}

private func str(_ v: Any?) -> String? {
    if let s = v as? String { return s }
    if let n = v as? NSNumber { return n.stringValue }
    if let d = v as? [String: Any], let x = dbl(d["x"]), let y = dbl(d["y"]) { return "(\(Int(x)), \(Int(y)))" }
    return nil
}

extension VamosSession {
    public init(engineJSON d: [String: Any]) throws {
        guard let id = d["id"] as? String else { throw EngineError(message: "session.json has no id") }
        let pts: [VamosPoint] = ((d["points"] as? [[String: Any]]) ?? []).compactMap { p in
            guard let pid = (p["id"] as? Int) ?? dbl(p["id"]).map(Int.init) else { return nil }
            return VamosPoint(id: pid, start: dbl(p["A"]) ?? dbl(p["start"]) ?? 0, end: dbl(p["B"]) ?? dbl(p["end"]) ?? 0,
                              winner: str(p["winner"]), favorite: (p["favorite"] as? Bool) ?? false,
                              winnerConfidence: dbl(p["winner_confidence"]), winnerSource: str(p["winner_src"]),
                              hits: (p["hits"] as? Int))
        }
        let rev: [ReviewItem] = ((d["review"] as? [[String: Any]]) ?? []).compactMap { r in
            guard let rid = r["id"] as? String else { return nil }
            return ReviewItem(id: rid, kind: (r["kind"] as? String) ?? "?", prompt: (r["prompt"] as? String) ?? "",
                              confidence: dbl(r["confidence"]) ?? 0.5, options: (r["options"] as? [String]) ?? [],
                              point: (r["point"] as? Int), t: dbl(r["t"]), current: str(r["current"]),
                              hardCondition: (r["hard_condition"] as? Bool) ?? false)
        }
        let q = d["quality"] as? [String: Any]
        let hints = (q?["hints"] as? [String]) ?? (q?["struggle_tags"] as? [String]) ?? []
        let pm = d["personal_model"] as? [String: Any]
        let ev = pm?["eval"] as? [String: Any]
        self.init(id: id, videoPath: (d["video"] as? String) ?? "", mode: (d["mode"] as? String) ?? "match",
                  teams: (d["teams"] as? [String]) ?? ["Team A", "Team B"], finalScore: d["final"] as? String,
                  shareOptIn: (d["share_opt_in"] as? Bool) ?? false, points: pts, review: rev, qualityHints: hints)
        self.nFixes = (d["n_fixes"] as? Int) ?? 0
        self.modelExamples = (pm?["n_examples"] as? Int) ?? 0
        self.modelLastTrained = pm?["last_trained"] as? String
        self.modelAccuracy = dbl(ev?["acc"])
        self.modeNote = d["mode_note"] as? String
    }
}

public enum VamosTier: String {
    case free, pro
}

public struct FeatureGate {
    public static let free: Set<String> = [
        "record", "import", "auto_edit", "scoring", "basic_stats", "fix_screen",
        "personal_model", "highlights", "watch_scoring", "share_fixes"
    ]
    public static let proPlaceholders: Set<String> = [
        "cloud_storage", "advanced_analytics", "coach_tools", "club_licence", "premium_styles"
    ]
    public static func isFree(_ id: String) -> Bool { free.contains(id) }
}
