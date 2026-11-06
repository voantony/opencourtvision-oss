import AppKit
import UniformTypeIdentifiers
import SwiftUI
import VamosShared

@main
struct VamosApp: App {
    @StateObject private var model = AppModel()
    init() {
        // Launched as a bare SwiftPM executable inside a hand-made bundle: make sure we come to the front with a Dock icon.
        NSApplication.shared.setActivationPolicy(.regular)
    }
    var body: some Scene {
        WindowGroup("Vamos") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 960, minHeight: 640)
                .onAppear { NSApplication.shared.activate(ignoringOtherApps: true); model.start() }
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Import Match Video…") { model.chooseVideo() }.keyboardShortcut("o")
                Button("Open Session Folder…") { model.chooseSession() }.keyboardShortcut("o", modifiers: [.command, .shift])
            }
        }
    }
}

enum Pane: Hashable { case importing, match, fix, highlights, model, pro(String) }

@MainActor
final class AppModel: ObservableObject {
    @Published var pane: Pane = .importing
    @Published var engineStatus = "Checking engine…"
    @Published var engineOK = false
    @Published var busy: String?
    @Published var lastError: String?
    @Published var lastMessage: String?
    @Published var session: VamosSession?
    @Published var work: URL?
    @Published var finalScore = ""
    @Published var lastExport: URL?
    // Highlights only
    @Published var hlSport = "tennis"
    @Published var hlPreset = "social"
    @Published var hlTitle = ""
    @Published var rcName = ""
    @Published var rcClass = ""
    @Published var rcRanking = ""
    @Published var rcContact = ""
    @Published var rcMute = false
    @Published var hlScoreSocial = true          // score overlay per preset (shown only when confirmed / high-confidence)
    @Published var hlScoreRecruit = false
    @Published var hlLongShot = "cut"            // 9:16 long cross-court shots: cut to the receiving side | fast pan
    @Published var hlOut: URL?
    @Published var hlSummary: String?

    let bridge = EngineBridge(workRoot: EngineBridge.defaultWorkRoot())

    func start() {
        let b = bridge
        Task.detached {
            do {
                let t = try b.tiers()
                let n = (t["free"] as? [Any])?.count ?? 0
                await MainActor.run {
                    self.engineOK = true
                    self.engineStatus = "Engine ready · \(n) free features · \(URL(fileURLWithPath: b.python).lastPathComponent)"
                }
            } catch {
                await MainActor.run { self.engineOK = false; self.engineStatus = "Engine not found: \(error.localizedDescription)" }
            }
        }
        // `--session <dir>` (or VAMOS_SESSION) opens an existing session at launch.
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--session"), i + 1 < args.count { open(work: URL(fileURLWithPath: args[i + 1])) }
        else if let s = ProcessInfo.processInfo.environment["VAMOS_SESSION"] { open(work: URL(fileURLWithPath: s)) }
    }

    func open(work: URL) {
        do {
            session = try EngineBridge.loadSession(work: work)
            self.work = work
            finalScore = session?.finalScore ?? ""
            pane = .match
            lastError = nil
        } catch { lastError = "Couldn't open session: \(error.localizedDescription)" }
    }

    func reload() { if let work { open(work: work) } }

    /// Runs an engine call off the main thread, then reloads session.json.
    func engine(_ label: String, reloadAfter: Bool = true, _ call: @escaping @Sendable (EngineBridge) throws -> [String: Any],
                then: (@MainActor ([String: Any]) -> Void)? = nil) {
        guard busy == nil else { return }
        busy = label; lastError = nil; lastMessage = nil
        let b = bridge
        Task.detached {
            do {
                let r = try call(b)
                await MainActor.run {
                    self.busy = nil
                    if reloadAfter { self.reload() }
                    then?(r)
                }
            } catch {
                await MainActor.run { self.busy = nil; self.lastError = error.localizedDescription }
            }
        }
    }

    func chooseVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = false
        panel.message = "Pick a match video. Everything runs on this Mac; no sign-in."
        guard panel.runModal() == .OK, let video = panel.url else { return }
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let work = bridge.workRoot.appendingPathComponent("\(video.deletingPathExtension().lastPathComponent)-\(stamp)")
        let final = finalScore.trimmingCharacters(in: .whitespaces)
        engine("Analysing \(video.lastPathComponent)…", reloadAfter: false, { b in
            try b.analyze(video: video, work: work, final: final.isEmpty ? nil : final)
        }, then: { r in
            self.open(work: work)
            self.lastMessage = "Found \(r["n_points"] ?? 0) points · \(r["n_review"] ?? 0) to check"
        })
    }

    /// One click: pick a video -> reel(s) + best clips in a folder next to the sessions. No scoring, no player ID.
    func highlightsOnly() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = false
        panel.message = "Pick any tennis or padel video: a match, a lesson or a drill."
        guard panel.runModal() == .OK, let video = panel.url else { return }
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let out = bridge.workRoot.appendingPathComponent("highlights/\(video.deletingPathExtension().lastPathComponent)-\(hlPreset)-\(stamp)")
        let sport = hlSport, preset = hlPreset, title = hlTitle, mute = rcMute
        let score = preset == "social" ? hlScoreSocial : hlScoreRecruit, longShot = hlLongShot
        let rec = ["name": rcName, "class_year": rcClass, "ranking": rcRanking, "contact": rcContact]
        let started = Date()
        engine("Making highlights from \(video.lastPathComponent)…", reloadAfter: false, { b in
            try b.highlights(video: video, out: out, sport: sport, preset: preset, title: title, recruit: rec, mute: mute,
                             score: score, longShot: longShot)
        }, then: { r in
            self.hlOut = out
            let ex = r["exports"] as? [String: Any] ?? [:]
            let first = (ex["9:16"] ?? ex["16:9"]) as? [String: Any]
            if let reel = first?["reel"] as? String { self.lastExport = URL(fileURLWithPath: reel) }
            let mins = Int(Date().timeIntervalSince(started) / 60.0 + 0.5)
            let rate = r["minutes_per_hour_of_footage"] as? Double ?? 0
            self.hlSummary = "\(r["rallies"] ?? 0) rallies found · done in ~\(mins) min (\(String(format: "%.1f", rate)) min per hour of video)"
            self.lastMessage = "Highlights ready"
        })
    }

    func chooseSession() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.directoryURL = bridge.workRoot
        panel.message = "Pick a Vamos session folder (contains session.json)."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        open(work: url)
    }

    func fix(_ item: ReviewItem, value: String) {
        guard let work else { return }
        engine("Saving fix…", { b in try b.applyFix(work: work, item: item.id, value: value) })
    }

    func refreshQueue() {
        guard let work else { return }
        engine("Building fix queue…", { b in try b.reviewQueue(work: work) })
    }

    func render(aspect: String) {
        guard let work else { return }
        let out = work.appendingPathComponent("exports/vamos_\(aspect.replacingOccurrences(of: ":", with: "x")).mp4")
        engine("Rendering \(aspect) reel…", { b in try b.render(work: work, aspect: aspect, out: out) }, then: { _ in
            self.lastExport = out; self.lastMessage = "Saved \(out.lastPathComponent)"
        })
    }

    func setShare(_ on: Bool) {
        guard let work else { return }
        engine(on ? "Opting in…" : "Opting out…", { b in try b.setShare(work: work, on: on) })
    }

    func retrain() {
        guard let work else { return }
        engine("Retraining on this Mac…", { b in try b.retrain(work: work) }, then: { r in
            if let ok = r["ok"] as? Bool, !ok { self.lastMessage = (r["reason"] as? String) ?? "Not enough fixes yet" }
            else if (r["rolled_back"] as? Bool) == true { self.lastMessage = "Kept your previous model (new one scored lower)" }
            else { self.lastMessage = "Model updated" }
        })
    }

    func deleteData() {
        guard let work else { return }
        engine("Deleting your training data…", { b in try b.deleteMyData(work: work) }, then: { _ in
            self.lastMessage = "Fixes, taps and personal model deleted"
        })
    }
}

struct ContentView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        NavigationSplitView {
            List(selection: $m.pane) {
                Label("Import", systemImage: "square.and.arrow.down").tag(Pane.importing)
                Label("Match", systemImage: "sportscourt").tag(Pane.match)
                Label("Fix", systemImage: "checkmark.circle")
                    .badge(m.session?.review.count ?? 0).tag(Pane.fix)
                Label("Highlights", systemImage: "film").tag(Pane.highlights)
                Label("Your model", systemImage: "brain").tag(Pane.model)
                Section("Vamos Pro") {
                    Label("Cloud library", systemImage: "cloud").badge("Pro").tag(Pane.pro("Cloud library"))
                    Label("Coach tools", systemImage: "person.2").badge("Pro").tag(Pane.pro("Coach tools"))
                }
            }
            .navigationTitle("Vamos")
            .listStyle(.sidebar)
            .safeAreaInset(edge: .bottom) {
                HStack(spacing: 6) {
                    Circle().fill(m.engineOK ? Color.green : Color.orange).frame(width: 8, height: 8)
                    Text(m.engineStatus).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }.padding(10)
            }
        } detail: {
            VStack(spacing: 0) {
                StatusBar()
                Group {
                    switch m.pane {
                    case .importing: ImportView()
                    case .match: MatchView()
                    case .fix: FixScreenView()
                    case .highlights: HighlightsView()
                    case .model: ModelView()
                    case .pro(let name): ProPlaceholder(name: name)
                    }
                }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
        }
    }
}

struct StatusBar: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        if let busy = m.busy {
            HStack { ProgressView().controlSize(.small); Text(busy) }.padding(8).frame(maxWidth: .infinity).background(.blue.opacity(0.12))
        } else if let e = m.lastError {
            Text(e).foregroundStyle(.red).padding(8).frame(maxWidth: .infinity).background(.red.opacity(0.08)).textSelection(.enabled)
        } else if let msg = m.lastMessage {
            Text(msg).padding(8).frame(maxWidth: .infinity).background(.green.opacity(0.10))
        }
    }
}

struct ImportView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Import a match").font(.largeTitle.bold())
            Text("Free · on your Mac · no sign-in").foregroundStyle(.secondary)
            HighlightsOnlyBox()
            HStack {
                TextField("Final score (optional, e.g. 6-4 3-6 10-7)", text: $m.finalScore).frame(maxWidth: 320)
                Text("Without it Vamos cuts rallies; add it to score the match.").font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("Choose video…") { m.chooseVideo() }.buttonStyle(.borderedProminent).disabled(!m.engineOK || m.busy != nil)
                Button("Open session…") { m.chooseSession() }
            }
            if let s = m.session {
                GroupBox("Current session") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(URL(fileURLWithPath: s.videoPath).lastPathComponent).font(.headline)
                        Text("\(s.points.count) points · \(s.review.count) to check · mode \(s.mode)").foregroundStyle(.secondary)
                        if !s.qualityHints.isEmpty {
                            Text("Recording tips").font(.subheadline.bold()).padding(.top, 4)
                            ForEach(s.qualityHints, id: \.self) { Text("• \($0)").foregroundStyle(.secondary) }
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(4)
                }
            } else {
                Text("• Phone, GoPro or club camera, any angle. Shaky or behind-the-fence footage still works; Vamos tells you what would help.")
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }.padding(32)
    }
}

struct HighlightsOnlyBox: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Button { m.highlightsOnly() } label: {
                    Label("Highlights only", systemImage: "sparkles.tv")
                        .font(.title2.bold()).frame(maxWidth: .infinity, minHeight: 56)
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(!m.engineOK || m.busy != nil)
                .help("One video in, a styled reel and the best clips out. No scoring needed.")
                HStack(spacing: 16) {
                    Picker("Sport", selection: $m.hlSport) { Text("Tennis").tag("tennis"); Text("Padel").tag("padel") }
                        .pickerStyle(.segmented).frame(maxWidth: 220)
                    Picker("Export", selection: $m.hlPreset) { Text("Social reel").tag("social"); Text("Recruiting").tag("recruiting") }
                        .pickerStyle(.segmented).frame(maxWidth: 280)
                }
                if m.hlPreset == "social" {
                    TextField("Title on the intro (optional, e.g. Club final)", text: $m.hlTitle).frame(maxWidth: 420)
                    HStack(spacing: 16) {
                        Toggle("Show score", isOn: $m.hlScoreSocial)
                            .help("Only when the score is confirmed or high-confidence; never guessed.")
                        Picker("Long shots (9:16)", selection: $m.hlLongShot) {
                            Text("Cut to receiver").tag("cut"); Text("Fast pan").tag("fast_pan")
                        }.pickerStyle(.segmented).frame(maxWidth: 360)
                    }
                    Text("9:16 + 16:9 reel and the best clips · broadcast camera: wide rally, reaction close-up.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    HStack {
                        TextField("Player name", text: $m.rcName)
                        TextField("Class of (e.g. 2028)", text: $m.rcClass).frame(maxWidth: 140)
                        TextField("UTR / ranking", text: $m.rcRanking).frame(maxWidth: 160)
                    }.frame(maxWidth: 640)
                    HStack {
                        TextField("Contact (email or phone)", text: $m.rcContact).frame(maxWidth: 320)
                        Toggle("Mute audio", isOn: $m.rcMute)
                        Toggle("Show score", isOn: $m.hlScoreRecruit)
                            .help("Only when the score is confirmed or high-confidence; never guessed.")
                    }
                    Text("3–5 min 16:9 cut for college coaches: title card, serves, net play, best points. No music.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let sum = m.hlSummary, let out = m.hlOut {
                    HStack {
                        Text(sum).font(.callout)
                        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([out]) }
                        if let reel = m.lastExport { Button("Play reel") { NSWorkspace.shared.open(reel) } }
                    }
                }
            }.padding(8)
        }
    }
}

func clock(_ s: Double) -> String { String(format: "%d:%02d", Int(s) / 60, Int(s) % 60) }

struct MatchView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Match").font(.largeTitle.bold())
            if let s = m.session {
                Text("\(s.teams.first ?? "A") vs \(s.teams.dropFirst().first ?? "B") · \(s.finalScore ?? "no final score yet")").font(.title3)
                let total = s.points.reduce(0) { $0 + $1.duration }
                Text("\(s.points.count) points · \(clock(total)) of play · mode \(s.mode)").foregroundStyle(.secondary)
                if let note = s.modeNote { Text(note).font(.caption).foregroundStyle(.secondary) }
                Table(s.points) {
                    TableColumn("#") { Text("\($0.id)") }.width(36)
                    TableColumn("Start") { Text(clock($0.start)) }.width(60)
                    TableColumn("Length") { Text(String(format: "%.0fs", $0.duration)) }.width(60)
                    TableColumn("Shots") { p in Text(p.hits.map(String.init) ?? "–") }.width(50)
                    TableColumn("Winner") { p in
                        Text(p.winner.map { w in teamName(s, w) } ?? "?")
                            .foregroundStyle(p.winnerSource == "user" ? .primary : .secondary)
                    }
                    TableColumn("Source") { p in Text(p.winnerSource ?? "engine").foregroundStyle(.secondary) }.width(70)
                    TableColumn("★") { p in Text(p.favorite ? "★" : "") }.width(28)
                }
            } else {
                Text("Import a video to see the auto-cut points and score.").foregroundStyle(.secondary)
                Spacer()
            }
        }.padding(32)
    }
}

func teamName(_ s: VamosSession, _ w: String) -> String {
    if w == "A", let t = s.teams.first { return t }
    if w == "B", s.teams.count > 1 { return s.teams[1] }
    return w
}

struct FixScreenView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Fix screen").font(.largeTitle.bold())
                Spacer()
                Button("Refresh queue") { m.refreshQueue() }.disabled(m.session == nil || m.busy != nil)
            }
            Text("Least-certain calls first. Hard conditions (low light, shaky, behind-fence) rise to the top. Your answers train your model.")
                .foregroundStyle(.secondary)
            if let s = m.session, !s.review.isEmpty {
                List(s.review) { item in
                    HStack(alignment: .center) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.prompt).font(.headline)
                            Text("\(item.kind.replacingOccurrences(of: "_", with: " ")) · point \(item.point.map(String.init) ?? "–") at \(clock(item.t ?? 0)) · \(Int((item.confidence * 100).rounded()))% sure\(item.hardCondition ? " · hard footage" : "")")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        ForEach(item.options.isEmpty ? ["Confirm", "skip"] : item.options, id: \.self) { opt in
                            Button(item.kind == "point_winner" ? teamName(s, opt) : opt) { m.fix(item, value: opt) }
                                .disabled(m.busy != nil)
                        }
                    }.padding(.vertical, 4)
                }
            } else {
                Text(m.session == nil ? "Import a match first." : "Nothing to fix. Nice.").foregroundStyle(.secondary)
                Spacer()
            }
        }.padding(32)
    }
}

struct HighlightsView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Highlights").font(.largeTitle.bold())
            Text("Style: Vamos Clean · whole points · 0–2 slow-mos · platform sound at share.").foregroundStyle(.secondary)
            HStack {
                Button("Export 9:16") { m.render(aspect: "9:16") }
                Button("Export 16:9") { m.render(aspect: "16:9") }
            }.disabled(m.session == nil || m.busy != nil)
            if let out = m.lastExport {
                HStack {
                    Text(out.lastPathComponent)
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([out]) }
                    Button("Play") { NSWorkspace.shared.open(out) }
                }
            }
            if let s = m.session {
                let favs = s.points.filter(\.favorite)
                Text("\(favs.count) favourites ★ always make the reel.").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }.padding(32)
    }
}

struct ModelView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Your model").font(.largeTitle.bold())
            if let s = m.session {
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(s.nFixes) fixes saved · trained on \(s.modelExamples) examples")
                        Text("Last trained: \(s.modelLastTrained ?? "never")").foregroundStyle(.secondary)
                        if let a = s.modelAccuracy { Text("Held-out accuracy: \(Int((a * 100).rounded()))%").foregroundStyle(.secondary) }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(4)
                }
                Toggle("Share my fixes to improve Vamos for everyone", isOn: Binding(
                    get: { s.shareOptIn }, set: { m.setShare($0) }
                )).disabled(m.busy != nil)
            } else {
                Text("Import a match first. Your fixes train a personal model that stays on this Mac.").foregroundStyle(.secondary)
            }
            Text("Default off. Delete my data is always available. No faces or names leave the device.").foregroundStyle(.secondary)
            HStack {
                Button("Retrain now (on this Mac)") { m.retrain() }
                Button("Delete my training data", role: .destructive) { m.deleteData() }
            }.disabled(m.session == nil || m.busy != nil)
            Spacer()
        }.padding(32)
    }
}

struct ProPlaceholder: View {
    let name: String
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(name).font(.largeTitle.bold())
            Text("Vamos Pro · coming later. Nothing to buy yet; everything free stays free.").foregroundStyle(.secondary)
            Spacer()
        }.padding(32)
    }
}
