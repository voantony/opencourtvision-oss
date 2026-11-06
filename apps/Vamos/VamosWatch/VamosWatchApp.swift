import SwiftUI
import WatchConnectivity
import WatchKit

@main
struct VamosWatchApp: App {
    @WKApplicationDelegateAdaptor(WatchDelegate.self) var delegate
    var body: some Scene {
        WindowGroup { ScoreView() }
    }
}

final class WatchDelegate: NSObject, WKApplicationDelegate, WCSessionDelegate {
    func applicationDidFinishLaunching() {
        if WCSession.isSupported() {
            WCSession.default.delegate = self
            WCSession.default.activate()
        }
    }
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {}
#if os(iOS)
    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) { session.activate() }
#endif
}

struct ScoreView: View {
    @State private var us = 0
    @State private var them = 0
    var body: some View {
        VStack(spacing: 8) {
            Text("Vamos").font(.headline)
            HStack {
                Button("Us") { us += 1; send(team: "us") }.buttonStyle(.borderedProminent)
                Button("Them") { them += 1; send(team: "them") }.buttonStyle(.bordered)
            }
            Text("\(us) – \(them)").font(.title2.monospacedDigit())
            HStack {
                Button("Undo") { send(kind: "undo") }
                Button("★") { send(kind: "highlight") }
            }.font(.caption)
            HStack {
                Button("REC") { send(kind: "record_start") }
                Button("Stop") { send(kind: "record_stop") }
            }.font(.caption2)
        }.padding()
    }
    func send(team: String? = nil, kind: String = "point") {
        var msg: [String: Any] = ["kind": kind, "t_s": Date().timeIntervalSince1970]
        if let team { msg["team"] = team }
        if WCSession.default.isReachable {
            WCSession.default.sendMessage(msg, replyHandler: nil, errorHandler: nil)
        } else {
            WCSession.default.transferUserInfo(msg)
        }
    }
}
