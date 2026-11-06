// swift-tools-version: 5.9
// SPDX-License-Identifier: AGPL-3.0-only
// Vamos Mac + Watch companion. `swift build --product Vamos` builds the Mac app binary; scripts/build_app.sh wraps it
// into an ad-hoc-signed Vamos.app. The Watch app is compile-checked against the watchOS simulator SDK by the same script
// (a real Watch install needs an Xcode project + signing team, which we don't do without Antony's go-ahead).
import PackageDescription

let package = Package(
    name: "Vamos",
    platforms: [.macOS(.v14), .iOS(.v17), .watchOS(.v10)],
    products: [
        .library(name: "VamosShared", targets: ["VamosShared"]),
        .executable(name: "VamosCLI", targets: ["VamosCLI"]),
        .executable(name: "Vamos", targets: ["Vamos"]),
    ],
    targets: [
        .target(name: "VamosShared", path: "VamosShared"),
        .executableTarget(name: "VamosCLI", dependencies: ["VamosShared"], path: "VamosCLI"),
        .executableTarget(name: "Vamos", dependencies: ["VamosShared"], path: "Vamos"),
    ]
)
