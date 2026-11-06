#!/bin/bash
# Build Vamos.app (Mac) from this SwiftPM package and compile-check the Watch app. Ad-hoc signing only ("-"):
# never signs with a personal Apple ID / developer team.
# Usage: scripts/build_app.sh [dest-dir]   (default: ../app-build next to this package; app lands at <dest>/Vamos.app)
set -euo pipefail
cd "$(dirname "$0")/.."
PKG="$PWD"
DEST="${1:-$PKG/..}"
NICE="nice -n 20"

$NICE swift build -c release --product Vamos
$NICE swift build -c release --product VamosCLI
BIN="$(swift build -c release --show-bin-path)"

APP="$DEST/Vamos.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/Vamos" "$APP/Contents/MacOS/Vamos"
VER="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Vamos</string>
  <key>CFBundleDisplayName</key><string>Vamos</string>
  <key>CFBundleExecutable</key><string>Vamos</string>
  <key>CFBundleIdentifier</key><string>com.opencourtvision.vamos</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>CFBundleVersion</key><string>$VER</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.sports</string>
</dict></plist>
PLIST
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --verbose=1 "$APP"
echo "APP: $APP"

# Watch: compile + link check against the watchOS simulator SDK (no runtime, device or signing needed).
if xcrun --sdk watchsimulator --show-sdk-path >/dev/null 2>&1; then
  OUT="$(mktemp -d)"
  $NICE xcrun --sdk watchsimulator swiftc -parse-as-library -target arm64-apple-watchos10.0-simulator \
    VamosWatch/*.swift -o "$OUT/VamosWatch" && echo "WATCH: compiled for watchOS simulator ($(xcrun --sdk watchsimulator --show-sdk-version))"
else
  echo "WATCH: skipped (watchOS simulator SDK not installed)"
fi
