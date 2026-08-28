#!/usr/bin/env bash
# Wraps the PyInstaller onedir build (dist/alliGAITor/) into an AppImage.
# Run after `pyinstaller packaging/alligaitor.spec` on Linux. Requires
# appimagetool on PATH (or at $APPIMAGETOOL).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist/alliGAITor"
APPDIR="$REPO_ROOT/dist/AppDir"
APPIMAGETOOL="${APPIMAGETOOL:-appimagetool}"

if [ ! -d "$DIST_DIR" ]; then
    echo "error: $DIST_DIR not found -- run pyinstaller packaging/alligaitor.spec first" >&2
    exit 1
fi

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -r "$DIST_DIR"/* "$APPDIR/usr/bin/"
cp "$REPO_ROOT/packaging/linux/alliGAITor.desktop" "$APPDIR/usr/share/applications/"
cp "$REPO_ROOT/packaging/linux/alliGAITor.desktop" "$APPDIR/"
cp "$REPO_ROOT/packaging/icons/alligaitor_256.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/alligaitor.png"
cp "$REPO_ROOT/packaging/icons/alligaitor_256.png" "$APPDIR/alligaitor.png"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "$HERE/usr/bin/alliGAITor" "$@"
EOF
chmod +x "$APPDIR/AppRun"

ARCH="$(uname -m)" "$APPIMAGETOOL" "$APPDIR" "$REPO_ROOT/dist/alliGAITor-$(uname -m).AppImage"
