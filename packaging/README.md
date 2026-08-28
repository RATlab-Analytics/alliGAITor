# Packaging

Builds a standalone alliGAITor GUI app with [PyInstaller](https://pyinstaller.org):
`.app` (macOS), `.exe` folder (Windows), and `.AppImage` (Linux). Models and
app data are **not** bundled — the app prompts for a models folder on first
run and stores its own settings in the OS-standard per-user data directory.

## Local build

```bash
python3 -m venv .build_venv && source .build_venv/bin/activate
pip install -r requirements.txt pyinstaller
pyinstaller packaging/alligaitor.spec --noconfirm
```

Output goes to `dist/`: `alliGAITor.app` on macOS, `alliGAITor/` (containing
`alliGAITor.exe`) on Windows, `alliGAITor/` (containing the `alliGAITor`
binary) on Linux. On Linux, wrap the output into an AppImage with:

```bash
./packaging/build_appimage.sh   # needs appimagetool on PATH
```

Build in a clean virtualenv, not a general dev environment — PyInstaller
bundles whatever is importable, and an environment with extra tooling
(Jupyter, matplotlib, etc.) installed will pull those in too and can trip
its single-Qt-binding check.

## CI

`.github/workflows/build.yml` builds all four targets (macOS arm64, Windows
x86_64, Linux x86_64, Linux aarch64) on GitHub-hosted runners and uploads
them as workflow artifacts. Trigger it manually (Actions tab) or by pushing
a `v*` tag.

## Signing

Builds are unsigned. macOS Gatekeeper and Windows SmartScreen will warn
first-time users; code signing needs an Apple Developer ID / a Windows
code-signing certificate, neither of which this workflow has configured.
