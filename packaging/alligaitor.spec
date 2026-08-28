# PyInstaller spec for the alliGAITor GUI. Build with:
#   pyinstaller packaging/alligaitor.spec --noconfirm
# Run from the repo root, or pass --distpath/--workpath to relocate output.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

REPO_ROOT = Path(SPECPATH).resolve().parent
GUI_DIR = REPO_ROOT / "gui"
TOOLS_DIR = REPO_ROOT / "tools"
ICONS_DIR = REPO_ROOT / "packaging" / "icons"

# The About dialog reads dependency versions via importlib.metadata, which
# needs each package's dist-info bundled explicitly -- PyInstaller doesn't
# include it by default. sleap-nn is optional (run as a subprocess, not
# imported), so it's skipped if not installed in the build environment.
METADATA_PACKAGES = [
    "aniposelib", "sleap-nn", "sleap-io", "numpy", "pandas", "openpyxl",
    "PyYAML", "PySide6", "opencv-python", "imageio-ffmpeg",
]
metadata_datas = []
for pkg in METADATA_PACKAGES:
    try:
        metadata_datas += copy_metadata(pkg)
    except Exception:
        pass

# gui/ and tools/ import each other with flat module names (see gui/__init__.py),
# so PyInstaller's static analysis needs these listed explicitly.
FLAT_MODULES = [
    "about_dialog", "add_job_dialog", "app_settings", "batch_runner",
    "batch_worker_process", "dark_theme", "group_config_dialog", "job_queue",
    "job_table_model", "main_window", "paw_colors", "regex_help",
    "validation_list_dialog", "validation_video_dialog", "video_player_widget",
    "crop_config", "crop_runner", "crop_setup_dialog", "crop_worker_process",
    "frame_utils", "make_config", "video_crop",
]

hiddenimports = list(FLAT_MODULES)
hiddenimports += collect_submodules("alligaitor")

if sys.platform == "darwin":
    icon_file = str(ICONS_DIR / "alligaitor.icns")
elif sys.platform.startswith("win"):
    icon_file = str(ICONS_DIR / "alligaitor.ico")
else:
    icon_file = str(ICONS_DIR / "alligaitor_256.png")

a = Analysis(
    [str(GUI_DIR / "app.py")],
    pathex=[str(REPO_ROOT), str(GUI_DIR), str(TOOLS_DIR)],
    binaries=[],
    datas=[(str(ICONS_DIR / "alligaitor_256.png"), "packaging/icons")] + metadata_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="alliGAITor",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="alliGAITor",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="alliGAITor.app",
        icon=icon_file,
        bundle_identifier="org.ratlab.alligaitor",
        info_plist={
            "CFBundleName": "alliGAITor",
            "CFBundleDisplayName": "alliGAITor",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
