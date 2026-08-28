# PyInstaller spec for the alliGAITor GUI. Build with:
#   pyinstaller packaging/alligaitor.spec --noconfirm
# Run from the repo root, or pass --distpath/--workpath to relocate output.

import sys
from importlib import metadata as _im
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

REPO_ROOT = Path(SPECPATH).resolve().parent
GUI_DIR = REPO_ROOT / "gui"
TOOLS_DIR = REPO_ROOT / "tools"
ICONS_DIR = REPO_ROOT / "packaging" / "icons"

# PyInstaller doesn't bundle a package's dist-info by default, which breaks
# two things: the About dialog's importlib.metadata.version() lookups, and
# -- less obviously -- some packages (e.g. imageio, pulled in transitively by
# sleap_io) call importlib.metadata.version() on *themselves* at import time
# to set __version__, and crash outright if their own metadata is missing.
# Bundling every installed distribution's metadata (cheap: just small text
# files) avoids chasing these one crash at a time.
metadata_datas = []
for dist_name in {d.metadata.get("Name") for d in _im.distributions() if d.metadata.get("Name")}:
    try:
        metadata_datas += copy_metadata(dist_name)
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
# sleap_io lazy-loads its io/model/codecs/rendering submodules via the
# lazy_loader package (see sleap_io/__init__.py's lazy.attach() call), which
# resolves them dynamically at runtime through __getattr__ -- invisible to
# PyInstaller's static import analysis, so they're never otherwise bundled.
hiddenimports += collect_submodules("sleap_io")

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
            "CFBundleShortVersionString": "1.0.1",
            "NSHighResolutionCapable": True,
        },
    )
