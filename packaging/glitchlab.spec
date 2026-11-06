# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: one folder ("onedir") bundle with two executables.

    dist/GlitchLab/GlitchLab(.exe)       windowed application
    dist/GlitchLab/glitchlab-cli(.exe)   console tool (self-test, contact sheet, batch render)
    dist/GlitchLab/ffmpeg/ffmpeg(.exe)   copied from packaging/ffmpeg/ if present
    dist/GlitchLab/docs/, LICENSE, THIRD_PARTY_NOTICES.md

Build:  pyinstaller --noconfirm --clean packaging/glitchlab.spec
"""
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
WIN = sys.platform == "win32"
ICON = os.path.join(SRC, "glitchlab", "resources", "glitchlab.ico")

datas = [
    (os.path.join(SRC, "glitchlab", "resources"), os.path.join("glitchlab", "resources")),
]
hidden = collect_submodules("glitchlab")
excludes = [
    "tkinter", "unittest", "pydoc_data", "pytest", "IPython",
    # Qt modules Glitch Lab never imports (keeps the bundle smaller)
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtQml",
    "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtPdf", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtBluetooth", "PySide6.QtPositioning",
]


def analysis(script):
    return Analysis(
        [os.path.join(ROOT, "packaging", script)],
        pathex=[SRC],
        binaries=[],
        datas=datas,
        hiddenimports=hidden,
        hookspath=[],
        runtime_hooks=[],
        excludes=excludes,
        noarchive=False,
    )


a_gui = analysis("launch_gui.py")
a_cli = analysis("launch_cli.py")

pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)

common = dict(
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    icon=ICON if WIN else None,
    version=os.path.join(ROOT, "packaging", "version_info.txt") if WIN else None,
)

exe_gui = EXE(pyz_gui, a_gui.scripts, [], name="GlitchLab", console=False, **common)
exe_cli = EXE(pyz_cli, a_cli.scripts, [], name="glitchlab-cli", console=True, **common)

# Extra files copied verbatim next to the executables.
extra = []
for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"):
    p = os.path.join(ROOT, name)
    if os.path.exists(p):
        extra.append((name, p, "DATA"))
docs = os.path.join(ROOT, "docs")
for name in sorted(os.listdir(docs)):
    if name.endswith(".md"):
        extra.append((os.path.join("docs", name), os.path.join(docs, name), "DATA"))
ffdir = os.path.join(ROOT, "packaging", "ffmpeg")
for name in sorted(os.listdir(ffdir)) if os.path.isdir(ffdir) else []:
    p = os.path.join(ffdir, name)
    if os.path.isfile(p) and name != "README.md":
        # plain copy (no dependency analysis / processing of the separate ffmpeg.exe)
        extra.append((os.path.join("ffmpeg", name), p, "DATA"))

coll = COLLECT(
    exe_gui, exe_cli,
    a_gui.binaries, a_gui.datas,
    a_cli.binaries, a_cli.datas,
    extra,
    strip=False,
    upx=False,
    name="GlitchLab",
)

# PyInstaller 6 puts data files into the "_internal" contents directory.
# Move user-facing files (FFmpeg, docs, licences) next to the executables so
# they are easy to find and ffmpeg.exe is easy to replace (LGPL).
import shutil  # noqa: E402

_dist = os.path.join(DISTPATH, "GlitchLab")
_internal = os.path.join(_dist, "_internal")
for _name in ("ffmpeg", "docs", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"):
    _src = os.path.join(_internal, _name)
    _dst = os.path.join(_dist, _name)
    if os.path.exists(_src):
        if os.path.isdir(_dst):
            shutil.rmtree(_dst)
        elif os.path.exists(_dst):
            os.remove(_dst)
        shutil.move(_src, _dst)
