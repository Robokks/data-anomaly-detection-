# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the TDMS/CSV/Excel Anomaly Detection desktop app.

Build (from the repo root, with pyinstaller + pyinstaller-hooks-contrib
installed in your environment):

    Windows:  packaging\\build_windows.bat
    Linux smoke test (does NOT produce a Windows exe -- see packaging/README.md):
              bash packaging/build_linux_smoke.sh

This targets a "onedir" layout (a folder full of the exe + its dependencies,
via the COLLECT step below) rather than "onefile". onedir starts faster and
makes missing-DLL/missing-module errors far easier to diagnose than a
self-extracting onefile exe, at the cost of shipping a folder instead of a
single file.

IMPORTANT: PyInstaller is not a cross-compiler. Running this spec on Linux
produces a Linux binary; running it on Windows produces a Windows .exe. A
real distributable Windows .exe can only be produced by running this spec
ON Windows (see packaging/build_windows.bat and packaging/README.md).
"""
from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Repo root == parent of this packaging/ directory. PyInstaller `exec`s
# spec files without setting `__file__`, so we use the `SPECPATH` global
# it injects into the spec namespace instead (rather than a hardcoded
# absolute path) -- this makes the spec work regardless of where the repo
# is checked out or which directory `pyinstaller` is invoked from.
REPO_ROOT = Path(SPECPATH).resolve().parent
ENTRY_POINT = REPO_ROOT / "gui" / "app.py"

APP_NAME = "AnomalyDetectionApp"

# --- hiddenimports -----------------------------------------------------
# PyInstaller's static analysis walks import statements it can see; it
# regularly misses dynamically-imported / plugin-style modules used by
# scientific-Python and Qt-binding packages. The list below covers the
# well-known gotchas for this project's dependency set (npTDMS, openpyxl,
# pyqtgraph, scipy, torch). This list is a *starting point*, not exhaustive:
# once real Windows build attempts are made, PyInstaller will surface
# `ModuleNotFoundError`s at runtime for anything still missing -- that is
# normal/expected for PyInstaller + scientific-Python stacks, and the fix
# is simply to add the missing module name here and rebuild.
hiddenimports = [
    # npTDMS
    "nptdms",
    "nptdms.tdms",
    # openpyxl (xlsx read/write) / xlrd (legacy .xls)
    "openpyxl",
    "openpyxl.cell._writer",
    "xlrd",
    # pyqtgraph (Qt plotting; imports its Qt binding dynamically)
    "pyqtgraph",
    "pyqtgraph.graphicsItems",
    "pyqtgraph.imageview",
    # scipy: cython_special is the classic PyInstaller/scipy gotcha --
    # scipy.special imports it lazily in a way static analysis misses.
    "scipy.special.cython_special",
    "scipy._lib.messagestream",
    "scipy.sparse.csgraph._validation",
    # scikit-learn: similarly has lazily-imported Cython extension modules.
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    # torch: commonly-missed hidden imports for CPU-only PyInstaller builds.
    "torch",
    "torch._C",
    "torch._VF",
    # pandas: Excel/IO backends are plugin-style and often need a nudge.
    "pandas._libs.tslibs.base",
]

# collect_submodules pulls in every submodule of a package, which is a
# blunt but reliable hammer for packages known to use dynamic/plugin-style
# imports. Keep this list short -- each entry adds build time and app size.
hiddenimports += collect_submodules("nptdms")

# pyqtgraph ships a large `examples` subpackage and an optional `opengl`
# subpackage (needs the separate PyOpenGL package, which this app does not
# depend on / use -- we only do 2D plotting). Collect pyqtgraph's
# submodules but filter those two out to avoid a large, unnecessary size/
# build-time hit and an "OpenGL not found" warning during the build.
hiddenimports += collect_submodules(
    "pyqtgraph",
    filter=lambda name: not (
        name.startswith("pyqtgraph.examples") or name.startswith("pyqtgraph.opengl")
    ),
)

# --- datas ---------------------------------------------------------------
# Non-Python data files that need to ship alongside the app. matplotlib
# ships font/style data files that collect_data_files picks up reliably;
# add more here (e.g. icon assets, once they exist) as needed.
datas = collect_data_files("matplotlib")

# --- excludes --------------------------------------------------------------
# Nothing excluded by default; add heavy unused torch/sklearn backends here
# later if final app size becomes a concern (e.g. torch.utils.tensorboard).
#
# NOTE on build size: if the `torch` wheel installed in the build
# environment is a CUDA-enabled build (the default from PyPI, as opposed
# to the CPU-only wheel from
# https://download.pytorch.org/whl/cpu -- see requirements.txt), the
# resulting onedir folder will be several GB, mostly NVIDIA/CUDA shared
# libraries this GUI app never uses. Installing the CPU-only torch wheel
# before building is the main lever for a much smaller distributable.
excludes: list[str] = []

block_cipher = None

a = Analysis(
    [str(ENTRY_POINT)],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # TODO: no icon asset exists in the repo yet. Once one is added (e.g.
    # packaging/assets/app.ico for Windows), set icon=str(REPO_ROOT / "packaging" / "assets" / "app.ico").
    icon=None,
)

# COLLECT (rather than a onefile EXE) produces a onedir build: a
# dist/AnomalyDetectionApp/ folder containing the exe plus all its
# dependencies as loose files/DLLs. This is the "--onedir-equivalent"
# layout requested for this app -- faster startup and much easier
# missing-DLL diagnosis than a self-extracting onefile exe.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
