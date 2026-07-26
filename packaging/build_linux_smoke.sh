#!/usr/bin/env bash
# Smoke-test the PyInstaller spec on Linux.
#
# This does NOT produce a real distributable -- PyInstaller is not a
# cross-compiler, so a build run on Linux produces a Linux binary, not a
# Windows .exe. The only way to produce a real Windows .exe is to run
# packaging/build_windows.bat ON a Windows machine. See packaging/README.md.
#
# What this script IS for: verifying that packaging/app.spec is correct --
# that PyInstaller's static analysis + our hiddenimports/datas actually
# resolve every module the app needs at runtime -- by building a onedir
# Linux binary here and launching it headlessly with
# --smoke-test-and-quit. A clean exit (code 0) means the spec is sound;
# any missing-module ModuleNotFoundError means packaging/app.spec's
# hiddenimports (or datas) needs another entry.
#
# Run from the repo root:
#     bash packaging/build_linux_smoke.sh
#
# Prerequisites: a venv with `pip install -r requirements.txt` plus
# `pip install pyinstaller pyinstaller-hooks-contrib` already done.

set -e

# Resolve the repo root as the parent of this script's directory, so this
# works regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

APP_NAME="AnomalyDetectionApp"

echo "== Installing PyInstaller build-time dependencies =="
pip install pyinstaller pyinstaller-hooks-contrib

echo "== Building onedir binary via packaging/app.spec (Linux build -- smoke test only) =="
pyinstaller packaging/app.spec --clean --noconfirm

BINARY="dist/${APP_NAME}/${APP_NAME}"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: expected binary not found at $BINARY" >&2
    exit 1
fi

echo "== Launching headlessly to confirm all imports resolve =="
QT_QPA_PLATFORM=offscreen "$BINARY" --smoke-test-and-quit
STATUS=$?

echo "Exit code: $STATUS"
exit "$STATUS"
