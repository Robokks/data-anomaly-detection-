# Packaging (PyInstaller)

This app is packaged into a standalone desktop distributable using
[PyInstaller](https://pyinstaller.org/), targeting a **onedir** build (a
folder containing the exe plus its dependencies) rather than a onefile
self-extracting exe -- onedir starts faster and makes missing-DLL/missing-module
issues far easier to diagnose.

Files in this directory:

| File | Purpose |
| --- | --- |
| `app.spec` | The PyInstaller spec file. Entry point: `gui/app.py`. |
| `build_windows.bat` | Run **on Windows** to produce the real Windows `.exe`. |
| `build_linux_smoke.sh` | Run **on Linux** to smoke-test that the spec resolves all imports. Does **not** produce a Windows exe. |

## Critical: PyInstaller is not a cross-compiler

PyInstaller packages an app for whatever OS and architecture it is *run
on* -- it does not cross-compile. That means:

- A real Windows `.exe` can only be produced by running
  `packaging\build_windows.bat` **on a Windows machine**.
- Running PyInstaller in this (or any) Linux CI runner / Linux sandbox
  will produce a **Linux** binary, never a Windows `.exe`, no matter what
  spec settings are used.

If you need a Windows exe, you must build it on Windows (a real Windows
machine, a Windows VM, or Windows CI runner such as a `windows-latest`
GitHub Actions job).

## Build-time dependencies

In addition to the app's runtime dependencies (`pip install -r
requirements.txt` from the repo root), building requires:

```
pip install pyinstaller pyinstaller-hooks-contrib
```

`pyinstaller-hooks-contrib` ships community-maintained PyInstaller hooks
for a number of packages in this app's dependency stack (notably PySide6
and torch), and significantly reduces the amount of manual
`hiddenimports`/`datas` tuning needed. It's a build-time-only dependency
(not needed at app runtime), so it's listed here and in the build scripts
rather than in the root `requirements.txt`.

## Building on Windows (the real distributable)

```
pip install -r requirements.txt
pip install pyinstaller pyinstaller-hooks-contrib
packaging\build_windows.bat
```

Output: `dist\AnomalyDetectionApp\AnomalyDetectionApp.exe` (plus its
supporting files in the same folder -- ship the whole
`dist\AnomalyDetectionApp\` folder, not just the exe).

Target Python version: match the repo's development Python (3.11) unless
you have a specific reason to differ.

## Smoke-testing the spec on Linux

`build_linux_smoke.sh` is **not** a way to produce a distributable. It
exists purely to validate that `app.spec`'s `hiddenimports`/`datas` are
correct -- i.e. that PyInstaller successfully bundles every module the app
actually needs at runtime -- without needing a Windows machine on every
iteration of spec-file changes. It builds a onedir Linux binary in
`dist/`, then launches it headlessly (`QT_QPA_PLATFORM=offscreen`) with
the app's `--smoke-test-and-quit` flag (see `gui/app.py`), which
constructs the main window and exits immediately without starting the Qt
event loop. A clean exit code 0 means all imports resolved; a
`ModuleNotFoundError` means an entry is missing from `hiddenimports` (or a
data file is missing from `datas`) in `app.spec`.

Run it from the repo root:

```
bash packaging/build_linux_smoke.sh
```

Because scientific-Python packages (scipy, scikit-learn, torch, npTDMS,
pyqtgraph, ...) commonly use dynamic/plugin-style imports that
PyInstaller's static analysis misses, it is normal to need to add more
entries to `hiddenimports` in `app.spec` as new `ModuleNotFoundError`s
surface -- both here and, later, on real Windows build attempts. Iterate:
add the missing module name, rerun, repeat.

## Manual verification checklist (after a real Windows build)

Once you have built the real `.exe` on Windows via `build_windows.bat`,
manually walk through the following before considering a packaged build
good:

1. Launch `dist\AnomalyDetectionApp\AnomalyDetectionApp.exe` by
   double-clicking it (not from a terminal) -- confirm it starts without
   a console window flashing errors and without needing Python/venv on
   the machine.
2. Confirm the main window opens (title bar reads "TDMS/CSV/Excel Anomaly
   Detection") and is responsive (resize the window, no crash/hang).
3. Use "Browse Folder..." -- pick a folder of test-rig recordings (or the
   synthetic demo data from `python -m src.synthetic_tdms`). Confirm the
   session tree in the left panel populates.
4. Try each **machine type** in the dropdown -- confirm the tree
   regroups (single-file sessions for General; grouped UUT sessions with
   phase/pocket tags for Transmission/Motor Test Bench/Endurance).
5. Select a session -- confirm the center plot renders each channel and
   the right-hand channel table lists them with working Plot/Train
   checkboxes; toggling Plot should add/remove that channel's curve.
6. Drag the shaded time-range region on the plot (and try typing into the
   Start/End fields, and the "Full Data" button) -- confirm it updates
   smoothly at real data sizes.
7. Click "Train..." -- run once with **Classic** and once with **Deep
   Learning**, confirm the dialog stays responsive during deep-learning
   training and that flagged windows appear in the results table and as
   markers on the plot afterward.
8. Click "Signature Analysis..." -- pick a channel, "Compute Signature",
   confirm the baseline-vs-current spectrum plot renders.
9. "Save Model..." and the results panel's "Export CSV..." -- confirm
   both write files to disk without error.
10. Close the app normally (no lingering process in Task Manager
    afterward).
