@echo off
REM Build the Windows distributable for the Anomaly Detection app.
REM
REM Run this ON WINDOWS from the repo root, e.g.:
REM     packaging\build_windows.bat
REM
REM PyInstaller is not a cross-compiler: it packages for whatever OS/
REM architecture it is run on. This script must be run on a Windows
REM machine to produce a real Windows .exe -- it will NOT work (and is
REM not intended to be run) inside Linux CI or a Linux sandbox. For a
REM Linux-side sanity check of the spec file itself, see
REM packaging\build_linux_smoke.sh instead.
REM
REM Target Python version: match the repo's development Python (3.11)
REM unless you have a specific reason to use a different version.
REM
REM Prerequisites (assumed already done in your environment/venv):
REM     pip install -r requirements.txt
REM     pip install pyinstaller pyinstaller-hooks-contrib

setlocal

REM Move to the repo root (parent of this packaging\ directory) so the
REM build always runs relative to the repo regardless of the caller's cwd.
cd /d "%~dp0\.."

echo Building AnomalyDetectionApp with PyInstaller...
pyinstaller packaging\app.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo Build FAILED. See the PyInstaller output above.
    exit /b 1
)

echo.
echo Build complete. Output: dist\AnomalyDetectionApp\
echo Run dist\AnomalyDetectionApp\AnomalyDetectionApp.exe to test it.

endlocal
