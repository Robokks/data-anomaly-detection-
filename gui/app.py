"""GUI entry point.

    python -m gui.app

``--smoke-test-and-quit`` constructs the main window and exits immediately
without starting the event loop -- used to verify the app launches cleanly
(all imports/plugins resolve) under a headless display, both in CI/tests and
when validating a PyInstaller build (see packaging/build_linux_smoke.sh).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be set before torch is imported anywhere in the process (transitively,
# via gui.main_window -> gui.train_dialog -> src.dl_model). This app is
# CPU-only by design (AutoencoderDetector defaults to device="cpu", no GPU
# assumed) -- but if a user's environment happens to have a GPU-enabled torch
# wheel installed without a properly configured CUDA driver, torch probing
# CUDA from the training background thread reliably crashes the process.
# Forcing no visible CUDA devices avoids that regardless of which wheel is
# installed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.theme import ThemeManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TDMS/CSV/Excel anomaly detection GUI.")
    p.add_argument(
        "--smoke-test-and-quit",
        action="store_true",
        help="Construct the main window and exit immediately, without starting the event loop.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    QCoreApplication.setOrganizationName("AnomalyDetection")
    QCoreApplication.setApplicationName("GUI")

    app = QApplication.instance() or QApplication(sys.argv)

    # Applied before MainWindow is constructed so there's no flash of the
    # wrong theme on launch.
    theme_manager = ThemeManager()
    theme_manager.apply(app)

    window = MainWindow(theme_manager=theme_manager)

    if args.smoke_test_and_quit:
        window.show()
        app.processEvents()
        window.close()
        return 0

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
