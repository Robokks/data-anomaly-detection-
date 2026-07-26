"""GUI entry point.

    python -m gui.app

``--smoke-test-and-quit`` constructs the main window and exits immediately
without starting the event loop -- used to verify the app launches cleanly
(all imports/plugins resolve) under a headless display, both in CI/tests and
when validating a PyInstaller build (see packaging/build_linux_smoke.sh).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


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

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    if args.smoke_test_and_quit:
        window.show()
        app.processEvents()
        window.close()
        return 0

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
