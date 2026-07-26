"""Main application window shell.

This wires up the overall layout (toolbar, session/channel/plot regions,
status bar) and holds the shared ``AppState``. The session/channel/plot
panels themselves are placeholders here -- they're built out in follow-up
work (gui/session_panel.py, gui/channel_panel.py, gui/plot_panel.py) and
swapped in without changing this file's overall structure.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gui.app_state import AppState
from gui.session_panel import SessionPanel


class _PlaceholderPanel(QWidget):
    """Simple labeled placeholder, swapped out by later panel implementations."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self.state = state or AppState()

        self.setWindowTitle("TDMS/CSV/Excel Anomaly Detection")
        self.resize(1200, 800)

        self._build_toolbar()
        self._build_central_widget()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Browse a folder to begin.")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        browse_action = QAction("Browse Folder...", self)
        browse_action.triggered.connect(self._on_browse_folder)
        toolbar.addAction(browse_action)
        self.browse_action = browse_action

    def _build_central_widget(self) -> None:
        # Left: session/file browser.
        self.session_panel = SessionPanel(self.state)
        # Center: plot (task: gui/plot_panel.py).
        self.plot_panel = _PlaceholderPanel("Plot panel\n(multi-channel plot + time-range selector)\ncoming soon")
        # Right: channel curation (task: gui/channel_panel.py).
        self.channel_panel = _PlaceholderPanel("Channel panel\n(Plot / Train checkboxes)\ncoming soon")

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.session_panel)
        splitter.addWidget(self.plot_panel)
        splitter.addWidget(self.channel_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        self.setCentralWidget(splitter)

    def _on_browse_folder(self) -> None:
        # Toolbar action delegates to the session panel's own Browse Folder
        # button, so there's one entry point for the folder-picking dialog
        # and the resulting scan+group, reachable from two places in the UI.
        self.session_panel._on_browse_clicked()
