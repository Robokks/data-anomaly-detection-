"""Main application window shell.

Wires up the overall layout (toolbar, session/plot/channel regions, status
bar) and holds the shared ``AppState``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)

from gui.app_state import AppState
from gui.channel_panel import ChannelPanel
from gui.icons import icon
from gui.live_monitor_panel import LiveMonitorPanel
from gui.plot_panel import PlotPanel
from gui.results_panel import ResultsPanel
from gui.session_panel import SessionPanel
from gui.signature_panel import SignatureDialog
from gui.theme import ThemeManager
from gui.train_dialog import TrainDialog
from src.pipeline import save_model


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, theme_manager: ThemeManager | None = None) -> None:
        super().__init__()
        self.state = state or AppState()
        self.theme_manager = theme_manager or ThemeManager()
        self.live_monitor_panel: LiveMonitorPanel | None = None

        self.setWindowTitle("TDMS/CSV/Excel Anomaly Detection")
        self.setWindowIcon(icon("live", self.theme_manager.current_palette.accent, size=32))
        self.resize(1200, 800)

        self._build_toolbar()
        self._build_central_widget()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Browse a folder to begin.")

        self.theme_manager.themeChanged.connect(self._on_theme_changed)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        accent = self.theme_manager.current_palette.text_primary

        browse_action = QAction(icon("folder", accent), "Browse Folder...", self)
        browse_action.triggered.connect(self._on_browse_folder)
        toolbar.addAction(browse_action)
        self.browse_action = browse_action

        train_action = QAction(icon("train", accent), "Train...", self)
        train_action.triggered.connect(self._on_train_clicked)
        toolbar.addAction(train_action)
        self.train_action = train_action

        signature_action = QAction(icon("signature", accent), "Signature Analysis...", self)
        signature_action.triggered.connect(self._on_signature_clicked)
        toolbar.addAction(signature_action)
        self.signature_action = signature_action

        live_monitor_action = QAction(icon("live", accent), "Live Monitor...", self)
        live_monitor_action.triggered.connect(self._on_live_monitor_clicked)
        toolbar.addAction(live_monitor_action)
        self.live_monitor_action = live_monitor_action

        save_model_action = QAction(icon("save", accent), "Save Model...", self)
        save_model_action.triggered.connect(self._on_save_model_clicked)
        save_model_action.setEnabled(False)
        toolbar.addAction(save_model_action)
        self.save_model_action = save_model_action
        self.state.modelTrained.connect(lambda *_: save_model_action.setEnabled(True))

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        theme_action = QAction(self._theme_action_icon(), self._theme_action_text(), self)
        theme_action.setCheckable(True)
        theme_action.setChecked(self.theme_manager.mode == "dark")
        theme_action.triggered.connect(self.theme_manager.toggle)
        toolbar.addAction(theme_action)
        self.theme_action = theme_action

    def _theme_action_icon(self):
        name = "sun" if self.theme_manager.mode == "dark" else "moon"
        return icon(name, self.theme_manager.current_palette.text_primary)

    def _theme_action_text(self) -> str:
        return "Light Mode" if self.theme_manager.mode == "dark" else "Dark Mode"

    def _on_theme_changed(self, _mode: str) -> None:
        self.theme_action.setChecked(self.theme_manager.mode == "dark")
        self.theme_action.setText(self._theme_action_text())
        self.theme_action.setIcon(self._theme_action_icon())
        self.setWindowIcon(icon("live", self.theme_manager.current_palette.accent, size=32))
        accent = self.theme_manager.current_palette.text_primary
        self.browse_action.setIcon(icon("folder", accent))
        self.train_action.setIcon(icon("train", accent))
        self.signature_action.setIcon(icon("signature", accent))
        self.live_monitor_action.setIcon(icon("live", accent))
        self.save_model_action.setIcon(icon("save", accent))

    def _build_central_widget(self) -> None:
        # Left: session/file browser.
        self.session_panel = SessionPanel(self.state)
        # Center: multi-channel plot + time-range selector.
        self.plot_panel = PlotPanel(self.state, theme_manager=self.theme_manager)
        # Right: channel curation (top) + training results (bottom).
        self.channel_panel = ChannelPanel(self.state)
        self.results_panel = ResultsPanel()

        for panel in (self.session_panel, self.plot_panel, self.channel_panel, self.results_panel):
            panel.setObjectName("card")

        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.setHandleWidth(6)
        right_splitter.addWidget(self.channel_panel)
        right_splitter.addWidget(self.results_panel)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(6)
        splitter.addWidget(self.session_panel)
        splitter.addWidget(self.plot_panel)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        self.setCentralWidget(splitter)

    def _on_browse_folder(self) -> None:
        # Toolbar action delegates to the session panel's own Browse Folder
        # button, so there's one entry point for the folder-picking dialog
        # and the resulting scan+group, reachable from two places in the UI.
        self.session_panel._on_browse_clicked()

    def _on_train_clicked(self) -> None:
        dialog = TrainDialog(self.state, self)
        dialog.trainingFinished.connect(self._on_training_finished)
        dialog.exec()

    def _on_training_finished(self, model, model_type: str, scores) -> None:
        self.results_panel.set_scores(scores)
        self.plot_panel.overlay_anomalies(scores)
        self.statusBar().showMessage(f"Trained {model_type} model: {int(scores['is_anomaly'].sum())} window(s) flagged.")

    def _on_signature_clicked(self) -> None:
        dialog = SignatureDialog(self.state, self, theme_manager=self.theme_manager)
        dialog.exec()

    def _on_live_monitor_clicked(self) -> None:
        # Non-modal: keep a single instance and raise it on repeat clicks,
        # rather than spawning duplicate dialogs (and duplicate servers).
        if self.live_monitor_panel is None:
            self.live_monitor_panel = LiveMonitorPanel(self.state, self, theme_manager=self.theme_manager)
        self.live_monitor_panel.show()
        self.live_monitor_panel.raise_()
        self.live_monitor_panel.activateWindow()

    def _on_save_model_clicked(self) -> None:
        if self.state.model is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model", "model.joblib", "Model files (*.joblib)")
        if not path:
            return
        try:
            save_model(self.state.model, self.state.model_type, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {self.state.model_type} model to {path}")
