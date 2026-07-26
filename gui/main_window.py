"""Main application window shell.

Wires up the overall layout (toolbar, session/plot/channel regions, status
bar) and holds the shared ``AppState``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QSplitter, QStatusBar, QToolBar

from gui.app_state import AppState
from gui.channel_panel import ChannelPanel
from gui.live_monitor_panel import LiveMonitorPanel
from gui.plot_panel import PlotPanel
from gui.results_panel import ResultsPanel
from gui.session_panel import SessionPanel
from gui.signature_panel import SignatureDialog
from gui.train_dialog import TrainDialog
from src.pipeline import save_model


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self.state = state or AppState()
        self.live_monitor_panel: LiveMonitorPanel | None = None

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

        train_action = QAction("Train...", self)
        train_action.triggered.connect(self._on_train_clicked)
        toolbar.addAction(train_action)
        self.train_action = train_action

        signature_action = QAction("Signature Analysis...", self)
        signature_action.triggered.connect(self._on_signature_clicked)
        toolbar.addAction(signature_action)
        self.signature_action = signature_action

        live_monitor_action = QAction("Live Monitor...", self)
        live_monitor_action.triggered.connect(self._on_live_monitor_clicked)
        toolbar.addAction(live_monitor_action)
        self.live_monitor_action = live_monitor_action

        save_model_action = QAction("Save Model...", self)
        save_model_action.triggered.connect(self._on_save_model_clicked)
        save_model_action.setEnabled(False)
        toolbar.addAction(save_model_action)
        self.save_model_action = save_model_action
        self.state.modelTrained.connect(lambda *_: save_model_action.setEnabled(True))

    def _build_central_widget(self) -> None:
        # Left: session/file browser.
        self.session_panel = SessionPanel(self.state)
        # Center: multi-channel plot + time-range selector.
        self.plot_panel = PlotPanel(self.state)
        # Right: channel curation (top) + training results (bottom).
        self.channel_panel = ChannelPanel(self.state)
        self.results_panel = ResultsPanel()

        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.addWidget(self.channel_panel)
        right_splitter.addWidget(self.results_panel)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
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
        dialog = SignatureDialog(self.state, self)
        dialog.exec()

    def _on_live_monitor_clicked(self) -> None:
        # Non-modal: keep a single instance and raise it on repeat clicks,
        # rather than spawning duplicate dialogs (and duplicate servers).
        if self.live_monitor_panel is None:
            self.live_monitor_panel = LiveMonitorPanel(self.state, self)
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
