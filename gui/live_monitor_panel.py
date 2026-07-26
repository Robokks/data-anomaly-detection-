"""Live monitor panel: receive streamed sensor data over TCP and score it live.

A non-modal dialog (deliberate deviation from TrainDialog/SignatureDialog's
``.exec()`` -- streaming must keep running while the user does other things
in the rest of the app). Shows a live-updating table: one row per channel
plus an "Overall" row, with per-second statistics, a deviation/anomaly score,
and an is-anomaly flag, updated as ``LiveWindowResult``s arrive.

Threading note: unlike ``TrainDialog`` (which moves genuinely slow work --
model training -- onto a QThread), ``TcpStreamServer.start()`` itself is
fast (it just binds a socket and launches its own internal background
thread for ``serve_forever()``), so there's no long-running call here that
needs to be kept off the UI thread. What *does* need care is that
``on_window``/``on_error`` are invoked from TcpStreamServer's per-connection
handler threads, not the UI thread -- they're wired directly to this
dialog's own Qt signals (``windowScored``/``serverError``), and Qt
auto-queues a cross-thread ``emit()`` to its connected slot, so the actual
``QTableWidgetItem`` writes in ``_on_window_scored`` always run on the UI
thread. No Qt widget is ever touched from a server thread.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.app_state import AppState
from gui.theme import ThemeManager
from src.data_loader import load_directory
from src.live_scorer import STAT_NAMES, LiveScorer, LiveWindowResult
from src.pipeline import get_model_window_size, load_model
from src.tcp_stream_server import TcpStreamServer

_TABLE_COLUMNS = ["Channel", *STAT_NAMES, "Score", "Anomaly"]
_OVERALL_ROW_LABEL = "Overall"


class LiveMonitorPanel(QDialog):
    windowScored = Signal(object)  # LiveWindowResult
    serverError = Signal(str)

    def __init__(self, state: AppState, parent: QWidget | None = None, theme_manager: ThemeManager | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.theme_manager = theme_manager or ThemeManager()
        self.setWindowTitle("Live Monitor")
        self.setModal(False)
        self.resize(900, 500)

        self._server: TcpStreamServer | None = None
        self._live_scorer: LiveScorer | None = None
        self._row_for_channel: dict[str, int] = {}

        self._model = None
        self._model_type: str | None = None

        layout = QVBoxLayout(self)

        self.model_label = QLabel("No model selected.")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        model_row = QHBoxLayout()
        self.use_current_model_button = QPushButton("Use Current Model")
        self.use_current_model_button.clicked.connect(self._on_use_current_model_clicked)
        model_row.addWidget(self.use_current_model_button)
        self.load_model_button = QPushButton("Load Model File...")
        self.load_model_button.clicked.connect(self._on_load_model_clicked)
        model_row.addWidget(self.load_model_button)
        layout.addLayout(model_row)

        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)

        baseline_row = QHBoxLayout()
        self.baseline_dir_edit = QLineEdit()
        if state.root_directory is not None:
            self.baseline_dir_edit.setText(str(state.root_directory))
        baseline_row.addWidget(self.baseline_dir_edit)
        self.browse_baseline_button = QPushButton("Browse...")
        self.browse_baseline_button.clicked.connect(self._on_browse_baseline_clicked)
        baseline_row.addWidget(self.browse_baseline_button)
        model_form.addRow("Baseline folder:", baseline_row)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(2, 1_000_000)
        self.window_size_spin.setValue(256)
        model_form.addRow("Model window size:", self.window_size_spin)
        layout.addWidget(model_group)

        windowing_group = QGroupBox("Windowing")
        windowing_form = QFormLayout(windowing_group)

        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(0.1, 1_000_000)
        self.sample_rate_spin.setValue(10000)
        windowing_form.addRow("Sample rate (Hz):", self.sample_rate_spin)

        self.window_duration_spin = QDoubleSpinBox()
        self.window_duration_spin.setRange(0.01, 3600)
        self.window_duration_spin.setValue(1.0)
        windowing_form.addRow("Stats window (s):", self.window_duration_spin)

        self.contamination_spin = QDoubleSpinBox()
        self.contamination_spin.setRange(0.001, 0.5)
        self.contamination_spin.setSingleStep(0.005)
        self.contamination_spin.setDecimals(3)
        self.contamination_spin.setValue(0.05)
        windowing_form.addRow("Channel contamination:", self.contamination_spin)
        layout.addWidget(windowing_group)

        connection_group = QGroupBox("Connection")
        connection_form = QFormLayout(connection_group)

        self.host_edit = QLineEdit("0.0.0.0")
        connection_form.addRow("Host:", self.host_edit)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(0, 65535)
        self.port_spin.setValue(9999)
        connection_form.addRow("Port (0 = auto):", self.port_spin)
        layout.addWidget(connection_group)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start_clicked)
        button_row.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.stop_button)
        layout.addLayout(button_row)

        self.status_label = QLabel("Not started.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(_TABLE_COLUMNS)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        self.windowScored.connect(self._on_window_scored)
        self.serverError.connect(self._on_server_error)

    # -- model selection --------------------------------------------------

    def _on_use_current_model_clicked(self) -> None:
        if self.state.model is None:
            self.status_label.setText("No trained model in the current session yet -- train one first, or load from file.")
            return
        self._set_model(self.state.model, self.state.model_type)

    def _on_load_model_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load model", "", "Model files (*.joblib)")
        if not path:
            return
        try:
            model, model_type = load_model(path)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Failed to load model: {exc}")
            return
        self._set_model(model, model_type)

    def _set_model(self, model, model_type: str) -> None:
        self._model = model
        self._model_type = model_type
        self.model_label.setText(f"Model: {model_type}")

        window_size, _step = get_model_window_size(model, model_type)
        if window_size is not None:
            self.window_size_spin.setValue(window_size)

    # -- baseline folder ----------------------------------------------------

    def _on_browse_baseline_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select a baseline data folder")
        if directory:
            self.baseline_dir_edit.setText(directory)

    # -- start/stop -----------------------------------------------------------

    def _on_start_clicked(self) -> None:
        if self._model is None:
            self.status_label.setText("Select a model first (Use Current Model or Load Model File...).")
            return
        baseline_dir = self.baseline_dir_edit.text().strip()
        if not baseline_dir:
            self.status_label.setText("Select a baseline folder first.")
            return

        try:
            baseline_frames = load_directory(baseline_dir, recursive=True)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Failed to load baseline data: {exc}")
            return

        channels = [c for c in baseline_frames[0].columns if c != "phase"]

        window_size = self.window_size_spin.value()
        live_scorer = LiveScorer(
            model=self._model,
            model_type=self._model_type,
            channels=channels,
            sample_rate_hz=self.sample_rate_spin.value(),
            model_window_size=window_size,
            window_duration_seconds=self.window_duration_spin.value(),
            channel_contamination=self.contamination_spin.value(),
        )
        try:
            live_scorer.fit_baseline(baseline_frames)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Failed to fit baseline: {exc}")
            return

        self._live_scorer = live_scorer
        self._populate_table_rows(channels)

        server = TcpStreamServer(
            self.host_edit.text().strip() or "0.0.0.0",
            self.port_spin.value(),
            live_scorer,
            on_window=self.windowScored.emit,
            on_error=self.serverError.emit,
        )
        try:
            bound_port = server.start()
        except OSError as exc:
            self.status_label.setText(f"Failed to start server: {exc}")
            return

        self._server = server
        self.port_spin.setValue(bound_port)
        self.status_label.setText(f"Listening on {self.host_edit.text().strip() or '0.0.0.0'}:{bound_port} ...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _on_stop_clicked(self) -> None:
        self._stop_server()

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self._live_scorer is not None:
            self.status_label.setText("Stopped.")

    def closeEvent(self, event) -> None:  # noqa: N802 -- Qt override naming
        self._stop_server()
        super().closeEvent(event)

    # -- table --------------------------------------------------------------

    def _populate_table_rows(self, channels: list[str]) -> None:
        self.table.setRowCount(0)
        self._row_for_channel.clear()

        rows = list(channels) + [_OVERALL_ROW_LABEL]
        self.table.setRowCount(len(rows))
        for row, name in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            for col in range(1, len(_TABLE_COLUMNS)):
                self.table.setItem(row, col, QTableWidgetItem(""))
            self._row_for_channel[name] = row

    def _anomaly_item(self, is_anomaly: bool | None) -> QTableWidgetItem:
        item = QTableWidgetItem("" if is_anomaly is None else ("Yes" if is_anomaly else "No"))
        if is_anomaly:
            palette = self.theme_manager.current_palette
            item.setBackground(QBrush(QColor(palette.danger)))
            item.setForeground(QBrush(QColor("#ffffff")))
        return item

    def _on_window_scored(self, result: LiveWindowResult) -> None:
        for channel, channel_result in result.channels.items():
            row = self._row_for_channel.get(channel)
            if row is None:
                continue
            for col, stat in enumerate(STAT_NAMES, start=1):
                self.table.setItem(row, col, QTableWidgetItem(f"{channel_result.stats[stat]:.4g}"))
            score_col = 1 + len(STAT_NAMES)
            self.table.setItem(row, score_col, QTableWidgetItem(f"{channel_result.deviation_score:.4g}"))
            self.table.setItem(row, score_col + 1, self._anomaly_item(channel_result.is_anomaly))

        overall_row = self._row_for_channel.get(_OVERALL_ROW_LABEL)
        if overall_row is not None:
            score_col = 1 + len(STAT_NAMES)
            score_text = "" if result.overall_anomaly_score is None else f"{result.overall_anomaly_score:.4g}"
            self.table.setItem(overall_row, score_col, QTableWidgetItem(score_text))
            self.table.setItem(overall_row, score_col + 1, self._anomaly_item(result.overall_is_anomaly))

        self.status_label.setText(
            f"Window {result.window_index} @ {result.n_samples_seen} samples seen -- listening."
        )

    def _on_server_error(self, message: str) -> None:
        self.status_label.setText(f"Stream error: {message}")
