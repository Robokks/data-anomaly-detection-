"""Modal dialog: configure and run model training on the currently loaded session.

Runs training on a background QThread (important for the deep-learning
model, which can take real time even on small data) so the UI stays
responsive. Trains on ``AppState.current_df`` filtered to the Train-checked
channels and the current time range -- i.e. whatever the user has curated
via the channel/plot panels. Multi-session batch training (training across
every session in a scanned folder at once) is available today via the CLI
(``src/train.py --data-dir ...``); this dialog trains on one loaded session
at a time.
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.app_state import AppState
from src.pipeline import score_model, train_model
from src.time_range import select_range


class _TrainWorker(QObject):
    finished = Signal(object, str, object)  # model, model_type, scores
    error = Signal(str)

    def __init__(self, model_type: str, frames: list[pd.DataFrame], window_size: int, step: int | None, model_kwargs: dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.frames = frames
        self.window_size = window_size
        self.step = step
        self.model_kwargs = model_kwargs

    def run(self) -> None:
        try:
            model = train_model(
                self.model_type, self.frames, window_size=self.window_size, step=self.step, **self.model_kwargs
            )
            scores = score_model(model, self.model_type, self.frames, window_size=self.window_size, step=self.step)
            self.finished.emit(model, self.model_type, scores)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the dialog, not raised on the worker thread
            self.error.emit(str(exc))


class TrainDialog(QDialog):
    trainingFinished = Signal(object, str, object)  # model, model_type, scores

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Train Anomaly Detector")
        self._thread: QThread | None = None
        self._worker: _TrainWorker | None = None

        layout = QVBoxLayout(self)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        general_group = QGroupBox("General")
        form = QFormLayout(general_group)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("Classic (Isolation Forest + PCA)", "classic")
        self.model_type_combo.addItem("Deep Learning (1D Conv Autoencoder)", "deep")
        self.model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        form.addRow("Model type:", self.model_type_combo)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(2, 1_000_000)
        self.window_size_spin.setValue(256)
        form.addRow("Window size:", self.window_size_spin)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(0, 1_000_000)
        self.step_spin.setValue(0)
        self.step_spin.setSpecialValueText("(= window size)")
        form.addRow("Step:", self.step_spin)

        self.contamination_spin = QDoubleSpinBox()
        self.contamination_spin.setRange(0.001, 0.5)
        self.contamination_spin.setSingleStep(0.005)
        self.contamination_spin.setDecimals(3)
        self.contamination_spin.setValue(0.02)
        form.addRow("Contamination:", self.contamination_spin)

        layout.addWidget(general_group)

        self.classic_widget = QGroupBox("Isolation Forest settings")
        classic_form = QFormLayout(self.classic_widget)
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 2000)
        self.n_estimators_spin.setValue(200)
        classic_form.addRow("Isolation Forest trees:", self.n_estimators_spin)

        self.deep_widget = QGroupBox("Autoencoder settings")
        deep_form = QFormLayout(self.deep_widget)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(30)
        deep_form.addRow("Epochs:", self.epochs_spin)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 4096)
        self.batch_size_spin.setValue(64)
        deep_form.addRow("Batch size:", self.batch_size_spin)
        self.latent_dim_spin = QSpinBox()
        self.latent_dim_spin.setRange(2, 512)
        self.latent_dim_spin.setValue(16)
        deep_form.addRow("Latent dim:", self.latent_dim_spin)

        self.model_options_stack = QStackedWidget()
        self.model_options_stack.addWidget(self.classic_widget)
        self.model_options_stack.addWidget(self.deep_widget)
        layout.addWidget(self.model_options_stack)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Indeterminate (busy) progress bar -- training has no reliable
        # step-by-step progress to report (classic fits IsolationForest+PCA
        # in one call; the deep-learning epoch loop isn't instrumented with
        # a progress callback), so this communicates "working" rather than
        # "N% done."
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Train")
        self.button_box.accepted.connect(self._on_train_clicked)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._update_info_label()

    def _on_model_type_changed(self, index: int) -> None:
        self.model_options_stack.setCurrentIndex(index)

    def _update_info_label(self) -> None:
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if self.state.current_df is None or self.state.current_df.empty:
            self.info_label.setText("No session loaded -- select one from the session panel first.")
            ok_button.setEnabled(False)
            return
        n_channels = len(self.state.train_channels)
        session_name = getattr(self.state.current_session, "uut_id", None) or "(loaded file)"
        self.info_label.setText(
            f"Training on session '{session_name}' using {n_channels} channel(s): "
            f"{', '.join(self.state.train_channels) or '(none selected)'}"
        )
        ok_button.setEnabled(n_channels > 0)

    def _prepare_frame(self) -> pd.DataFrame:
        df = self.state.current_df
        if df is None:
            raise RuntimeError("No session loaded.")
        columns = [c for c in self.state.train_channels if c in df.columns]
        if not columns:
            raise RuntimeError("No channels selected for training (check at least one Train checkbox).")
        frame = df[columns].copy()
        start, end = self.state.time_range
        if start is not None or end is not None:
            frame = select_range(frame, start=start, end=end)
        return frame

    def _on_train_clicked(self) -> None:
        try:
            frame = self._prepare_frame()
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Error: {exc}")
            return

        model_type = self.model_type_combo.currentData()
        window_size = self.window_size_spin.value()
        step = self.step_spin.value() or None

        model_kwargs: dict = {"contamination": self.contamination_spin.value()}
        if model_type == "classic":
            model_kwargs["n_estimators"] = self.n_estimators_spin.value()
        else:
            model_kwargs.update(
                epochs=self.epochs_spin.value(),
                batch_size=self.batch_size_spin.value(),
                latent_dim=self.latent_dim_spin.value(),
            )

        self.button_box.setEnabled(False)
        self.status_label.setText("Training...")
        self.progress_bar.show()

        self._thread = QThread(self)
        self._worker = _TrainWorker(model_type, [frame], window_size, step, model_kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_training_finished)
        self._worker.error.connect(self._on_training_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_training_finished(self, model, model_type: str, scores: pd.DataFrame) -> None:
        n_anom = int(scores["is_anomaly"].sum())
        self.status_label.setText(f"Training complete: {n_anom} / {len(scores)} windows flagged.")
        self.progress_bar.hide()
        self.button_box.setEnabled(True)
        self.state.set_model(model, model_type)
        self.trainingFinished.emit(model, model_type, scores)
        self.accept()

    def _on_training_error(self, message: str) -> None:
        self.status_label.setText(f"Training failed: {message}")
        self.progress_bar.hide()
        self.button_box.setEnabled(True)
