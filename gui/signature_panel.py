"""Signature analysis dialog: a channel's FFT spectrum vs. a learned baseline.

Fits a SignatureBaseline (mean +/- std spectral envelope) from the currently
loaded session's windows for the chosen channel, then overlays the spectrum
of the current time-range selection against that envelope -- the
frequency/spectral "signature analysis" clarified earlier: compare a live
spectrum against a learned normal baseline to spot drift or new frequency
components.
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gui.app_state import AppState
from src.spectral import SignatureBaseline, compute_spectrum, fit_baselines_from_frames
from src.time_range import select_range


class SignatureDialog(QDialog):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Signature Analysis")
        self.resize(700, 500)
        self._baseline: SignatureBaseline | None = None

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.channel_combo = QComboBox()
        form.addRow("Channel:", self.channel_combo)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(2, 1_000_000)
        self.window_size_spin.setValue(256)
        form.addRow("Baseline window size:", self.window_size_spin)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.compute_button = QPushButton("Compute Signature")
        self.compute_button.clicked.connect(self._on_compute_clicked)
        button_row.addWidget(self.compute_button)
        layout.addLayout(button_row)

        self.status_label = QLabel(
            "Fits a baseline spectrum from the loaded session, then compares the "
            "current time-range selection's spectrum against it."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", "Frequency (Hz) / bin")
        self.plot_widget.setLabel("left", "Magnitude")
        layout.addWidget(self.plot_widget)

        self._populate_channels()

    def _populate_channels(self) -> None:
        self.channel_combo.clear()
        df = self.state.current_df
        if df is None:
            return
        channels = [c for c in df.columns if c != "phase"]
        self.channel_combo.addItems(channels)

    def _on_compute_clicked(self) -> None:
        df = self.state.current_df
        channel = self.channel_combo.currentText()
        if df is None or df.empty or not channel or channel not in df.columns:
            self.status_label.setText("No session/channel available -- load a session first.")
            return

        window_size = self.window_size_spin.value()
        if len(df) < window_size:
            self.status_label.setText(f"Session has {len(df)} samples, shorter than window size {window_size}.")
            return

        sample_rate = df.attrs.get("sample_rate_hz")
        single_channel_df = df[[channel]]

        baselines = fit_baselines_from_frames(
            [single_channel_df], channels=[channel], window_size=window_size, sample_rate=sample_rate
        )
        baseline = baselines.get(channel)
        if baseline is None:
            self.status_label.setText(f"Could not fit a baseline for channel '{channel}'.")
            return
        self._baseline = baseline

        start, end = self.state.time_range
        current_df = select_range(single_channel_df, start=start, end=end) if (start is not None or end is not None) else single_channel_df
        current_values = current_df[channel].to_numpy(dtype=float)

        self._plot(baseline, current_values, sample_rate, window_size)

    def _plot(self, baseline: SignatureBaseline, current_values, sample_rate, window_size: int) -> None:
        self.plot_widget.clear()
        self.plot_widget.addLegend()

        upper = baseline.mean_ + baseline.std_
        lower = baseline.mean_ - baseline.std_
        upper_curve = self.plot_widget.plot(baseline.freqs_, upper, pen=pg.mkPen(None))
        lower_curve = self.plot_widget.plot(baseline.freqs_, lower, pen=pg.mkPen(None))
        fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(31, 119, 180, 60))
        self.plot_widget.addItem(fill)
        self.plot_widget.plot(baseline.freqs_, baseline.mean_, pen=pg.mkPen("#1f77b4", width=2), name="Baseline mean")

        current_freqs, current_mag = compute_spectrum(current_values, sample_rate=sample_rate)
        self.plot_widget.plot(current_freqs, current_mag, pen=pg.mkPen("#d62728", width=2), name="Current selection")

        summary = f"Fit baseline from {baseline.n_windows_} window(s) of '{baseline.channel}'."
        if len(current_values) == window_size:
            deviation = baseline.deviation(current_values, sample_rate=sample_rate)
            summary += f" Current-selection deviation from baseline: {deviation:.3f}."
        else:
            summary += (
                f" Current selection has {len(current_values)} sample(s), which differs from the "
                f"baseline window size ({window_size}); spectrum shown, but a deviation score needs "
                "matching lengths -- adjust the time-range selection or window size to compare."
            )
        self.status_label.setText(summary)
