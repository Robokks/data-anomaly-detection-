"""Channel panel: per-channel Plot/Train checkboxes.

Right-hand panel. Plot and Train selection are independent -- a user can
visually inspect a channel (Plot checked) without including it in model
training (Train unchecked), which is how "isolate the unwanted data" works:
overlay several channels on the plot, spot the bad one, uncheck its Train
box (and optionally its Plot box too) without losing the rest of the
session's data.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.app_state import AppState

_COLUMNS = ["Channel", "Plot", "Train", "Range"]


def _centered(widget: QWidget) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(widget)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.setContentsMargins(0, 0, 0, 0)
    return container


class ChannelPanel(QWidget):
    plotSelectionChanged = Signal(list)
    trainSelectionChanged = Signal(list)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self._checkboxes: dict[str, tuple[QCheckBox, QCheckBox]] = {}

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        self.state.fileLoaded.connect(self._on_file_loaded)

    def _on_file_loaded(self, df) -> None:
        self.table.setRowCount(0)
        self._checkboxes.clear()
        if df is None or df.empty:
            return

        channels = [c for c in df.columns if c != "phase"]
        self.table.setRowCount(len(channels))
        for row, channel in enumerate(channels):
            name_item = QTableWidgetItem(channel)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            plot_checkbox = QCheckBox()
            plot_checkbox.setChecked(True)
            plot_checkbox.stateChanged.connect(self._on_checkbox_changed)
            self.table.setCellWidget(row, 1, _centered(plot_checkbox))

            train_checkbox = QCheckBox()
            train_checkbox.setChecked(True)
            train_checkbox.stateChanged.connect(self._on_checkbox_changed)
            self.table.setCellWidget(row, 2, _centered(train_checkbox))

            series = df[channel]
            range_text = f"{series.min():.4g} .. {series.max():.4g}"
            range_item = QTableWidgetItem(range_text)
            range_item.setFlags(range_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 3, range_item)

            self._checkboxes[channel] = (plot_checkbox, train_checkbox)

    def _on_checkbox_changed(self, *_args) -> None:
        plot_channels = [ch for ch, (plot_cb, _train_cb) in self._checkboxes.items() if plot_cb.isChecked()]
        train_channels = [ch for ch, (_plot_cb, train_cb) in self._checkboxes.items() if train_cb.isChecked()]
        self.plotSelectionChanged.emit(plot_channels)
        self.trainSelectionChanged.emit(train_channels)
        self.state.set_channel_selection(plot_channels, train_channels)
