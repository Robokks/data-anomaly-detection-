"""Plot panel: multi-channel time-series plot with a draggable time-range region.

pyqtgraph (not matplotlib) so it stays responsive at real sensor sample
counts via setDownsampling/setClipToView, and so the time-range selector
can be a native, draggable LinearRegionItem synced to explicit start/end
fields -- matches "full data or from this point in time to that point in
time."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from gui.app_state import AppState

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


class PlotPanel(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self._df: pd.DataFrame | None = None
        self._x: np.ndarray | None = None
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._is_datetime_index = False

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Start:"))
        self.start_field = QLineEdit()
        controls.addWidget(self.start_field)
        controls.addWidget(QLabel("End:"))
        self.end_field = QLineEdit()
        controls.addWidget(self.end_field)
        self.full_data_button = QPushButton("Full Data")
        controls.addWidget(self.full_data_button)
        layout.addLayout(controls)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.setDownsampling(auto=True, mode="peak")
        self.plot_widget.setClipToView(True)
        layout.addWidget(self.plot_widget)

        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.region.hide()
        self.plot_widget.addItem(self.region)

        self.region.sigRegionChanged.connect(self._update_fields_from_region)
        self.region.sigRegionChangeFinished.connect(self._on_region_changed)
        self.start_field.editingFinished.connect(self._on_fields_edited)
        self.end_field.editingFinished.connect(self._on_fields_edited)
        self.full_data_button.clicked.connect(self._on_full_data_clicked)

        self.state.fileLoaded.connect(self._on_file_loaded)
        self.state.channelSelectionChanged.connect(self._on_channel_selection_changed)

    # -- data loading -----------------------------------------------------

    def _on_file_loaded(self, df: pd.DataFrame | None) -> None:
        self._df = df
        self._curves.clear()
        self.plot_widget.clear()
        self.plot_widget.addItem(self.region)

        if df is None or df.empty:
            self._x = None
            self.region.hide()
            return

        self._is_datetime_index = bool(np.issubdtype(df.index.dtype, np.datetime64))
        if self._is_datetime_index:
            self._x = df.index.values.astype("datetime64[ns]").astype(np.int64) / 1e9
        else:
            self._x = np.asarray(df.index.values, dtype=float)

        lo, hi = float(self._x.min()), float(self._x.max())
        self.region.setBounds([lo, hi])
        self.region.setRegion([lo, hi])
        self.region.show()
        self._update_fields_from_region()
        self._redraw_channels(self.state.plot_channels)

    def _on_channel_selection_changed(self, plot_channels: list[str], _train_channels: list[str]) -> None:
        self._redraw_channels(plot_channels)

    def _redraw_channels(self, channels: list[str]) -> None:
        if self._df is None or self._x is None:
            return
        for channel in list(self._curves):
            if channel not in channels:
                self.plot_widget.removeItem(self._curves.pop(channel))
        for i, channel in enumerate(channels):
            if channel == "phase" or channel not in self._df.columns or channel in self._curves:
                continue
            color = _COLORS[i % len(_COLORS)]
            curve = self.plot_widget.plot(
                self._x,
                self._df[channel].to_numpy(dtype=float),
                pen=pg.mkPen(color=color, width=1),
                name=channel,
            )
            self._curves[channel] = curve

    # -- time-range region <-> fields --------------------------------------

    def _format_index_value(self, x_value: float) -> str:
        if self._is_datetime_index:
            return str(pd.Timestamp(int(x_value * 1e9)))
        return f"{x_value:.6g}"

    def _parse_index_value(self, text: str):
        text = text.strip()
        if self._is_datetime_index:
            return pd.Timestamp(text)
        return float(text)

    def _to_x(self, index_value) -> float:
        if self._is_datetime_index:
            return pd.Timestamp(index_value).value / 1e9
        return float(index_value)

    def _to_index_value(self, x_value: float):
        if self._is_datetime_index:
            return pd.Timestamp(int(x_value * 1e9))
        return x_value

    def _update_fields_from_region(self) -> None:
        if self._x is None:
            return
        lo, hi = self.region.getRegion()
        self.start_field.setText(self._format_index_value(lo))
        self.end_field.setText(self._format_index_value(hi))

    def _on_fields_edited(self) -> None:
        if self._x is None:
            return
        try:
            lo = self._to_x(self._parse_index_value(self.start_field.text()))
            hi = self._to_x(self._parse_index_value(self.end_field.text()))
        except (ValueError, TypeError):
            return
        if lo > hi:
            return
        self.region.setRegion([lo, hi])

    def _on_full_data_clicked(self) -> None:
        if self._x is None:
            return
        self.region.setRegion([float(self._x.min()), float(self._x.max())])

    def _on_region_changed(self) -> None:
        if self._x is None:
            return
        lo, hi = self.region.getRegion()
        full_lo, full_hi = float(self._x.min()), float(self._x.max())
        start = None if np.isclose(lo, full_lo) else self._to_index_value(lo)
        end = None if np.isclose(hi, full_hi) else self._to_index_value(hi)
        self.state.set_time_range(start, end)

    def overlay_anomalies(self, scores: pd.DataFrame) -> None:
        """Shade flagged windows on the plot. scores must have an 'is_anomaly'
        boolean column indexed the same way as the loaded DataFrame's windows."""
        for item in list(self.plot_widget.items()):
            if isinstance(item, pg.LinearRegionItem) and item is not self.region:
                self.plot_widget.removeItem(item)
        if scores is None or scores.empty or "is_anomaly" not in scores.columns:
            return
        flagged = scores.index[scores["is_anomaly"]]
        for idx in flagged:
            try:
                x_value = self._to_x(idx)
            except (ValueError, TypeError):
                continue
            marker = pg.LinearRegionItem(
                values=[x_value, x_value], movable=False, brush=pg.mkBrush(220, 20, 60, 60)
            )
            marker.setZValue(-10)
            self.plot_widget.addItem(marker)
