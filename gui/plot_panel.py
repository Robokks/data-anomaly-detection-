"""Plot panel: multi-channel time-series plot with a draggable time-range region.

pyqtgraph (not matplotlib) so it stays responsive at real sensor sample
counts via setDownsampling/setClipToView, and so the time-range selector
can be a native, draggable LinearRegionItem synced to explicit start/end
fields -- matches "full data or from this point in time to that point in
time."

**Multiple Y scales**: channels can have wildly different magnitudes (e.g.
vibration ~0-1 vs. speed ~0-6000), so plotting them all against one shared
Y-axis makes the smaller-magnitude ones unreadable. Each additional channel
beyond the first gets its own Y-axis + independently-auto-ranging
``pg.ViewBox``, all X-linked to the main plot so panning/zooming stays in
sync -- the standard pyqtgraph "multiple Y axes" pattern (see their
MultiplePlotAxes example). The axis pen/label color matches that channel's
curve color, since there's no room for a legend entry per axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from gui.app_state import AppState
from gui.theme import ThemeManager, plot_colors


class PlotPanel(QWidget):
    def __init__(self, state: AppState, parent: QWidget | None = None, theme_manager: ThemeManager | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.theme_manager = theme_manager or ThemeManager()
        self._df: pd.DataFrame | None = None
        self._x: np.ndarray | None = None
        self._is_datetime_index = False

        # Primary channel (first in the current selection) uses the
        # PlotItem's own built-in left axis/ViewBox. Every channel after
        # that gets its own secondary ViewBox + AxisItem (see _add_secondary_axis).
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._channel_viewboxes: dict[str, pg.ViewBox] = {}
        self._channel_axes: dict[str, pg.AxisItem] = {}

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
        self.plot_item = self.plot_widget.getPlotItem()
        self.main_vb = self.plot_item.vb
        self.plot_widget.addLegend()
        self.plot_widget.setDownsampling(auto=True, mode="peak")
        self.plot_widget.setClipToView(True)
        layout.addWidget(self.plot_widget)

        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.region.hide()
        self.plot_widget.addItem(self.region)

        # Keeps every secondary channel ViewBox's screen geometry (and X
        # range) matched to the main plot's -- required boilerplate for the
        # multi-ViewBox-per-axis pattern; without it, secondary curves drift
        # out of alignment on resize/zoom/pan.
        self.main_vb.sigResized.connect(self._update_secondary_viewbox_geometry)

        self.region.sigRegionChanged.connect(self._update_fields_from_region)
        self.region.sigRegionChangeFinished.connect(self._on_region_changed)
        self.start_field.editingFinished.connect(self._on_fields_edited)
        self.end_field.editingFinished.connect(self._on_fields_edited)
        self.full_data_button.clicked.connect(self._on_full_data_clicked)

        self.state.fileLoaded.connect(self._on_file_loaded)
        self.state.channelSelectionChanged.connect(self._on_channel_selection_changed)

        self.theme_manager.themeChanged.connect(self._apply_theme)
        self._apply_theme(self.theme_manager.mode)

    # -- theming ------------------------------------------------------------

    def _apply_theme(self, mode: str) -> None:
        palette = self.theme_manager.current_palette
        self.plot_widget.setBackground(palette.surface)
        grid_alpha = 0.15 if mode == "dark" else 0.3
        self.plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)
        bottom_axis = self.plot_item.getAxis("bottom")
        bottom_axis.setPen(pg.mkPen(palette.border))
        bottom_axis.setTextPen(pg.mkPen(palette.text_secondary))

        # A very low-alpha fill + accent-colored edges: at the default "full
        # data" selection the region spans the whole plot, so a stronger fill
        # would otherwise wash out every curve underneath it.
        region_fill = pg.mkColor(palette.accent)
        region_fill.setAlpha(18)
        self.region.setBrush(pg.mkBrush(region_fill))
        self.region.setHoverBrush(pg.mkBrush(region_fill))
        edge_pen = pg.mkPen(palette.accent, width=1.5)
        for line in self.region.lines:
            line.setPen(edge_pen)

        # Curve/axis colors are palette-dependent -- full rebuild so they
        # pick up the new theme's colors.
        self._redraw_channels(self.state.plot_channels)

    # -- data loading -----------------------------------------------------

    def _on_file_loaded(self, df: pd.DataFrame | None) -> None:
        self._df = df
        self._clear_channel_items()
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

    # -- multi-axis channel rendering ----------------------------------------

    def _clear_channel_items(self) -> None:
        for channel, vb in list(self._channel_viewboxes.items()):
            curve = self._curves.pop(channel, None)
            if curve is not None:
                vb.removeItem(curve)
            axis = self._channel_axes.pop(channel, None)
            if axis is not None:
                self.plot_item.layout.removeItem(axis)
                axis.setParentItem(None)
            self.plot_item.scene().removeItem(vb)
        self._channel_viewboxes.clear()
        self._channel_axes.clear()

        for curve in self._curves.values():
            self.plot_item.removeItem(curve)
        self._curves.clear()

        if self.plot_item.legend is not None:
            self.plot_item.legend.clear()

    def _add_secondary_axis(self, channel: str, color: str) -> tuple[pg.ViewBox, pg.AxisItem]:
        axis = pg.AxisItem("right")
        axis.setPen(pg.mkPen(color))
        axis.setTextPen(pg.mkPen(color))
        axis.setLabel(channel, color=color)
        # Column 3 is the first free slot to the right of the main plot's
        # own left-axis(1)/viewbox(2) columns; each further secondary axis
        # takes the next column over.
        col = 3 + len(self._channel_viewboxes)
        self.plot_item.layout.addItem(axis, 2, col)

        vb = pg.ViewBox()
        self.plot_item.scene().addItem(vb)
        axis.linkToView(vb)
        vb.setXLink(self.main_vb)
        return vb, axis

    def _update_secondary_viewbox_geometry(self) -> None:
        rect = self.main_vb.sceneBoundingRect()
        for vb in self._channel_viewboxes.values():
            vb.setGeometry(rect)
            vb.linkedViewChanged(self.main_vb, vb.XAxis)

    def _redraw_channels(self, channels: list[str]) -> None:
        self._clear_channel_items()
        if self._df is None or self._x is None:
            return

        colors = plot_colors(self.theme_manager.mode)
        palette = self.theme_manager.current_palette
        valid_channels = [c for c in channels if c != "phase" and c in self._df.columns]

        for i, channel in enumerate(valid_channels):
            color = colors[i % len(colors)]
            y = self._df[channel].to_numpy(dtype=float)
            if i == 0:
                curve = self.plot_item.plot(self._x, y, pen=pg.mkPen(color=color, width=1.5), name=channel)
                left_axis = self.plot_item.getAxis("left")
                left_axis.setPen(pg.mkPen(color))
                left_axis.setTextPen(pg.mkPen(color))
                left_axis.setLabel(channel, color=color)
            else:
                vb, axis = self._add_secondary_axis(channel, color)
                curve = pg.PlotDataItem(self._x, y, pen=pg.mkPen(color=color, width=1.5))
                vb.addItem(curve)
                vb.autoRange()
                if self.plot_item.legend is not None:
                    self.plot_item.legend.addItem(curve, channel)
                self._channel_viewboxes[channel] = vb
                self._channel_axes[channel] = axis
            self._curves[channel] = curve

        if not valid_channels:
            left_axis = self.plot_item.getAxis("left")
            left_axis.setPen(pg.mkPen(palette.border))
            left_axis.setTextPen(pg.mkPen(palette.text_secondary))
            left_axis.setLabel(None)

        self._update_secondary_viewbox_geometry()

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
        danger = pg.mkColor(self.theme_manager.current_palette.danger)
        danger.setAlpha(60)
        for idx in flagged:
            try:
                x_value = self._to_x(idx)
            except (ValueError, TypeError):
                continue
            marker = pg.LinearRegionItem(
                values=[x_value, x_value], movable=False, brush=pg.mkBrush(danger)
            )
            marker.setZValue(-10)
            self.plot_widget.addItem(marker)
