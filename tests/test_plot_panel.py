import numpy as np
import pandas as pd

from gui.app_state import AppState
from gui.plot_panel import PlotPanel


def _make_sample_df():
    df = pd.DataFrame({"vibration": np.sin(np.linspace(0, 10, 50)), "current": np.linspace(0, 1, 50)})
    df.index.name = "sample"
    return df


def _make_datetime_df():
    idx = pd.date_range("2024-01-01", periods=50, freq="s")
    df = pd.DataFrame({"vibration": np.sin(np.linspace(0, 10, 50))}, index=idx)
    df.index.name = "time"
    return df


def test_plot_panel_draws_curves_for_selected_channels(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)

    state.set_current_session(None, _make_sample_df())

    assert set(panel._curves.keys()) == {"vibration", "current"}


def test_plot_panel_region_bounds_match_full_data(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)

    df = _make_sample_df()
    state.set_current_session(None, df)

    lo, hi = panel.region.getRegion()
    assert lo == 0
    assert hi == len(df) - 1


def test_channel_deselection_removes_curve(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_sample_df())

    state.set_channel_selection(["vibration"], ["vibration", "current"])
    assert set(panel._curves.keys()) == {"vibration"}


def test_dragging_region_updates_app_state_time_range(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_sample_df())

    with qtbot.waitSignal(state.timeRangeChanged, timeout=1000) as blocker:
        panel.region.setRegion([10, 30])

    start, end = blocker.args
    assert start == 10
    assert end == 30


def test_full_data_button_resets_region_and_clears_time_range(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)
    df = _make_sample_df()
    state.set_current_session(None, df)
    panel.region.setRegion([10, 30])

    with qtbot.waitSignal(state.timeRangeChanged, timeout=1000) as blocker:
        panel.full_data_button.click()

    start, end = blocker.args
    assert start is None
    assert end is None
    lo, hi = panel.region.getRegion()
    assert lo == 0
    assert hi == len(df) - 1


def test_editing_start_end_fields_updates_region(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_sample_df())

    panel.start_field.setText("5")
    panel.end_field.setText("20")
    with qtbot.waitSignal(state.timeRangeChanged, timeout=1000):
        panel.end_field.editingFinished.emit()

    lo, hi = panel.region.getRegion()
    assert lo == 5
    assert hi == 20


def test_plot_panel_handles_datetime_index(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)

    df = _make_datetime_df()
    state.set_current_session(None, df)

    assert panel._is_datetime_index is True
    assert "vibration" in panel._curves
    assert panel.start_field.text() != ""
    assert panel.end_field.text() != ""


def test_overlay_anomalies_adds_markers_for_flagged_windows(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)
    df = _make_sample_df()
    state.set_current_session(None, df)

    scores = pd.DataFrame(
        {"anomaly_score": [0.1, 5.0, 0.2], "is_anomaly": [False, True, False]},
        index=[0, 10, 20],
    )
    panel.overlay_anomalies(scores)
    # No exception, and at least one extra item beyond curves/region was added.
    assert len(panel.plot_widget.items()) > len(panel._curves) + 1
