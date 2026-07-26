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


def _make_multiscale_df():
    # Wildly different magnitudes -- vibration ~[-0.5, 0.5], speed ~[0, 6000],
    # temperature ~[39, 41] -- the exact case multi-axis scaling is for.
    df = pd.DataFrame(
        {
            "vibration": 0.5 * np.sin(np.linspace(0, 10, 50)),
            "speed": np.linspace(0, 6000, 50),
            "temperature": 40 + 0.2 * np.sin(np.linspace(0, 5, 50)),
        }
    )
    df.index.name = "sample"
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


def test_multi_channel_plot_gets_one_secondary_axis_per_extra_channel(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)

    state.set_current_session(None, _make_multiscale_df())

    assert set(panel._curves.keys()) == {"vibration", "speed", "temperature"}
    # First channel uses the plot's own built-in axis/viewbox; the other two
    # each get their own secondary ViewBox + AxisItem.
    assert len(panel._channel_viewboxes) == 2
    assert len(panel._channel_axes) == 2
    assert set(panel._channel_viewboxes.keys()) == {"speed", "temperature"}


def test_secondary_viewbox_autoranges_to_its_own_channel_data(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)

    state.set_current_session(None, _make_multiscale_df())

    speed_vb = panel._channel_viewboxes["speed"]
    y_lo, y_hi = speed_vb.viewRange()[1]
    # speed ranges 0..6000 -- its own viewbox should scale to that, not to
    # vibration's ~[-0.5, 0.5] range (which the shared-axis version would
    # have flattened it against).
    assert y_hi > 1000


def test_deselecting_a_channel_cleans_up_its_secondary_axis(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_multiscale_df())
    assert "speed" in panel._channel_viewboxes

    state.set_channel_selection(["vibration", "temperature"], state.train_channels)

    assert "speed" not in panel._curves
    assert "speed" not in panel._channel_viewboxes
    assert "speed" not in panel._channel_axes
    assert set(panel._curves.keys()) == {"vibration", "temperature"}
    assert len(panel._channel_viewboxes) == 1


def test_redraw_does_not_duplicate_legend_entries(qtbot):
    state = AppState()
    panel = PlotPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_multiscale_df())

    # Toggling the theme (or any other trigger of _redraw_channels) with the
    # same channels selected must not accumulate duplicate legend rows.
    panel._apply_theme(panel.theme_manager.mode)
    panel._apply_theme(panel.theme_manager.mode)

    assert len(panel.plot_item.legend.items) == 3
