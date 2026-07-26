import pandas as pd

from gui.app_state import AppState
from gui.channel_panel import ChannelPanel


def _make_df():
    df = pd.DataFrame({"vibration": [1.0, 2.0, 3.0], "current": [4.0, 5.0, 6.0]})
    df.index.name = "sample"
    return df


def test_channel_panel_populates_rows_on_file_loaded(qtbot):
    state = AppState()
    panel = ChannelPanel(state)
    qtbot.addWidget(panel)

    state.set_current_session(None, _make_df())

    assert panel.table.rowCount() == 2
    names = {panel.table.item(row, 0).text() for row in range(panel.table.rowCount())}
    assert names == {"vibration", "current"}


def test_channel_panel_all_checked_by_default(qtbot):
    state = AppState()
    panel = ChannelPanel(state)
    qtbot.addWidget(panel)

    state.set_current_session(None, _make_df())

    for plot_cb, train_cb in panel._checkboxes.values():
        assert plot_cb.isChecked()
        assert train_cb.isChecked()


def test_unchecking_train_updates_app_state_without_affecting_plot(qtbot):
    state = AppState()
    panel = ChannelPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_df())

    plot_cb, train_cb = panel._checkboxes["current"]
    with qtbot.waitSignal(state.channelSelectionChanged, timeout=1000) as blocker:
        train_cb.setChecked(False)

    plot_channels, train_channels = blocker.args
    assert set(plot_channels) == {"vibration", "current"}
    assert set(train_channels) == {"vibration"}
    assert state.plot_channels == plot_channels
    assert state.train_channels == train_channels


def test_unchecking_plot_removes_from_plot_selection(qtbot):
    state = AppState()
    panel = ChannelPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_df())

    plot_cb, _train_cb = panel._checkboxes["vibration"]
    with qtbot.waitSignal(state.channelSelectionChanged, timeout=1000) as blocker:
        plot_cb.setChecked(False)

    plot_channels, train_channels = blocker.args
    assert set(plot_channels) == {"current"}
    assert set(train_channels) == {"vibration", "current"}


def test_channel_panel_clears_on_empty_dataframe(qtbot):
    state = AppState()
    panel = ChannelPanel(state)
    qtbot.addWidget(panel)
    state.set_current_session(None, _make_df())
    assert panel.table.rowCount() == 2

    state.set_current_session(None, pd.DataFrame())
    assert panel.table.rowCount() == 0
