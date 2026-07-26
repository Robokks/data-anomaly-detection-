"""End-to-end smoke test driving the full workflow through MainWindow itself,
not just individual panels in isolation: browse -> scan/group -> select
session -> curate channels/time-range -> train -> results + plot overlay.
"""
from __future__ import annotations

from gui.main_window import MainWindow
from gui.train_dialog import TrainDialog
from src.session_grouping import MachineType
from src.synthetic_sessions import write_transmission_session
from src.synthetic_tdms import _make_normal_signals, write_tdms


def test_browse_folder_action_delegates_to_session_panel(qtbot, tmp_path, monkeypatch):
    for i in range(2):
        write_tdms(tmp_path / f"run_{i}.tdms", _make_normal_signals(n_samples=200, seed=i))

    window = MainWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr(
        "gui.session_panel.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )

    with qtbot.waitSignal(window.state.sessionsGrouped, timeout=5000):
        window._on_browse_folder()

    assert len(window.state.sessions) == 2


def test_full_classic_training_workflow_through_main_window(qtbot, tmp_path):
    for i in range(3):
        write_tdms(tmp_path / f"normal_{i}.tdms", _make_normal_signals(n_samples=4000, seed=i))

    window = MainWindow()
    qtbot.addWidget(window)

    with qtbot.waitSignal(window.state.sessionsGrouped, timeout=5000):
        window.session_panel.load_directory(tmp_path)

    top_item = window.session_panel.tree.topLevelItem(0)
    with qtbot.waitSignal(window.state.fileLoaded, timeout=2000):
        window.session_panel.tree.setCurrentItem(top_item)

    # ChannelPanel should have populated rows for the loaded session's channels.
    assert window.channel_panel.table.rowCount() == 3

    # Deselect one channel from training to confirm the curation step feeds through.
    _plot_cb, train_cb = window.channel_panel._checkboxes["temperature"]
    train_cb.setChecked(False)
    assert "temperature" not in window.state.train_channels

    # Drive training the same way the Train... toolbar action does, minus the
    # blocking exec() call (a modal dialog can't be driven headlessly here).
    dialog = TrainDialog(window.state, window)
    dialog.trainingFinished.connect(window._on_training_finished)
    dialog.window_size_spin.setValue(200)

    with qtbot.waitSignal(dialog.trainingFinished, timeout=15000):
        dialog._on_train_clicked()

    assert window.state.model is not None
    assert window.state.model_type == "classic"
    assert all(col.startswith(("vibration_", "pressure_")) for col in window.state.model.feature_columns_)

    assert "windows flagged" in window.results_panel.summary_label.text()
    assert window.results_panel.export_button.isEnabled()
    assert window.save_model_action.isEnabled()

    # Anomaly overlay should have added items to the plot beyond the curves + region.
    assert len(window.plot_panel.plot_widget.items()) > len(window.plot_panel._curves) + 1


def test_full_workflow_with_transmission_machine_type(qtbot, tmp_path):
    write_transmission_session(tmp_path, uut_id="unit001", n_samples=2000, seed=0)

    window = MainWindow()
    qtbot.addWidget(window)

    transmission_index = window.session_panel.machine_type_combo.findData(MachineType.TRANSMISSION)
    window.session_panel.machine_type_combo.setCurrentIndex(transmission_index)

    with qtbot.waitSignal(window.state.sessionsGrouped, timeout=5000):
        window.session_panel.load_directory(tmp_path)

    top_item = window.session_panel.tree.topLevelItem(0)
    with qtbot.waitSignal(window.state.fileLoaded, timeout=2000):
        window.session_panel.tree.setCurrentItem(top_item)

    # Default transmission phase filter (steady_state + coasting) should be
    # reflected in the loaded session's data, and flow through to the plot.
    assert set(window.state.current_df["phase"].unique()) == {"steady_state", "coasting"}
    assert "phase" not in window.channel_panel._checkboxes

    dialog = TrainDialog(window.state, window)
    dialog.trainingFinished.connect(window._on_training_finished)
    dialog.window_size_spin.setValue(100)

    with qtbot.waitSignal(dialog.trainingFinished, timeout=15000):
        dialog._on_train_clicked()

    assert window.state.model is not None
