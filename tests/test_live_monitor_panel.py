import threading

from gui.app_state import AppState
from gui.live_monitor_panel import LiveMonitorPanel
from src.data_loader import load_directory
from src.live_scorer import STAT_NAMES
from src.pipeline import train_model
from src.stream_simulator import stream_signals
from src.synthetic_tdms import _make_normal_signals, write_tdms

SAMPLE_RATE_HZ = 1000
WINDOW_SAMPLES = 200


def _write_baseline_files(directory, n_files=3, n_samples=4000):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        write_tdms(directory / f"normal_{i:02d}.tdms", _make_normal_signals(n_samples=n_samples, seed=i))


def test_live_monitor_panel_use_current_model_autofills_window_size(qtbot, tmp_path):
    baseline_dir = tmp_path / "baseline"
    _write_baseline_files(baseline_dir)
    frames = load_directory(baseline_dir)
    model = train_model("classic", frames, window_size=WINDOW_SAMPLES, contamination=0.05)

    state = AppState()
    state.set_model(model, "classic")

    panel = LiveMonitorPanel(state)
    qtbot.addWidget(panel)
    panel._on_use_current_model_clicked()

    assert panel._model is model
    assert panel._model_type == "classic"
    assert panel.window_size_spin.value() == WINDOW_SAMPLES


def test_live_monitor_panel_start_stop_and_receives_windows(qtbot, tmp_path):
    baseline_dir = tmp_path / "baseline"
    _write_baseline_files(baseline_dir)
    frames = load_directory(baseline_dir)
    model = train_model("classic", frames, window_size=WINDOW_SAMPLES, contamination=0.05)

    state = AppState()
    state.set_model(model, "classic")

    panel = LiveMonitorPanel(state)
    qtbot.addWidget(panel)
    panel._on_use_current_model_clicked()

    panel.baseline_dir_edit.setText(str(baseline_dir))
    panel.sample_rate_spin.setValue(SAMPLE_RATE_HZ)
    panel.window_duration_spin.setValue(WINDOW_SAMPLES / SAMPLE_RATE_HZ)
    panel.host_edit.setText("127.0.0.1")
    panel.port_spin.setValue(0)

    panel._on_start_clicked()
    assert panel._server is not None
    assert not panel.start_button.isEnabled()
    assert panel.stop_button.isEnabled()

    bound_port = panel.port_spin.value()
    assert bound_port != 0

    n_channels = len(frames[0].columns)
    assert panel.table.rowCount() == n_channels + 1

    signals = _make_normal_signals(n_samples=2000, seed=50)
    stream_thread = threading.Thread(
        target=stream_signals,
        kwargs=dict(
            host="127.0.0.1", port=bound_port, signals=signals, sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=200, batch_interval_ms=5, realtime=False,
        ),
    )
    stream_thread.start()

    with qtbot.waitSignal(panel.windowScored, timeout=10000):
        pass
    stream_thread.join(timeout=10)

    # At least one channel row and the Overall row should have a populated score cell.
    score_col = 1 + len(STAT_NAMES)
    overall_row = panel._row_for_channel["Overall"]
    any_channel_row = next(r for name, r in panel._row_for_channel.items() if name != "Overall")
    assert panel.table.item(any_channel_row, score_col).text() != ""
    assert panel.table.item(overall_row, 0).text() == "Overall"

    panel._on_stop_clicked()
    assert panel._server is None
    assert panel.start_button.isEnabled()
    assert not panel.stop_button.isEnabled()


def test_live_monitor_panel_close_event_stops_server(qtbot, tmp_path):
    baseline_dir = tmp_path / "baseline"
    _write_baseline_files(baseline_dir)
    frames = load_directory(baseline_dir)
    model = train_model("classic", frames, window_size=WINDOW_SAMPLES, contamination=0.05)

    state = AppState()
    state.set_model(model, "classic")

    panel = LiveMonitorPanel(state)
    qtbot.addWidget(panel)
    panel._on_use_current_model_clicked()
    panel.baseline_dir_edit.setText(str(baseline_dir))
    panel.sample_rate_spin.setValue(SAMPLE_RATE_HZ)
    panel.window_duration_spin.setValue(WINDOW_SAMPLES / SAMPLE_RATE_HZ)
    panel.host_edit.setText("127.0.0.1")
    panel.port_spin.setValue(0)

    panel._on_start_clicked()
    assert panel._server is not None

    panel.close()
    assert panel._server is None


def test_live_monitor_panel_requires_model_before_start(qtbot, tmp_path):
    baseline_dir = tmp_path / "baseline"
    _write_baseline_files(baseline_dir)

    state = AppState()
    panel = LiveMonitorPanel(state)
    qtbot.addWidget(panel)
    panel.baseline_dir_edit.setText(str(baseline_dir))
    panel.port_spin.setValue(0)

    panel._on_start_clicked()

    assert panel._server is None
    assert "Select a model" in panel.status_label.text()
