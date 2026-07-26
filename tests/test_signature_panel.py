import numpy as np
import pandas as pd

from gui.app_state import AppState
from gui.signature_panel import SignatureDialog


def _make_signal_df(n_samples=4000, freq=5.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 100.0
    vibration = np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.05, n_samples)
    other = np.cos(2 * np.pi * 2.0 * t) + rng.normal(0, 0.05, n_samples)
    df = pd.DataFrame({"vibration": vibration, "other": other})
    df.index.name = "sample"
    df.attrs["sample_rate_hz"] = 100.0
    return df


def test_signature_dialog_populates_channels(qtbot):
    state = AppState()
    state.set_current_session(None, _make_signal_df())

    dialog = SignatureDialog(state)
    qtbot.addWidget(dialog)

    items = [dialog.channel_combo.itemText(i) for i in range(dialog.channel_combo.count())]
    assert set(items) == {"vibration", "other"}


def test_compute_signature_fits_baseline_and_plots(qtbot):
    state = AppState()
    state.set_current_session(None, _make_signal_df())

    dialog = SignatureDialog(state)
    qtbot.addWidget(dialog)
    dialog.window_size_spin.setValue(200)

    dialog._on_compute_clicked()

    assert dialog._baseline is not None
    assert dialog._baseline.channel == "vibration"
    assert dialog._baseline.n_windows_ == 4000 // 200
    assert "Fit baseline from" in dialog.status_label.text()
    assert len(dialog.plot_widget.items()) > 0


def test_compute_signature_shows_deviation_when_lengths_match(qtbot):
    state = AppState()
    df = _make_signal_df()
    state.set_current_session(None, df)
    # Restrict the time range to exactly one window's worth of samples.
    state.set_time_range(0, 199)

    dialog = SignatureDialog(state)
    qtbot.addWidget(dialog)
    dialog.window_size_spin.setValue(200)

    dialog._on_compute_clicked()

    assert "deviation from baseline" in dialog.status_label.text()


def test_compute_signature_handles_too_short_session(qtbot):
    state = AppState()
    df = _make_signal_df(n_samples=50)
    state.set_current_session(None, df)

    dialog = SignatureDialog(state)
    qtbot.addWidget(dialog)
    dialog.window_size_spin.setValue(200)

    dialog._on_compute_clicked()

    assert "shorter than window size" in dialog.status_label.text()


def test_compute_signature_with_no_session_shows_message(qtbot):
    state = AppState()
    dialog = SignatureDialog(state)
    qtbot.addWidget(dialog)

    dialog._on_compute_clicked()

    assert "load a session" in dialog.status_label.text()
