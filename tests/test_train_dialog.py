from gui.app_state import AppState
from gui.train_dialog import TrainDialog
from src.data_loader import load_dataframe
from src.synthetic_tdms import _make_normal_signals, write_tdms


def _load_normal_df(tmp_path, n_samples=4000, seed=0):
    path = tmp_path / "normal.tdms"
    write_tdms(path, _make_normal_signals(n_samples=n_samples, seed=seed))
    return load_dataframe(path)


def test_train_dialog_ok_disabled_without_session(qtbot):
    state = AppState()
    dialog = TrainDialog(state)
    qtbot.addWidget(dialog)

    ok_button = dialog.button_box.button(dialog.button_box.StandardButton.Ok)
    assert not ok_button.isEnabled()
    assert "No session loaded" in dialog.info_label.text()


def test_train_dialog_classic_training_end_to_end(qtbot, tmp_path):
    state = AppState()
    state.set_current_session(None, _load_normal_df(tmp_path))

    dialog = TrainDialog(state)
    qtbot.addWidget(dialog)

    ok_button = dialog.button_box.button(dialog.button_box.StandardButton.Ok)
    assert ok_button.isEnabled()
    assert dialog.progress_bar.isHidden()

    dialog._on_train_clicked()
    assert not dialog.progress_bar.isHidden()

    with qtbot.waitSignal(dialog.trainingFinished, timeout=15000) as blocker:
        pass

    model, model_type, scores = blocker.args
    assert model_type == "classic"
    assert "anomaly_score" in scores.columns
    assert "is_anomaly" in scores.columns
    assert state.model is model
    assert state.model_type == "classic"
    assert dialog.progress_bar.isHidden()


def test_train_dialog_respects_train_channel_selection(qtbot, tmp_path):
    state = AppState()
    state.set_current_session(None, _load_normal_df(tmp_path))
    state.set_channel_selection(state.plot_channels, ["vibration"])

    dialog = TrainDialog(state)
    qtbot.addWidget(dialog)

    with qtbot.waitSignal(dialog.trainingFinished, timeout=15000) as blocker:
        dialog._on_train_clicked()

    model, _model_type, _scores = blocker.args
    assert all(col.startswith("vibration_") for col in model.feature_columns_)


def test_train_dialog_deep_learning_end_to_end(qtbot, tmp_path):
    state = AppState()
    state.set_current_session(None, _load_normal_df(tmp_path, n_samples=2000))

    dialog = TrainDialog(state)
    qtbot.addWidget(dialog)

    deep_index = dialog.model_type_combo.findData("deep")
    dialog.model_type_combo.setCurrentIndex(deep_index)
    dialog.window_size_spin.setValue(100)
    dialog.epochs_spin.setValue(5)

    with qtbot.waitSignal(dialog.trainingFinished, timeout=60000) as blocker:
        dialog._on_train_clicked()

    model, model_type, scores = blocker.args
    assert model_type == "deep"
    assert "recon_error" in scores.columns
    assert state.model_type == "deep"


def test_train_dialog_error_with_no_channels_selected(qtbot, tmp_path):
    state = AppState()
    state.set_current_session(None, _load_normal_df(tmp_path))
    state.set_channel_selection(state.plot_channels, [])

    dialog = TrainDialog(state)
    qtbot.addWidget(dialog)

    # Info label reflects zero selected channels and disables the button.
    ok_button = dialog.button_box.button(dialog.button_box.StandardButton.Ok)
    assert not ok_button.isEnabled()
