from gui.app_state import AppState
from gui.session_panel import SessionPanel
from src.session_grouping import MachineType, Phase
from src.synthetic_sessions import write_transmission_session
from src.synthetic_tdms import _make_normal_signals, write_tdms


def test_load_directory_general_populates_tree(qtbot, tmp_path):
    for i in range(3):
        write_tdms(tmp_path / f"run_{i}.tdms", _make_normal_signals(n_samples=200, seed=i))

    state = AppState()
    panel = SessionPanel(state)
    qtbot.addWidget(panel)

    with qtbot.waitSignal(state.sessionsGrouped, timeout=5000):
        panel.load_directory(tmp_path)

    assert panel.tree.topLevelItemCount() == 3
    assert len(state.sessions) == 3
    assert all(s.machine_type == MachineType.GENERAL for s in state.sessions)


def test_selecting_session_updates_app_state(qtbot, tmp_path):
    write_tdms(tmp_path / "run_0.tdms", _make_normal_signals(n_samples=200, seed=0))

    state = AppState()
    panel = SessionPanel(state)
    qtbot.addWidget(panel)

    with qtbot.waitSignal(state.sessionsGrouped, timeout=5000):
        panel.load_directory(tmp_path)

    top_item = panel.tree.topLevelItem(0)
    with qtbot.waitSignal(state.fileLoaded, timeout=2000):
        panel.tree.setCurrentItem(top_item)

    assert state.current_df is not None
    assert set(state.plot_channels) == {"vibration", "temperature", "pressure"}
    assert set(state.train_channels) == {"vibration", "temperature", "pressure"}


def test_transmission_machine_type_defaults_to_steady_and_coasting(qtbot, tmp_path):
    write_transmission_session(tmp_path, uut_id="unit001", n_samples=200, seed=0)

    state = AppState()
    panel = SessionPanel(state)
    qtbot.addWidget(panel)

    transmission_index = panel.machine_type_combo.findData(MachineType.TRANSMISSION)
    assert transmission_index >= 0
    panel.machine_type_combo.setCurrentIndex(transmission_index)

    with qtbot.waitSignal(state.sessionsGrouped, timeout=5000):
        panel.load_directory(tmp_path)

    assert len(state.sessions) == 1
    session = state.sessions[0]
    assert session.machine_type == MachineType.TRANSMISSION
    assert set(session.phase_map.values()) == {
        Phase.RAMP_UP,
        Phase.STEADY_STATE,
        Phase.COASTING,
        Phase.RAMP_DOWN,
    }

    top_item = panel.tree.topLevelItem(0)
    with qtbot.waitSignal(state.fileLoaded, timeout=2000):
        panel.tree.setCurrentItem(top_item)

    assert state.current_df is not None
    assert set(state.current_df["phase"].unique()) == {"steady_state", "coasting"}


def test_non_recursive_scan_skips_subfolders(qtbot, tmp_path):
    write_tdms(tmp_path / "top.tdms", _make_normal_signals(n_samples=200, seed=0))
    nested = tmp_path / "sub"
    nested.mkdir()
    write_tdms(nested / "nested.tdms", _make_normal_signals(n_samples=200, seed=1))

    state = AppState()
    panel = SessionPanel(state)
    qtbot.addWidget(panel)
    panel.recursive_checkbox.setChecked(False)

    with qtbot.waitSignal(state.sessionsGrouped, timeout=5000):
        panel.load_directory(tmp_path)

    assert len(state.sessions) == 1
    assert state.sessions[0].files[0].name == "top.tdms"


def test_selecting_child_file_row_loads_its_parent_session(qtbot, tmp_path):
    write_transmission_session(tmp_path, uut_id="unit001", n_samples=200, seed=0)

    state = AppState()
    panel = SessionPanel(state)
    qtbot.addWidget(panel)

    transmission_index = panel.machine_type_combo.findData(MachineType.TRANSMISSION)
    panel.machine_type_combo.setCurrentIndex(transmission_index)

    with qtbot.waitSignal(state.sessionsGrouped, timeout=5000):
        panel.load_directory(tmp_path)

    top_item = panel.tree.topLevelItem(0)
    child_item = top_item.child(0)
    assert child_item is not None

    with qtbot.waitSignal(state.fileLoaded, timeout=2000):
        panel.tree.setCurrentItem(child_item)

    assert state.current_session is not None
    assert state.current_session.uut_id == "unit001"
