import subprocess
import sys
from pathlib import Path

import pandas as pd

from gui.app_state import AppState
from gui.main_window import MainWindow
from src.session_grouping import MachineType

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_main_window_constructs_and_shows(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    assert window.windowTitle() == "TDMS/CSV/Excel Anomaly Detection"
    assert window.centralWidget() is not None
    assert window.state.machine_type == MachineType.GENERAL


def test_browse_folder_action_exists(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.browse_action is not None
    assert window.browse_action.text() == "Browse Folder..."


def test_app_state_emits_on_session_load(qtbot):
    state = AppState()
    df = pd.DataFrame({"vibration": [1.0, 2.0, 3.0], "current": [4.0, 5.0, 6.0]})
    df.index.name = "sample"

    with qtbot.waitSignal(state.fileLoaded, timeout=1000) as blocker:
        state.set_current_session(None, df)

    assert blocker.args[0] is df
    assert state.plot_channels == ["vibration", "current"]
    assert state.train_channels == ["vibration", "current"]


def test_app_state_emits_on_channel_selection_change(qtbot):
    state = AppState()
    with qtbot.waitSignal(state.channelSelectionChanged, timeout=1000) as blocker:
        state.set_channel_selection(["vibration"], ["vibration"])
    assert blocker.args == [["vibration"], ["vibration"]]


def test_app_state_emits_on_time_range_change(qtbot):
    state = AppState()
    with qtbot.waitSignal(state.timeRangeChanged, timeout=1000) as blocker:
        state.set_time_range(10.0, 20.0)
    assert blocker.args == [10.0, 20.0]
    assert state.time_range == (10.0, 20.0)


def test_gui_launches_headlessly_via_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "gui.app", "--smoke-test-and-quit"],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
