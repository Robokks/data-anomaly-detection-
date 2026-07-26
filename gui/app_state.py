"""Shared application state, exposed as Qt signals so panels stay decoupled.

Each panel reacts to the signals it cares about instead of reaching into
other panels directly. Nothing here talks to Qt widgets -- this module has
no dependency on any specific panel, only on the ``src`` pipeline types.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Signal

from src.data_loader import FileInfo
from src.session_grouping import MachineType, TestSession


class AppState(QObject):
    # Emitted after a background folder scan finds files under the chosen root.
    filesScanned = Signal(list)  # list[FileInfo]
    # Emitted after scanned files are grouped into per-UUT sessions for the
    # current machine type.
    sessionsGrouped = Signal(list)  # list[TestSession]
    # Emitted once a session/file has been loaded into a working DataFrame.
    fileLoaded = Signal(object)  # pd.DataFrame
    # Emitted when the set of sessions selected in the session tree changes
    # (one when a single row is selected, more when multiple are Ctrl/Shift-
    # selected for combined/batch training).
    trainingSessionsChanged = Signal(list)  # list[TestSession]
    # Emitted when the user toggles Plot/Train checkboxes in the channel panel.
    channelSelectionChanged = Signal(list, list)  # (plot_channels, train_channels)
    # Emitted when the time-range region/fields change.
    timeRangeChanged = Signal(object, object)  # (start, end)
    # Emitted once a model finishes training.
    modelTrained = Signal(object, str)  # (model, model_type)

    def __init__(self) -> None:
        super().__init__()
        self.root_directory: Path | None = None
        self.machine_type: MachineType = MachineType.GENERAL
        self.recursive: bool = True

        self.files: list[FileInfo] = []
        self.sessions: list[TestSession] = []
        self.current_session: TestSession | None = None
        self.current_df: pd.DataFrame | None = None
        self.training_sessions: list[TestSession] = []

        self.plot_channels: list[str] = []
        self.train_channels: list[str] = []
        self.time_range: tuple = (None, None)

        self.model = None
        self.model_type: str | None = None
        self.last_scores: pd.DataFrame | None = None

    def set_root_directory(self, path: str | Path) -> None:
        self.root_directory = Path(path)

    def set_machine_type(self, machine_type: MachineType) -> None:
        self.machine_type = machine_type

    def set_files(self, files: list[FileInfo]) -> None:
        self.files = files
        self.filesScanned.emit(files)

    def set_sessions(self, sessions: list[TestSession]) -> None:
        self.sessions = sessions
        self.sessionsGrouped.emit(sessions)

    def set_current_session(self, session: TestSession | None, df: pd.DataFrame | None) -> None:
        self.current_session = session
        self.current_df = df
        all_channels = [c for c in df.columns if c != "phase"] if df is not None else []
        self.plot_channels = list(all_channels)
        self.train_channels = list(all_channels)
        self.time_range = (None, None)
        self.fileLoaded.emit(df)
        self.channelSelectionChanged.emit(self.plot_channels, self.train_channels)
        self.timeRangeChanged.emit(None, None)

    def set_training_sessions(self, sessions: list[TestSession]) -> None:
        self.training_sessions = list(sessions)
        self.trainingSessionsChanged.emit(self.training_sessions)

    def set_channel_selection(self, plot_channels: list[str], train_channels: list[str]) -> None:
        self.plot_channels = list(plot_channels)
        self.train_channels = list(train_channels)
        self.channelSelectionChanged.emit(self.plot_channels, self.train_channels)

    def set_time_range(self, start, end) -> None:
        self.time_range = (start, end)
        self.timeRangeChanged.emit(start, end)

    def set_model(self, model, model_type: str) -> None:
        self.model = model
        self.model_type = model_type
        self.modelTrained.emit(model, model_type)
