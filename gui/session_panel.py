"""Session panel: machine-type selector + background folder scan/group.

Lets the user pick a machine type, scans a folder in the background,
groups the discovered files into per-UUT sessions via
``src.session_grouping``, and lets the user pick one to load into the rest
of the app (plot/channel panels) via ``AppState``.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.app_state import AppState
from src.data_loader import scan_directory
from src.pipeline import DEFAULT_TRANSMISSION_PHASES
from src.session_grouping import MachineType, Phase, TestSession, group_files, load_session


class _ScanWorkerSignals(QObject):
    finished = Signal(list)  # list[TestSession]
    error = Signal(str)


class _ScanWorker(QRunnable):
    """Runs scan_directory + group_files off the UI thread."""

    def __init__(self, directory: Path, machine_type: MachineType, recursive: bool) -> None:
        super().__init__()
        self.directory = directory
        self.machine_type = machine_type
        self.recursive = recursive
        self.signals = _ScanWorkerSignals()

    def run(self) -> None:
        try:
            files = scan_directory(self.directory, recursive=self.recursive)
            sessions = group_files(files, machine_type=self.machine_type)
            self.signals.finished.emit(sessions)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, not raised on a worker thread
            self.signals.error.emit(str(exc))


class SessionPanel(QWidget):
    """Machine-type selector + session tree; loading a session updates AppState."""

    sessionLoaded = Signal(object, object)  # (TestSession, pd.DataFrame)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self._threadpool = QThreadPool.globalInstance()
        self._current_directory: Path | None = None
        self._sessions: list[TestSession] = []

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.machine_type_combo = QComboBox()
        for machine_type in MachineType:
            self.machine_type_combo.addItem(machine_type.value, machine_type)
        self.machine_type_combo.currentIndexChanged.connect(self._on_machine_type_changed)
        form.addRow("Machine type:", self.machine_type_combo)
        layout.addLayout(form)

        self.recursive_checkbox = QCheckBox("Scan subfolders")
        self.recursive_checkbox.setChecked(True)
        layout.addWidget(self.recursive_checkbox)

        browse_row = QHBoxLayout()
        self.browse_button = QPushButton("Browse Folder...")
        self.browse_button.clicked.connect(self._on_browse_clicked)
        browse_row.addWidget(self.browse_button)
        layout.addLayout(browse_row)

        self.status_label = QLabel("No folder selected.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Session / File", "Detail"])
        self.tree.setAlternatingRowColors(True)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)

    # -- public API -----------------------------------------------------

    def load_directory(self, directory: str | Path) -> None:
        """Kick off a background scan+group of ``directory`` using the current machine type."""
        self._current_directory = Path(directory)
        self.state.set_root_directory(self._current_directory)
        self.status_label.setText(f"Scanning {self._current_directory} ...")
        self.tree.clear()

        worker = _ScanWorker(
            self._current_directory,
            self.state.machine_type,
            self.recursive_checkbox.isChecked(),
        )
        worker.signals.finished.connect(self._on_scan_finished)
        worker.signals.error.connect(self._on_scan_error)
        self._threadpool.start(worker)

    # -- internals --------------------------------------------------------

    def _on_browse_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select a data folder")
        if directory:
            self.load_directory(directory)

    def _on_machine_type_changed(self, index: int) -> None:
        raw = self.machine_type_combo.itemData(index)
        if raw is None:
            return
        # PySide6's QVariant round-trip coerces a str-subclassed Enum (MachineType
        # is `str, Enum`) back into a plain str -- rebuild the real enum member
        # rather than trust itemData() to have preserved the type.
        machine_type = MachineType(raw)
        self.state.set_machine_type(machine_type)
        if self._current_directory is not None:
            self.load_directory(self._current_directory)

    def _on_scan_finished(self, sessions: list[TestSession]) -> None:
        self._sessions = sessions
        self.state.set_sessions(sessions)
        self._populate_tree(sessions)
        n_files = sum(len(s.files) for s in sessions)
        self.status_label.setText(
            f"Found {len(sessions)} session(s), {n_files} file(s) under {self._current_directory}."
        )

    def _on_scan_error(self, message: str) -> None:
        self.status_label.setText(f"Scan failed: {message}")

    def _populate_tree(self, sessions: list[TestSession]) -> None:
        self.tree.clear()
        for session in sessions:
            top = QTreeWidgetItem([session.uut_id, session.machine_type.value])
            top.setData(0, Qt.ItemDataRole.UserRole, session)
            for f in sorted(session.files):
                detail = ""
                if session.phase_map:
                    detail = session.phase_map.get(f, Phase.UNKNOWN).value
                elif session.pocket_map:
                    detail = f"pocket {session.pocket_map.get(f, '?')}"
                child = QTreeWidgetItem([f.name, detail])
                child.setData(0, Qt.ItemDataRole.UserRole, None)
                top.addChild(child)
            self.tree.addTopLevelItem(top)
        self.tree.expandAll()

    def _on_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        session = item.data(0, Qt.ItemDataRole.UserRole)
        if session is None:
            # A child (individual file) row was selected -- walk up to its session.
            parent = item.parent()
            session = parent.data(0, Qt.ItemDataRole.UserRole) if parent is not None else None
        if session is None:
            return
        self._load_session(session)

    def _load_session(self, session: TestSession) -> None:
        phases = DEFAULT_TRANSMISSION_PHASES if session.machine_type == MachineType.TRANSMISSION else None
        try:
            df = load_session(session, phases=phases)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, not raised
            self.status_label.setText(f"Failed to load session {session.uut_id}: {exc}")
            return
        self.state.set_current_session(session, df)
        self.sessionLoaded.emit(session, df)
