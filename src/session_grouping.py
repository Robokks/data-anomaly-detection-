"""Group multi-file test-rig data into logical UUT (unit-under-test) sessions.

Depending on ``MachineType`` a single UUT's test run is logged to a different
number of files on disk:

- ``GENERAL``: one file == one UUT. Pass-through.
- ``TRANSMISSION``: one UUT's test cycle is split across several files, one
  per gear/operating phase (ramp-up, ramp-down, steady state, coasting).
  Analysts mostly care about steady-state and coasting; ramp phases are
  transients.
- ``MOTOR_TEST_BENCH``: one UUT's files are split by test step / "pocket"
  number (a pocket is a physical test station/slot).
- ``ENDURANCE``: a UUT may log to one file, or to several chunked files that
  were rolled over purely due to file-size limits and need chronological
  reassembly.

``group_files`` turns a flat list of discovered files (e.g. from
``src.data_loader.scan_directory``) into a list of ``TestSession`` objects.
``load_session`` loads and concatenates the files belonging to one session
into a single normalized DataFrame, with optional phase/pocket filtering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from src.data_loader import load_dataframe


class MachineType(str, Enum):
    GENERAL = "general"
    TRANSMISSION = "transmission"
    MOTOR_TEST_BENCH = "motor_test_bench"
    ENDURANCE = "endurance"


class Phase(str, Enum):
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    STEADY_STATE = "steady_state"
    COASTING = "coasting"
    UNKNOWN = "unknown"


# Order matters only in that the first matching pattern wins; the patterns
# themselves don't overlap for realistic filenames.
_PHASE_PATTERNS: list[tuple[re.Pattern, Phase]] = [
    (re.compile(r"ramp.?up", re.IGNORECASE), Phase.RAMP_UP),
    (re.compile(r"(ramp.?down|ramp.?dn|de.?ramp)", re.IGNORECASE), Phase.RAMP_DOWN),
    (re.compile(r"steady", re.IGNORECASE), Phase.STEADY_STATE),
    (re.compile(r"coast", re.IGNORECASE), Phase.COASTING),
]

_POCKET_PATTERNS = [
    re.compile(r"pocket[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"step[_-]?(\d+)", re.IGNORECASE),
]

_TIMESTAMP_DATETIME_PATTERN = re.compile(r"(\d{8}_\d{6})")
_TIMESTAMP_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
_CHUNK_INDEX_PATTERN = re.compile(r"(?:chunk[_-]?|_)(\d{2,})$", re.IGNORECASE)


@dataclass
class TestSession:
    uut_id: str
    machine_type: MachineType
    files: list[Path]
    phase_map: dict[Path, Phase] = field(default_factory=dict)     # transmission
    pocket_map: dict[Path, str] = field(default_factory=dict)      # motor test bench
    order: list[Path] = field(default_factory=list)                # endurance, chronological order of files


def _detect_phase(filename: str) -> Phase:
    for pattern, phase in _PHASE_PATTERNS:
        if pattern.search(filename):
            return phase
    return Phase.UNKNOWN


def _detect_pocket(stem: str) -> str:
    for pattern in _POCKET_PATTERNS:
        m = pattern.search(stem)
        if m:
            return m.group(1)
    # Neither pocket nor step pattern matched: fall back to the filename stem
    # so the file is still grouped (as its own pocket bucket) rather than dropped.
    return stem


def _extract_chrono_key(path: Path) -> tuple | None:
    """Best-effort sortable key extracted from a filename: (kind, value).

    Tries an embedded YYYYMMDD_HHMMSS timestamp, then a YYYY-MM-DD date, then
    a trailing chunk/index suffix like ``_001`` or ``_chunk002``. Returns None
    if nothing usable is found.
    """
    stem = path.stem

    m = _TIMESTAMP_DATETIME_PATTERN.search(stem)
    if m:
        try:
            return (0, datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"))
        except ValueError:
            pass

    m = _TIMESTAMP_DATE_PATTERN.search(stem)
    if m:
        try:
            return (0, datetime.strptime(m.group(1), "%Y-%m-%d"))
        except ValueError:
            pass

    m = _CHUNK_INDEX_PATTERN.search(stem)
    if m:
        return (1, int(m.group(1)))

    return None


def _order_endurance_files(files: list[Path]) -> list[Path]:
    keys = {f: _extract_chrono_key(f) for f in files}
    if all(k is None for k in keys.values()):
        return sorted(files, key=lambda f: f.stat().st_mtime)

    def sort_key(f: Path):
        k = keys[f]
        # Files without an extractable key sort after keyed ones, ordered by mtime.
        return k if k is not None else (2, f.stat().st_mtime)

    return sorted(files, key=sort_key)


def _group_key_and_uut_id(path: Path, uut_pattern: str | None) -> tuple[str, str]:
    """Return (grouping_key, display uut_id) for a file."""
    if uut_pattern:
        m = re.search(uut_pattern, path.name)
        uid = m.group(1) if m else path.stem
        return uid, uid
    # Default: group by parent folder (files under the same directory == same UUT).
    key = str(path.parent)
    uid = path.parent.name or key
    return key, uid


def group_files(
    files: list[Path],
    machine_type: MachineType,
    uut_pattern: str | None = None,
) -> list[TestSession]:
    """Group a flat list of files into per-UUT ``TestSession`` objects.

    - GENERAL: every file becomes its own single-file session.
    - TRANSMISSION / MOTOR_TEST_BENCH / ENDURANCE: files are grouped by parent
      folder by default, or by a captured group from ``uut_pattern`` (a regex
      with one capture group applied to the filename) when given. Each file
      that can't be matched by ``uut_pattern`` falls back to its own filename
      stem as its uut_id, rather than being dropped.
    """
    files = sorted(Path(f) for f in files)

    if machine_type == MachineType.GENERAL:
        return [
            TestSession(uut_id=f.stem, machine_type=machine_type, files=[f])
            for f in files
        ]

    buckets: dict[str, list[Path]] = {}
    uid_for_key: dict[str, str] = {}
    key_order: list[str] = []
    for f in files:
        key, uid = _group_key_and_uut_id(f, uut_pattern)
        if key not in buckets:
            buckets[key] = []
            uid_for_key[key] = uid
            key_order.append(key)
        buckets[key].append(f)

    sessions: list[TestSession] = []
    for key in key_order:
        group_files_list = buckets[key]
        session = TestSession(
            uut_id=uid_for_key[key],
            machine_type=machine_type,
            files=group_files_list,
        )
        if machine_type == MachineType.TRANSMISSION:
            session.phase_map = {f: _detect_phase(f.name) for f in group_files_list}
        elif machine_type == MachineType.MOTOR_TEST_BENCH:
            session.pocket_map = {f: _detect_pocket(f.stem) for f in group_files_list}
        elif machine_type == MachineType.ENDURANCE:
            session.order = _order_endurance_files(group_files_list)
        sessions.append(session)

    return sessions


def load_session(
    session: TestSession,
    phases: set[Phase] | None = None,
    pockets: set[str] | None = None,
) -> pd.DataFrame:
    """Load every file in a session via ``load_dataframe`` and concatenate.

    Files are loaded in ``session.order`` when set (endurance, chronological),
    otherwise in sorted filename order. A ``"phase"`` string column is added
    when ``session.phase_map`` is non-empty (``"unknown"`` for untagged
    files). ``phases``/``pockets`` filter which files' rows are kept and are
    no-ops when the session has no phase_map/pocket_map respectively (e.g.
    filtering by phase on a MOTOR_TEST_BENCH session does nothing).

    Callers at the CLI/GUI layer should default
    ``phases={Phase.STEADY_STATE, Phase.COASTING}`` for TRANSMISSION sessions,
    since that matches what users actually care about day to day; ramp phases
    remain available but are off by default at the UI layer, not baked in here.

    Index choice: rather than trying to offset each file's original time/
    sample index (which breaks down when files mix time-indexed and
    sample-indexed data, or have overlapping/reset clocks), the concatenated
    frame gets a fresh ``sample`` RangeIndex spanning the whole session. This
    keeps the combined index monotonic and simple for downstream windowed
    feature extraction; per-file boundaries are still recoverable from the
    ``phase``/pocket info and the original files if ever needed.
    """
    if session.machine_type == MachineType.ENDURANCE and session.order:
        ordered_files = session.order
    else:
        ordered_files = sorted(session.files)

    frames: list[pd.DataFrame] = []
    for f in ordered_files:
        phase = session.phase_map.get(f, Phase.UNKNOWN) if session.phase_map else None
        pocket = session.pocket_map.get(f) if session.pocket_map else None

        if phases is not None and session.phase_map and phase not in phases:
            continue
        if pockets is not None and session.pocket_map and pocket not in pockets:
            continue

        df = load_dataframe(f)
        if session.phase_map:
            df = df.copy()
            df["phase"] = phase.value
        frames.append(df)

    if not frames:
        combined = pd.DataFrame()
        combined.attrs["source_file"] = session.uut_id
        return combined

    combined = pd.concat(frames, ignore_index=True)
    combined.index.name = "sample"
    # downstream code only uses source_file for labeling, not parsing.
    combined.attrs["source_file"] = session.uut_id
    return combined
