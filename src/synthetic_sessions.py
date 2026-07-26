"""Multi-file synthetic dataset generators for exercising session_grouping.

Builds small folders of synthetic .tdms files, named the way each machine
type actually logs data, so tests (and demos) can exercise
``src.session_grouping`` without needing real LabVIEW data. Signal content is
reused as-is from ``src.synthetic_tdms`` — this module only handles naming
and folder layout.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from src.synthetic_tdms import _make_normal_signals, write_tdms

_TRANSMISSION_PHASES = ["ramp_up", "steady_state", "coasting", "ramp_down"]


def write_transmission_session(
    root: str | Path,
    uut_id: str = "unit001",
    n_samples: dict[str, int] | int = 200,
    seed: int = 0,
) -> dict[str, Path]:
    """Write one ramp_up/steady_state/coasting/ramp_down file per phase.

    Files are written under ``root/uut_id/`` as
    ``{uut_id}_{phase}.tdms``. Returns {phase: path}.
    """
    out_dir = Path(root) / uut_id
    if isinstance(n_samples, int):
        n_samples = {phase: n_samples for phase in _TRANSMISSION_PHASES}

    paths: dict[str, Path] = {}
    for i, phase in enumerate(_TRANSMISSION_PHASES):
        signals = _make_normal_signals(n_samples=n_samples[phase], seed=seed + i)
        path = out_dir / f"{uut_id}_{phase}.tdms"
        write_tdms(path, signals)
        paths[phase] = path
    return paths


def write_motor_test_bench_session(
    root: str | Path,
    uut_id: str = "unit001",
    pockets: list[int] = (1, 2),
    n_samples: dict[int, int] | int = 200,
    seed: int = 0,
) -> dict[str, Path]:
    """Write one file per pocket/step under ``root/uut_id/``.

    Files are named ``{uut_id}_pocket{n}_step1.tdms``. Returns
    {str(pocket): path}.
    """
    out_dir = Path(root) / uut_id
    pockets = list(pockets)
    if isinstance(n_samples, int):
        n_samples = {pocket: n_samples for pocket in pockets}

    paths: dict[str, Path] = {}
    for i, pocket in enumerate(pockets):
        signals = _make_normal_signals(n_samples=n_samples[pocket], seed=seed + i)
        path = out_dir / f"{uut_id}_pocket{pocket}_step1.tdms"
        write_tdms(path, signals)
        paths[str(pocket)] = path
    return paths


def write_endurance_session(
    root: str | Path,
    uut_id: str = "unit001",
    n_chunks: int = 2,
    n_samples: list[int] | int = 200,
    seed: int = 0,
    naming: str = "timestamp",
) -> list[Path]:
    """Write chunked files for one UUT under ``root/uut_id/``, in chronological order.

    ``naming`` is ``"timestamp"`` (``{uut_id}_YYYYMMDD_HHMMSS.tdms``, one
    minute apart) or ``"index"`` (``{uut_id}_chunk001.tdms`` style). Returns
    the paths in the chronological order they were written (chunk 0 first).
    """
    out_dir = Path(root) / uut_id
    if isinstance(n_samples, int):
        n_samples = [n_samples] * n_chunks

    base_time = datetime(2024, 1, 1, 0, 0, 0)
    paths: list[Path] = []
    for i in range(n_chunks):
        signals = _make_normal_signals(n_samples=n_samples[i], seed=seed + i)
        if naming == "index":
            name = f"{uut_id}_chunk{i + 1:03d}.tdms"
        else:
            ts = (base_time + timedelta(minutes=i)).strftime("%Y%m%d_%H%M%S")
            name = f"{uut_id}_{ts}.tdms"
        path = out_dir / name
        write_tdms(path, signals)
        paths.append(path)
    return paths
