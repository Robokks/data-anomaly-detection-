"""Load LabVIEW TDMS files into pandas DataFrames."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile


def load_tdms_dataframe(
    path: str | Path,
    group_name: str | None = None,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Read a .tdms file and return a single DataFrame of aligned channels.

    Each TDMS channel becomes a column. If a channel carries waveform timing
    properties (``wf_start_time`` / ``wf_increment``), a ``time`` index is
    built from them; otherwise a plain sample-index is used. Channels of
    differing length are aligned by truncating to the shortest channel, since
    TDMS channels within a group are normally sampled together.
    """
    path = Path(path)
    tdms_file = TdmsFile.read(str(path))

    groups = tdms_file.groups()
    if not groups:
        raise ValueError(f"No groups found in TDMS file: {path}")

    group = next((g for g in groups if g.name == group_name), groups[0]) if group_name else groups[0]

    columns: dict[str, np.ndarray] = {}
    time_index = None

    for ch in group.channels():
        if channels and ch.name not in channels:
            continue
        data = ch[:]
        if data.size == 0:
            continue
        columns[ch.name] = data

        if time_index is None:
            try:
                time_index = ch.time_track()
            except (KeyError, AttributeError):
                time_index = None

    if not columns:
        raise ValueError(f"No matching channels found in group '{group.name}' of {path}")

    min_len = min(len(v) for v in columns.values())
    columns = {name: values[:min_len] for name, values in columns.items()}

    df = pd.DataFrame(columns)
    if time_index is not None and len(time_index) >= min_len:
        df.index = pd.Index(time_index[:min_len], name="time")
    else:
        df.index.name = "sample"

    df.attrs["source_file"] = str(path)
    df.attrs["group"] = group.name
    return df


def load_tdms_directory(
    directory: str | Path,
    group_name: str | None = None,
    channels: list[str] | None = None,
    pattern: str = "*.tdms",
) -> list[pd.DataFrame]:
    """Load every TDMS file in a directory into a list of DataFrames."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")
    return [load_tdms_dataframe(f, group_name=group_name, channels=channels) for f in files]
