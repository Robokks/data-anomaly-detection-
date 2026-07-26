import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    infer_time_column,
    load_csv,
    load_dataframe,
    load_excel,
    probe_file,
)
from src.synthetic_tdms import _make_normal_signals, write_tdms


def _make_raw_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 100.0
    return pd.DataFrame(
        {
            "Time (s)": t,
            "vibration": np.sin(2 * np.pi * 5 * t) + rng.normal(0, 0.05, n),
            "current": 10 + rng.normal(0, 0.2, n),
        }
    )


def test_infer_time_column():
    assert infer_time_column(["Time (s)", "vibration", "current"]) == "Time (s)"
    assert infer_time_column(["timestamp", "x"]) == "timestamp"
    assert infer_time_column(["vibration", "current"]) is None


def test_load_csv_normalizes_time_index_and_channels(tmp_path):
    raw = _make_raw_df()
    path = tmp_path / "run1.csv"
    raw.to_csv(path, index=False)

    df = load_csv(path)
    assert set(df.columns) == {"vibration", "current"}
    assert df.index.name == "time"
    assert len(df) == len(raw)
    np.testing.assert_allclose(df["vibration"].values, raw["vibration"].values, atol=1e-9)


def test_load_csv_without_time_column_falls_back_to_sample_index(tmp_path):
    raw = _make_raw_df().drop(columns=["Time (s)"])
    path = tmp_path / "no_time.csv"
    raw.to_csv(path, index=False)

    df = load_csv(path)
    assert df.index.name == "sample"
    assert list(df.index) == list(range(len(raw)))


def test_load_csv_channel_filter(tmp_path):
    raw = _make_raw_df()
    path = tmp_path / "run.csv"
    raw.to_csv(path, index=False)

    df = load_csv(path, channels=["vibration"])
    assert list(df.columns) == ["vibration"]


def test_load_excel_roundtrip(tmp_path):
    raw = _make_raw_df(n=200, seed=1)
    path = tmp_path / "run1.xlsx"
    raw.to_excel(path, index=False)

    df = load_excel(path)
    assert set(df.columns) == {"vibration", "current"}
    assert df.index.name == "time"
    assert len(df) == len(raw)


def test_load_dataframe_dispatches_by_extension(tmp_path):
    raw = _make_raw_df(n=100, seed=2)
    csv_path = tmp_path / "a.csv"
    xlsx_path = tmp_path / "b.xlsx"
    raw.to_csv(csv_path, index=False)
    raw.to_excel(xlsx_path, index=False)

    signals = _make_normal_signals(n_samples=1000, seed=3)
    tdms_path = tmp_path / "c.tdms"
    write_tdms(tdms_path, signals)

    df_csv = load_dataframe(csv_path)
    df_xlsx = load_dataframe(xlsx_path)
    df_tdms = load_dataframe(tdms_path)

    assert df_csv.attrs["format"] == "csv"
    assert df_xlsx.attrs["format"] == "excel"
    assert df_tdms.attrs["format"] == "tdms"
    assert set(df_tdms.columns) == set(signals.keys())


def test_load_dataframe_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError):
        load_dataframe(path)


def test_probe_file_csv(tmp_path):
    raw = _make_raw_df(n=50, seed=4)
    path = tmp_path / "probe.csv"
    raw.to_csv(path, index=False)

    info = probe_file(path)
    assert info.error is None
    assert set(info.channels) == {"Time (s)", "vibration", "current"}
    assert info.n_samples == 50


def test_probe_file_tdms(tmp_path):
    signals = _make_normal_signals(n_samples=300, seed=5)
    path = tmp_path / "probe.tdms"
    write_tdms(path, signals)

    info = probe_file(path)
    assert info.error is None
    assert set(info.channels) == set(signals.keys())
    assert info.n_samples == 300


def test_probe_file_reports_error_instead_of_raising(tmp_path):
    path = tmp_path / "broken.tdms"
    path.write_bytes(b"not a real tdms file")

    info = probe_file(path)
    assert info.error is not None
