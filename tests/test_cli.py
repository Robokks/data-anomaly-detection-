import sys

import pandas as pd
import pytest

from src import detect, train
from src.synthetic_sessions import write_transmission_session
from src.synthetic_tdms import _make_normal_signals, inject_anomalies, write_tdms

WINDOW_SIZE = 200


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


def test_train_and_detect_cli_classic_general_roundtrip(tmp_path, monkeypatch, capsys):
    normal_dir = tmp_path / "normal"
    for i in range(4):
        write_tdms(normal_dir / f"normal_{i}.tdms", _make_normal_signals(n_samples=4000, seed=i))

    model_path = tmp_path / "models" / "classic.joblib"
    _run(
        train,
        [
            "--data-dir", str(normal_dir),
            "--window-size", str(WINDOW_SIZE),
            "--contamination", "0.05",
            "--model-out", str(model_path),
        ],
        monkeypatch,
    )
    assert model_path.exists()

    test_signals, ranges = inject_anomalies(_make_normal_signals(n_samples=4000, seed=99), seed=99, n_events=3)
    test_path = tmp_path / "test_run.tdms"
    write_tdms(test_path, test_signals)

    out_csv = tmp_path / "scores.csv"
    _run(
        detect,
        [
            "--model", str(model_path),
            "--input", str(test_path),
            "--window-size", str(WINDOW_SIZE),
            "--output", str(out_csv),
        ],
        monkeypatch,
    )
    assert out_csv.exists()
    scores = pd.read_csv(out_csv)
    assert {"anomaly_score", "is_anomaly", "iso_forest_score", "pca_recon_error"} <= set(scores.columns)

    anomalous_windows = set()
    for start, end in ranges:
        anomalous_windows.update(range(start // WINDOW_SIZE, (end - 1) // WINDOW_SIZE + 1))
    anomalous_windows &= set(range(len(scores)))
    normal_windows = set(range(len(scores))) - anomalous_windows
    assert anomalous_windows
    assert scores["anomaly_score"].values[list(anomalous_windows)].mean() > scores["anomaly_score"].values[list(normal_windows)].mean()


def test_train_and_detect_cli_deep_model(tmp_path, monkeypatch):
    normal_dir = tmp_path / "normal"
    for i in range(3):
        write_tdms(normal_dir / f"normal_{i}.tdms", _make_normal_signals(n_samples=2000, seed=i))

    model_path = tmp_path / "models" / "deep.joblib"
    _run(
        train,
        [
            "--data-dir", str(normal_dir),
            "--window-size", "100",
            "--model-type", "deep",
            "--epochs", "5",
            "--contamination", "0.05",
            "--model-out", str(model_path),
        ],
        monkeypatch,
    )
    assert model_path.exists()

    test_path = tmp_path / "test_run.tdms"
    write_tdms(test_path, _make_normal_signals(n_samples=2000, seed=50))

    out_csv = tmp_path / "deep_scores.csv"
    _run(
        detect,
        [
            "--model", str(model_path),
            "--input", str(test_path),
            "--window-size", "100",
            "--output", str(out_csv),
        ],
        monkeypatch,
    )
    scores = pd.read_csv(out_csv)
    assert {"anomaly_score", "is_anomaly", "recon_error"} <= set(scores.columns)


def test_train_and_detect_cli_transmission_machine_type(tmp_path, monkeypatch):
    data_dir = tmp_path / "rig_data"
    for i, uid in enumerate(["unit001", "unit002", "unit003", "unit004"]):
        write_transmission_session(data_dir, uut_id=uid, n_samples=2000, seed=i * 10)

    model_path = tmp_path / "models" / "transmission.joblib"
    _run(
        train,
        [
            "--data-dir", str(data_dir),
            "--machine-type", "transmission",
            "--window-size", "200",
            "--contamination", "0.05",
            "--model-out", str(model_path),
        ],
        monkeypatch,
    )
    assert model_path.exists()

    out_csv = tmp_path / "transmission_scores.csv"
    _run(
        detect,
        [
            "--model", str(model_path),
            "--input", str(data_dir),
            "--machine-type", "transmission",
            "--window-size", "200",
            "--output", str(out_csv),
        ],
        monkeypatch,
    )
    scores = pd.read_csv(out_csv)
    assert len(scores) > 0
    assert {"anomaly_score", "is_anomaly"} <= set(scores.columns)


def test_train_cli_no_data_raises_system_exit(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--data-dir", str(empty_dir), "--model-out", str(tmp_path / "m.joblib")],
    )
    with pytest.raises(SystemExit):
        train.main()
