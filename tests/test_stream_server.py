import csv
import sys
import time

import pytest

from src import stream_server
from src.anomaly_model import AnomalyDetector
from src.features import extract_features_from_frames
from src.pipeline import load_sessions_from_directory, save_model, train_model
from src.stream_simulator import stream_signals
from src.synthetic_tdms import _make_normal_signals, write_tdms

SAMPLE_RATE_HZ = 1000
WINDOW_SIZE = 200


def _wait_until(predicate, timeout=10.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_baseline_dir(tmp_path, n_files=4, n_samples=4000, seed_offset=0, name="baseline"):
    baseline_dir = tmp_path / name
    for i in range(n_files):
        write_tdms(baseline_dir / f"normal_{i}.tdms", _make_normal_signals(n_samples=n_samples, seed=seed_offset + i))
    return baseline_dir


def _train_and_save_classic_model(baseline_dir, model_path, window_size=WINDOW_SIZE):
    frames = load_sessions_from_directory(baseline_dir)
    model = train_model("classic", frames, window_size=window_size, contamination=0.05)
    save_model(model, "classic", model_path)
    return model


def _base_argv(model_path, baseline_dir, extra=None):
    # window-duration-seconds is set to line up exactly one stats-tick with
    # one model window (matching tests/test_tcp_stream_server.py's approach)
    # so expected-tick-count math in these tests is simple/exact.
    argv = [
        "prog",
        "--model", str(model_path),
        "--baseline-dir", str(baseline_dir),
        "--sample-rate", str(SAMPLE_RATE_HZ),
        "--window-size", str(WINDOW_SIZE),
        "--window-duration-seconds", str(WINDOW_SIZE / SAMPLE_RATE_HZ),
        "--host", "127.0.0.1",
        "--port", "0",
    ]
    if extra:
        argv.extend(extra)
    return argv


def test_stream_server_build_and_stream_end_to_end(tmp_path, monkeypatch, capsys):
    baseline_dir = _make_baseline_dir(tmp_path)
    model_path = tmp_path / "model.joblib"
    _train_and_save_classic_model(baseline_dir, model_path)

    monkeypatch.setattr(sys, "argv", _base_argv(model_path, baseline_dir))
    args = stream_server.parse_args()

    results = []
    server, live_scorer = stream_server.build_server(args, extra_on_window=results.append)
    assert set(live_scorer.channels) == {"vibration", "temperature", "pressure"}

    port = server.start()
    try:
        signals = _make_normal_signals(n_samples=4000, seed=99)
        n_batches = stream_signals(
            "127.0.0.1", port, signals, sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=137, batch_interval_ms=10, realtime=False,
        )
        assert n_batches > 0

        expected_ticks = 4000 // WINDOW_SIZE
        assert _wait_until(lambda: len(results) >= expected_ticks, timeout=10.0)
    finally:
        server.stop()

    assert len(results) == expected_ticks

    out = capsys.readouterr().out
    assert "[window 0]" in out
    assert "Listening" not in out  # that banner is only printed by main(), not build_server()


def test_stream_server_output_csv_has_expected_shape(tmp_path, monkeypatch):
    baseline_dir = _make_baseline_dir(tmp_path)
    model_path = tmp_path / "model.joblib"
    _train_and_save_classic_model(baseline_dir, model_path)

    out_csv = tmp_path / "live_log.csv"
    monkeypatch.setattr(sys, "argv", _base_argv(model_path, baseline_dir, extra=["--output-csv", str(out_csv)]))
    args = stream_server.parse_args()

    results = []
    server, live_scorer = stream_server.build_server(args, extra_on_window=results.append)
    port = server.start()
    try:
        signals = _make_normal_signals(n_samples=4000, seed=7)
        stream_signals(
            "127.0.0.1", port, signals, sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=200, batch_interval_ms=10, realtime=False,
        )
        expected_ticks = 4000 // WINDOW_SIZE
        assert _wait_until(lambda: len(results) >= expected_ticks, timeout=10.0)
    finally:
        server.stop()

    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(results)
    fieldnames = set(rows[0].keys())
    assert {"window_index", "timestamp", "n_samples_seen", "overall_anomaly_score", "overall_is_anomaly"} <= fieldnames
    for channel in live_scorer.channels:
        assert f"{channel}_deviation_score" in fieldnames
        assert f"{channel}_is_anomaly" in fieldnames
        assert f"{channel}_rms" in fieldnames
        assert f"{channel}_crest_factor" in fieldnames


def test_stream_server_legacy_model_without_window_size_raises_system_exit(tmp_path, monkeypatch, capsys):
    baseline_dir = _make_baseline_dir(tmp_path)

    # Build a "legacy" classic model the way pipeline.train_model USED to
    # (before it started stashing window_size_/step_) by fitting an
    # AnomalyDetector directly, bypassing train_model.
    frames = load_sessions_from_directory(baseline_dir)
    features = extract_features_from_frames(frames, window_size=WINDOW_SIZE, step=WINDOW_SIZE)
    legacy_model = AnomalyDetector(contamination=0.05)
    legacy_model.fit(features)
    assert legacy_model.window_size_ is None

    model_path = tmp_path / "legacy.joblib"
    save_model(legacy_model, "classic", model_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model", str(model_path),
            "--baseline-dir", str(baseline_dir),
            "--sample-rate", str(SAMPLE_RATE_HZ),
            "--host", "127.0.0.1",
            "--port", "0",
        ],
    )

    with pytest.raises(SystemExit):
        stream_server.main()

    captured = capsys.readouterr()
    assert "window-size" in captured.err.lower() or "window_size" in captured.err.lower()
    assert "traceback" not in captured.err.lower()


def test_stream_server_window_size_override_warns_on_mismatch(tmp_path, monkeypatch, capsys):
    baseline_dir = _make_baseline_dir(tmp_path)
    model_path = tmp_path / "model.joblib"
    _train_and_save_classic_model(baseline_dir, model_path, window_size=WINDOW_SIZE)

    override_window_size = 150
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model", str(model_path),
            "--baseline-dir", str(baseline_dir),
            "--sample-rate", str(SAMPLE_RATE_HZ),
            "--window-size", str(override_window_size),
            "--host", "127.0.0.1",
            "--port", "0",
        ],
    )
    args = stream_server.parse_args()

    server, live_scorer = stream_server.build_server(args)
    assert live_scorer.model_window_size == override_window_size

    out = capsys.readouterr().out
    assert "mismatch" in out.lower()
    assert str(WINDOW_SIZE) in out
    assert str(override_window_size) in out


def test_stream_server_no_baseline_data_raises_system_exit(tmp_path, monkeypatch):
    baseline_dir = _make_baseline_dir(tmp_path, name="baseline_for_model")
    model_path = tmp_path / "model.joblib"
    _train_and_save_classic_model(baseline_dir, model_path)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model", str(model_path),
            "--baseline-dir", str(empty_dir),
            "--sample-rate", str(SAMPLE_RATE_HZ),
            "--window-size", str(WINDOW_SIZE),
            "--host", "127.0.0.1",
            "--port", "0",
        ],
    )

    with pytest.raises(SystemExit):
        stream_server.main()
