import threading
import time

from src.live_scorer import LiveScorer
from src.pipeline import train_model
from src.stream_simulator import stream_signals
from src.synthetic_tdms import _make_normal_signals, inject_anomalies
from src.tcp_stream_server import TcpStreamServer

SAMPLE_RATE_HZ = 1000
WINDOW_SAMPLES = 200


def _synthetic_df(n_samples, seed):
    import pandas as pd

    signals = _make_normal_signals(n_samples=n_samples, seed=seed)
    df = pd.DataFrame(signals)
    df.index.name = "sample"
    df.attrs["sample_rate_hz"] = SAMPLE_RATE_HZ
    return df


def _make_live_scorer():
    baseline_frames = [_synthetic_df(4000, seed=i) for i in range(4)]
    model = train_model("classic", baseline_frames, window_size=WINDOW_SAMPLES, contamination=0.05)

    scorer = LiveScorer(
        model=model,
        model_type="classic",
        channels=list(baseline_frames[0].columns),
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=WINDOW_SAMPLES,
        window_duration_seconds=WINDOW_SAMPLES / SAMPLE_RATE_HZ,
        channel_contamination=0.05,
    )
    scorer.fit_baseline(baseline_frames)
    return scorer


def _wait_until(predicate, timeout=10.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_tcp_stream_server_end_to_end_scores_windows():
    scorer = _make_live_scorer()
    results = []
    errors = []
    lock = threading.Lock()

    def on_window(result):
        with lock:
            results.append(result)

    def on_error(message):
        with lock:
            errors.append(message)

    server = TcpStreamServer("127.0.0.1", 0, scorer, on_window=on_window, on_error=on_error)
    port = server.start()
    try:
        signals = _make_normal_signals(n_samples=4000, seed=99)
        n_batches = stream_signals(
            "127.0.0.1", port, signals, sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=137, batch_interval_ms=10, realtime=False,
        )
        assert n_batches > 0

        expected_ticks = 4000 // WINDOW_SAMPLES
        assert _wait_until(lambda: len(results) >= expected_ticks, timeout=10.0)
    finally:
        server.stop()

    assert not errors
    assert len(results) == expected_ticks
    for result in results:
        assert set(result.channels.keys()) == set(scorer.channels)
        for channel_result in result.channels.values():
            assert "rms" in channel_result.stats
            assert "crest_factor" in channel_result.stats

    window_indices = [r.window_index for r in results]
    assert window_indices == sorted(window_indices)


def test_tcp_stream_server_detects_injected_anomalies():
    scorer = _make_live_scorer()
    results = []
    lock = threading.Lock()

    def on_window(result):
        with lock:
            results.append(result)

    server = TcpStreamServer("127.0.0.1", 0, scorer, on_window=on_window)
    port = server.start()
    try:
        test_signals = _make_normal_signals(n_samples=4000, seed=200)
        test_signals, anomaly_ranges = inject_anomalies(
            test_signals, seed=200, n_events=3, sample_rate=SAMPLE_RATE_HZ
        )
        stream_signals(
            "127.0.0.1", port, test_signals, sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=250, batch_interval_ms=10, realtime=False,
        )

        expected_ticks = 4000 // WINDOW_SAMPLES
        assert _wait_until(lambda: len(results) >= expected_ticks, timeout=10.0)
    finally:
        server.stop()

    results.sort(key=lambda r: r.window_index)

    anomalous_ticks = set()
    for start, end in anomaly_ranges:
        first = start // WINDOW_SAMPLES
        last = (end - 1) // WINDOW_SAMPLES
        anomalous_ticks.update(range(first, last + 1))
    anomalous_ticks &= set(range(len(results)))
    normal_ticks = set(range(len(results))) - anomalous_ticks
    assert anomalous_ticks

    overall_scores = [r.overall_anomaly_score for r in results if r.overall_anomaly_score is not None]
    assert overall_scores  # model warmup should have completed within this many ticks

    anomalous_overall = [results[i].overall_anomaly_score for i in anomalous_ticks if results[i].overall_anomaly_score is not None]
    normal_overall = [results[i].overall_anomaly_score for i in normal_ticks if results[i].overall_anomaly_score is not None]
    if anomalous_overall and normal_overall:
        assert sum(anomalous_overall) / len(anomalous_overall) > sum(normal_overall) / len(normal_overall)


def test_tcp_stream_server_rejects_channel_mismatch():
    scorer = _make_live_scorer()
    errors = []
    lock = threading.Lock()

    def on_error(message):
        with lock:
            errors.append(message)

    server = TcpStreamServer("127.0.0.1", 0, scorer, on_window=lambda r: None, on_error=on_error)
    port = server.start()
    try:
        bad_signals = {"only_one_channel": _make_normal_signals(n_samples=500, seed=1)["vibration"]}
        try:
            stream_signals(
                "127.0.0.1", port, bad_signals, sample_rate_hz=SAMPLE_RATE_HZ,
                batch_samples=100, batch_interval_ms=10, realtime=False,
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Expected/legitimate: the server rejects and closes right after
            # `hello` (fire-and-forget protocol, no ack), so the client can
            # race ahead and try to send a batch or two before noticing.
            pass
        assert _wait_until(lambda: len(errors) > 0, timeout=5.0)
    finally:
        server.stop()

    assert any("channel set mismatch" in e for e in errors)


def test_tcp_stream_server_stop_is_idempotent_and_frees_the_port():
    scorer = _make_live_scorer()
    server = TcpStreamServer("127.0.0.1", 0, scorer, on_window=lambda r: None)
    server.start()
    server.stop()
    server.stop()  # must not raise
