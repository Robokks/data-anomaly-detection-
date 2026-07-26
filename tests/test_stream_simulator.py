import socket
import threading
import time

import numpy as np
import pytest

from src.stream_protocol import BatchMessage, HelloMessage, decode_message
from src.stream_simulator import stream_signals
from src.synthetic_tdms import _make_normal_signals

SAMPLE_RATE_HZ = 1000


def _run_capture_server(server_sock, received, done_event):
    conn, _ = server_sock.accept()
    with conn:
        f = conn.makefile("r")
        for line in f:
            line = line.strip()
            if not line:
                continue
            received.append(decode_message(line))
    done_event.set()


def _start_server():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]

    received: list = []
    done_event = threading.Event()
    thread = threading.Thread(target=_run_capture_server, args=(server_sock, received, done_event), daemon=True)
    thread.start()
    return server_sock, port, received, done_event, thread


def test_stream_signals_roundtrip_reconstructs_signals_exactly():
    server_sock, port, received, done_event, thread = _start_server()
    try:
        signals = _make_normal_signals(n_samples=950, seed=0)

        batch_count = stream_signals(
            host="127.0.0.1",
            port=port,
            signals=signals,
            sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=200,
            batch_interval_ms=10,
            realtime=False,
        )

        thread.join(timeout=5)
        assert done_event.is_set()
    finally:
        server_sock.close()

    assert len(received) >= 1
    hello = received[0]
    assert isinstance(hello, HelloMessage)
    assert set(hello.channels) == set(signals.keys())
    assert hello.sample_rate_hz == SAMPLE_RATE_HZ

    batches = received[1:]
    assert batches
    for msg in batches:
        assert isinstance(msg, BatchMessage)

    # seq values monotonically increasing
    seqs = [b.seq for b in batches]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)

    # returned batch count matches what was actually received
    assert batch_count == len(batches)

    # concatenating batches per channel reconstructs the original signals exactly
    for channel in signals:
        reconstructed = np.concatenate([np.asarray(b.channels[channel]) for b in batches])
        np.testing.assert_allclose(reconstructed, signals[channel])


def test_stream_signals_last_batch_is_shorter_and_no_samples_dropped():
    server_sock, port, received, done_event, thread = _start_server()
    try:
        signals = _make_normal_signals(n_samples=950, seed=1)

        batch_count = stream_signals(
            host="127.0.0.1",
            port=port,
            signals=signals,
            sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=200,
            batch_interval_ms=10,
            realtime=False,
        )
        thread.join(timeout=5)
        assert done_event.is_set()
    finally:
        server_sock.close()

    batches = [m for m in received if isinstance(m, BatchMessage)]
    assert batch_count == 5  # 4 full batches of 200 + 1 of 150
    assert batch_count == len(batches)

    lengths = [len(b.channels["vibration"]) for b in batches]
    assert lengths == [200, 200, 200, 200, 150]
    assert sum(lengths) == 950


def test_stream_signals_realtime_false_is_fast():
    server_sock, port, received, done_event, thread = _start_server()
    try:
        signals = _make_normal_signals(n_samples=5000, seed=2)

        start = time.monotonic()
        stream_signals(
            host="127.0.0.1",
            port=port,
            signals=signals,
            sample_rate_hz=SAMPLE_RATE_HZ,
            batch_samples=200,
            batch_interval_ms=200,  # would take 25 batches * 0.2s = 5s+ if realtime
            realtime=False,
        )
        elapsed = time.monotonic() - start
        thread.join(timeout=5)
    finally:
        server_sock.close()

    assert elapsed < 2.0
