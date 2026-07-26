"""CLI/library: stream synthetic sensor data over a live TCP connection.

Reuses this repo's existing synthetic signal generators
(``src/synthetic_tdms.py``) -- it does not reimplement any signal math --
and sends them over the wire using ``src/stream_protocol.py``'s NDJSON
format, chunked into batches, so the rest of the live-scoring pipeline can
be developed/tested without real hardware.

Example:
    python -m src.stream_simulator --port 9999 --channel-set cycle \
        --batch-samples 1000 --batch-interval-ms 100
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.stream_protocol import encode_batch, encode_hello
from src.synthetic_tdms import (
    SAMPLE_RATE_HZ,
    _make_normal_signals,
    _make_rig_signals,
    _make_test_cycle_signals,
    inject_anomalies,
)


def stream_signals(
    host: str,
    port: int,
    signals: dict[str, np.ndarray],
    sample_rate_hz: float,
    batch_samples: int = 1000,
    batch_interval_ms: int = 100,
    realtime: bool = True,
) -> int:
    """Connects to (host, port) over TCP, sends a ``hello``, then streams
    ``signals`` sliced into consecutive chunks of ``batch_samples`` samples
    each (the last chunk may be shorter -- it is sent, not dropped/padded)
    as one ``batch`` message per chunk, in order.

    ``seq`` starts at 0 and increases by 1 per batch. ``t0`` is the
    wall-clock ``time.time()`` at which that particular batch is sent.

    When ``realtime`` is True, sleeps ``batch_interval_ms / 1000`` seconds
    between sends (best-effort pacing). When False, sends all batches
    back-to-back with no sleeping.

    Closes the socket cleanly (plain close, no goodbye message) after the
    last batch. Returns the total number of batches sent.
    """
    channel_names = list(signals.keys())
    n_samples = len(next(iter(signals.values())))
    interval_s = batch_interval_ms / 1000.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        sock.sendall(encode_hello(channel_names, sample_rate_hz).encode())

        seq = 0
        batch_count = 0
        for start in range(0, n_samples, batch_samples):
            end = min(start + batch_samples, n_samples)
            chunk = {name: signals[name][start:end] for name in channel_names}
            t0 = time.time()
            sock.sendall(encode_batch(seq, t0, chunk).encode())
            seq += 1
            batch_count += 1
            if realtime:
                time.sleep(interval_s)
    finally:
        sock.close()

    return batch_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream synthetic sensor data over a live TCP connection.")
    p.add_argument("--host", default="localhost", help="Host to connect to (default: localhost)")
    p.add_argument("--port", type=int, required=True, help="TCP port to connect to")
    p.add_argument(
        "--channel-set",
        choices=["basic", "full", "cycle"],
        default="cycle",
        help="'basic' = vibration/temperature/pressure at 1 kHz. "
        "'full' = the steady-state 10-channel rig. "
        "'cycle' = the same 10 channels following a ramp/hold/de-ramp test "
        "cycle (default) -- see synthetic_tdms._make_test_cycle_signals.",
    )
    p.add_argument("--sample-rate", type=int, default=10000, help="Sample rate in Hz, --channel-set full/cycle only")
    p.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Total signal duration (--channel-set full/cycle). Default: 20.0s for "
        "'full', the generator's own randomized range for 'cycle'.",
    )
    p.add_argument("--n-samples", type=int, default=20000, help="Samples to generate, --channel-set basic only")
    p.add_argument("--batch-samples", type=int, default=1000, help="Samples per streamed batch message")
    p.add_argument("--batch-interval-ms", type=int, default=100, help="Delay between batches in realtime mode")
    p.add_argument("--inject-anomalies", action="store_true", help="Inject synthetic anomalies before streaming")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-realtime", action="store_true", help="Send all batches back-to-back with no pacing delay")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.channel_set == "cycle":
        signals = _make_test_cycle_signals(
            seed=args.seed,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration_seconds,
        )
        inject_kwargs = {"sample_rate": args.sample_rate}
    elif args.channel_set == "full":
        duration_seconds = args.duration_seconds if args.duration_seconds is not None else 20.0
        n_samples = int(args.sample_rate * duration_seconds)
        signals = _make_rig_signals(n_samples, seed=args.seed, sample_rate=args.sample_rate)
        inject_kwargs = {"sample_rate": args.sample_rate}
    else:
        signals = _make_normal_signals(args.n_samples, seed=args.seed)
        inject_kwargs = {}

    if args.inject_anomalies:
        signals, anomaly_windows = inject_anomalies(signals, seed=args.seed, **inject_kwargs)
        print(f"Injected anomaly sample-ranges: {anomaly_windows}")

    n_channels = len(signals)
    n_samples = len(next(iter(signals.values())))
    sample_rate = args.sample_rate if args.channel_set != "basic" else SAMPLE_RATE_HZ

    print(f"Connecting to {args.host}:{args.port} and streaming {n_channels} channel(s), {n_samples} samples ...")
    batch_count = stream_signals(
        host=args.host,
        port=args.port,
        signals=signals,
        sample_rate_hz=sample_rate,
        batch_samples=args.batch_samples,
        batch_interval_ms=args.batch_interval_ms,
        realtime=not args.no_realtime,
    )
    print(f"Sent {batch_count} batch(es) to {args.host}:{args.port} ({n_channels} channels, {n_samples} samples total)")


if __name__ == "__main__":
    main()
