"""Generate synthetic multi-channel sensor .tdms files for demo/testing.

Simulates a small test rig with vibration, temperature, and pressure
channels. Useful for exercising the training/detection pipeline before you
have real LabVIEW data on hand, and for the automated tests in this repo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from nptdms import ChannelObject, TdmsWriter

SAMPLE_RATE_HZ = 1000


def _make_normal_signals(n_samples: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SAMPLE_RATE_HZ

    vibration = 0.5 * np.sin(2 * np.pi * 50 * t) + rng.normal(0, 0.05, n_samples)
    temperature = 40 + 0.5 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.1, n_samples)
    pressure = 100 + 2 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.3, n_samples)

    return {"vibration": vibration, "temperature": temperature, "pressure": pressure}


def inject_anomalies(signals: dict[str, np.ndarray], seed: int, n_events: int = 3) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    """Inject spike/drift/noise-burst anomalies; returns (signals, [(start, end), ...])."""
    rng = np.random.default_rng(seed + 1)
    n_samples = len(next(iter(signals.values())))
    signals = {k: v.copy() for k, v in signals.items()}
    windows: list[tuple[int, int]] = []

    for _ in range(n_events):
        kind = rng.choice(["spike", "drift", "noise_burst"])
        start = int(rng.integers(0, n_samples - SAMPLE_RATE_HZ * 2))
        duration = int(rng.integers(SAMPLE_RATE_HZ // 4, SAMPLE_RATE_HZ * 2))
        end = min(start + duration, n_samples)
        channel = rng.choice(list(signals.keys()))

        if kind == "spike":
            signals[channel][start:end] += rng.choice([-1, 1]) * rng.uniform(3, 6)
        elif kind == "drift":
            ramp = np.linspace(0, rng.uniform(3, 8), end - start)
            signals[channel][start:end] += ramp
        else:  # noise_burst
            signals[channel][start:end] += rng.normal(0, 2.0, end - start)

        windows.append((start, end))

    return signals, windows


def write_tdms(path: str | Path, signals: dict[str, np.ndarray], group: str = "rig") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TdmsWriter(str(path)) as writer:
        channels = [ChannelObject(group, name, data) for name, data in signals.items()]
        writer.write_segment(channels)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic TDMS files for demo/testing.")
    p.add_argument("--out-dir", default="data", help="Output directory")
    p.add_argument("--n-normal", type=int, default=5, help="Number of normal (training) files to generate")
    p.add_argument("--n-samples", type=int, default=20000, help="Samples per file")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    normal_dir = out_dir / "normal"
    test_dir = out_dir / "test"

    for i in range(args.n_normal):
        signals = _make_normal_signals(args.n_samples, seed=args.seed + i)
        write_tdms(normal_dir / f"normal_{i:02d}.tdms", signals)
    print(f"Wrote {args.n_normal} normal training file(s) to {normal_dir}")

    test_signals = _make_normal_signals(args.n_samples, seed=args.seed + 100)
    test_signals, anomaly_windows = inject_anomalies(test_signals, seed=args.seed + 100)
    write_tdms(test_dir / "test_run.tdms", test_signals)
    print(f"Wrote 1 test file with {len(anomaly_windows)} injected anomalies to {test_dir}")
    print(f"Injected anomaly sample-ranges: {anomaly_windows}")


if __name__ == "__main__":
    main()
