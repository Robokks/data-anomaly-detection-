"""Generate synthetic multi-channel sensor .tdms files for demo/testing.

Two channel sets are available:

- ``_make_normal_signals`` (the "basic" set, unchanged since this module was
  first written): vibration, temperature, pressure at 1 kHz. Used throughout
  this repo's test suite -- do not change its signature/defaults, or every
  test asserting an exact 3-channel set breaks.
- ``_make_rig_signals`` (the "full" set): a wider, more realistic
  motor/transmission-style rig -- vibration, broadband noise, shaft speed,
  motor current, 3-phase voltage, temperature, pressure, torque (10
  channels) -- at a configurable sample rate/duration, e.g. 10 kHz for 10
  seconds. This is what ``--channel-set full`` on the CLI below generates.

Useful for exercising the training/detection pipeline before you have real
LabVIEW data on hand, and for the automated tests in this repo.
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


def _make_rig_signals(n_samples: int, seed: int, sample_rate: int = 10000) -> dict[str, np.ndarray]:
    """Wider, more realistic 10-channel rig: vibration, noise, speed, current,
    3-phase voltage, temperature, pressure, torque -- at ``sample_rate`` Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate

    line_freq = 50.0  # Hz -- mains/electrical frequency
    shaft_freq = 25.0  # Hz -- ~1500 RPM

    vibration = (
        0.5 * np.sin(2 * np.pi * shaft_freq * t)
        + 0.15 * np.sin(2 * np.pi * shaft_freq * 3 * t)  # a harmonic
        + rng.normal(0, 0.05, n_samples)
    )
    noise = rng.normal(0, 1.0, n_samples)  # broadband background/acoustic noise channel
    speed = 1500 + 5 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 2.0, n_samples)  # RPM
    current = 10 + 2 * np.sin(2 * np.pi * line_freq * t) + rng.normal(0, 0.1, n_samples)  # A
    voltage_a = 230 * np.sin(2 * np.pi * line_freq * t) + rng.normal(0, 1.0, n_samples)
    voltage_b = 230 * np.sin(2 * np.pi * line_freq * t - 2 * np.pi / 3) + rng.normal(0, 1.0, n_samples)
    voltage_c = 230 * np.sin(2 * np.pi * line_freq * t + 2 * np.pi / 3) + rng.normal(0, 1.0, n_samples)
    temperature = 40 + 0.5 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.1, n_samples)
    pressure = 100 + 2 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.3, n_samples)
    torque = 5 + 0.5 * np.sin(2 * np.pi * shaft_freq * t) + rng.normal(0, 0.05, n_samples)

    return {
        "vibration": vibration,
        "noise": noise,
        "speed": speed,
        "current": current,
        "voltage_a": voltage_a,
        "voltage_b": voltage_b,
        "voltage_c": voltage_c,
        "temperature": temperature,
        "pressure": pressure,
        "torque": torque,
    }


def _ramp(x: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    frac = np.clip((x - x0) / max(x1 - x0, 1e-9), 0.0, 1.0)
    return y0 + frac * (y1 - y0)


def _make_test_cycle_signals(
    seed: int,
    sample_rate: int = 10000,
    duration_seconds: float | None = None,
    duration_range: tuple[float, float] = (5.0, 10.0),
) -> dict[str, np.ndarray]:
    """Simulate a speed/torque ramp-hold-de-ramp test cycle (motor/transmission
    test-bench style), with these normal (non-anomalous) physical
    relationships baked into the signal shapes:

    - **Speed (RPM)**: 0 -> 1000 (ramp) -> hold 1000 -> 1000 -> 6000 (ramp) ->
      hold 6000 -> 6000 -> 1000 (de-ramp) -> 1000 -> 0 (de-ramp). Phase
      durations are proportional to a nominal 10 s cycle (ramp 1s, hold at
      1000 rpm 3s, ramp 2s, hold at 6000 rpm 2s, de-ramp 1s, de-ramp 1s),
      scaled linearly to fit this file's actual duration (drawn uniformly
      from ``duration_range`` unless ``duration_seconds`` is given).
    - **Torque (Nm)**: 0 -> 50 -> 250 -> 0, ramping across the same three
      broad stages (low-speed, high-speed, de-ramp) as the speed profile --
      load increases as the rig spins up and unloads as it spins down.
    - **Vibration/noise**: amplitude rises during ramp/de-ramp transitions
      (mechanical transients) and is elevated -- but still stable -- during
      the 6000 rpm hold relative to the 1000 rpm hold. Both are normal
      operating behavior, not anomalies; a model trained on many of these
      cycles should learn this profile-correlated pattern as the baseline.
    - **Current**: tracks the torque profile (proportional, plus ripple).
    - **Voltage**: constant-amplitude 3-phase sinusoid throughout,
      independent of the speed/torque profile.

    Returns a 10-channel dict: vibration, noise, speed, torque, current,
    voltage_a/b/c, temperature, pressure.
    """
    rng = np.random.default_rng(seed)
    if duration_seconds is None:
        duration_seconds = rng.uniform(*duration_range)
    n_samples = max(int(sample_rate * duration_seconds), sample_rate)
    t = np.arange(n_samples) / sample_rate
    total_t = n_samples / sample_rate

    # Nominal (10s) phase durations, linearly scaled to this file's actual duration.
    nominal = {
        "ramp_up_a": 1.0,   # 0 -> 1000 rpm
        "hold_a": 3.0,      # hold at 1000 rpm
        "ramp_up_b": 2.0,   # 1000 -> 6000 rpm
        "hold_b": 2.0,      # hold at 6000 rpm
        "ramp_down_a": 1.0,  # 6000 -> 1000 rpm
        "ramp_down_b": 1.0,  # 1000 -> 0 rpm
    }
    scale = total_t / sum(nominal.values())
    durs = {k: v * scale for k, v in nominal.items()}

    b0 = 0.0
    b1 = b0 + durs["ramp_up_a"]
    b2 = b1 + durs["hold_a"]
    b3 = b2 + durs["ramp_up_b"]
    b4 = b3 + durs["hold_b"]
    b5 = b4 + durs["ramp_down_a"]
    b6 = b5 + durs["ramp_down_b"]  # == total_t

    speed = np.piecewise(
        t,
        [t < b1, (t >= b1) & (t < b2), (t >= b2) & (t < b3), (t >= b3) & (t < b4), (t >= b4) & (t < b5), t >= b5],
        [
            lambda x: _ramp(x, b0, b1, 0, 1000),
            lambda x: np.full_like(x, 1000.0),
            lambda x: _ramp(x, b2, b3, 1000, 6000),
            lambda x: np.full_like(x, 6000.0),
            lambda x: _ramp(x, b4, b5, 6000, 1000),
            lambda x: _ramp(x, b5, b6, 1000, 0),
        ],
    )

    torque = np.piecewise(
        t,
        [t < b2, (t >= b2) & (t < b4), t >= b4],
        [
            lambda x: _ramp(x, b0, b2, 0, 50),
            lambda x: _ramp(x, b2, b4, 50, 250),
            lambda x: _ramp(x, b4, b6, 250, 0),
        ],
    )

    # Elevated (smoothed) during ramp/de-ramp transients.
    in_ramp = ((t >= b0) & (t < b1)) | ((t >= b2) & (t < b3)) | ((t >= b4) & (t < b6))
    transient_boost = np.where(in_ramp, 2.5, 1.0).astype(float)
    smooth_n = max(int(sample_rate * 0.05), 1)
    kernel = np.ones(smooth_n) / smooth_n
    transient_boost = np.convolve(transient_boost, kernel, mode="same")

    # Extra baseline vibration while holding at the high-speed (6000 rpm) stage.
    high_speed_boost = np.where((t >= b3) & (t < b4), 1.4, 1.0)

    # Continuous phase from the (non-constant) instantaneous shaft frequency,
    # so vibration stays a physically consistent oscillation through the
    # speed ramps rather than a discontinuous sin(2*pi*f*t) at each instant.
    shaft_freq = speed / 60.0
    phase = 2 * np.pi * np.cumsum(shaft_freq) / sample_rate

    vib_amplitude = 0.3 * transient_boost * high_speed_boost
    vibration = (
        vib_amplitude * np.sin(phase)
        + 0.3 * transient_boost * np.sin(3 * phase)
        + rng.normal(0, 0.03 * transient_boost, n_samples)
    )
    noise = rng.normal(0, 0.5 * transient_boost, n_samples)

    line_freq = 50.0
    current = 0.5 + 0.04 * torque + 0.3 * np.sin(2 * np.pi * line_freq * t) + rng.normal(0, 0.05, n_samples)
    voltage_a = 230 * np.sin(2 * np.pi * line_freq * t) + rng.normal(0, 1.0, n_samples)
    voltage_b = 230 * np.sin(2 * np.pi * line_freq * t - 2 * np.pi / 3) + rng.normal(0, 1.0, n_samples)
    voltage_c = 230 * np.sin(2 * np.pi * line_freq * t + 2 * np.pi / 3) + rng.normal(0, 1.0, n_samples)

    temperature = 40 + 0.02 * torque + rng.normal(0, 0.1, n_samples)
    pressure = 100 + 0.05 * speed / 60 + rng.normal(0, 0.3, n_samples)

    # Measurement noise for the *output* speed/torque channels -- vibration's
    # phase and current/temperature above deliberately use the clean/exact
    # speed/torque profile so they stay physically consistent, but a real
    # sensor reading (and the windowed feature stats computed from it) is
    # never a mathematically perfect constant, even while "holding" -- a
    # perfectly flat window has zero variance, which produces NaN skew/
    # kurtosis downstream (scipy divides by std=0). Small sensor jitter
    # avoids that and is realistic besides.
    speed_measured = speed + rng.normal(0, 2.0, n_samples)
    torque_measured = torque + rng.normal(0, 0.5, n_samples)

    return {
        "vibration": vibration,
        "noise": noise,
        "speed": speed_measured,
        "torque": torque_measured,
        "current": current,
        "voltage_a": voltage_a,
        "voltage_b": voltage_b,
        "voltage_c": voltage_c,
        "temperature": temperature,
        "pressure": pressure,
    }


def inject_anomalies(
    signals: dict[str, np.ndarray],
    seed: int,
    n_events: int = 3,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    """Inject spike/drift/noise-burst anomalies; returns (signals, [(start, end), ...]).

    ``sample_rate`` scales anomaly durations to real time (a fraction of a
    second up to ~2 seconds) -- pass the same rate the signals were
    generated at (e.g. 10000 for ``_make_rig_signals``'s default).
    """
    rng = np.random.default_rng(seed + 1)
    n_samples = len(next(iter(signals.values())))
    signals = {k: v.copy() for k, v in signals.items()}
    windows: list[tuple[int, int]] = []

    for _ in range(n_events):
        kind = rng.choice(["spike", "drift", "noise_burst"])
        start = int(rng.integers(0, max(1, n_samples - sample_rate * 2)))
        duration = int(rng.integers(max(1, sample_rate // 4), max(2, sample_rate * 2)))
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
    p.add_argument("--n-samples", type=int, default=20000, help="Samples per file (ignored if --duration-seconds is given)")
    p.add_argument(
        "--channel-set",
        choices=["basic", "full", "cycle"],
        default="basic",
        help="'basic' = vibration/temperature/pressure at 1 kHz (default). "
        "'full' = a steady-state 10-channel rig (vibration, noise, speed, "
        "current, voltage_a/b/c, temperature, pressure, torque). "
        "'cycle' = the same 10 channels but following a ramp/hold/de-ramp "
        "speed+torque test cycle (0->1000->6000->1000->0 rpm, 0->50->250->0 Nm) "
        "with vibration/noise/current tied to the profile -- see "
        "_make_test_cycle_signals for the exact shape.",
    )
    p.add_argument("--sample-rate", type=int, default=10000, help="Sample rate in Hz, --channel-set full/cycle only")
    p.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Fixed samples-per-file = sample-rate * duration-seconds (--channel-set full/cycle), overriding "
        "--n-samples / --duration-min/--duration-max",
    )
    p.add_argument("--duration-min", type=float, default=5.0, help="--channel-set cycle only: min per-file duration in seconds (varies per file)")
    p.add_argument("--duration-max", type=float, default=10.0, help="--channel-set cycle only: max per-file duration in seconds (varies per file)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    normal_dir = out_dir / "normal"
    test_dir = out_dir / "test"

    if args.channel_set == "cycle":

        def make_signals(seed: int) -> dict[str, np.ndarray]:
            return _make_test_cycle_signals(
                seed=seed,
                sample_rate=args.sample_rate,
                duration_seconds=args.duration_seconds,
                duration_range=(args.duration_min, args.duration_max),
            )

        inject_kwargs = {"sample_rate": args.sample_rate}
        size_note = f"{args.duration_min}-{args.duration_max}s" if args.duration_seconds is None else f"{args.duration_seconds}s"
        size_note = f"@ {args.sample_rate} Hz, {size_note}/file (duration varies per file)"
    elif args.channel_set == "full":
        n_samples = int(args.sample_rate * args.duration_seconds) if args.duration_seconds else args.n_samples

        def make_signals(seed: int) -> dict[str, np.ndarray]:
            return _make_rig_signals(n_samples, seed=seed, sample_rate=args.sample_rate)

        inject_kwargs = {"sample_rate": args.sample_rate}
        size_note = f"@ {args.sample_rate} Hz = {n_samples / args.sample_rate:.1f}s/file"
    else:
        n_samples = args.n_samples

        def make_signals(seed: int) -> dict[str, np.ndarray]:
            return _make_normal_signals(n_samples, seed=seed)

        inject_kwargs = {}
        size_note = f"{n_samples} samples/file"

    n_channels = len(make_signals(args.seed))
    for i in range(args.n_normal):
        signals = make_signals(seed=args.seed + i)
        write_tdms(normal_dir / f"normal_{i:02d}.tdms", signals)
    print(f"Wrote {args.n_normal} normal training file(s) to {normal_dir} ({n_channels} channels, {size_note})")

    test_signals = make_signals(seed=args.seed + 100)
    test_signals, anomaly_windows = inject_anomalies(test_signals, seed=args.seed + 100, **inject_kwargs)
    write_tdms(test_dir / "test_run.tdms", test_signals)
    print(f"Wrote 1 test file with {len(anomaly_windows)} injected anomalies to {test_dir}")
    print(f"Injected anomaly sample-ranges: {anomaly_windows}")


if __name__ == "__main__":
    main()
