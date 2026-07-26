import numpy as np
import pandas as pd
import pytest

from src.spectral import SignatureBaseline, compute_spectrum, fit_baselines_from_frames

SAMPLE_RATE = 1000.0
WINDOW_SIZE = 256


def _sine(freq_hz: float, n_samples: int, sample_rate: float, seed: int, noise_std: float = 0.02, amplitude: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, noise_std, n_samples)


def _windows(signal: np.ndarray, window_size: int) -> np.ndarray:
    n_windows = len(signal) // window_size
    return signal[: n_windows * window_size].reshape(n_windows, window_size)


# ---------------------------------------------------------------------------
# compute_spectrum
# ---------------------------------------------------------------------------


def test_compute_spectrum_sine_peak_at_known_frequency():
    freq_hz = 50.0
    n_samples = 1024
    t = np.arange(n_samples) / SAMPLE_RATE
    x = np.sin(2 * np.pi * freq_hz * t)

    freqs, magnitude = compute_spectrum(x, sample_rate=SAMPLE_RATE)

    peak_bin = np.argmax(magnitude)
    peak_freq = freqs[peak_bin]
    assert peak_freq == pytest.approx(freq_hz, abs=SAMPLE_RATE / n_samples * 2)


def test_compute_spectrum_bin_indices_without_sample_rate():
    x = np.sin(2 * np.pi * 10 * np.arange(256) / 256)
    freqs, magnitude = compute_spectrum(x)

    assert len(freqs) == len(magnitude)
    np.testing.assert_array_equal(freqs, np.arange(len(magnitude)))


def test_compute_spectrum_removes_dc_offset():
    n_samples = 512
    t = np.arange(n_samples) / SAMPLE_RATE
    x_no_offset = np.sin(2 * np.pi * 20 * t)
    x_with_offset = x_no_offset + 100.0  # huge DC offset

    _, mag_no_offset = compute_spectrum(x_no_offset, sample_rate=SAMPLE_RATE)
    _, mag_with_offset = compute_spectrum(x_with_offset, sample_rate=SAMPLE_RATE)

    # DC bin (index 0) should not blow up due to the offset -- mean removal
    # means both spectra are effectively identical.
    np.testing.assert_allclose(mag_no_offset, mag_with_offset, atol=1e-8)
    # And the DC bin should not dominate the spectrum.
    assert mag_with_offset[0] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# SignatureBaseline.fit / deviation
# ---------------------------------------------------------------------------


def test_baseline_deviation_low_for_matching_signal_high_for_different():
    freq_hz = 50.0
    n_train_windows = 40

    train_signal = _sine(freq_hz, n_train_windows * WINDOW_SIZE, SAMPLE_RATE, seed=1, noise_std=0.02)
    train_windows = _windows(train_signal, WINDOW_SIZE)

    baseline = SignatureBaseline(channel="vibration")
    baseline.fit(train_windows, sample_rate=SAMPLE_RATE)

    assert baseline.n_windows_ == n_train_windows
    assert baseline.mean_.shape == baseline.std_.shape
    assert baseline.freqs_.shape == baseline.mean_.shape

    # New window, same frequency + noise -> low deviation.
    normal_signal = _sine(freq_hz, WINDOW_SIZE, SAMPLE_RATE, seed=999, noise_std=0.02)
    normal_dev = baseline.deviation(normal_signal, sample_rate=SAMPLE_RATE)

    # Window at a noticeably different frequency plus a much larger amplitude
    # -> high deviation.
    anomalous_signal = _sine(freq_hz * 4, WINDOW_SIZE, SAMPLE_RATE, seed=999, noise_std=0.02, amplitude=5.0)
    anomalous_dev = baseline.deviation(anomalous_signal, sample_rate=SAMPLE_RATE)

    assert normal_dev < anomalous_dev


def test_baseline_fit_rejects_empty_windows():
    baseline = SignatureBaseline(channel="vibration")
    with pytest.raises(ValueError):
        baseline.fit(np.empty((0, WINDOW_SIZE)))


def test_baseline_fit_rejects_non_2d_windows():
    baseline = SignatureBaseline(channel="vibration")
    with pytest.raises(ValueError):
        baseline.fit(np.zeros(WINDOW_SIZE))


def test_baseline_deviation_before_fit_raises():
    baseline = SignatureBaseline(channel="vibration")
    with pytest.raises(RuntimeError):
        baseline.deviation(np.zeros(WINDOW_SIZE))


# ---------------------------------------------------------------------------
# save/load round-trip
# ---------------------------------------------------------------------------


def test_baseline_save_load_roundtrip(tmp_path):
    freq_hz = 30.0
    train_signal = _sine(freq_hz, 20 * WINDOW_SIZE, SAMPLE_RATE, seed=5, noise_std=0.02)
    train_windows = _windows(train_signal, WINDOW_SIZE)

    baseline = SignatureBaseline(channel="pressure")
    baseline.fit(train_windows, sample_rate=SAMPLE_RATE)

    test_signal = _sine(freq_hz, WINDOW_SIZE, SAMPLE_RATE, seed=42, noise_std=0.02)
    dev_before = baseline.deviation(test_signal, sample_rate=SAMPLE_RATE)

    path = tmp_path / "baseline.joblib"
    baseline.save(path)
    loaded = SignatureBaseline.load(path)

    assert loaded.channel == baseline.channel
    dev_after = loaded.deviation(test_signal, sample_rate=SAMPLE_RATE)

    assert dev_after == pytest.approx(dev_before)
    np.testing.assert_allclose(loaded.mean_, baseline.mean_)
    np.testing.assert_allclose(loaded.std_, baseline.std_)


# ---------------------------------------------------------------------------
# fit_baselines_from_frames
# ---------------------------------------------------------------------------


def test_fit_baselines_from_frames_returns_one_per_channel():
    n_samples = 10 * WINDOW_SIZE
    frame_a = pd.DataFrame(
        {
            "vibration": _sine(50.0, n_samples, SAMPLE_RATE, seed=1),
            "temperature": _sine(0.5, n_samples, SAMPLE_RATE, seed=2, noise_std=0.1),
            "pressure": _sine(1.0, n_samples, SAMPLE_RATE, seed=3, noise_std=0.1),
        }
    )
    frame_b = pd.DataFrame(
        {
            "vibration": _sine(50.0, n_samples, SAMPLE_RATE, seed=4),
            "temperature": _sine(0.5, n_samples, SAMPLE_RATE, seed=5, noise_std=0.1),
            "pressure": _sine(1.0, n_samples, SAMPLE_RATE, seed=6, noise_std=0.1),
        }
    )

    baselines = fit_baselines_from_frames(
        [frame_a, frame_b], window_size=WINDOW_SIZE, sample_rate=SAMPLE_RATE
    )

    assert set(baselines.keys()) == {"vibration", "temperature", "pressure"}
    for name, baseline in baselines.items():
        assert baseline.channel == name
        assert baseline.mean_.size > 0
        assert baseline.std_.size > 0
        assert baseline.n_windows_ == 2 * (n_samples // WINDOW_SIZE)


def test_fit_baselines_from_frames_skips_channels_missing_from_any_frame():
    n_samples = 5 * WINDOW_SIZE
    frame_a = pd.DataFrame(
        {
            "vibration": _sine(50.0, n_samples, SAMPLE_RATE, seed=1),
            "temperature": _sine(0.5, n_samples, SAMPLE_RATE, seed=2, noise_std=0.1),
        }
    )
    frame_b = pd.DataFrame(
        {
            "vibration": _sine(50.0, n_samples, SAMPLE_RATE, seed=3),
        }
    )

    baselines = fit_baselines_from_frames([frame_a, frame_b], window_size=WINDOW_SIZE)

    assert set(baselines.keys()) == {"vibration"}


def test_fit_baselines_from_frames_respects_explicit_channel_list():
    n_samples = 5 * WINDOW_SIZE
    frame = pd.DataFrame(
        {
            "vibration": _sine(50.0, n_samples, SAMPLE_RATE, seed=1),
            "temperature": _sine(0.5, n_samples, SAMPLE_RATE, seed=2, noise_std=0.1),
            "pressure": _sine(1.0, n_samples, SAMPLE_RATE, seed=3, noise_std=0.1),
        }
    )

    baselines = fit_baselines_from_frames([frame], channels=["vibration", "pressure"], window_size=WINDOW_SIZE)

    assert set(baselines.keys()) == {"vibration", "pressure"}
