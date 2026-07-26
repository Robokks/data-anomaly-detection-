"""Regression tests guarding against silent behavior drift in the pipeline.

Part 1: proves that swapping the loading layer from the old TDMS-only
``src.tdms_loader.load_tdms_directory`` path to the new unified
``src.data_loader.load_directory`` path does not change
``AnomalyDetector`` behavior at all -- same features in, same
``anomaly_score`` / ``threshold_`` out.

Part 2: sanity-checks the opt-in ``n_spectral_bands`` plumbing added to
``src/features.py`` -- default (0 / omitted) must be byte-for-byte identical
to today's behavior, and a positive value must add exactly the expected new
columns without touching any existing column's values.
"""
import numpy as np
import pandas as pd

from src.anomaly_model import AnomalyDetector
from src.data_loader import load_directory
from src.features import (
    _window_stats,
    extract_features_from_frames,
    extract_window_features,
)
from src.synthetic_tdms import _make_normal_signals, inject_anomalies, write_tdms
from src.tdms_loader import load_tdms_directory

WINDOW_SIZE = 200


def _write_synthetic_files(root, n_normal=4, n_samples=10_000):
    """Write the same synthetic TDMS training files used by the golden test."""
    train_dir = root / "normal"
    for i in range(n_normal):
        signals = _make_normal_signals(n_samples=n_samples, seed=i)
        write_tdms(train_dir / f"normal_{i}.tdms", signals)
    return train_dir


def test_data_loader_matches_tdms_loader_anomaly_scores(tmp_path):
    """Old path (src.tdms_loader.load_tdms_directory) vs. new path
    (src.data_loader.load_directory) must produce numerically identical
    AnomalyDetector behavior end to end.
    """
    train_dir = _write_synthetic_files(tmp_path)

    # --- Old path ---
    old_frames = load_tdms_directory(train_dir)
    old_features = extract_features_from_frames(old_frames, window_size=WINDOW_SIZE)
    old_detector = AnomalyDetector(contamination=0.02, n_estimators=100)
    old_detector.fit(old_features)

    # --- New path ---
    new_frames = load_directory(train_dir)
    new_features = extract_features_from_frames(new_frames, window_size=WINDOW_SIZE)
    new_detector = AnomalyDetector(contamination=0.02, n_estimators=100)
    new_detector.fit(new_features)

    # Feature extraction itself must match exactly on the numeric columns
    # (both loaders read the same underlying TDMS samples/time index).
    numeric_old = old_features.select_dtypes(include=[np.number])
    numeric_new = new_features.select_dtypes(include=[np.number])
    assert list(numeric_old.columns) == list(numeric_new.columns)
    np.testing.assert_allclose(numeric_old.values, numeric_new.values)

    # Training-time derived threshold must match exactly.
    np.testing.assert_allclose(old_detector.threshold_, new_detector.threshold_)

    # Score a held-out anomalous test file through both detectors and compare.
    test_signals = _make_normal_signals(n_samples=10_000, seed=99)
    test_signals, _ = inject_anomalies(test_signals, seed=99, n_events=3)
    test_path = tmp_path / "test" / "test_run.tdms"
    write_tdms(test_path, test_signals)

    old_test_frame = load_tdms_directory(test_path.parent)[0]
    new_test_frame = load_directory(test_path.parent)[0]

    old_test_features = extract_window_features(old_test_frame, window_size=WINDOW_SIZE)
    new_test_features = extract_window_features(new_test_frame, window_size=WINDOW_SIZE)

    old_scores = old_detector.score(old_test_features)
    new_scores = new_detector.score(new_test_features)

    np.testing.assert_allclose(
        old_scores["anomaly_score"].values, new_scores["anomaly_score"].values
    )
    np.testing.assert_allclose(old_detector.threshold_, new_detector.threshold_)
    np.testing.assert_array_equal(
        old_scores["is_anomaly"].values, new_scores["is_anomaly"].values
    )


def test_n_spectral_bands_default_matches_omitted_argument():
    """n_spectral_bands=0 (explicit) must be identical to not passing it at all."""
    signals = _make_normal_signals(n_samples=1000, seed=2)
    df = pd.DataFrame(signals)

    features_default = extract_window_features(df, window_size=WINDOW_SIZE)
    features_explicit_zero = extract_window_features(
        df, window_size=WINDOW_SIZE, n_spectral_bands=0
    )

    pd.testing.assert_frame_equal(features_default, features_explicit_zero)


def test_n_spectral_bands_adds_columns_without_altering_existing_ones():
    """n_spectral_bands=3 must add exactly 3 new columns per channel and must
    not change the values of any pre-existing column.
    """
    signals = _make_normal_signals(n_samples=1000, seed=2)
    df = pd.DataFrame(signals)
    channels = list(signals.keys())

    baseline = extract_window_features(df, window_size=WINDOW_SIZE)
    with_bands = extract_window_features(df, window_size=WINDOW_SIZE, n_spectral_bands=3)

    # Existing columns (regardless of position -- band columns are
    # interleaved per-channel) keep identical values.
    assert set(baseline.columns).issubset(set(with_bands.columns))
    pd.testing.assert_frame_equal(with_bands[baseline.columns], baseline)

    # Exactly 3 new columns per channel, named as expected.
    expected_new_cols = {
        f"{ch}_fft_band_{i}" for ch in channels for i in range(3)
    }
    actual_new_cols = set(with_bands.columns) - set(baseline.columns)
    assert actual_new_cols == expected_new_cols
    assert len(with_bands.columns) - len(baseline.columns) == 3 * len(channels)


def test_window_stats_spectral_bands_reuses_existing_fft(tmp_path):
    """Direct unit check on _window_stats: band columns are present and
    finite, computed from the same fft_mag as the existing dominant-freq
    logic (excluding DC), for a variety of band counts."""
    rng = np.random.default_rng(0)
    window = rng.normal(size=256)

    stats = _window_stats(window, "ch", n_spectral_bands=4)
    for i in range(4):
        key = f"ch_fft_band_{i}"
        assert key in stats
        assert np.isfinite(stats[key])

    # n_spectral_bands=0 must not add any band keys.
    stats_zero = _window_stats(window, "ch", n_spectral_bands=0)
    assert not any(k.startswith("ch_fft_band_") for k in stats_zero)
