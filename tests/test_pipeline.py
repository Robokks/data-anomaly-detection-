import numpy as np
import pandas as pd
import pytest

from src.anomaly_model import AnomalyDetector
from src.features import extract_features_from_frames, extract_window_features
from src.synthetic_tdms import _make_normal_signals, inject_anomalies, write_tdms
from src.tdms_loader import load_tdms_dataframe, load_tdms_directory

WINDOW_SIZE = 200
SAMPLE_RATE_HZ = 1000


def test_write_and_load_tdms_roundtrip(tmp_path):
    signals = _make_normal_signals(n_samples=2000, seed=1)
    path = tmp_path / "sample.tdms"
    write_tdms(path, signals)

    df = load_tdms_dataframe(path)
    assert set(df.columns) == set(signals.keys())
    assert len(df) == 2000
    np.testing.assert_allclose(df["vibration"].values, signals["vibration"], atol=1e-6)


def test_extract_window_features_shape():
    signals = _make_normal_signals(n_samples=1000, seed=2)
    df = pd.DataFrame(signals)
    features = extract_window_features(df, window_size=WINDOW_SIZE)
    assert len(features) == 1000 // WINDOW_SIZE
    for ch in signals:
        for stat in ["mean", "std", "min", "max", "ptp", "rms", "crest_factor", "skew", "kurtosis", "dom_freq_mag", "zero_crossing_rate"]:
            assert f"{ch}_{stat}" in features.columns


def test_crest_factor_hand_computed():
    # A single spike among otherwise-zero samples has an easy-to-verify
    # crest factor: peak / rms = 10 / sqrt(100 / 10) = 10 / sqrt(10).
    x = np.zeros(10)
    x[0] = 10.0
    df = pd.DataFrame({"ch": x})
    features = extract_window_features(df, window_size=10)
    expected = 10.0 / np.sqrt(np.mean(x**2))
    assert features["ch_crest_factor"].iloc[0] == pytest.approx(expected)

    # All-zero window: rms == 0, crest_factor defined as 0.0 rather than NaN/inf.
    df_zero = pd.DataFrame({"ch": np.zeros(10)})
    features_zero = extract_window_features(df_zero, window_size=10)
    assert features_zero["ch_crest_factor"].iloc[0] == 0.0


def test_anomaly_detector_flags_injected_anomalies(tmp_path):
    train_dir = tmp_path / "normal"
    for i in range(4):
        signals = _make_normal_signals(n_samples=10_000, seed=i)
        write_tdms(train_dir / f"normal_{i}.tdms", signals)

    normal_frames = load_tdms_directory(train_dir)
    train_features = extract_features_from_frames(normal_frames, window_size=WINDOW_SIZE)

    detector = AnomalyDetector(contamination=0.02, n_estimators=100)
    detector.fit(train_features)

    test_signals = _make_normal_signals(n_samples=10_000, seed=99)
    test_signals, anomaly_ranges = inject_anomalies(test_signals, seed=99, n_events=3)
    test_path = tmp_path / "test_run.tdms"
    write_tdms(test_path, test_signals)

    test_df = load_tdms_dataframe(test_path)
    test_features = extract_window_features(test_df, window_size=WINDOW_SIZE)
    scores = detector.score(test_features)

    anomalous_window_indices = set()
    for start, end in anomaly_ranges:
        first_window = start // WINDOW_SIZE
        last_window = (end - 1) // WINDOW_SIZE
        anomalous_window_indices.update(range(first_window, last_window + 1))
    anomalous_window_indices &= set(range(len(scores)))
    normal_window_indices = set(range(len(scores))) - anomalous_window_indices

    assert anomalous_window_indices, "test setup should produce at least one anomalous window"
    mean_anomalous_score = scores["anomaly_score"].values[list(anomalous_window_indices)].mean()
    mean_normal_score = scores["anomaly_score"].values[list(normal_window_indices)].mean()

    assert mean_anomalous_score > mean_normal_score


def test_save_and_load_model(tmp_path):
    signals = _make_normal_signals(n_samples=5000, seed=3)
    import pandas as pd

    features = extract_window_features(pd.DataFrame(signals), window_size=WINDOW_SIZE)
    detector = AnomalyDetector(contamination=0.05, n_estimators=50).fit(features)

    model_path = tmp_path / "model.joblib"
    detector.save(model_path)
    loaded = AnomalyDetector.load(model_path)

    scores_original = detector.score(features)
    scores_loaded = loaded.score(features)
    np.testing.assert_allclose(
        scores_original["anomaly_score"].values, scores_loaded["anomaly_score"].values
    )
