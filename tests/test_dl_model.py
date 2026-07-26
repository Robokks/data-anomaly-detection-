import numpy as np
import pandas as pd

from src.dl_model import AutoencoderDetector
from src.features import extract_raw_windows, extract_raw_windows_from_frames
from src.synthetic_tdms import _make_normal_signals, inject_anomalies, write_tdms
from src.tdms_loader import load_tdms_dataframe, load_tdms_directory

WINDOW_SIZE = 128


def test_extract_raw_windows_shape():
    signals = _make_normal_signals(n_samples=1000, seed=2)
    df = pd.DataFrame(signals)
    windows, index = extract_raw_windows(df, window_size=WINDOW_SIZE)

    assert windows.shape == (1000 // WINDOW_SIZE, len(signals), WINDOW_SIZE)
    assert len(index) == windows.shape[0]
    # First window should match the raw samples of the first channel exactly.
    first_channel = list(signals.keys())[0]
    np.testing.assert_allclose(windows[0, 0, :], signals[first_channel][:WINDOW_SIZE])


def test_extract_raw_windows_from_frames_concatenates():
    signals_a = _make_normal_signals(n_samples=600, seed=1)
    signals_b = _make_normal_signals(n_samples=900, seed=2)
    df_a = pd.DataFrame(signals_a)
    df_b = pd.DataFrame(signals_b)

    windows, index = extract_raw_windows_from_frames([df_a, df_b], window_size=WINDOW_SIZE)

    expected_n = (600 // WINDOW_SIZE) + (900 // WINDOW_SIZE)
    assert windows.shape == (expected_n, len(signals_a), WINDOW_SIZE)
    assert len(index) == expected_n


def test_extract_raw_windows_from_frames_rejects_mismatched_columns():
    df_a = pd.DataFrame(_make_normal_signals(n_samples=500, seed=1))
    df_b = df_a.drop(columns=["temperature"])

    try:
        extract_raw_windows_from_frames([df_a, df_b], window_size=WINDOW_SIZE)
        assert False, "expected ValueError for mismatched columns"
    except ValueError:
        pass


def test_autoencoder_detector_flags_injected_anomalies(tmp_path):
    train_dir = tmp_path / "normal"
    for i in range(3):
        signals = _make_normal_signals(n_samples=3000, seed=i)
        write_tdms(train_dir / f"normal_{i}.tdms", signals)

    normal_frames = load_tdms_directory(train_dir)

    detector = AutoencoderDetector(
        window_size=WINDOW_SIZE,
        latent_dim=8,
        epochs=15,
        batch_size=16,
        contamination=0.05,
        random_state=42,
    )
    detector.fit(normal_frames)

    test_signals = _make_normal_signals(n_samples=3000, seed=99)
    test_signals, anomaly_ranges = inject_anomalies(test_signals, seed=99, n_events=3)
    test_path = tmp_path / "test_run.tdms"
    write_tdms(test_path, test_signals)

    test_df = load_tdms_dataframe(test_path)
    scores = detector.score([test_df])

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


def test_autoencoder_save_and_load_roundtrip(tmp_path):
    signals = _make_normal_signals(n_samples=3000, seed=3)
    df = pd.DataFrame(signals)

    detector = AutoencoderDetector(
        window_size=WINDOW_SIZE,
        latent_dim=8,
        epochs=10,
        batch_size=16,
        contamination=0.05,
        random_state=7,
    ).fit([df])

    model_path = tmp_path / "ae_model.joblib"
    detector.save(model_path)
    loaded = AutoencoderDetector.load(model_path)

    scores_original = detector.score([df])
    scores_loaded = loaded.score([df])

    np.testing.assert_allclose(
        scores_original["anomaly_score"].values,
        scores_loaded["anomaly_score"].values,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        scores_original["is_anomaly"].values, scores_loaded["is_anomaly"].values
    )
