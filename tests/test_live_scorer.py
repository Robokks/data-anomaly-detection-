import numpy as np
import pandas as pd
import pytest

from src.features import _window_stats
from src.live_scorer import LiveScorer, LiveWindowResult
from src.pipeline import train_model
from src.synthetic_tdms import _make_normal_signals, inject_anomalies

CHANNELS = ["vibration", "temperature", "pressure"]
SAMPLE_RATE_HZ = 1000.0
STATS_WINDOW_SAMPLES = 200
WINDOW_DURATION_SECONDS = STATS_WINDOW_SAMPLES / SAMPLE_RATE_HZ


def _chunk_sizes(total: int, sizes: list[int]) -> list[int]:
    """Split ``total`` into chunks cycling through ``sizes``, trimming the last."""
    chunks = []
    remaining = total
    i = 0
    while remaining > 0:
        size = min(sizes[i % len(sizes)], remaining)
        chunks.append(size)
        remaining -= size
        i += 1
    return chunks


def _baseline_df(n_samples: int = 4000, seed: int = 1) -> pd.DataFrame:
    return pd.DataFrame(_make_normal_signals(n_samples=n_samples, seed=seed))


def _fitted_minimal_scorer() -> LiveScorer:
    scorer = LiveScorer(
        model=None,
        model_type="classic",
        channels=CHANNELS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=10_000_000,  # never fills -- keeps this scorer model-free
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    scorer.fit_baseline([_baseline_df()])
    return scorer


def test_push_matches_direct_window_stats_regardless_of_chunking():
    n_samples = 5000
    signals = _make_normal_signals(n_samples=n_samples, seed=7)

    scorer = LiveScorer(
        model=None,
        model_type="classic",
        channels=CHANNELS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=10_000_000,  # never fills -- overall score stays None throughout
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    scorer.fit_baseline([_baseline_df()])

    chunk_sizes = _chunk_sizes(n_samples, [37, 250, 4, 999, 133, 501, 17, 89])

    all_results: list[LiveWindowResult] = []
    offset = 0
    for size in chunk_sizes:
        batch = {ch: signals[ch][offset : offset + size] for ch in CHANNELS}
        all_results.extend(scorer.push(batch))
        offset += size

    n_full_windows = n_samples // STATS_WINDOW_SAMPLES
    assert len(all_results) == n_full_windows

    for i, result in enumerate(all_results):
        assert result.window_index == i
        assert result.overall_anomaly_score is None
        assert result.overall_is_anomaly is None
        for ch in CHANNELS:
            window = signals[ch][i * STATS_WINDOW_SAMPLES : (i + 1) * STATS_WINDOW_SAMPLES]
            raw_expected = _window_stats(window, ch)
            prefix = f"{ch}_"
            expected = {k[len(prefix) :]: v for k, v in raw_expected.items() if k.startswith(prefix)}

            actual = result.channels[ch].stats
            assert set(actual.keys()) == set(expected.keys())
            for stat_name, expected_value in expected.items():
                np.testing.assert_allclose(actual[stat_name], expected_value, rtol=1e-9, atol=1e-9)


def test_warmup_until_model_deque_fills():
    model_window_size = 500
    baseline = _baseline_df()
    model = train_model(
        "classic",
        [baseline],
        window_size=model_window_size,
        step=model_window_size,
        contamination=0.1,
        n_estimators=50,
    )

    scorer = LiveScorer(
        model=model,
        model_type="classic",
        channels=CHANNELS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=model_window_size,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    scorer.fit_baseline([baseline])

    n_ticks = 10
    signals = _make_normal_signals(n_samples=STATS_WINDOW_SAMPLES * n_ticks, seed=42)

    cumulative = 0
    ticks_with_cumulative = []
    for i in range(n_ticks):
        batch = {
            ch: signals[ch][i * STATS_WINDOW_SAMPLES : (i + 1) * STATS_WINDOW_SAMPLES] for ch in CHANNELS
        }
        results = scorer.push(batch)
        assert len(results) == 1
        cumulative += STATS_WINDOW_SAMPLES
        ticks_with_cumulative.append((results[0], cumulative))

    # Sanity: this test is only meaningful if it actually straddles the warmup boundary.
    assert any(cum < model_window_size for _, cum in ticks_with_cumulative)
    assert any(cum >= model_window_size for _, cum in ticks_with_cumulative)

    for result, cumulative_after in ticks_with_cumulative:
        if cumulative_after < model_window_size:
            assert result.overall_anomaly_score is None
            assert result.overall_is_anomaly is None
        else:
            assert result.overall_anomaly_score is not None
            assert result.overall_is_anomaly is not None


def test_live_anomaly_detection_flags_injected_anomalies():
    model_window_size = STATS_WINDOW_SAMPLES  # aligned -- warmup completes almost immediately

    train_frames = [pd.DataFrame(_make_normal_signals(n_samples=10_000, seed=s)) for s in range(4)]
    model = train_model(
        "classic",
        train_frames,
        window_size=model_window_size,
        step=model_window_size,
        contamination=0.05,
        n_estimators=100,
    )

    scorer = LiveScorer(
        model=model,
        model_type="classic",
        channels=CHANNELS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=model_window_size,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
        channel_contamination=0.05,
    )
    scorer.fit_baseline(train_frames)

    n_samples = 10_000
    original_signals = _make_normal_signals(n_samples=n_samples, seed=99)
    test_signals, anomaly_ranges = inject_anomalies(dict(original_signals), seed=99, n_events=3)

    # inject_anomalies doesn't report which channel each event perturbed --
    # recover it by diffing against the pre-injection signal.
    event_channels = []
    for start, end in anomaly_ranges:
        diffs = {
            ch: float(np.max(np.abs(test_signals[ch][start:end] - original_signals[ch][start:end])))
            for ch in CHANNELS
        }
        event_channels.append(max(diffs, key=diffs.get))

    chunk_sizes = _chunk_sizes(n_samples, [211, 97, 480, 33, 650])
    all_results: list[LiveWindowResult] = []
    offset = 0
    for size in chunk_sizes:
        batch = {ch: test_signals[ch][offset : offset + size] for ch in CHANNELS}
        all_results.extend(scorer.push(batch))
        offset += size

    anomalous_indices_by_channel: dict[str, set[int]] = {ch: set() for ch in CHANNELS}
    all_anomalous_indices: set[int] = set()
    valid_range = set(range(len(all_results)))
    for (start, end), ch in zip(anomaly_ranges, event_channels):
        first_window = start // STATS_WINDOW_SAMPLES
        last_window = (end - 1) // STATS_WINDOW_SAMPLES
        idx_range = set(range(first_window, last_window + 1)) & valid_range
        anomalous_indices_by_channel[ch] |= idx_range
        all_anomalous_indices |= idx_range

    assert all_anomalous_indices, "test setup should produce at least one anomalous tick"

    for ch, anomalous_idx in anomalous_indices_by_channel.items():
        if not anomalous_idx:
            continue
        normal_idx = valid_range - anomalous_idx
        anomalous_devs = [all_results[i].channels[ch].deviation_score for i in anomalous_idx]
        normal_devs = [all_results[i].channels[ch].deviation_score for i in normal_idx]
        assert np.mean(anomalous_devs) > np.mean(normal_devs)

    normal_indices = valid_range - all_anomalous_indices
    overall_anomalous = [
        all_results[i].overall_anomaly_score
        for i in all_anomalous_indices
        if all_results[i].overall_anomaly_score is not None
    ]
    overall_normal = [
        all_results[i].overall_anomaly_score
        for i in normal_indices
        if all_results[i].overall_anomaly_score is not None
    ]
    assert overall_anomalous, "expected at least one post-warmup anomalous tick"
    assert overall_normal, "expected at least one post-warmup normal tick"
    assert np.mean(overall_anomalous) > np.mean(overall_normal)


def test_push_before_fit_baseline_raises():
    scorer = LiveScorer(
        model=None,
        model_type="classic",
        channels=CHANNELS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=100,
    )
    with pytest.raises(RuntimeError):
        scorer.push({ch: np.zeros(10) for ch in CHANNELS})


def test_push_rejects_mismatched_channel_set():
    scorer = _fitted_minimal_scorer()
    batch = {ch: np.zeros(10) for ch in CHANNELS if ch != "pressure"}
    with pytest.raises(ValueError):
        scorer.push(batch)


def test_push_rejects_mismatched_array_lengths():
    scorer = _fitted_minimal_scorer()
    batch = {"vibration": np.zeros(10), "temperature": np.zeros(20), "pressure": np.zeros(10)}
    with pytest.raises(ValueError):
        scorer.push(batch)


def test_push_single_large_batch_yields_multiple_ticks():
    scorer = LiveScorer(
        model=None,
        model_type="classic",
        channels=CHANNELS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=10_000_000,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    scorer.fit_baseline([_baseline_df()])

    n_ticks = 5
    signals = _make_normal_signals(n_samples=STATS_WINDOW_SAMPLES * n_ticks, seed=11)
    batch = {ch: signals[ch] for ch in CHANNELS}
    results = scorer.push(batch)

    assert len(results) == n_ticks
    assert [r.window_index for r in results] == list(range(n_ticks))
    # n_samples_seen is incremented once per push() call, not once per tick,
    # so every tick produced by this single call shares the same value.
    expected_n_samples_seen = STATS_WINDOW_SAMPLES * n_ticks
    assert all(r.n_samples_seen == expected_n_samples_seen for r in results)


def test_fit_baseline_warns_on_sample_rate_mismatch():
    signals = _make_normal_signals(n_samples=1000, seed=5)
    df = pd.DataFrame(signals)
    df.attrs["sample_rate_hz"] = 50_000  # far off from the configured 1000 Hz below

    scorer = LiveScorer(
        model=None,
        model_type="classic",
        channels=CHANNELS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_window_size=100,
        window_duration_seconds=0.1,  # -> 100-sample stats windows
    )

    with pytest.warns(UserWarning):
        scorer.fit_baseline([df])
