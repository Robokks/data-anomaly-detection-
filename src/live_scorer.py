"""Protocol-agnostic live/streaming anomaly scoring core.

Sensor data arrives continuously in arbitrary-sized chunks from whatever
transport layer is feeding it (network socket, simulator, etc. -- none of
that lives here). ``LiveScorer`` buffers incoming samples per channel and
produces two kinds of scored output as data streams in:

- **Per-channel deviation**: cheap, always-available z-score-style deviation
  of each channel's per-second (configurable) summary stats from a baseline
  learned by ``fit_baseline()``, computed with the same ``_window_stats``
  used everywhere else in this codebase.
- **Overall model score**: the existing "classic"/"deep" ``AnomalyDetector``/
  ``AutoencoderDetector`` scored (via ``src.pipeline.score_model``) against a
  trailing window of the most recent ``model_window_size`` samples per
  channel, once that trailing window has filled ("warmup").

No networking/GUI dependencies -- just numpy/pandas and this repo's existing
model/feature code, so it can be exercised entirely in-process in tests and
reused by whatever transport layer ends up calling it.
"""
from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import _window_stats, extract_features_from_frames
from src.pipeline import score_model

# Stat names produced by _window_stats (bare, unprefixed) -- kept in sync
# with src/features.py::_window_stats's returned dict keys.
STAT_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "ptp",
    "rms",
    "crest_factor",
    "skew",
    "kurtosis",
    "dom_freq_mag",
    "zero_crossing_rate",
]

_SAMPLE_RATE_RELATIVE_TOLERANCE = 0.01  # 1%


@dataclass
class ChannelResult:
    channel: str
    stats: dict[str, float]
    deviation_score: float
    is_anomaly: bool


@dataclass
class LiveWindowResult:
    window_index: int
    timestamp: float
    n_samples_seen: int
    channels: dict[str, ChannelResult]
    overall_anomaly_score: float | None
    overall_is_anomaly: bool | None


class LiveScorer:
    """Buffers streaming per-channel samples and produces scored ticks.

    Two independent buffering timescales are maintained per channel:

    1. A growable "stats" buffer that accumulates toward
       ``_stats_window_samples`` (derived from ``window_duration_seconds`` *
       ``sample_rate_hz``); each time it fills, a non-overlapping window's
       worth of per-channel summary stats is computed and compared against
       the ``fit_baseline()``-learned per-channel/per-stat mean/std.
    2. A trailing ``deque(maxlen=model_window_size)`` fed continuously,
       independent of (1), since the model's trained window size need not
       match one stats-tick's worth of samples. Once every channel's deque
       has filled, each stats-tick also gets an overall model score.
    """

    def __init__(
        self,
        model,
        model_type: str,
        channels: list[str],
        sample_rate_hz: float,
        model_window_size: int,
        model_step: int | None = None,
        window_duration_seconds: float = 1.0,
        channel_contamination: float = 0.05,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.channels = list(channels)
        self.sample_rate_hz = sample_rate_hz
        self.model_window_size = model_window_size
        self.model_step = model_step if model_step is not None else model_window_size
        self.window_duration_seconds = window_duration_seconds
        self.channel_contamination = channel_contamination

        self.baseline_mean_: dict[str, dict[str, float]] = {}
        self.baseline_std_: dict[str, dict[str, float]] = {}
        self.channel_thresholds_: dict[str, float] = {}
        self._stats_window_samples: int | None = None
        self._baseline_fit = False

        self._window_index = 0
        self._n_samples_seen = 0

        self._stats_buffers: dict[str, list[float]] = {ch: [] for ch in self.channels}
        self._model_deques: dict[str, deque] = {
            ch: deque(maxlen=self.model_window_size) for ch in self.channels
        }

    def fit_baseline(self, baseline_frames: list[pd.DataFrame]) -> None:
        stats_window_samples = round(self.window_duration_seconds * self.sample_rate_hz)
        if stats_window_samples < 2:
            raise ValueError(
                "window_duration_seconds * sample_rate_hz must be >= 2 samples; got "
                f"{stats_window_samples}"
            )
        self._stats_window_samples = stats_window_samples

        if baseline_frames:
            baseline_rate = baseline_frames[0].attrs.get("sample_rate_hz")
            if baseline_rate is not None:
                rel_diff = abs(baseline_rate - self.sample_rate_hz) / max(abs(self.sample_rate_hz), 1e-12)
                if rel_diff > _SAMPLE_RATE_RELATIVE_TOLERANCE:
                    warnings.warn(
                        f"baseline_frames[0].attrs['sample_rate_hz']={baseline_rate!r} differs from "
                        f"LiveScorer's configured sample_rate_hz={self.sample_rate_hz!r} by more than "
                        f"{_SAMPLE_RATE_RELATIVE_TOLERANCE:.0%}; stats-window durations may not match "
                        "the real sample rate of this data."
                    )

        features = extract_features_from_frames(
            baseline_frames,
            window_size=stats_window_samples,
            step=stats_window_samples,
        )
        features = features.drop(columns=["source_file"], errors="ignore")

        baseline_mean: dict[str, dict[str, float]] = {}
        baseline_std: dict[str, dict[str, float]] = {}
        for channel in self.channels:
            mean_for_channel: dict[str, float] = {}
            std_for_channel: dict[str, float] = {}
            for stat in STAT_NAMES:
                col = f"{channel}_{stat}"
                raw = features[col]
                mean_for_channel[stat] = float(raw.mean())
                std_for_channel[stat] = float(raw.std() or 1.0)
            baseline_mean[channel] = mean_for_channel
            baseline_std[channel] = std_for_channel

        self.baseline_mean_ = baseline_mean
        self.baseline_std_ = baseline_std

        # Baseline thresholds: deviation score for every baseline window, per channel.
        channel_thresholds: dict[str, float] = {}
        for channel in self.channels:
            deviations = []
            for _, row in features.iterrows():
                stats_dict = {stat: float(row[f"{channel}_{stat}"]) for stat in STAT_NAMES}
                deviations.append(self._compute_deviation(channel, stats_dict))
            deviations_arr = np.asarray(deviations, dtype=float)
            channel_thresholds[channel] = float(
                np.percentile(deviations_arr, 100 * (1 - self.channel_contamination))
            )
        self.channel_thresholds_ = channel_thresholds

        self._baseline_fit = True

    def _compute_deviation(self, channel: str, stats_dict: dict[str, float]) -> float:
        mean_for_channel = self.baseline_mean_[channel]
        std_for_channel = self.baseline_std_[channel]
        z_scores = [
            abs((stats_dict[stat] - mean_for_channel[stat]) / std_for_channel[stat])
            for stat in STAT_NAMES
        ]
        return float(np.mean(z_scores))

    def push(self, batch: dict[str, np.ndarray]) -> list[LiveWindowResult]:
        if not self._baseline_fit:
            raise RuntimeError("LiveScorer.push() called before fit_baseline().")

        if set(batch.keys()) != set(self.channels):
            raise ValueError(
                f"batch channel keys {sorted(batch.keys())} do not match "
                f"configured channels {sorted(self.channels)}"
            )

        lengths = {ch: len(arr) for ch, arr in batch.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"all arrays in batch must have equal length; got lengths {lengths}")

        batch_len = next(iter(lengths.values()))

        for channel in self.channels:
            arr = np.asarray(batch[channel], dtype=float)
            self._stats_buffers[channel].extend(arr.tolist())
            self._model_deques[channel].extend(arr.tolist())

        self._n_samples_seen += batch_len

        results: list[LiveWindowResult] = []
        stats_window_samples = self._stats_window_samples

        while all(len(self._stats_buffers[ch]) >= stats_window_samples for ch in self.channels):
            channel_results: dict[str, ChannelResult] = {}
            for channel in self.channels:
                buf = self._stats_buffers[channel]
                window = np.asarray(buf[:stats_window_samples], dtype=float)
                del buf[:stats_window_samples]

                raw_stats = _window_stats(window, channel)
                prefix = f"{channel}_"
                stats_dict = {
                    key[len(prefix):]: value for key, value in raw_stats.items() if key.startswith(prefix)
                }

                deviation_score = self._compute_deviation(channel, stats_dict)
                is_anomaly = deviation_score >= self.channel_thresholds_[channel]

                channel_results[channel] = ChannelResult(
                    channel=channel,
                    stats=stats_dict,
                    deviation_score=deviation_score,
                    is_anomaly=is_anomaly,
                )

            overall_anomaly_score: float | None = None
            overall_is_anomaly: bool | None = None
            if all(len(self._model_deques[ch]) == self.model_window_size for ch in self.channels):
                model_df = pd.DataFrame(
                    {ch: list(self._model_deques[ch]) for ch in self.channels}
                )
                scores = score_model(
                    self.model,
                    self.model_type,
                    [model_df],
                    window_size=self.model_window_size,
                    step=self.model_step,
                )
                overall_anomaly_score = float(scores["anomaly_score"].iloc[0])
                overall_is_anomaly = bool(scores["is_anomaly"].iloc[0])

            result = LiveWindowResult(
                window_index=self._window_index,
                timestamp=time.time(),
                n_samples_seen=self._n_samples_seen,
                channels=channel_results,
                overall_anomaly_score=overall_anomaly_score,
                overall_is_anomaly=overall_is_anomaly,
            )
            results.append(result)
            self._window_index += 1

        return results
