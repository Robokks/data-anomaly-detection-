"""Windowed statistical feature extraction for time-series sensor channels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def _window_stats(window: np.ndarray, channel: str) -> dict[str, float]:
    x = window.astype(float)
    n = len(x)
    rms = np.sqrt(np.mean(x**2))

    fft_mag = np.abs(np.fft.rfft(x - x.mean()))
    dominant_freq_bin = int(np.argmax(fft_mag[1:]) + 1) if len(fft_mag) > 1 else 0
    dominant_freq_mag = float(fft_mag[dominant_freq_bin]) if len(fft_mag) > 1 else 0.0

    zero_crossings = np.sum(np.diff(np.sign(x - x.mean())) != 0)

    return {
        f"{channel}_mean": float(x.mean()),
        f"{channel}_std": float(x.std()),
        f"{channel}_min": float(x.min()),
        f"{channel}_max": float(x.max()),
        f"{channel}_ptp": float(x.max() - x.min()),
        f"{channel}_rms": float(rms),
        f"{channel}_skew": float(skew(x)) if n > 2 else 0.0,
        f"{channel}_kurtosis": float(kurtosis(x)) if n > 2 else 0.0,
        f"{channel}_dom_freq_mag": dominant_freq_mag,
        f"{channel}_zero_crossing_rate": float(zero_crossings) / n,
    }


def extract_window_features(
    df: pd.DataFrame,
    window_size: int = 256,
    step: int | None = None,
) -> pd.DataFrame:
    """Slide a fixed-size window over every channel and compute summary stats.

    Returns one row of features per window, indexed by the window's starting
    timestamp/sample-index from ``df``. ``step`` defaults to ``window_size``
    (non-overlapping windows); use a smaller step for overlap.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    step = step or window_size
    n = len(df)
    if n < window_size:
        raise ValueError(f"DataFrame has {n} rows, shorter than window_size={window_size}")

    channels = list(df.columns)
    rows = []
    index = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        row: dict[str, float] = {}
        for ch in channels:
            row.update(_window_stats(df[ch].values[start:end], ch))
        rows.append(row)
        index.append(df.index[start])

    features = pd.DataFrame(rows, index=pd.Index(index, name=df.index.name))
    return features


def extract_features_from_frames(
    frames: list[pd.DataFrame],
    window_size: int = 256,
    step: int | None = None,
) -> pd.DataFrame:
    """Extract windowed features from multiple DataFrames and concatenate them."""
    feature_frames = []
    for df in frames:
        f = extract_window_features(df, window_size=window_size, step=step)
        f["source_file"] = df.attrs.get("source_file", "")
        feature_frames.append(f)
    return pd.concat(feature_frames, axis=0, ignore_index=True)
