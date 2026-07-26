"""Spectral signature analysis for sensor channels.

Standard vibration/process-signal fault-detection technique: compute a
channel's FFT magnitude spectrum and compare it against a learned "normal"
baseline spectrum (mean +/- std envelope across many training windows) to
spot drift or new frequency components. Intended to back a GUI panel where a
user picks a channel, hits "Compute Signature", and sees the current
spectrum overlaid on the baseline's mean +/- std envelope.

This module is standalone -- it is not wired into ``src/anomaly_model.py``'s
combined ``anomaly_score`` (that integration is deliberately deferred; see
``AnomalyDetector._spectral_deviation``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import extract_raw_windows_from_frames


def compute_spectrum(x: np.ndarray, sample_rate: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """FFT magnitude spectrum of a single 1D signal.

    ``x`` can be one window or a whole channel's samples. Returns
    ``(freqs, magnitude)``. If ``sample_rate`` is given, ``freqs`` are in Hz
    via ``np.fft.rfftfreq(len(x), d=1/sample_rate)``; otherwise ``freqs`` are
    just bin indices (``0..len(magnitude)-1``). ``magnitude`` is
    ``np.abs(np.fft.rfft(x - x.mean()))`` (DC offset removed first, matching
    the convention already used in ``src/features.py``'s ``_window_stats``).
    """
    x = np.asarray(x, dtype=float)
    magnitude = np.abs(np.fft.rfft(x - x.mean()))
    if sample_rate is not None:
        freqs = np.fft.rfftfreq(len(x), d=1 / sample_rate)
    else:
        freqs = np.arange(len(magnitude))
    return freqs, magnitude


@dataclass
class SignatureBaseline:
    channel: str
    freqs_: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_: np.ndarray = field(default_factory=lambda: np.array([]))
    std_: np.ndarray = field(default_factory=lambda: np.array([]))
    n_windows_: int = 0

    def fit(self, windows: np.ndarray, sample_rate: float | None = None) -> "SignatureBaseline":
        """Fit the mean/std spectral envelope from raw windows.

        ``windows``: ``(n_windows, window_size)`` raw samples for this
        channel. Computes ``compute_spectrum`` for every window (all windows
        must be the same length, so every spectrum has the same freq bins),
        stacks them, and stores the per-frequency-bin ``mean_``/``std_``
        envelope (``std_`` floored at ``1.0`` where zero, to avoid
        divide-by-zero in ``deviation()``). ``freqs_`` is stored from
        ``compute_spectrum``'s first window.
        """
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 2 or windows.shape[0] == 0:
            raise ValueError("windows must be a non-empty 2D array of shape (n_windows, window_size)")

        spectra = []
        freqs = None
        for i in range(windows.shape[0]):
            f, mag = compute_spectrum(windows[i], sample_rate=sample_rate)
            if freqs is None:
                freqs = f
            spectra.append(mag)
        stacked = np.stack(spectra, axis=0)

        self.freqs_ = freqs
        self.mean_ = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        self.n_windows_ = windows.shape[0]
        return self

    def deviation(self, x: np.ndarray, sample_rate: float | None = None) -> float:
        """Scalar deviation of a single window's spectrum from the baseline.

        ``x``: a single 1D window (same length as the training windows).
        Computes ``compute_spectrum(x, sample_rate)``, then returns the mean
        of ``abs((magnitude - mean_) / std_)`` across frequency bins.
        """
        if self.mean_.size == 0 or self.std_.size == 0:
            raise RuntimeError("SignatureBaseline must be fit() (or loaded) before calling deviation().")

        _, magnitude = compute_spectrum(x, sample_rate=sample_rate)
        return float(np.mean(np.abs((magnitude - self.mean_) / self.std_)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "SignatureBaseline":
        return joblib.load(Path(path))


def fit_baselines_from_frames(
    frames: list[pd.DataFrame],
    channels: list[str] | None = None,
    window_size: int = 256,
    step: int | None = None,
    sample_rate: float | None = None,
) -> dict[str, SignatureBaseline]:
    """Fit one ``SignatureBaseline`` per channel from raw windowed waveforms.

    For each channel (default: all columns present in every frame, in
    ``frames[0]``'s column order -- reuses the same channel-consistency
    expectations as ``extract_raw_windows_from_frames``, which this calls
    under the hood), extracts that channel's raw windows across all frames
    and fits one ``SignatureBaseline``. Returns
    ``{channel_name: SignatureBaseline}``.

    A channel that isn't present in every frame is skipped gracefully
    (rather than raising), matching ``extract_raw_windows_from_frames``'s
    "all frames must share the same columns" contract: only the columns
    common to every frame (in ``frames[0]``'s order) are eligible.
    """
    if not frames:
        raise ValueError("frames must be a non-empty list of DataFrames")

    common_columns = [c for c in frames[0].columns if all(c in df.columns for df in frames)]
    if channels is None:
        target_channels = common_columns
    else:
        target_channels = [c for c in channels if c in common_columns]

    if not target_channels:
        return {}

    # extract_raw_windows_from_frames requires identical columns/order across
    # frames, so slice every frame down to the shared, ordered channel set
    # before extracting windows once for all target channels together.
    aligned_frames = [df[target_channels] for df in frames]
    windows, _ = extract_raw_windows_from_frames(aligned_frames, window_size=window_size, step=step)

    baselines: dict[str, SignatureBaseline] = {}
    for i, channel in enumerate(target_channels):
        baseline = SignatureBaseline(channel=channel)
        baseline.fit(windows[:, i, :], sample_rate=sample_rate)
        baselines[channel] = baseline
    return baselines
