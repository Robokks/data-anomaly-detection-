"""Deep-learning anomaly detector: a 1D convolutional autoencoder.

A second, selectable model type alongside the "classic" ``AnomalyDetector``
in ``src/anomaly_model.py``. Where the classic model trains on hand-crafted
per-window statistics, this one trains directly on raw, normalized window
waveforms -- so it can learn shape/temporal structure that the classic
model's summary stats don't capture.

Architecture is a small 1D convolutional autoencoder (not LSTM -- chosen for
CPU training speed and simpler packaging, no GPU assumed) operating on
``(n_channels, window_size)`` tensors per window. It is trained on windows
drawn from user-selected "normal" data and scored by per-window mean-squared
reconstruction error.

The anomaly-score/threshold convention mirrors the classic model exactly, so
the two are interchangeable from the UI/CLI's point of view: the raw
reconstruction error is z-normalized against the *training* distribution's
mean/std, and the detector flags windows whose z-score is at or above the
``contamination``-percentile of the training scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import extract_raw_windows_from_frames


class Conv1dAutoencoder(nn.Module):
    """Small 1D convolutional autoencoder over ``(n_channels, window_size)`` windows.

    Encoder: two stride-2 Conv1d+ReLU blocks (halving the temporal length
    each time) flattened into a Linear bottleneck of size ``latent_dim``.
    Decoder mirrors this with a Linear expansion followed by two
    ConvTranspose1d+ReLU blocks, using an explicit ``output_size`` on each
    transpose-conv so the output lands back on exactly ``window_size``
    regardless of parity. Kept intentionally small so it trains in seconds
    on tiny synthetic fixtures during tests and stays fast on CPU for real
    use.
    """

    def __init__(self, n_channels: int, window_size: int, latent_dim: int = 16):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.latent_dim = latent_dim

        # Temporal length after each stride-2 conv (ceil-division, matching
        # Conv1d's output-length formula for kernel_size=5, padding=2, stride=2).
        self.len1 = (window_size + 1) // 2
        self.len2 = (self.len1 + 1) // 2

        self.enc_conv1 = nn.Conv1d(n_channels, 8, kernel_size=5, stride=2, padding=2)
        self.enc_conv2 = nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2)
        self.enc_fc = nn.Linear(16 * self.len2, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 16 * self.len2)
        self.dec_conv1 = nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=2)
        self.dec_conv2 = nn.ConvTranspose1d(8, n_channels, kernel_size=5, stride=2, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.enc_conv1(x))
        h = self.relu(self.enc_conv2(h))
        h = h.reshape(h.size(0), -1)
        z = self.enc_fc(h)

        h = self.relu(self.dec_fc(z))
        h = h.reshape(h.size(0), 16, self.len2)
        h = self.relu(self.dec_conv1(h, output_size=(h.size(0), 8, self.len1)))
        out = self.dec_conv2(h, output_size=(h.size(0), self.n_channels, self.window_size))
        return out


@dataclass
class AutoencoderDetector:
    window_size: int = 256
    step: int | None = None
    latent_dim: int = 16
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    contamination: float = 0.02
    random_state: int = 42
    device: str = "cpu"

    channels_: list[str] = field(default_factory=list, repr=False)
    channel_mean_: np.ndarray | None = field(default=None, repr=False)
    channel_std_: np.ndarray | None = field(default=None, repr=False)
    state_dict_: dict | None = field(default=None, repr=False)  # picklable weights, not the live nn.Module
    recon_error_mean_: float = 0.0
    recon_error_std_: float = 1.0
    threshold_: float = 0.0

    def __post_init__(self) -> None:
        # Live nn.Module, rebuilt lazily from state_dict_ + architecture params
        # (never stored in the pickled state -- see __getstate__/__setstate__).
        self._model: Conv1dAutoencoder | None = None

    def _normalize(self, windows: np.ndarray) -> np.ndarray:
        mean = self.channel_mean_[None, :, None]
        std = self.channel_std_[None, :, None]
        return (windows - mean) / std

    def _ensure_model(self) -> Conv1dAutoencoder:
        if self._model is None:
            if not self.channels_:
                raise RuntimeError("AutoencoderDetector must be fit() (or loaded) before use.")
            model = Conv1dAutoencoder(
                n_channels=len(self.channels_),
                window_size=self.window_size,
                latent_dim=self.latent_dim,
            )
            if self.state_dict_ is not None:
                model.load_state_dict(self.state_dict_)
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def _reconstruction_error(self, normalized: np.ndarray) -> np.ndarray:
        model = self._ensure_model()
        model.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(normalized), self.batch_size):
                batch = torch.as_tensor(
                    normalized[start : start + self.batch_size], dtype=torch.float32
                ).to(self.device)
                recon = model(batch)
                mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
                errors.append(mse.cpu().numpy())
        return np.concatenate(errors) if errors else np.empty((0,))

    def fit(self, frames: list[pd.DataFrame]) -> "AutoencoderDetector":
        windows, _ = extract_raw_windows_from_frames(frames, window_size=self.window_size, step=self.step)
        if windows.shape[0] == 0:
            raise ValueError("No training windows produced -- check window_size against input data length")

        self.channels_ = list(frames[0].columns)
        n_channels = len(self.channels_)

        self.channel_mean_ = windows.mean(axis=(0, 2))
        channel_std = windows.std(axis=(0, 2))
        channel_std = np.where(channel_std == 0, 1.0, channel_std)
        self.channel_std_ = channel_std

        normalized = self._normalize(windows)

        torch.manual_seed(self.random_state)
        model = Conv1dAutoencoder(n_channels=n_channels, window_size=self.window_size, latent_dim=self.latent_dim)
        model.to(self.device)
        model.train()

        dataset = TensorDataset(torch.as_tensor(normalized, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

        model.eval()
        self._model = model
        self.state_dict_ = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        raw_recon = self._reconstruction_error(normalized)
        self.recon_error_mean_ = float(raw_recon.mean())
        self.recon_error_std_ = float(raw_recon.std() or 1.0)

        anomaly_score = (raw_recon - self.recon_error_mean_) / self.recon_error_std_
        self.threshold_ = float(np.percentile(anomaly_score, 100 * (1 - self.contamination)))
        return self

    def score(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not self.channels_ or self.channel_mean_ is None:
            raise RuntimeError("AutoencoderDetector must be fit() (or loaded) before scoring.")

        for df in frames:
            if list(df.columns) != self.channels_:
                raise ValueError(
                    "DataFrame columns do not match the channels this detector was fit on; "
                    f"expected {self.channels_}, got {list(df.columns)}"
                )

        windows, index = extract_raw_windows_from_frames(frames, window_size=self.window_size, step=self.step)
        normalized = self._normalize(windows)
        raw_recon = self._reconstruction_error(normalized)

        anomaly_score = (raw_recon - self.recon_error_mean_) / self.recon_error_std_
        return pd.DataFrame(
            {
                "recon_error": raw_recon,
                "anomaly_score": anomaly_score,
                "is_anomaly": anomaly_score >= self.threshold_,
            },
            index=pd.Index(index) if index else None,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "AutoencoderDetector":
        return joblib.load(Path(path))

    def __getstate__(self) -> dict:
        # Drop the live torch.nn.Module (and any device handles it holds) --
        # it isn't picklable/portable across devices. state_dict_ (a plain
        # dataclass field, already CPU tensors) is what actually gets saved;
        # self._model = None in the restored state acts as the "needs
        # rebuild" flag, resolved lazily by _ensure_model() on next use.
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._model = None
