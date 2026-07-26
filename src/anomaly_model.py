"""Unsupervised anomaly detector for TDMS sensor-window features.

Combines two complementary unsupervised signals so the model does not
depend on labeled anomaly data (rare in practice for LabVIEW test rigs):

- Isolation Forest: isolates points that are easy to separate in feature
  space -- good at catching multivariate outliers.
- PCA reconstruction error: fits a low-rank model of "normal" operation and
  flags windows that don't project well onto it -- good at catching drift
  and subtle multivariate shape changes.

The two scores are z-normalized against the training distribution and
averaged into a single ``anomaly_score``; a threshold (default: the
``contamination``-percentile of the training scores) turns that into a
boolean flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyDetector:
    contamination: float = 0.05
    n_estimators: int = 200
    pca_variance: float = 0.95
    random_state: int = 42

    # Opt-in hook for a future spectral-deviation signal (see src/spectral.py,
    # not yet implemented). Both default to values that make
    # ``_spectral_deviation`` a no-op, so behavior is unchanged today.
    spectral_weight: float = 0.0
    # Will eventually hold ``dict[str, SignatureBaseline]`` (per-channel
    # baselines from the not-yet-built src/spectral.py). Loosely typed for
    # now since that class doesn't exist yet.
    signature_baselines: dict | None = None

    feature_columns_: list[str] = field(default_factory=list, repr=False)
    scaler_: StandardScaler | None = field(default=None, repr=False)
    iso_forest_: IsolationForest | None = field(default=None, repr=False)
    pca_: PCA | None = field(default=None, repr=False)
    iso_score_mean_: float = 0.0
    iso_score_std_: float = 1.0
    recon_error_mean_: float = 0.0
    recon_error_std_: float = 1.0
    # Populated only when the (currently dead) spectral-deviation branch runs.
    spectral_dev_mean_: float = 0.0
    spectral_dev_std_: float = 1.0
    threshold_: float = 0.0

    def _select_numeric(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.select_dtypes(include=[np.number])

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        numeric = self._select_numeric(features)
        self.feature_columns_ = list(numeric.columns)

        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(numeric.values)

        self.iso_forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iso_forest_.fit(X)
        raw_iso = -self.iso_forest_.score_samples(X)  # higher = more anomalous

        n_components = min(self.pca_variance, X.shape[1], X.shape[0])
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        transformed = self.pca_.fit_transform(X)
        reconstructed = self.pca_.inverse_transform(transformed)
        raw_recon = np.mean((X - reconstructed) ** 2, axis=1)

        self.iso_score_mean_, self.iso_score_std_ = raw_iso.mean(), raw_iso.std() or 1.0
        self.recon_error_mean_, self.recon_error_std_ = raw_recon.mean(), raw_recon.std() or 1.0

        spectral_dev = self._spectral_deviation(features)
        if spectral_dev is None:
            combined = self._combine(raw_iso, raw_recon)
        else:
            self.spectral_dev_mean_, self.spectral_dev_std_ = (
                spectral_dev.mean(),
                spectral_dev.std() or 1.0,
            )
            combined = self._combine(raw_iso, raw_recon, spectral_dev)
        self.threshold_ = float(np.percentile(combined, 100 * (1 - self.contamination)))
        return self

    def _spectral_deviation(self, features: pd.DataFrame) -> np.ndarray | None:
        """Optional third anomaly signal: deviation from per-channel spectral
        signature baselines (see the future src/spectral.py::SignatureBaseline).

        Returns ``None`` (today, always) when spectral scoring is disabled --
        i.e. ``spectral_weight == 0.0`` or no ``signature_baselines`` are set.
        Real deviation scoring against ``SignatureBaseline`` objects is left
        for a later PR once that class exists.
        """
        if self.spectral_weight == 0.0 or not self.signature_baselines:
            return None
        raise NotImplementedError(
            "Spectral deviation scoring requires src.spectral.SignatureBaseline, "
            "which is not yet implemented."
        )

    def _combine(
        self,
        raw_iso: np.ndarray,
        raw_recon: np.ndarray,
        raw_spectral: np.ndarray | None = None,
    ) -> np.ndarray:
        z_iso = (raw_iso - self.iso_score_mean_) / self.iso_score_std_
        z_recon = (raw_recon - self.recon_error_mean_) / self.recon_error_std_
        if raw_spectral is None:
            return (z_iso + z_recon) / 2.0
        z_spectral = (raw_spectral - self.spectral_dev_mean_) / self.spectral_dev_std_
        base_weight = 1.0 - self.spectral_weight
        return base_weight * ((z_iso + z_recon) / 2.0) + self.spectral_weight * z_spectral

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None or self.iso_forest_ is None or self.pca_ is None:
            raise RuntimeError("AnomalyDetector must be fit() before scoring.")

        numeric = self._select_numeric(features).reindex(columns=self.feature_columns_, fill_value=0.0)
        X = self.scaler_.transform(numeric.values)

        raw_iso = -self.iso_forest_.score_samples(X)
        transformed = self.pca_.transform(X)
        reconstructed = self.pca_.inverse_transform(transformed)
        raw_recon = np.mean((X - reconstructed) ** 2, axis=1)

        spectral_dev = self._spectral_deviation(features)
        if spectral_dev is None:
            combined = self._combine(raw_iso, raw_recon)
        else:
            combined = self._combine(raw_iso, raw_recon, spectral_dev)
        return pd.DataFrame(
            {
                "iso_forest_score": raw_iso,
                "pca_recon_error": raw_recon,
                "anomaly_score": combined,
                "is_anomaly": combined >= self.threshold_,
            },
            index=features.index,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "AnomalyDetector":
        return joblib.load(Path(path))
