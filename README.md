# LabVIEW TDMS Anomaly Detection

Train an unsupervised anomaly-detection model on LabVIEW TDMS sensor data and
use it to flag anomalous time windows in new recordings.

## How it works

1. **Load** — `src/tdms_loader.py` reads `.tdms` files (via
   [npTDMS](https://nptdms.readthedocs.io/)) into pandas DataFrames, one
   column per channel.
2. **Feature extraction** — `src/features.py` slides a fixed-size window over
   every channel and computes per-window statistics (mean, std, min/max,
   RMS, skew, kurtosis, dominant FFT magnitude, zero-crossing rate).
3. **Model** — `src/anomaly_model.py` fits an `AnomalyDetector` on windows of
   *normal* operating data only (no labeled anomalies required):
   - **Isolation Forest** catches multivariate outlier windows.
   - **PCA reconstruction error** catches drift and subtle shape changes
     relative to the learned "normal" subspace.
   - The two signals are z-normalized and averaged into a single
     `anomaly_score`; windows above a `contamination`-percentile threshold
     (learned from the training data) are flagged `is_anomaly`.
4. **Detect** — `src/detect.py` scores new TDMS files against a saved model
   and writes a CSV (and optional PNG plot) of per-window scores.

This approach needs no labeled fault data, which fits how most LabVIEW test
rigs are set up: you have plenty of recordings of normal operation and few
(or no) confirmed anomaly examples.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (with synthetic demo data)

No real TDMS files yet? Generate a synthetic rig (vibration / temperature /
pressure channels) with injected spikes, drift, and noise bursts:

```bash
python -m src.synthetic_tdms --out-dir data --n-normal 5 --n-samples 20000
```

This writes normal training files to `data/normal/` and one test file with
labeled anomaly ranges to `data/test/test_run.tdms`.

Train the model on the normal data:

```bash
python -m src.train \
  --data-dir data/normal \
  --window-size 256 \
  --contamination 0.02 \
  --model-out models/anomaly_detector.joblib
```

Score a new recording:

```bash
python -m src.detect \
  --model models/anomaly_detector.joblib \
  --input data/test/test_run.tdms \
  --window-size 256 \
  --output results/scores.csv \
  --plot
```

`results/scores.csv` gets one row per window (`iso_forest_score`,
`pca_recon_error`, `anomaly_score`, `is_anomaly`); `results/scores.png`
plots the anomaly score over time with the threshold and flagged windows
marked.

## Using your own TDMS data

1. Put a folder of TDMS recordings that represent **normal** operation in a
   directory, e.g. `data/normal/`.
2. If a file has multiple groups or you only want specific channels, pass
   `--group <name>` and/or `--channels ch1 ch2 ...` to `train.py`/`detect.py`.
3. Pick `--window-size` based on your sample rate — it should span enough
   samples to characterize one "cycle" of behavior (e.g. a few periods of
   your dominant vibration frequency), typically 100s–1000s of samples.
4. `--contamination` is your best estimate of what fraction of normal-data
   windows are already borderline/noisy; 0.01–0.05 is a reasonable start.
5. Train, then run `detect.py` on held-out or live recordings. Re-train
   periodically as your rig's "normal" baseline drifts (seasonal, wear, etc).

## Tests

```bash
pytest tests/ -v
```

Tests generate synthetic TDMS data, train a model, and assert that injected
anomalies score higher than normal windows — this exercises the full
load → feature-extract → train → score pipeline without needing real data.

## Project layout

```
src/
  tdms_loader.py     TDMS -> pandas DataFrame
  features.py        windowed statistical feature extraction
  anomaly_model.py    AnomalyDetector (Isolation Forest + PCA reconstruction)
  train.py            CLI: fit a model on normal data
  detect.py            CLI: score new data against a saved model
  synthetic_tdms.py    generate synthetic TDMS files for demo/testing
tests/
  test_pipeline.py    end-to-end pipeline tests on synthetic data
data/, models/, results/   generated at runtime (gitignored)
```

## Next steps / extension ideas

- Swap or ensemble in other detectors (One-Class SVM, LOF, autoencoders) by
  following the `AnomalyDetector` interface in `anomaly_model.py`.
- If you later collect confirmed anomaly labels, use them to validate/tune
  `--contamination` and the threshold rather than to train supervised — or
  add a supervised classifier on top of the same feature set.
- Wire `detect.py` into a scheduled job or LabVIEW post-processing step to
  score new runs automatically.
