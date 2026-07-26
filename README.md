# LabVIEW Test-Rig Anomaly Detection

A desktop app (and a scriptable CLI) for training an AI anomaly-detection
model on sensor data from LabVIEW test rigs — TDMS, CSV, or Excel — and
using it to flag anomalous time windows in new recordings.

## What it does

- **Loads TDMS, CSV, and Excel** data uniformly, scanning a folder
  recursively regardless of file naming or subfolder layout.
- **Understands multi-file test rigs.** A single unit-under-test (UUT) isn't
  always one file — pick a **machine type** and the app reassembles files
  back into one logical session per UUT:
  - **General** — one file = one UUT (the default).
  - **Transmission** — a UUT's cycle is split across ramp-up / ramp-down /
    steady-state / coasting files; analysis defaults to steady-state +
    coasting (ramp phases are transients, available but off by default).
  - **Motor test bench** — a UUT's files are split by test step/"pocket"
    number (a physical test station/slot).
  - **Endurance test rig** — a UUT may log to one file or several
    size-rolled-over chunks, reassembled chronologically.
- **Lets you curate the data visually** before training: plot any number of
  channels together to spot a bad one, independently choose which channels
  feed the model (Plot vs. Train checkboxes), and restrict analysis to the
  full recording or a specific time range.
- **Trains two kinds of model**, picked per run:
  - **Classic** — Isolation Forest + PCA reconstruction error on
    hand-crafted per-window statistics. Fast, no GPU, good default.
  - **Deep learning** — a 1D convolutional autoencoder (PyTorch) trained
    directly on raw waveform windows, learning shape/temporal structure the
    classic model's summary stats don't capture.
  Both are unsupervised (trained on "normal" data only — no labeled fault
  examples needed) and produce the same `anomaly_score`/`is_anomaly` shape,
  so they're interchangeable.
- **Signature analysis** — compares a channel's live FFT spectrum against a
  learned normal baseline (mean ± std envelope) to spot drift or new
  frequency components.
- Everything is available both as a **native desktop GUI** (PySide6/Qt,
  packagable into a standalone `.exe`) and as **CLI scripts** for scripted,
  headless, or scheduled use — both go through the same underlying pipeline.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- **torch**: the plain `pip install -r requirements.txt` pulls whichever
  wheel PyPI resolves, which may be a large GPU-enabled build. This app is
  CPU-only by design; for a smaller install use the CPU wheel instead:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **Linux GUI system libraries**: Qt needs a couple of libraries pip doesn't
  install. If you hit `libEGL.so.1: cannot open shared object file`, run
  `sudo apt-get install libegl1 libegl-mesa0` (Debian/Ubuntu; other distros
  have equivalent packages). Not needed on Windows/macOS.

## Running the GUI

```bash
python -m gui.app
```

Workflow: **Browse Folder** → pick a **machine type** → the app scans the
folder in the background and groups files into per-UUT sessions → select a
session in the tree → toggle **Plot**/**Train** checkboxes per channel and
drag the time-range region on the plot to curate what goes into training →
**Train...** (choose Classic or Deep Learning, set hyperparameters) →
flagged windows show up in the results table and as an overlay on the plot
→ **Signature Analysis...** for the FFT-vs-baseline view → **Save Model...**
/ export scores to CSV.

## Running the CLI

Generate synthetic demo data if you don't have real files yet (vibration /
temperature / pressure channels with injected spikes, drift, and noise
bursts):

```bash
python -m src.synthetic_tdms --out-dir data --n-normal 5 --n-samples 20000
```

Train on a folder of normal recordings:

```bash
python -m src.train \
  --data-dir data/normal \
  --window-size 256 --contamination 0.02 \
  --model-type classic \
  --model-out models/anomaly_detector.joblib
```

Use `--model-type deep` for the autoencoder instead (see `--epochs`,
`--batch-size`, `--latent-dim`). For a multi-file rig, add
`--machine-type {transmission,motor_test_bench,endurance}` (and optionally
`--phases`/`--pockets`/`--uut-pattern`); add `--start`/`--end` to restrict to
a time range. `--recursive`/`--no-recursive` controls subfolder scanning.

Score a new recording or folder:

```bash
python -m src.detect \
  --model models/anomaly_detector.joblib \
  --input data/test/test_run.tdms \
  --window-size 256 \
  --output results/scores.csv \
  --plot
```

`results/scores.csv` has one row per window (`anomaly_score`, `is_anomaly`,
plus model-specific columns); `results/scores.png` plots the score over time
with the threshold and flagged windows marked. Run `python -m src.train -h`
/ `python -m src.detect -h` for the full flag list.

## Packaging as a standalone executable

See `packaging/README.md`. Short version: `packaging/app.spec` is a
PyInstaller spec for `gui/app.py`; run `packaging/build_windows.bat` **on a
Windows machine** to get a real `.exe` (PyInstaller isn't a
cross-compiler — it can't be built here). `packaging/build_linux_smoke.sh`
builds+headlessly-launches a Linux binary in this repo's environment purely
to smoke-test that the spec's imports resolve, not as a distributable.

## Tests

```bash
pytest tests/ -v
```

All GUI tests run headlessly (via `pytest-qt` + `QT_QPA_PLATFORM=offscreen`,
set automatically in `tests/conftest.py`) — no display needed. Tests use
synthetic TDMS/CSV/session fixtures throughout (`src/synthetic_tdms.py`,
`src/synthetic_sessions.py`), so the full pipeline is exercised without
needing real data: loading, scanning, session grouping, time-range
selection, feature extraction, both model types, spectral analysis, every
GUI panel individually, and a full end-to-end run through `MainWindow`
(`tests/test_main_window_integration.py`).

## Project layout

```
src/
  data_loader.py       unified .tdms/.csv/.xlsx loading + recursive folder scanning
  tdms_loader.py         TDMS -> pandas DataFrame (wrapped by data_loader.py)
  session_grouping.py    machine-type-aware multi-file -> per-UUT session grouping
  time_range.py           full-data or [start, end] slicing
  features.py               windowed statistical + raw-window feature extraction
  anomaly_model.py            AnomalyDetector (classic: Isolation Forest + PCA)
  dl_model.py                   AutoencoderDetector (deep: 1D conv autoencoder)
  spectral.py                     SignatureBaseline (FFT signature analysis)
  pipeline.py                       dispatch layer: load -> train/score, either model type
  train.py / detect.py                CLIs, built on pipeline.py
  synthetic_tdms.py, synthetic_sessions.py   synthetic data generators for demo/testing
gui/
  app.py                entry point (python -m gui.app)
  main_window.py           overall layout + toolbar actions
  app_state.py               shared cross-panel state (Qt signals)
  session_panel.py             machine-type selector + scan/group tree
  plot_panel.py                  pyqtgraph multi-channel plot + time-range selector
  channel_panel.py                 Plot/Train channel curation checkboxes
  train_dialog.py                    model training (background thread)
  results_panel.py                     flagged-window table + CSV export
  signature_panel.py                     FFT-vs-baseline signature analysis
packaging/            PyInstaller spec + build scripts
tests/                 pytest + pytest-qt, all headless
data/, models/, results/   generated at runtime (gitignored)
```

## Extension ideas

- Fold spectral deviation into the combined `anomaly_score` (the hook
  already exists — `AnomalyDetector.spectral_weight`/`signature_baselines`,
  currently an opt-in no-op) once real spectral behavior has been validated
  on actual rig data.
- Per-session (rather than one shared) channel/time-range curation when
  training across many sessions in a folder at once.
- If you later collect confirmed anomaly labels, use them to validate/tune
  `--contamination` and the threshold rather than to train supervised — or
  add a supervised classifier on top of the same feature set.
- Tighten the filename/folder heuristics used to auto-detect phase
  (ramp/steady/coast) and pocket/step numbers once you can share real
  filename examples from your machines.
