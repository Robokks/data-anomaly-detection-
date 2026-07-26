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
- **Live streaming scoring** — in addition to batch (upload-a-file) scoring,
  a running test rig can stream sensor data over TCP and get scored
  continuously: an overall anomaly score plus a per-channel score, and a
  live per-second statistics table (min/max/rms/crest factor/skew/kurtosis/
  etc). Available as a GUI panel, a standalone headless receiver, and a
  test data simulator for trying it without real hardware — see
  [Live streaming](#live-streaming) below.
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

Generate synthetic demo data if you don't have real files yet. Three
`--channel-set` options:

```bash
# basic: vibration/temperature/pressure at 1 kHz, small files -- quick smoke test
python -m src.synthetic_tdms --out-dir data --n-normal 5 --n-samples 20000

# full: a steady-state 10-channel rig (vibration, noise, speed, current,
# voltage_a/b/c, temperature, pressure, torque) at a real sample rate/duration
python -m src.synthetic_tdms --out-dir data --n-normal 100 --channel-set full \
  --sample-rate 10000 --duration-seconds 10

# cycle: the same 10 channels, but following a ramp/hold/de-ramp speed+torque
# test cycle (0->1000->6000->1000->0 rpm, 0->50->250->0 Nm) with vibration/
# noise/current tied to the profile -- see _make_test_cycle_signals in
# src/synthetic_tdms.py for the exact physical relationships modeled.
# Note: 100 files at these settings is ~500-600MB -- regenerate locally
# rather than committing it (it's gitignored and seeded, so this exact
# command reproduces byte-identical output every time).
python -m src.synthetic_tdms --out-dir data --n-normal 100 --channel-set cycle \
  --sample-rate 10000 --duration-min 5 --duration-max 10 --seed 0
```

All three write normal training files to `data/normal/` and one test file
with labeled injected anomalies to `data/test/test_run.tdms`.

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

## Live streaming

Batch scoring (above) needs a finished recording. For scoring a **running**
rig continuously, a sender (e.g. LabVIEW, or the test simulator below)
streams sensor data over TCP; the exact wire format is documented in
[`docs/streaming_protocol.md`](docs/streaming_protocol.md) — hand that file
to whoever builds the sending side. Whatever transport receives the data,
scoring works the same way: an **overall** anomaly score from your trained
model (classic or deep, same as batch scoring) plus an independent
**per-channel deviation score** (each channel's live min/max/rms/crest
factor/skew/kurtosis/etc compared against a learned baseline), on a
configurable cadence (default: once per second).

### Try it without real hardware

Generate a small baseline dataset and train a model (see "Running the CLI"
above), then in one terminal:

```bash
python -m src.stream_server \
  --model models/anomaly_detector.joblib \
  --baseline-dir data/normal \
  --sample-rate 1000 --window-size 256 \
  --port 9999 --output-csv results/live_scores.csv
```

and in another, stream synthetic data at it (optionally with injected
anomalies to confirm detection):

```bash
python -m src.stream_simulator --port 9999 --channel-set basic \
  --n-samples 20000 --inject-anomalies --no-realtime
```

The server prints one line per scored window (overall score, anomaly flag,
which channels are currently flagged) and appends the full per-channel
statistics to the CSV if `--output-csv` is given.

### GUI panel

`python -m gui.app` → **Live Monitor...** toolbar action → pick a model
(reuse the one just trained in this session, or load a saved `.joblib`) →
point at a baseline folder (same "normal" data you'd train on) → set the
stream's sample rate and the model's window size (auto-filled when the
model has it stored) → **Start**. The table shows one row per channel plus
an **Overall** row, live-updating as windows complete; **Stop** when done
(or just close the panel — it stops the server automatically).

### Standalone headless receiver

`src/stream_server.py` is the same scoring core with no GUI/Qt dependency —
for running unattended next to a rig. It supports the same
`--machine-type`/`--phases`/`--pockets`/`--channels`/`--start`/`--end` flags
as `train.py` for building its baseline. Run `python -m src.stream_server -h`
for the full flag list; Ctrl+C stops it cleanly.

#### Running it with no arguments (e.g. from an IDE's Run button)

Typing out the flags above every run is awkward from an IDE like PyCharm,
which by default runs a script with no arguments. Instead, put the values
in a JSON file and pass just `--config PATH` — an example is checked in at
[`examples/stream_config.example.json`](examples/stream_config.example.json)
(points at this repo's own checked-in demo model/data, so it runs as-is):

```bash
python -m src.stream_server --config examples/stream_config.example.json
```

In PyCharm: Run/Debug Configurations → your `stream_server` config →
Parameters field → `--config examples/stream_config.example.json`. Copy the
example file, edit the paths/values for your own model and baseline data,
and point `--config` at your copy. JSON keys match the flags' long names
with dashes replaced by underscores (`--baseline-dir` → `baseline_dir`,
`--window-size` → `window_size`, etc.); any flag also given on the command
line overrides the same key from the config file.

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
(`tests/test_main_window_integration.py`). Live streaming is tested the
same way — `tests/test_tcp_stream_server.py` and
`tests/test_live_monitor_panel.py` stream synthetic data (via
`src/stream_simulator.py`) over a real loopback socket and assert on the
scored results, no real network sender needed.

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
  stream_protocol.py    live-streaming wire format (NDJSON) encode/decode
  live_scorer.py           protocol-agnostic live scoring core (buffering, per-channel + overall scores)
  tcp_stream_server.py       TCP ingestion server, feeds LiveScorer
  stream_server.py              standalone headless CLI receiver, built on the above
  stream_simulator.py           streams synthetic data over TCP, for testing/demo
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
  live_monitor_panel.py                    live streaming scoring panel
docs/
  streaming_protocol.md   live-streaming wire format spec (LabVIEW integration doc)
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
- Add UDP and/or REST ingestion for live streaming alongside TCP — the
  scoring core (`LiveScorer.push()`) already takes a plain
  `dict[str, np.ndarray]` batch, transport-agnostic, so a new adapter is a
  thin wrapper around `TcpStreamServer`'s pattern, not a rework.
