# Example TDMS files

Small, checked-in sample data so you can try the app immediately without
generating anything. Not a training set on its own (only 3 normal files) --
for real training data, generate more with `src/synthetic_tdms.py` (see the
root README's "Running the CLI" section); this folder is just here to let
you open the GUI/CLI and see real files with the expected shape right away.

All 10-channel (vibration, noise, speed, torque, current, voltage_a/b/c,
temperature, pressure), 10 kHz, following the ramp/hold/de-ramp test-cycle
profile in `src/synthetic_tdms.py::_make_test_cycle_signals`
(0 -> 1000 -> 6000 -> 1000 -> 0 rpm, 0 -> 50 -> 250 -> 0 Nm).

| File | Contents |
| --- | --- |
| `normal_00.tdms`, `normal_01.tdms`, `normal_02.tdms` | Normal cycles (no injected anomalies) |
| `test_with_anomalies.tdms` | Same profile, with 3 injected anomalies at sample ranges `(9615, 16507)`, `(44451, 52101)`, `(67697, 82387)` -- ground truth for testing detection |

Regenerate (or make more) with:

```bash
python -m src.synthetic_tdms --out-dir data --n-normal 100 --channel-set cycle \
  --sample-rate 10000 --duration-min 5 --duration-max 10 --seed 0
```
