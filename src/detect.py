"""CLI: score TDMS/CSV/Excel data against a trained anomaly model.

--input may be a single file or a directory (scanned/grouped the same way
--data-dir is in train.py). Works with models trained by either --model-type.

Example:
    python -m src.detect --model models/anomaly_detector.joblib \
        --input data/test_run.tdms --output results/scores.csv --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataframe
from src.pipeline import load_sessions_from_directory, load_model, parse_range_bound, score_model
from src.session_grouping import MachineType, Phase
from src.time_range import select_range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score TDMS/CSV/Excel data against a trained anomaly model.")
    p.add_argument("--model", required=True, help="Path to a trained model (.joblib)")
    p.add_argument("--input", required=True, help="A data file, or a directory of data files/sessions")
    p.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Scan subdirectories when --input is a directory (default)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only scan the top-level directory")
    p.add_argument(
        "--machine-type",
        choices=[m.value for m in MachineType],
        default=MachineType.GENERAL.value,
        help="How multi-file UUT recordings are grouped when --input is a directory (default: general)",
    )
    p.add_argument("--uut-pattern", default=None, help="Regex (one capture group) extracting a UUT id from filenames")
    p.add_argument(
        "--phases",
        nargs="*",
        choices=[ph.value for ph in Phase],
        default=None,
        help="Transmission only: phases to include (default: steady_state coasting)",
    )
    p.add_argument("--pockets", nargs="*", default=None, help="Motor-test-bench only: pocket/step ids to include (default: all)")
    p.add_argument("--group", default=None, help="TDMS group name to read (default: first group in each file)")
    p.add_argument("--channels", nargs="*", default=None, help="Subset of channel names to use (default: all; must match training)")
    p.add_argument("--start", default=None, help="Trim to start at this index/time (default: full data)")
    p.add_argument("--end", default=None, help="Trim to end at this index/time (default: full data)")
    p.add_argument("--window-size", type=int, default=256, help="Samples per feature/raw window (match training)")
    p.add_argument("--step", type=int, default=None, help="Window stride (default: window-size)")
    p.add_argument("--output", default=None, help="Output CSV path for per-window scores")
    p.add_argument("--plot", action="store_true", help="Save a PNG plot of the anomaly score over time")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    machine_type = MachineType(args.machine_type)
    phases = {Phase(ph) for ph in args.phases} if args.phases else None
    pockets = set(args.pockets) if args.pockets else None
    start, end = parse_range_bound(args.start), parse_range_bound(args.end)

    model, model_type = load_model(args.model)

    input_path = Path(args.input)
    try:
        if input_path.is_dir():
            frames = load_sessions_from_directory(
                input_path,
                machine_type=machine_type,
                recursive=args.recursive,
                uut_pattern=args.uut_pattern,
                group=args.group,
                channels=args.channels,
                phases=phases,
                pockets=pockets,
                start=start,
                end=end,
            )
        else:
            df = load_dataframe(input_path, group=args.group, channels=args.channels)
            if start is not None or end is not None:
                df = select_range(df, start=start, end=end)
            frames = [df]
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None

    if not frames:
        raise SystemExit(f"No usable data found for {args.input}")

    scores = score_model(model, model_type, frames, window_size=args.window_size, step=args.step)

    n_anom = int(scores["is_anomaly"].sum())
    print(f"Scored {len(scores)} windows ({model_type} model): {n_anom} flagged as anomalous ({n_anom / len(scores):.2%})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_path, index_label=scores.index.name or "window")
        print(f"Wrote scores to {out_path}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(scores["anomaly_score"].values, label="anomaly_score", color="steelblue")
        ax.axhline(model.threshold_, color="crimson", linestyle="--", label="threshold")
        anomalies = scores.index[scores["is_anomaly"]]
        ax.scatter(
            [scores.index.get_loc(i) for i in anomalies],
            scores.loc[anomalies, "anomaly_score"],
            color="crimson",
            zorder=5,
            label="flagged",
        )
        ax.set_xlabel("window #")
        ax.set_ylabel("anomaly score")
        ax.legend()
        fig.tight_layout()

        plot_path = Path(args.output).with_suffix(".png") if args.output else Path("anomaly_scores.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
