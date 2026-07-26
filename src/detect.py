"""CLI: score a TDMS file (or directory) against a trained anomaly model.

Example:
    python -m src.detect --model models/anomaly_detector.joblib \
        --input data/test_run.tdms --output results/scores.csv --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_model import AnomalyDetector
from src.features import extract_features_from_frames
from src.tdms_loader import load_tdms_dataframe, load_tdms_directory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score TDMS data against a trained anomaly model.")
    p.add_argument("--model", required=True, help="Path to a trained model (.joblib)")
    p.add_argument("--input", required=True, help="A .tdms file or a directory of .tdms files")
    p.add_argument("--group", default=None, help="TDMS group name to read (default: first group in each file)")
    p.add_argument("--channels", nargs="*", default=None, help="Subset of channel names to use (default: all)")
    p.add_argument("--window-size", type=int, default=256, help="Samples per feature window (match training)")
    p.add_argument("--step", type=int, default=None, help="Window stride (default: window-size)")
    p.add_argument("--output", default=None, help="Output CSV path for per-window scores")
    p.add_argument("--plot", action="store_true", help="Save a PNG plot of the anomaly score over time")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    detector = AnomalyDetector.load(args.model)

    input_path = Path(args.input)
    if input_path.is_dir():
        frames = load_tdms_directory(input_path, group_name=args.group, channels=args.channels)
    else:
        frames = [load_tdms_dataframe(input_path, group_name=args.group, channels=args.channels)]

    features = extract_features_from_frames(frames, window_size=args.window_size, step=args.step)
    scores = detector.score(features)
    scores["source_file"] = features["source_file"].values

    n_anom = int(scores["is_anomaly"].sum())
    print(f"Scored {len(scores)} windows: {n_anom} flagged as anomalous ({n_anom / len(scores):.2%})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_path, index_label=features.index.name or "window")
        print(f"Wrote scores to {out_path}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(scores["anomaly_score"].values, label="anomaly_score", color="steelblue")
        ax.axhline(detector.threshold_, color="crimson", linestyle="--", label="threshold")
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
