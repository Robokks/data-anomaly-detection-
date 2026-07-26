"""CLI: train an anomaly-detection model on a folder of "normal" TDMS files.

Example:
    python -m src.train --data-dir data/normal --window-size 256 \
        --contamination 0.02 --model-out models/anomaly_detector.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_model import AnomalyDetector
from src.features import extract_features_from_frames
from src.tdms_loader import load_tdms_directory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a TDMS anomaly-detection model.")
    p.add_argument("--data-dir", required=True, help="Directory of .tdms files representing normal operation")
    p.add_argument("--group", default=None, help="TDMS group name to read (default: first group in each file)")
    p.add_argument("--channels", nargs="*", default=None, help="Subset of channel names to use (default: all)")
    p.add_argument("--window-size", type=int, default=256, help="Samples per feature window")
    p.add_argument("--step", type=int, default=None, help="Window stride (default: window-size, no overlap)")
    p.add_argument("--contamination", type=float, default=0.02, help="Expected fraction of anomalous windows")
    p.add_argument("--n-estimators", type=int, default=200, help="Isolation Forest tree count")
    p.add_argument("--model-out", default="models/anomaly_detector.joblib", help="Output path for the trained model")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading TDMS files from {args.data_dir} ...")
    frames = load_tdms_directory(args.data_dir, group_name=args.group, channels=args.channels)
    print(f"Loaded {len(frames)} file(s); channels: {list(frames[0].columns)}")

    print(f"Extracting windowed features (window_size={args.window_size}, step={args.step or args.window_size}) ...")
    features = extract_features_from_frames(frames, window_size=args.window_size, step=args.step)
    print(f"Extracted {len(features)} feature windows x {features.shape[1] - 1} features")

    detector = AnomalyDetector(contamination=args.contamination, n_estimators=args.n_estimators)
    detector.fit(features)

    detector.save(args.model_out)
    print(f"Saved trained model to {args.model_out}")

    scores = detector.score(features)
    print(f"Training-set anomaly rate: {scores['is_anomaly'].mean():.2%} (threshold={detector.threshold_:.3f})")


if __name__ == "__main__":
    main()
