"""CLI: train an anomaly-detection model on TDMS/CSV/Excel data.

Supports single-file-per-UUT layouts ("general", the default) as well as
multi-file UUT layouts used on other rig types -- see --machine-type. Both
the classic (Isolation Forest + PCA) and deep-learning (1D conv autoencoder)
model types are selectable via --model-type.

Examples:
    python -m src.train --data-dir data/normal --window-size 256 \
        --contamination 0.02 --model-out models/anomaly_detector.joblib

    python -m src.train --data-dir data/normal --machine-type transmission \
        --model-type deep --epochs 20 --model-out models/deep.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import load_sessions_from_directory, parse_range_bound, save_model, score_model, train_model
from src.session_grouping import MachineType, Phase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an anomaly-detection model on TDMS/CSV/Excel data.")
    p.add_argument("--data-dir", required=True, help="Directory of data files representing normal operation")
    p.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Scan subdirectories (default)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only scan the top-level directory")
    p.add_argument(
        "--machine-type",
        choices=[m.value for m in MachineType],
        default=MachineType.GENERAL.value,
        help="How multi-file UUT recordings are grouped (default: general = one file per UUT)",
    )
    p.add_argument(
        "--uut-pattern",
        default=None,
        help="Regex (one capture group) extracting a UUT id from filenames; non-general machine types default to grouping by parent folder",
    )
    p.add_argument(
        "--phases",
        nargs="*",
        choices=[ph.value for ph in Phase],
        default=None,
        help="Transmission only: phases to include (default: steady_state coasting)",
    )
    p.add_argument("--pockets", nargs="*", default=None, help="Motor-test-bench only: pocket/step ids to include (default: all)")
    p.add_argument("--group", default=None, help="TDMS group name to read (general machine-type only; default: first group in each file)")
    p.add_argument("--channels", nargs="*", default=None, help="Subset of channel names to use (default: all)")
    p.add_argument("--start", default=None, help="Trim each session/file to start at this index/time (default: full data)")
    p.add_argument("--end", default=None, help="Trim each session/file to end at this index/time (default: full data)")
    p.add_argument("--window-size", type=int, default=256, help="Samples per feature/raw window")
    p.add_argument("--step", type=int, default=None, help="Window stride (default: window-size, no overlap)")
    p.add_argument("--model-type", choices=["classic", "deep"], default="classic", help="Anomaly detector to train")
    p.add_argument("--contamination", type=float, default=0.02, help="Expected fraction of anomalous windows")
    p.add_argument("--n-estimators", type=int, default=200, help="[classic] Isolation Forest tree count")
    p.add_argument("--epochs", type=int, default=30, help="[deep] training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="[deep] training batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="[deep] learning rate")
    p.add_argument("--latent-dim", type=int, default=16, help="[deep] autoencoder bottleneck size")
    p.add_argument("--device", default="cpu", help="[deep] torch device")
    p.add_argument("--model-out", default="models/anomaly_detector.joblib", help="Output path for the trained model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    machine_type = MachineType(args.machine_type)
    phases = {Phase(ph) for ph in args.phases} if args.phases else None
    pockets = set(args.pockets) if args.pockets else None

    print(f"Scanning {args.data_dir} (machine_type={machine_type.value}, recursive={args.recursive}) ...")
    try:
        frames = load_sessions_from_directory(
            args.data_dir,
            machine_type=machine_type,
            recursive=args.recursive,
            uut_pattern=args.uut_pattern,
            group=args.group,
            channels=args.channels,
            phases=phases,
            pockets=pockets,
            start=parse_range_bound(args.start),
            end=parse_range_bound(args.end),
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    if not frames:
        raise SystemExit(f"No usable sessions/files found under {args.data_dir}")
    channel_names = [c for c in frames[0].columns if c != "phase"]
    print(f"Loaded {len(frames)} session(s)/file(s); channels: {channel_names}")

    model_kwargs: dict = {"contamination": args.contamination}
    if args.model_type == "classic":
        model_kwargs["n_estimators"] = args.n_estimators
    else:
        model_kwargs.update(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            latent_dim=args.latent_dim,
            device=args.device,
        )

    print(f"Training {args.model_type} model (window_size={args.window_size}, step={args.step or args.window_size}) ...")
    model = train_model(args.model_type, frames, window_size=args.window_size, step=args.step, **model_kwargs)

    save_model(model, args.model_type, args.model_out)
    print(f"Saved trained {args.model_type} model to {args.model_out}")

    scores = score_model(model, args.model_type, frames, window_size=args.window_size, step=args.step)
    print(f"Training-set anomaly rate: {scores['is_anomaly'].mean():.2%} (threshold={model.threshold_:.3f})")


if __name__ == "__main__":
    main()
