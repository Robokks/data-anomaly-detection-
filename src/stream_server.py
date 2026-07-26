"""CLI: run the live anomaly scorer against a TCP sensor stream, headless.

The standalone (no Qt/GUI) equivalent of this app's live-monitoring feature --
built for running unattended next to a test rig: fits a ``LiveScorer``
baseline from a folder of reference "normal" data, listens on a TCP port for
a live stream (see ``src/stream_protocol.py``/``src/stream_simulator.py`` for
the wire format), and prints a compact per-window summary to the console as
data comes in, optionally also logging every scored window to a CSV file.

Example:
    python -m src.stream_server --model models/anomaly_detector.joblib \
        --baseline-dir data/normal --sample-rate 1000 --window-size 256 \
        --port 9999 --output-csv results/live_log.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live_scorer import STAT_NAMES, LiveScorer, LiveWindowResult
from src.pipeline import get_model_window_size, load_model, load_sessions_from_directory, parse_range_bound
from src.session_grouping import MachineType, Phase
from src.tcp_stream_server import TcpStreamServer


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the live anomaly scorer against a TCP sensor stream (headless).")
    p.add_argument("--model", required=True, help="Path to a trained model (.joblib)")
    p.add_argument("--baseline-dir", required=True, help="Directory of 'normal' reference data to fit the live baseline from")
    p.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Scan subdirectories under --baseline-dir (default)")
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
    p.add_argument("--start", default=None, help="Trim baseline data to start at this index/time (default: full data)")
    p.add_argument("--end", default=None, help="Trim baseline data to end at this index/time (default: full data)")
    p.add_argument("--sample-rate", type=float, required=True, help="Sample rate of the live stream, in Hz")
    p.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Model window size in samples (default: read from the saved model; required if the model doesn't have one stored)",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="Model window stride in samples (default: the model's stored step if known, else --window-size)",
    )
    p.add_argument("--window-duration-seconds", type=float, default=1.0, help="Per-channel stats-tick duration, in seconds (default: 1.0)")
    p.add_argument("--channel-contamination", type=float, default=0.05, help="Expected fraction of anomalous per-channel stats-ticks (default: 0.05)")
    p.add_argument("--host", default="0.0.0.0", help="Host/interface to listen on (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=9999, help="TCP port to listen on (default: 9999)")
    p.add_argument("--output-csv", default=None, help="Append one flattened row per scored window to this CSV")
    return p


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def _csv_fieldnames(channels: list[str]) -> list[str]:
    fieldnames = ["window_index", "timestamp", "n_samples_seen", "overall_anomaly_score", "overall_is_anomaly"]
    for channel in channels:
        fieldnames.append(f"{channel}_deviation_score")
        fieldnames.append(f"{channel}_is_anomaly")
        for stat in STAT_NAMES:
            fieldnames.append(f"{channel}_{stat}")
    return fieldnames


def _flatten_result(result: LiveWindowResult, channels: list[str]) -> dict:
    row = {
        "window_index": result.window_index,
        "timestamp": result.timestamp,
        "n_samples_seen": result.n_samples_seen,
        "overall_anomaly_score": result.overall_anomaly_score,
        "overall_is_anomaly": result.overall_is_anomaly,
    }
    for channel in channels:
        channel_result = result.channels.get(channel)
        row[f"{channel}_deviation_score"] = channel_result.deviation_score if channel_result else None
        row[f"{channel}_is_anomaly"] = channel_result.is_anomaly if channel_result else None
        for stat in STAT_NAMES:
            row[f"{channel}_{stat}"] = channel_result.stats.get(stat) if channel_result else None
    return row


def _append_csv_row(path: Path, channels: list[str], result: LiveWindowResult) -> None:
    """Appends one flattened row to ``path``, writing the header first if the
    file is new/empty. Reopens the file per call (rather than holding a
    handle open for the process lifetime) -- simple and safe for a
    long-running unattended process with no explicit shutdown hook needed for
    the log file, at the cost of a bit of per-window I/O overhead that's
    negligible at the ~1 tick/second scale this is used at.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(channels))
        if write_header:
            writer.writeheader()
        writer.writerow(_flatten_result(result, channels))


def _format_console_line(result: LiveWindowResult) -> str:
    ts = datetime.fromtimestamp(result.timestamp).isoformat(timespec="seconds")

    if result.overall_anomaly_score is None:
        overall_summary = "overall=warming up"
    else:
        flag = "ANOMALY" if result.overall_is_anomaly else "normal"
        overall_summary = f"overall={result.overall_anomaly_score:.3f} ({flag})"

    flagged_channels = [channel for channel, cr in result.channels.items() if cr.is_anomaly]
    channel_summary = "flagged: " + ", ".join(flagged_channels) if flagged_channels else "all channels normal"

    return (
        f"[window {result.window_index}] {ts} n_samples_seen={result.n_samples_seen} "
        f"{overall_summary} | {channel_summary}"
    )


def build_server(
    args: argparse.Namespace,
    extra_on_window: Callable[[LiveWindowResult], None] | None = None,
    extra_on_error: Callable[[str], None] | None = None,
) -> tuple[TcpStreamServer, LiveScorer]:
    """Loads the model, fits the live baseline, and wires up a (not-yet-started)
    ``TcpStreamServer``.

    ``extra_on_window``/``extra_on_error`` are optional additional callbacks
    invoked alongside the console-print/CSV-log handlers -- primarily so
    tests can observe results without parsing stdout or needing to run
    ``main()``'s own blocking loop.
    """
    machine_type = MachineType(args.machine_type)
    phases = {Phase(ph) for ph in args.phases} if args.phases else None
    pockets = set(args.pockets) if args.pockets else None

    model, model_type = load_model(args.model)
    stored_window_size, stored_step = get_model_window_size(model, model_type)

    if args.window_size is not None:
        window_size = args.window_size
        if stored_window_size is not None and stored_window_size != window_size:
            print(
                f"Warning: --window-size={window_size} does not match this model's stored "
                f"window size ({stored_window_size}) -- window-size mismatch; proceeding with "
                "the explicitly-provided value."
            )
    else:
        window_size = stored_window_size
        if window_size is None:
            _build_arg_parser().error(
                "This model has no stored window size (a legacy classic model trained before "
                "window_size_ tracking was added); pass --window-size explicitly."
            )

    if args.step is not None:
        step = args.step
    elif stored_step is not None:
        step = stored_step
    else:
        step = window_size

    print(f"Scanning {args.baseline_dir} (machine_type={machine_type.value}, recursive={args.recursive}) ...")
    try:
        baseline_frames = load_sessions_from_directory(
            args.baseline_dir,
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
    if not baseline_frames:
        raise SystemExit(f"No usable baseline sessions/files found under {args.baseline_dir}")

    channel_names = [c for c in baseline_frames[0].columns if c != "phase"]
    print(f"Loaded {len(baseline_frames)} baseline session(s)/file(s); channels: {channel_names}")

    live_scorer = LiveScorer(
        model=model,
        model_type=model_type,
        channels=channel_names,
        sample_rate_hz=args.sample_rate,
        model_window_size=window_size,
        model_step=step,
        window_duration_seconds=args.window_duration_seconds,
        channel_contamination=args.channel_contamination,
    )
    live_scorer.fit_baseline(baseline_frames)

    csv_path = Path(args.output_csv) if args.output_csv else None

    def on_window(result: LiveWindowResult) -> None:
        print(_format_console_line(result))
        if csv_path is not None:
            _append_csv_row(csv_path, channel_names, result)
        if extra_on_window is not None:
            extra_on_window(result)

    def on_error(message: str) -> None:
        print(f"[stream error] {message}", file=sys.stderr)
        if extra_on_error is not None:
            extra_on_error(message)

    server = TcpStreamServer(args.host, args.port, live_scorer, on_window=on_window, on_error=on_error)
    return server, live_scorer


def main() -> None:
    args = parse_args()
    server, live_scorer = build_server(args)

    port = server.start()
    print(
        f"Listening on {args.host}:{port} ... (window_size={live_scorer.model_window_size}, "
        f"step={live_scorer.model_step}, channels={live_scorer.channels})"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        print("Shutting down.")


if __name__ == "__main__":
    main()
