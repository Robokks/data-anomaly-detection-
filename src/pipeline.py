"""Dispatch layer tying the loader, session grouping, time-range trimming, and
the two selectable anomaly detectors into one training/scoring flow.

This is the single place that knows how to go from "a folder path plus some
settings" to "a trained/scored model" -- both ``train.py``/``detect.py`` and,
later, the GUI call through here rather than duplicating this logic.
"""
from __future__ import annotations

import os

# Must be set before torch actually initializes its runtime (triggered below,
# directly or via src.dl_model) in a process that also imports scikit-learn
# (src.anomaly_model, imported right below too) -- both ship their own
# bundled OpenMP runtime, and loading both in one process without this can
# abort the whole process with a bare "Fatal Python error: Aborted" and no
# Python-level exception at all, typically the first time a deep-learning
# model actually trains/scores after a classic (sklearn) model has already
# run. This is the standard, safe mitigation for that whole class of
# duplicate-OpenMP-runtime issue: it tells the runtime to tolerate the
# duplicate load instead of aborting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from src.anomaly_model import AnomalyDetector
from src.data_loader import load_directory, scan_directory
from src.dl_model import AutoencoderDetector
from src.features import extract_features_from_frames
from src.session_grouping import MachineType, Phase, group_files, load_session
from src.time_range import select_range

ModelType = Literal["classic", "deep"]

# Analysts mostly care about steady-state/coasting data; ramp phases are
# transients. This is the CLI/GUI-layer default -- src.session_grouping keeps
# it out of load_session() itself so callers can opt into ramp phases too.
DEFAULT_TRANSMISSION_PHASES = {Phase.STEADY_STATE, Phase.COASTING}


def parse_range_bound(value: str | None):
    """Best-effort CLI-string -> range-bound coercion for --start/--end.

    Tries a float (sample index / seconds-since-start) first; falls back to
    the raw string (parsed as a datetime later by time_range.select_range if
    the target DataFrame has a datetime index).
    """
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _select_channels(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    keep = [c for c in channels if c in df.columns]
    if "phase" in df.columns and "phase" not in keep:
        keep = keep + ["phase"]
    out = df[keep].copy()
    out.attrs.update(df.attrs)
    return out


def _strip_phase_column(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" in df.columns:
        out = df.drop(columns=["phase"])
        out.attrs.update(df.attrs)
        return out
    return df


def load_sessions_from_directory(
    directory: str | Path,
    machine_type: MachineType = MachineType.GENERAL,
    recursive: bool = True,
    uut_pattern: str | None = None,
    group: str | None = None,
    channels: list[str] | None = None,
    phases: set[Phase] | None = None,
    pockets: set[str] | None = None,
    start=None,
    end=None,
) -> list[pd.DataFrame]:
    """Discover files under ``directory`` and return one working DataFrame per
    logical UUT session (or per file, for ``MachineType.GENERAL``), each
    optionally channel-filtered and time-range-trimmed.

    For ``TRANSMISSION`` sessions, ``phases`` defaults to
    ``{STEADY_STATE, COASTING}`` when not given explicitly.
    """
    if machine_type == MachineType.GENERAL:
        frames = load_directory(directory, recursive=recursive, group=group, channels=channels)
    else:
        files = scan_directory(directory, recursive=recursive)
        sessions = group_files(files, machine_type=machine_type, uut_pattern=uut_pattern)

        effective_phases = phases
        if effective_phases is None and machine_type == MachineType.TRANSMISSION:
            effective_phases = DEFAULT_TRANSMISSION_PHASES

        frames = [load_session(s, phases=effective_phases, pockets=pockets) for s in sessions]
        frames = [f for f in frames if not f.empty]
        if channels:
            frames = [_select_channels(f, channels) for f in frames]

    if start is not None or end is not None:
        frames = [select_range(f, start=start, end=end) for f in frames]

    return frames


def train_model(
    model_type: ModelType,
    frames: list[pd.DataFrame],
    window_size: int = 256,
    step: int | None = None,
    **model_kwargs,
) -> AnomalyDetector | AutoencoderDetector:
    """Fit either the classic or deep-learning detector on the given frames."""
    clean_frames = [_strip_phase_column(f) for f in frames]

    if model_type == "classic":
        features = extract_features_from_frames(clean_frames, window_size=window_size, step=step)
        model = AnomalyDetector(**model_kwargs)
        model.fit(features)
        # AutoencoderDetector already stores its own window_size/step (it needs
        # them for windowing raw arrays); AnomalyDetector doesn't otherwise
        # track them since it only ever sees the post-windowed feature table.
        # Stash them here so callers (e.g. the live-stream scorer) can recover
        # the window size a saved model expects without re-entering it.
        model.window_size_ = window_size
        model.step_ = step or window_size
        return model

    if model_type == "deep":
        model = AutoencoderDetector(window_size=window_size, step=step, **model_kwargs)
        model.fit(clean_frames)
        return model

    raise ValueError(f"Unknown model_type {model_type!r}; expected 'classic' or 'deep'")


def get_model_window_size(model: AnomalyDetector | AutoencoderDetector, model_type: ModelType) -> tuple[int | None, int | None]:
    """Return (window_size, step) the model was trained with, if known.

    Both are ``None`` for a classic model saved before ``window_size_``/
    ``step_`` existed -- callers should fall back to asking the operator.
    Uses ``getattr`` with a default as extra insurance rather than direct
    attribute access, though in practice a pickled instance missing these
    from its restored ``__dict__`` still resolves them via the class-level
    ``= None`` default (dataclass fields with a plain default are also class
    attributes) -- ``getattr`` just keeps this contract explicit rather than
    relying on that fallback.
    """
    if model_type == "deep":
        return model.window_size, model.step
    return getattr(model, "window_size_", None), getattr(model, "step_", None)


def score_model(
    model: AnomalyDetector | AutoencoderDetector,
    model_type: ModelType,
    frames: list[pd.DataFrame],
    window_size: int = 256,
    step: int | None = None,
) -> pd.DataFrame:
    """Score frames with either detector type; both return an anomaly_score/is_anomaly-shaped DataFrame."""
    clean_frames = [_strip_phase_column(f) for f in frames]

    if model_type == "classic":
        features = extract_features_from_frames(clean_frames, window_size=window_size, step=step)
        scores = model.score(features)
        scores["source_file"] = features["source_file"].values
        return scores

    if model_type == "deep":
        return model.score(clean_frames)

    raise ValueError(f"Unknown model_type {model_type!r}; expected 'classic' or 'deep'")


def save_model(model: AnomalyDetector | AutoencoderDetector, model_type: ModelType, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_type": model_type, "model": model}, path)


def load_model(path: str | Path) -> tuple[AnomalyDetector | AutoencoderDetector, ModelType]:
    payload = joblib.load(Path(path))
    return payload["model"], payload["model_type"]
