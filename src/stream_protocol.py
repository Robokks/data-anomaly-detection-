"""Wire protocol for live/streaming sensor scoring.

TCP, newline-delimited JSON (NDJSON) -- one JSON object per line, ``\\n``
terminated. Three message types:

- ``hello``: sent first on a connection, declares the full channel set
  (order doesn't matter, matched as a set downstream) and the sender's
  sample rate.
- ``batch``: a chunk of samples for every channel declared in ``hello``.
  All per-channel arrays within one batch message must have equal length.
  ``seq`` and ``t0`` are diagnostic metadata (sequence number, Unix epoch
  timestamp of the batch's first sample) -- not validated for
  monotonicity/contiguity here, that's the receiver's business.
- ``error``: ``{"type": "error", "message": ...}``, used by either side to
  report a protocol-level problem.

This module only encodes/decodes messages -- it has no socket code of its
own. See ``src/stream_simulator.py`` for a TCP sender built on top of it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

PROTOCOL_VERSION = 1


@dataclass
class HelloMessage:
    channels: list[str]
    sample_rate_hz: float
    protocol_version: int = PROTOCOL_VERSION


@dataclass
class BatchMessage:
    seq: int
    t0: float
    channels: dict[str, list[float]]


def encode_hello(channels: list[str], sample_rate_hz: float) -> str:
    """Returns a JSON string including the trailing '\\n', ready for
    ``socket.sendall(s.encode())``."""
    msg = {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "channels": list(channels),
        "sample_rate_hz": sample_rate_hz,
    }
    return json.dumps(msg) + "\n"


def encode_batch(seq: int, t0: float, channels: dict[str, "object"]) -> str:
    """``channels`` values may be numpy arrays or python lists/sequences of
    numbers -- both are accepted and normalized to plain python lists so
    JSON serialization works either way."""
    encoded_channels = {}
    for name, values in channels.items():
        if hasattr(values, "tolist"):
            encoded_channels[name] = values.tolist()
        else:
            encoded_channels[name] = list(values)
    msg = {
        "type": "batch",
        "seq": seq,
        "t0": t0,
        "channels": encoded_channels,
    }
    return json.dumps(msg) + "\n"


def encode_error(message: str) -> str:
    """Returns a JSON string including the trailing '\\n'."""
    msg = {"type": "error", "message": message}
    return json.dumps(msg) + "\n"


def decode_message(line: str) -> HelloMessage | BatchMessage | dict:
    """Parses one line (with or without trailing newline/whitespace) of
    NDJSON. Raises ValueError with a clear, specific message on any
    malformed input -- see module docstring for the wire format."""
    line = line.strip()
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")

    msg_type = obj.get("type")
    if msg_type is None:
        raise ValueError("message missing required 'type' field")

    if msg_type == "hello":
        return _decode_hello(obj)
    elif msg_type == "batch":
        return _decode_batch(obj)
    elif msg_type == "error":
        return _decode_error(obj)
    else:
        raise ValueError(f"unrecognized message type: {msg_type!r}")


def _decode_hello(obj: dict) -> HelloMessage:
    if "channels" not in obj:
        raise ValueError("hello message missing required 'channels' field")
    if "sample_rate_hz" not in obj:
        raise ValueError("hello message missing required 'sample_rate_hz' field")

    channels = obj["channels"]
    if not channels:
        raise ValueError("hello message 'channels' must not be empty")

    sample_rate_hz = obj["sample_rate_hz"]
    if not isinstance(sample_rate_hz, (int, float)) or isinstance(sample_rate_hz, bool) or sample_rate_hz <= 0:
        raise ValueError(f"hello message 'sample_rate_hz' must be > 0, got {sample_rate_hz!r}")

    return HelloMessage(
        channels=list(channels),
        sample_rate_hz=sample_rate_hz,
        protocol_version=obj.get("protocol_version", PROTOCOL_VERSION),
    )


def _decode_batch(obj: dict) -> BatchMessage:
    for field in ("seq", "t0", "channels"):
        if field not in obj:
            raise ValueError(f"batch message missing required '{field}' field")

    channels = obj["channels"]
    if not channels:
        raise ValueError("batch message 'channels' must not be empty")

    lengths = {}
    for name, values in channels.items():
        if not isinstance(values, list):
            raise ValueError(f"batch message channel {name!r} must be an array, got {type(values).__name__}")
        for v in values:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"batch message channel {name!r} contains a non-numeric value: {v!r}")
        lengths[name] = len(values)

    distinct_lengths = set(lengths.values())
    if len(distinct_lengths) > 1:
        raise ValueError(f"batch message channel arrays have mismatched lengths: {lengths}")

    return BatchMessage(seq=obj["seq"], t0=obj["t0"], channels=channels)


def _decode_error(obj: dict) -> dict:
    if "message" not in obj:
        raise ValueError("error message missing required 'message' field")
    return obj
