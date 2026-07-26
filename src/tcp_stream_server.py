"""TCP ingestion server for live/streaming anomaly scoring.

Wraps stdlib ``socketserver.ThreadingTCPServer``: validates the wire
protocol's ``hello`` handshake, decodes ``batch`` messages, and feeds them
into a ``LiveScorer`` under a lock, forwarding each produced
``LiveWindowResult`` to a caller-supplied callback. No Qt/GUI dependency --
both the GUI live-monitor panel and the standalone headless receiver build
on this.

A future UDP/REST adapter only needs to decode its transport into the same
``dict[str, np.ndarray]`` batch shape and call the same
``live_scorer.push(...)`` + ``on_window`` -- ``LiveScorer`` itself has zero
networking imports.
"""
from __future__ import annotations

import socketserver
import threading
from typing import Callable

import numpy as np

from src.live_scorer import LiveScorer, LiveWindowResult
from src.stream_protocol import BatchMessage, HelloMessage, decode_message, encode_error

# Guards against a sender that never emits '\n' letting readline() buffer
# unboundedly. Not a strict "reject anything longer" check (readline(size)
# may return a partial, newline-less line right at the boundary rather than
# raising) -- a truncated line simply fails JSON parsing in decode_message
# and gets reported as a malformed message, which is an acceptable outcome
# for this edge case.
MAX_LINE_BYTES = 16 * 1024 * 1024
_SAMPLE_RATE_RELATIVE_TOLERANCE = 0.01  # 1%, matches LiveScorer.fit_baseline's own tolerance


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    owner: "TcpStreamServer"


class _StreamHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        owner: TcpStreamServer = self.server.owner

        self.connection.settimeout(30.0)
        try:
            hello_line = self._readline()
        except OSError as exc:
            owner.on_error(f"connection closed before hello: {exc}")
            return
        if hello_line is None:
            return  # closed before sending anything

        try:
            hello = decode_message(hello_line)
        except ValueError as exc:
            self._send_error(f"invalid hello message: {exc}")
            return
        if not isinstance(hello, HelloMessage):
            self._send_error(f"expected 'hello' as the first message, got {type(hello).__name__}")
            return

        expected_channels = set(owner.live_scorer.channels)
        if set(hello.channels) != expected_channels:
            self._send_error(
                f"channel set mismatch: expected {sorted(expected_channels)}, got {sorted(hello.channels)}"
            )
            return

        expected_rate = owner.live_scorer.sample_rate_hz
        rel_diff = abs(hello.sample_rate_hz - expected_rate) / max(abs(expected_rate), 1e-12)
        if rel_diff > _SAMPLE_RATE_RELATIVE_TOLERANCE:
            self._send_error(f"sample_rate_hz mismatch: expected ~{expected_rate}, got {hello.sample_rate_hz}")
            return

        self.connection.settimeout(None)
        while True:
            try:
                line = self._readline()
            except OSError as exc:
                owner.on_error(f"connection error: {exc}")
                return
            if line is None:
                return

            try:
                msg = decode_message(line)
            except ValueError as exc:
                owner.on_error(f"malformed batch message: {exc}")
                continue
            if not isinstance(msg, BatchMessage):
                owner.on_error(f"expected 'batch' message, got {type(msg).__name__}")
                continue

            batch = {name: np.asarray(values, dtype=float) for name, values in msg.channels.items()}
            with owner._lock:
                try:
                    results = owner.live_scorer.push(batch)
                except (ValueError, RuntimeError) as exc:
                    owner.on_error(f"scoring error: {exc}")
                    continue

            for result in results:
                owner.on_window(result)

    def _readline(self) -> str | None:
        raw = self.rfile.readline(MAX_LINE_BYTES)
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace")

    def _send_error(self, message: str) -> None:
        try:
            self.wfile.write(encode_error(message).encode("utf-8"))
        except OSError:
            pass
        self.server.owner.on_error(message)


class TcpStreamServer:
    """Start/stop a background TCP server feeding a ``LiveScorer``.

    ``on_window``/``on_error`` are called from a per-connection background
    thread (``daemon_threads=True``) -- callers that update UI state (e.g.
    Qt widgets) must marshal back to their own thread themselves (a plain
    Qt signal ``emit()`` from a non-owning thread is auto-queued, which is
    exactly what the GUI live-monitor panel relies on).
    """

    def __init__(
        self,
        host: str,
        port: int,
        live_scorer: LiveScorer,
        on_window: Callable[[LiveWindowResult], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.live_scorer = live_scorer
        self.on_window = on_window
        self.on_error = on_error or (lambda message: None)
        self._lock = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        """Binds and starts serving on a background thread; returns the bound port."""
        if self._server is not None:
            raise RuntimeError("TcpStreamServer.start() called while already running")
        self._server = _Server((self.host, self.port), _StreamHandler)
        self._server.owner = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_address[1]

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
