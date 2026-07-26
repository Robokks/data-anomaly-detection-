# Live Streaming Protocol

Integration spec for sending live sensor data to this app's live-monitor
(GUI panel or standalone `src/stream_server.py`) over a network connection —
e.g. from a LabVIEW test rig. This is the contract a sender must follow;
the receiving side is already built and tested (`src/stream_protocol.py`,
`src/tcp_stream_server.py`, `src/live_scorer.py`).

## Transport (v1)

**TCP only.** One persistent TCP connection per sender. UDP and REST are
deliberately out of scope for now — the receiving core (`LiveScorer`) has
no networking code in it at all, so adding another transport later is a
thin adapter, not a rework, but only TCP is implemented and tested today.

## Wire format

Newline-delimited JSON (NDJSON): one JSON object per line, `\n`-terminated.
No length prefix — `\n` is the frame boundary. Chosen over a binary format
because it needs no new dependency (stdlib `socket` + `json` on the
receiving end), is debuggable by hand (`nc`, `telnet`, or just eyeballing a
captured line), and JSON overhead is a non-issue at these data rates
(tens of KB/sec at 10,000 samples/sec).

There are three message types.

### `hello` — must be the first message on a connection

```json
{"type": "hello", "protocol_version": 1, "channels": ["vibration", "temperature", "pressure"], "sample_rate_hz": 10000}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"hello"` | required |
| `protocol_version` | int | currently `1` |
| `channels` | array of strings | the full channel set this connection will send. **Order doesn't matter** — the receiver matches it as a set against the model/baseline's channel set, not positionally. Must be non-empty. |
| `sample_rate_hz` | number > 0 | the sender's sample rate, in Hz. Used only for a **validation check** against the receiver's configured expected rate (within ~1% tolerance) — sending this does not dynamically reconfigure anything server-side. A mismatch (either the channel set or the sample rate) gets you an `error` message back and the connection is closed. |

### `batch` — one chunk of samples, for every declared channel

```json
{"type": "batch", "seq": 1, "t0": 1737901234.512, "channels": {"vibration": [0.101, 0.114, 0.098], "temperature": [40.02, 40.03, 40.01], "pressure": [100.1, 100.2, 100.1]}}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"batch"` | required |
| `seq` | int | a sequence number for this batch. Monotonically increasing is expected but not strictly enforced by the receiver — it's diagnostic metadata for logging/debugging, not used for correctness. |
| `t0` | number | Unix epoch timestamp (seconds, float) of this batch's first sample. Also diagnostic only. |
| `channels` | object | one array per channel. **Every channel declared in `hello` must be present**, and **all arrays in one `batch` message must have the same length** — this matches how synchronized multi-channel DAQ scanning naturally produces data, and lets the receiver skip per-channel skew bookkeeping entirely. A batch that violates either rule is rejected (logged as an error on the receiving side; the connection is not necessarily closed for this — only a bad `hello` closes the connection). |

**Batch size and cadence are entirely the sender's choice.** The receiver
never assumes anything about batch boundaries — it concatenates arriving
samples into its own rolling per-channel buffer and slices that into fixed-
size scoring windows purely by cumulative sample count. Concretely, all of
these are equally valid and interchangeable from the sender's side:

- 10 batches/second of 1,000 samples each (`"1000 samples per 100ms"`)
- 1 batch/second of 10,000 samples (`"10000 samples/sec"`)
- One large batch containing many seconds' worth of samples (e.g. after
  buffering/compressing on the sender's side before transmitting)

There's no per-batch acknowledgment — this is fire-and-forget, which also
keeps the door open for a future UDP transport using the same message
shapes. Session end is a plain TCP connection close (FIN); no explicit
"goodbye" message is needed or sent.

### `error` — either side may send this to report a protocol problem

```json
{"type": "error", "message": "channel set mismatch: expected ['pressure', 'temperature', 'vibration'], got ['pressure', 'temperature']"}
```

Today the receiver sends this (and then closes the connection) only in
response to a bad `hello` — a channel-set or sample-rate mismatch. A
malformed `batch` message is logged on the receiving side but does not
close the connection (the sender can just keep going with the next batch).

## Minimal integration checklist for a sender (e.g. LabVIEW)

1. Open a TCP connection to the host:port the receiver is listening on
   (shown in the GUI panel's status line, or printed at startup by
   `stream_server.py`).
2. Send exactly one `hello` line declaring your channel names and sample
   rate, matching what the receiver was configured to expect (same channel
   *names*, not necessarily the same order; sample rate within ~1%).
3. Send `batch` lines continuously as data becomes available, at whatever
   size/cadence is natural for your acquisition loop. Every batch must
   include every channel from `hello`, all arrays the same length.
4. Close the connection when done. No goodbye message needed.

## Reference implementation

`src/stream_protocol.py` (`encode_hello`/`encode_batch`/`decode_message`)
and `src/stream_simulator.py` (`stream_signals()` — a working Python sender,
useful both as a live-monitor test tool and as a second reference
implementation of this spec beyond the prose above) are the canonical,
tested implementations of everything described in this document.
