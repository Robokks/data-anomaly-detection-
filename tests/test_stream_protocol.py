import numpy as np
import pytest

from src.stream_protocol import (
    BatchMessage,
    HelloMessage,
    decode_message,
    encode_batch,
    encode_error,
    encode_hello,
)


def test_encode_hello_roundtrip():
    encoded = encode_hello(["vibration", "temperature", "pressure"], 10000)
    decoded = decode_message(encoded)

    assert isinstance(decoded, HelloMessage)
    assert set(decoded.channels) == {"vibration", "temperature", "pressure"}
    assert decoded.channels == ["vibration", "temperature", "pressure"]  # order preserved
    assert decoded.sample_rate_hz == 10000


def test_encode_batch_roundtrip_with_python_lists():
    channels = {"vibration": [0.1, 0.2, 0.3], "temperature": [40.0, 40.1, 40.2]}
    encoded = encode_batch(1, 1737901234.512, channels)
    decoded = decode_message(encoded)

    assert isinstance(decoded, BatchMessage)
    assert decoded.seq == 1
    assert decoded.t0 == pytest.approx(1737901234.512)
    assert decoded.channels["vibration"] == pytest.approx([0.1, 0.2, 0.3])
    assert decoded.channels["temperature"] == pytest.approx([40.0, 40.1, 40.2])


def test_encode_batch_roundtrip_with_numpy_arrays():
    channels = {
        "vibration": np.array([0.1, 0.2, 0.3]),
        "temperature": np.array([40.0, 40.1, 40.2]),
    }
    encoded = encode_batch(2, 1737901235.0, channels)
    decoded = decode_message(encoded)

    assert isinstance(decoded, BatchMessage)
    assert decoded.seq == 2
    np.testing.assert_allclose(decoded.channels["vibration"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(decoded.channels["temperature"], [40.0, 40.1, 40.2])


def test_encode_error_roundtrip():
    encoded = encode_error("channel set mismatch")
    decoded = decode_message(encoded)

    assert decoded == {"type": "error", "message": "channel set mismatch"}


def test_encoded_messages_end_with_single_trailing_newline_and_are_single_line():
    for encoded in (
        encode_hello(["a", "b"], 1000),
        encode_batch(0, 0.0, {"a": [1, 2], "b": [3, 4]}),
        encode_error("oops"),
    ):
        assert encoded.endswith("\n")
        stripped = encoded[:-1]
        assert "\n" not in stripped
        # decode_message should work fine given the raw (newline-included) string
        decode_message(encoded)


def test_decode_message_strips_extra_whitespace():
    encoded = encode_hello(["a"], 1000)
    decoded = decode_message("  " + encoded.strip() + "  \n\n")
    assert isinstance(decoded, HelloMessage)


# -- Malformed-input cases -------------------------------------------------


def test_decode_invalid_json_raises():
    with pytest.raises(ValueError):
        decode_message("{not valid json")


def test_decode_missing_type_raises():
    with pytest.raises(ValueError):
        decode_message('{"channels": ["a"]}')


def test_decode_unknown_type_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "foo"}')


def test_decode_hello_missing_channels_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "hello", "sample_rate_hz": 1000}')


def test_decode_hello_missing_sample_rate_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "hello", "channels": ["a"]}')


def test_decode_hello_empty_channels_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "hello", "channels": [], "sample_rate_hz": 1000}')


@pytest.mark.parametrize("rate", [0, -5, -0.1])
def test_decode_hello_non_positive_sample_rate_raises(rate):
    with pytest.raises(ValueError):
        decode_message(f'{{"type": "hello", "channels": ["a"], "sample_rate_hz": {rate}}}')


def test_decode_batch_missing_seq_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "batch", "t0": 0.0, "channels": {"a": [1, 2]}}')


def test_decode_batch_missing_t0_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "batch", "seq": 0, "channels": {"a": [1, 2]}}')


def test_decode_batch_missing_channels_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "batch", "seq": 0, "t0": 0.0}')


def test_decode_batch_empty_channels_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "batch", "seq": 0, "t0": 0.0, "channels": {}}')


def test_decode_batch_mismatched_array_lengths_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "batch", "seq": 0, "t0": 0.0, "channels": {"a": [1,2,3], "b": [1,2]}}')


def test_decode_batch_non_numeric_value_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "batch", "seq": 0, "t0": 0.0, "channels": {"a": ["not a number", 2, 3]}}')


def test_decode_batch_null_value_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "batch", "seq": 0, "t0": 0.0, "channels": {"a": [1, null, 3]}}')


def test_decode_error_missing_message_raises():
    with pytest.raises(ValueError):
        decode_message('{"type": "error"}')
