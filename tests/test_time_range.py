import numpy as np
import pandas as pd
import pytest

from src.time_range import range_bounds, select_range


def _make_sample_df(n=50):
    """int sample RangeIndex, as produced when the loader has no time column."""
    return pd.DataFrame(
        {"vibration": np.arange(n, dtype=float), "current": np.arange(n, dtype=float) * 2},
        index=pd.RangeIndex(n, name="sample"),
    )


def _make_float_time_df(n=50, dt=0.1):
    """float seconds-since-start time index."""
    t = np.arange(n) * dt
    return pd.DataFrame(
        {"vibration": np.arange(n, dtype=float), "current": np.arange(n, dtype=float) * 2},
        index=pd.Index(t, name="time"),
    )


def _make_datetime_df(n=50, freq="s"):
    """pandas-datetime time index."""
    idx = pd.date_range("2024-01-01 00:00:00", periods=n, freq=freq, name="time")
    return pd.DataFrame(
        {"vibration": np.arange(n, dtype=float), "current": np.arange(n, dtype=float) * 2},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Both None => full copy, mutation-independent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "make_df", [_make_sample_df, _make_float_time_df, _make_datetime_df]
)
def test_select_range_none_none_returns_full_copy(make_df):
    df = make_df()
    result = select_range(df, None, None)

    assert len(result) == len(df)
    pd.testing.assert_frame_equal(result, df)

    # mutating result must not mutate original
    result.iloc[0, 0] = 12345.0
    assert df.iloc[0, 0] != 12345.0


# ---------------------------------------------------------------------------
# int sample index
# ---------------------------------------------------------------------------

def test_select_range_sample_index_start_only():
    df = _make_sample_df(50)
    result = select_range(df, start=10, end=None)
    assert len(result) == 40
    assert result.index.min() == 10
    assert result.index.max() == 49


def test_select_range_sample_index_end_only():
    df = _make_sample_df(50)
    result = select_range(df, start=None, end=10)
    assert len(result) == 11
    assert result.index.min() == 0
    assert result.index.max() == 10


def test_select_range_sample_index_start_and_end():
    df = _make_sample_df(50)
    result = select_range(df, start=10, end=20)
    assert len(result) == 11  # inclusive
    assert result.index.min() == 10
    assert result.index.max() == 20


def test_select_range_sample_index_start_gt_end_raises():
    df = _make_sample_df(50)
    with pytest.raises(ValueError):
        select_range(df, start=20, end=10)


def test_select_range_sample_index_out_of_bounds_raises():
    df = _make_sample_df(50)
    with pytest.raises(ValueError):
        select_range(df, start=100, end=200)


# ---------------------------------------------------------------------------
# float seconds time index
# ---------------------------------------------------------------------------

def test_select_range_float_time_index_start_only():
    df = _make_float_time_df(50, dt=0.1)
    result = select_range(df, start=1.0, end=None)
    assert len(result) == 40
    assert result.index.min() == pytest.approx(1.0)
    assert result.index.max() == pytest.approx(4.9)


def test_select_range_float_time_index_end_only():
    df = _make_float_time_df(50, dt=0.1)
    result = select_range(df, start=None, end=1.0)
    assert len(result) == 11
    assert result.index.min() == pytest.approx(0.0)
    assert result.index.max() == pytest.approx(1.0)


def test_select_range_float_time_index_start_and_end():
    df = _make_float_time_df(50, dt=0.1)
    result = select_range(df, start=1.0, end=2.0)
    assert len(result) == 11  # inclusive
    assert result.index.min() == pytest.approx(1.0)
    assert result.index.max() == pytest.approx(2.0)


def test_select_range_float_time_index_start_gt_end_raises():
    df = _make_float_time_df(50, dt=0.1)
    with pytest.raises(ValueError):
        select_range(df, start=3.0, end=1.0)


def test_select_range_float_time_index_out_of_bounds_raises():
    df = _make_float_time_df(50, dt=0.1)  # index spans [0, 4.9]
    with pytest.raises(ValueError):
        select_range(df, start=100.0, end=200.0)


# ---------------------------------------------------------------------------
# datetime time index
# ---------------------------------------------------------------------------

def test_select_range_datetime_index_start_only():
    df = _make_datetime_df(50)
    result = select_range(df, start=pd.Timestamp("2024-01-01 00:00:10"), end=None)
    assert len(result) == 40
    assert result.index.min() == pd.Timestamp("2024-01-01 00:00:10")
    assert result.index.max() == pd.Timestamp("2024-01-01 00:00:49")


def test_select_range_datetime_index_end_only():
    df = _make_datetime_df(50)
    result = select_range(df, start=None, end=pd.Timestamp("2024-01-01 00:00:10"))
    assert len(result) == 11
    assert result.index.min() == pd.Timestamp("2024-01-01 00:00:00")
    assert result.index.max() == pd.Timestamp("2024-01-01 00:00:10")


def test_select_range_datetime_index_start_and_end():
    df = _make_datetime_df(50)
    result = select_range(
        df,
        start=pd.Timestamp("2024-01-01 00:00:10"),
        end=pd.Timestamp("2024-01-01 00:00:20"),
    )
    assert len(result) == 11  # inclusive
    assert result.index.min() == pd.Timestamp("2024-01-01 00:00:10")
    assert result.index.max() == pd.Timestamp("2024-01-01 00:00:20")


def test_select_range_datetime_index_start_gt_end_raises():
    df = _make_datetime_df(50)
    with pytest.raises(ValueError):
        select_range(
            df,
            start=pd.Timestamp("2024-01-01 00:00:20"),
            end=pd.Timestamp("2024-01-01 00:00:10"),
        )


def test_select_range_datetime_index_out_of_bounds_raises():
    df = _make_datetime_df(50)  # spans 2024-01-01 00:00:00 .. 00:00:49
    with pytest.raises(ValueError):
        select_range(
            df,
            start=pd.Timestamp("2024-01-02 00:00:00"),
            end=pd.Timestamp("2024-01-03 00:00:00"),
        )


def test_select_range_datetime_index_accepts_iso_strings():
    """GUI text fields supply plain strings, not Timestamp objects."""
    df = _make_datetime_df(50)
    result = select_range(df, start="2024-01-01 00:00:10", end="2024-01-01 00:00:20")
    assert len(result) == 11
    assert result.index.min() == pd.Timestamp("2024-01-01 00:00:10")
    assert result.index.max() == pd.Timestamp("2024-01-01 00:00:20")


def test_select_range_datetime_index_single_sided_iso_string():
    df = _make_datetime_df(50)
    result = select_range(df, start="2024-01-01 00:00:40", end=None)
    assert len(result) == 10
    assert result.index.min() == pd.Timestamp("2024-01-01 00:00:40")


def test_select_range_datetime_index_iso_date_only_string_out_of_bounds_raises():
    df = _make_datetime_df(50)
    with pytest.raises(ValueError):
        select_range(df, start="2025-01-01", end="2025-01-02")


# ---------------------------------------------------------------------------
# range_bounds
# ---------------------------------------------------------------------------

def test_range_bounds_sample_index():
    df = _make_sample_df(50)
    lo, hi = range_bounds(df)
    assert lo == 0
    assert hi == 49


def test_range_bounds_float_time_index():
    df = _make_float_time_df(50, dt=0.1)
    lo, hi = range_bounds(df)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(4.9)


def test_range_bounds_datetime_index():
    df = _make_datetime_df(50)
    lo, hi = range_bounds(df)
    assert lo == pd.Timestamp("2024-01-01 00:00:00")
    assert hi == pd.Timestamp("2024-01-01 00:00:49")


def test_range_bounds_empty_raises():
    df = _make_sample_df(0)
    with pytest.raises(ValueError):
        range_bounds(df)
