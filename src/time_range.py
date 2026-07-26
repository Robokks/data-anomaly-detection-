"""Restrict a loaded DataFrame to a "full data" or explicit [start, end] range.

Works uniformly across the three index kinds produced by ``src.data_loader``
/ ``src.tdms_loader``: a plain integer ``sample`` RangeIndex, a float
seconds-since-start ``time`` index, or a pandas ``datetime64`` ``time``
index. Since the loader always hands back a sorted index, label-based
``.loc[start:end]`` slicing already does the right thing for all three -- the
work here is coercing/validating the bounds and refusing to silently return
an empty slice.
"""
from __future__ import annotations

import pandas as pd


def _coerce_bound(df: pd.DataFrame, value):
    """Coerce a start/end bound to something comparable to df.index's dtype."""
    if value is None:
        return None
    if isinstance(df.index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(df.index):
        return pd.to_datetime(value)
    return value


def select_range(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Slice df to [start, end] (inclusive) along its index.

    start/end are None (open-ended: full data on that side) or a value
    comparable to df.index's dtype (int sample number, float seconds, or a
    datetime/str-parseable-as-datetime when df.index is datetime64).

    Both None => returns an equivalent full copy of df.

    Raises ValueError if start > end, or if the requested range does not
    overlap df.index at all (the resulting slice would be empty).
    """
    start_c = _coerce_bound(df, start)
    end_c = _coerce_bound(df, end)

    if start_c is not None and end_c is not None and start_c > end_c:
        raise ValueError(f"start ({start!r}) must be <= end ({end!r})")

    if start_c is None and end_c is None:
        return df.copy()

    result = df.loc[start_c:end_c]

    if result.empty:
        raise ValueError(
            f"Requested range [{start!r}, {end!r}] does not overlap the "
            f"DataFrame's index range [{df.index.min()!r}, {df.index.max()!r}]"
        )

    return result.copy()


def range_bounds(df: pd.DataFrame) -> tuple:
    """Return (df.index.min(), df.index.max()) for GUI defaults / a reset button.

    Raises ValueError on an empty DataFrame.
    """
    if df.empty:
        raise ValueError("Cannot compute range bounds of an empty DataFrame")
    return (df.index.min(), df.index.max())
