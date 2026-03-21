from __future__ import annotations

import narwhals as nw


def rolling_max(series: nw.Series, window_size: int, *, min_samples: int | None = None) -> nw.Series:
    """Compute rolling max using native backend (not yet available in narwhals)."""
    native = nw.to_native(series)
    ms = min_samples if min_samples is not None else window_size
    if hasattr(native, "rolling_max"):
        # Polars Series
        result = native.rolling_max(window_size=window_size, min_samples=ms)
    else:
        # Pandas Series
        result = native.rolling(window=window_size, min_periods=ms).max()
    return nw.from_native(result, series_only=True)


def rolling_min(series: nw.Series, window_size: int, *, min_samples: int | None = None) -> nw.Series:
    """Compute rolling min using native backend (not yet available in narwhals)."""
    native = nw.to_native(series)
    ms = min_samples if min_samples is not None else window_size
    if hasattr(native, "rolling_min"):
        # Polars Series
        result = native.rolling_min(window_size=window_size, min_samples=ms)
    else:
        # Pandas Series
        result = native.rolling(window=window_size, min_periods=ms).min()
    return nw.from_native(result, series_only=True)
