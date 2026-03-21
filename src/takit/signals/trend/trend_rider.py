from __future__ import annotations

from typing import Annotated

import narwhals as nw
from pydantic import PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.validation import config_dict, validate_series


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def trend_rider(
    series: nw.Series,
    fast_length: PositiveInt = 20,
    slow_length: PositiveInt = 50,
    mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma",
) -> nw.DataFrame:
    """
    Trend Rider.

    Reference:
        https://tradeciety.com/trend-rider-indicator

    Logic:
       If Value > Fast MA and Value > Slow MA -> Uptrend
       Else If Value < Fast MA and Value < Slow MA -> Downtrend
       Else -> Sideways / Undetermined

    Args:
        series: Input series
        fast_length: Length of the fast moving average window
        slow_length: Length of the slow moving average window
        mode: Mode of the moving average (sma or ema)

    Returns:
        Trend values of the input series
    """
    series = validate_series(series, min_length=slow_length)

    fast_col = f"{mode.upper()}{fast_length}"
    slow_col = f"{mode.upper()}{slow_length}"

    fast_ma = nw.from_native(getattr(ma, mode)(series, fast_length), series_only=True).alias(fast_col)
    slow_ma = nw.from_native(getattr(ma, mode)(series, slow_length), series_only=True).alias(slow_col)

    # Build DataFrame from series
    native_ns = nw.get_native_namespace(series)
    df = nw.from_native(
        native_ns.DataFrame({
            fast_col: fast_ma.to_native(),
            slow_col: slow_ma.to_native(),
            "_series": series.to_native(),
        }),
        eager_only=True,
    )

    uptrend = (nw.col(fast_col) < nw.col("_series")) & (nw.col(slow_col) < nw.col("_series"))
    downtrend = (nw.col(fast_col) > nw.col("_series")) & (nw.col(slow_col) > nw.col("_series"))
    df = df.with_columns(
        nw.when(uptrend)
        .then(1)
        .otherwise(nw.when(downtrend).then(-1).otherwise(0))
        .alias("TR")
    )

    return df.select(fast_col, slow_col, "TR")
