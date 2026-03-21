from __future__ import annotations

import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.indicators.trend.ma import exponential_moving_average
from takit.validation import config_dict, validate_series


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def moving_average_convergence_divergence(
    series: nw.Series,
    fast_length: PositiveInt = 12,
    slow_length: PositiveInt = 26,
    signal_length: PositiveInt = 9,
) -> nw.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502344-macd-moving-average-convergence-divergence/

    Equations:
        1. MACD Line = EMA(series, fast_length) - EMA(series, slow_length)
        2. Signal Line = EMA(MACD Line, signal_length)
        3. Histogram = MACD Line - Signal Line

    Args:
        series: Input series
        fast_length: Fast EMA period
        slow_length: Slow EMA period
        signal_length: Signal line EMA period

    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    series = validate_series(series, min_length=slow_length)

    fast_ema = nw.from_native(exponential_moving_average(series, fast_length), series_only=True)
    slow_ema = nw.from_native(exponential_moving_average(series, slow_length), series_only=True)

    macd_line = (fast_ema - slow_ema).alias("MACD")
    signal_line = nw.from_native(
        exponential_moving_average(macd_line, signal_length), series_only=True
    ).alias("MACD_SIGNAL")
    histogram = (macd_line - signal_line).alias("MACD_HIST")

    native_ns = nw.get_native_namespace(series)
    data = {s.name: s.to_native() for s in [macd_line, signal_line, histogram]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


macd = moving_average_convergence_divergence
