from __future__ import annotations

import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict, validate_series


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def simple_moving_average(
    series: nw.Series, length: PositiveInt, *, validate: bool = True, min_periods: int | None = None
) -> nw.Series:
    """
    Simple Moving Average (SMA).

    Args:
        series: Input series
        length: Length of the window
        validate: Validate input series
        min_periods: Minimum number of periods to calculate the moving average

    Returns:
        Moving average values of the input series
    """
    if validate:
        min_length = min(length, min_periods) if min_periods else length
        series = validate_series(series, min_length=min_length)

    return series.rolling_mean(window_size=length, min_samples=min_periods).alias(f"SMA{length}")


sma = simple_moving_average


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def exponential_moving_average(
    series: nw.Series, length: PositiveInt, *, validate: bool = True, min_periods: int | None = None
) -> nw.Series:
    """
    Exponential Moving Average (EMA).

    Args:
        series: Input series
        length: Length of the window
        validate: Validate input series
        min_periods: Minimum number of periods to calculate the moving average

    Returns:
        Moving average values of the input series
    """
    if validate:
        min_length = min(length, min_periods) if min_periods else length
        series = validate_series(series, min_length=min_length)

    min_periods = length if min_periods is None else min_periods
    return series.ewm_mean(span=length, adjust=False, min_samples=min_periods)


ema = exponential_moving_average


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def relative_moving_average(series: nw.Series, length: PositiveInt, *, validate: bool = True) -> nw.Series:
    """
    Relative Moving Average (RMA).

    Equations:

        1. RMA_i = (RMA_(i-1) * (length - 1) + series_i) / length

    Args:
        series: Input series
        length: Length of the window
        validate: Validate input series

    Returns:
        Relative moving average values of the input series
    """
    if validate:
        series = validate_series(series, min_length=length)

    # RMA is equivalent to EWM with alpha=1/length, adjust=False
    return series.ewm_mean(alpha=1.0 / length, adjust=False, min_samples=1)


rma = relative_moving_average
