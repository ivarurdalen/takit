import numpy as np
import pandas as pd
import pandera.pandas as pa
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict, ta_series_schema


@pa.check_types
def simple_moving_average(
    series: pd.Series, length: PositiveInt, *, validate: bool = True, min_periods: int | None = None
) -> pd.Series:
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
        series = ta_series_schema(min_length=min_length).validate(series)

    return series.rolling(window=length, min_periods=min_periods).mean().rename(f"SMA{length}")


sma = simple_moving_average


@validate_call(config=config_dict)
def exponential_moving_average(
    series: pd.Series, length: PositiveInt, *, validate: bool = True, min_periods: int | None = None
) -> pd.Series:
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
        series = ta_series_schema(min_length=min_length).validate(series)

    min_periods = length if min_periods is None else min_periods
    return series.ewm(span=length, adjust=False, min_periods=min_periods).mean()


ema = exponential_moving_average


@validate_call(config=config_dict)
def relative_moving_average(series: pd.Series, length: PositiveInt, *, validate: bool = True) -> pd.Series:
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
        series = ta_series_schema(min_length=length).validate(series)

    data = series.to_numpy()
    rma = np.empty_like(data)
    rma[0] = data[0]  # Initialize with the first value
    for i in range(1, len(data)):
        rma[i] = (rma[i - 1] * (length - 1) + data[i]) / length
    return pd.Series(rma, index=series.index)


rma = relative_moving_average
