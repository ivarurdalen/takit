from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.validation import config_dict, ta_series_schema


@validate_call(config=config_dict)
def trend_rider(
    series: pd.Series,
    fast_length: PositiveInt = 20,
    slow_length: PositiveInt = 50,
    mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma",
) -> pd.DataFrame:
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
    series = ta_series_schema(min_length=slow_length).validate(series)

    df = series.to_frame()

    fast_col = f"{mode.upper()}{fast_length}"
    slow_col = f"{mode.upper()}{slow_length}"

    df[fast_col] = getattr(ma, mode)(series, fast_length)
    df[slow_col] = getattr(ma, mode)(series, slow_length)

    df["TR"] = np.where(
        (df[fast_col] < series) & (df[slow_col] < series),
        1,
        np.where((df[fast_col] > series) & (df[slow_col] > series), -1, 0),
    )

    return df.iloc[:, -3:]
