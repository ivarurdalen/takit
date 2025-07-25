from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.validation import config_dict, ta_series_schema


class MA(BaseModel):
    length: PositiveInt
    mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma"
    multiplier: float = 1

    def get_col_name(self) -> str:
        """Get the column name for the moving average."""
        col = f"{self.mode.upper()}{self.length}"
        if self.multiplier != 1.0:
            col += f"x{self.multiplier}"
        return col


@validate_call(config=config_dict)
def ma_cross(
    series: pd.Series, fast_ma: MA, slow_ma: MA, signal_name: str | None = None, *, only_crosses: bool = False
) -> pd.DataFrame:
    """
    MA Cross.

    Reference:
        https://tradeciety.com/trend-rider-indicator

    Logic:
       If Value > Fast MA and Value > Slow MA -> Uptrend
       Else If Value < Fast MA and Value < Slow MA -> Downtrend
       Else -> Sideways / Undetermined

    Args:
        series: Input series
        fast_ma: Fast moving average
        slow_ma: Slow moving average
        signal_name: Name of the signal
        only_crosses: If True, only set 1 when fast_ma crosses slow_ma, otherwise set 1 when fast_ma > slow_ma

    Returns:
        Trend values of the input series
    """
    series = ta_series_schema(min_length=slow_ma.length).validate(series)

    df = series.to_frame()

    fast_col = fast_ma.get_col_name()
    slow_col = slow_ma.get_col_name()

    df[fast_col] = getattr(ma, fast_ma.mode)(series, fast_ma.length) * fast_ma.multiplier
    df[slow_col] = getattr(ma, slow_ma.mode)(series, slow_ma.length) * slow_ma.multiplier

    if signal_name is None:
        signal_name = f"{fast_col}X{slow_col}"

    df[signal_name] = np.where(df[fast_col] > df[slow_col], 1, -1)

    if only_crosses:
        df[signal_name] = df[signal_name].diff().replace({2: 1, -2: -1})

    return df.iloc[:, -3:]


BMSB_FAST = MA(length=21 * 7, mode="ema")  # 21 weeks
BMSB_SLOW = MA(length=20 * 7, mode="sma")  # 20 weeks


def bull_market_support_band(series: pd.Series, *, only_crosses: bool = False) -> pd.DataFrame:
    """
    Bull Market Support Band.

    Reference:
        1. Benjamin Cowen - https://intothecryptoverse.com/
    """
    return ma_cross(series, BMSB_FAST, BMSB_SLOW, signal_name="BMSB", only_crosses=only_crosses)


bmsb = bull_market_support_band

LARSSON_FAST = MA(length=35, mode="ema")
LARSSON_SLOW = MA(length=50, mode="ema")


def larsson_line(series: pd.Series, *, only_crosses: bool = False) -> pd.DataFrame:
    """
    Larsson Line.

    Reference:
        1. CTO Larsen - https://www.ctolarsson.com/
    """
    return ma_cross(series, LARSSON_FAST, LARSSON_SLOW, signal_name="LL", only_crosses=only_crosses)
