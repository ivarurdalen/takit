import pandas as pd
from pydantic import PositiveInt, validate_call

from takit.indicators.trend import simple_moving_average
from takit.validation import config_dict, ohlc_schema


@validate_call(config=config_dict)
def average_true_range(ohlc: pd.DataFrame, length: PositiveInt = 14) -> pd.Series:
    """
    Average True Range (ATR).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000501823-average-true-range-atr/

    Equations:
        1. TR_i = True Range = max(high_i - low_i, abs(high_i - close_(i-1)), abs(low_i - close_(i-1)))
        2. ATR = SMA(TR, length)

    Args:
        ohlc: DataFrame with OHLC data
        length: Length of the window

    Returns:
        ATR values of the input series
    """
    ohlc = ohlc_schema(min_length=length).validate(ohlc)

    ohlc["high_low"] = ohlc["high"] - ohlc["low"]
    ohlc["high_prev_close"] = (ohlc["high"] - ohlc["close"].shift(1)).abs()
    ohlc["low_prev_close"] = (ohlc["low"] - ohlc["close"].shift(1)).abs()

    ohlc["true_range"] = ohlc[["high_low", "high_prev_close", "low_prev_close"]].max(axis=1)

    return simple_moving_average(ohlc["true_range"], length).rename(f"ATR{length}")


atr = average_true_range
