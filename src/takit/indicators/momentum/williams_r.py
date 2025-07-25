import pandas as pd
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict, ohlc_schema


@validate_call(config=config_dict)
def williams_r(ohlc: pd.DataFrame, lookback_length: PositiveInt = 14) -> pd.Series:
    """
    Williams %R.

    Momentum-based oscillator to identify overbought and oversold conditions.
    -100 is oversold, -50 is neutral and 0 is overbought.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000501985-williams-r-r/

    Equations:
        1. LH = Lookback High = max_over_range(high, lookback_length)
        2. LL = Lookback Low = min_over_range(low, lookback_length)
        3. WR = -100 * (LH - close) / (LH - LL)

    Args:
        ohlc: DataFrame with OHLC data
        lookback_length: Length of the window

    Returns:
        Williams %R values
    """
    ohlc = ohlc_schema(min_length=lookback_length).validate(ohlc)

    lookback_high = ohlc["high"].rolling(window=lookback_length, min_periods=1).max()
    lookback_low = ohlc["low"].rolling(window=lookback_length, min_periods=1).min()
    return (-100 * (lookback_high - ohlc["close"]) / (lookback_high - lookback_low)).rename(f"WR{lookback_length}")


wr = williams_r
