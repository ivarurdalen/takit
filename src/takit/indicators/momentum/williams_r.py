from __future__ import annotations

import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.util.rolling import rolling_max, rolling_min
from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def williams_r(ohlc: nw.DataFrame, lookback_length: PositiveInt = 14) -> nw.Series:
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
    ohlc = validate_ohlc(ohlc, min_length=lookback_length)

    lookback_high = rolling_max(ohlc.get_column("high"), lookback_length, min_samples=1)
    lookback_low = rolling_min(ohlc.get_column("low"), lookback_length, min_samples=1)
    close = ohlc.get_column("close")

    return (-100 * (lookback_high - close) / (lookback_high - lookback_low)).alias(f"WR{lookback_length}")


wr = williams_r
