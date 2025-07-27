import pandas as pd
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict, ohlc_schema


@validate_call(config=config_dict)
def donchian_channel(ohlc: pd.DataFrame, length: PositiveInt = 20) -> pd.DataFrame:
    """
    Donchian Channel.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/

    Equations:
        1. DCH = Lookback High = max_over_range(high, length)
        2. DCL = Lookback Low = min_over_range(low, length)
        3. DCM = Lookback Mid = (DCH + DCL) / 2

    Args:
        ohlc: DataFrame with OHLC data
        length: Length of the lookback period

    Returns:
        DataFrame with DCM, DCL, DCH columns
    """
    ohlc = ohlc_schema(min_length=length).validate(ohlc)

    dc_high = ohlc["high"].rolling(window=length, min_periods=1).max().rename(f"DCH{length}")
    dc_low = ohlc["low"].rolling(window=length, min_periods=1).min().rename(f"DCL{length}")
    dc_mid = ((dc_high + dc_low) / 2).rename(f"DCM{length}")
    return pd.concat([dc_mid, dc_low, dc_high], axis=1)


dc = donchian_channel
