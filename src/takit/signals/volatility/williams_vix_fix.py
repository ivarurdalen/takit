import pandas as pd
from pydantic import PositiveFloat, PositiveInt, validate_call

from takit.indicators.volatility import bollinger_bands
from takit.validation import PriceCol, config_dict, ohlc_schema


@validate_call(config=config_dict)
def williams_vix_fix(
    ohlc: pd.DataFrame,
    lookback_high_length: PositiveInt = 22,
    lookback_range_high_length: PositiveInt = 50,
    bb_length: PositiveInt = 20,
    n_std_dev: PositiveFloat = 2.0,
    percentile_high: PositiveFloat = 0.85,
    source_col: PriceCol = "close",
) -> pd.DataFrame:
    """
    Williams VIX Fix (WVF).

    Reference:
        1. Active Trader Magazine 2007. Article by Larry Williams.

    Returns:
        WVF_BBM, WVF_BBL, WVF_BBU, WVF_RH, WVF, WVF_XBBU|RH columns.
        Note that Bollinger Band columns also include the Bollinger Band parameters (length and n_std_dev).
    """
    ohlc = ohlc_schema(min_length=bb_length).validate(ohlc)

    # Williams VIX Fix
    lookback_high = ohlc[source_col].rolling(window=lookback_high_length, min_periods=1).max()
    williams_vix_fix = (((lookback_high - ohlc["low"]) / lookback_high) * 100).rename("WVF")

    # Range high for calculating signal
    range_high = (
        williams_vix_fix.rolling(window=lookback_range_high_length, min_periods=1).max() * percentile_high
    ).rename("WVF_RH")

    # Williams VIX Fix Bollinger Bands for calculating upper band
    williams_vix_fix_bb = bollinger_bands(williams_vix_fix, bb_length, n_std_dev).rename(lambda x: f"WVF_{x}", axis=1)
    upper_band_col_name = f"WVF_BBU{bb_length}_{n_std_dev}"

    # Williams VIX Fix signal
    williams_vix_fix_signal = (
        ((williams_vix_fix >= williams_vix_fix_bb[upper_band_col_name]) | (williams_vix_fix >= range_high))
        .astype(int)
        .rename("WVF_XBBU|RH")
    )

    return pd.concat([williams_vix_fix_bb, range_high, williams_vix_fix, williams_vix_fix_signal], axis=1)


wvf = williams_vix_fix
