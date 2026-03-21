import narwhals as nw
from pydantic import PositiveFloat, PositiveInt, validate_call

from takit.indicators.volatility import bollinger_bands
from takit.util.rolling import rolling_max
from takit.validation import PriceCol, config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def williams_vix_fix(
    ohlc: nw.DataFrame,
    lookback_high_length: PositiveInt = 22,
    lookback_range_high_length: PositiveInt = 50,
    bb_length: PositiveInt = 20,
    n_std_dev: PositiveFloat = 2.0,
    percentile_high: PositiveFloat = 0.85,
    source_col: PriceCol = "close",
) -> nw.DataFrame:
    """
    Williams VIX Fix (WVF).

    Reference:
        1. Active Trader Magazine 2007. Article by Larry Williams.

    Returns:
        WVF_BBM, WVF_BBL, WVF_BBU, WVF_RH, WVF, WVF_XBBU|RH columns.
        Note that Bollinger Band columns also include the Bollinger Band parameters (length and n_std_dev).
    """
    ohlc = validate_ohlc(ohlc, min_length=bb_length)

    # Williams VIX Fix
    source = ohlc.get_column(source_col)
    low = ohlc.get_column("low")
    lookback_high = rolling_max(source, lookback_high_length, min_samples=1)
    wvf = ((lookback_high - low) / lookback_high * 100).alias("WVF")

    # Range high for calculating signal
    range_high = (rolling_max(wvf, lookback_range_high_length, min_samples=1) * percentile_high).alias("WVF_RH")

    # Williams VIX Fix Bollinger Bands
    bb_result = nw.from_native(bollinger_bands(wvf, bb_length, n_std_dev), eager_only=True)

    # Rename BB columns with WVF_ prefix
    renamed_cols = {col: f"WVF_{col}" for col in bb_result.columns}
    bb_result = bb_result.rename(renamed_cols)

    upper_band_col_name = f"WVF_BBU{bb_length}_{n_std_dev}"
    upper_band = bb_result.get_column(upper_band_col_name)

    # Williams VIX Fix signal: WVF >= upper band OR WVF >= range high
    signal_values = [
        1 if (w is not None and u is not None and w >= u) or (w is not None and r is not None and w >= r) else 0
        for w, u, r in zip(wvf.to_list(), upper_band.to_list(), range_high.to_list(), strict=True)
    ]
    native_ns = nw.get_native_namespace(ohlc)
    signal = nw.new_series("WVF_XBBU|RH", signal_values, backend=native_ns)

    # Combine all columns
    data = {}
    for col in bb_result.columns:
        data[col] = bb_result.get_column(col).to_native()
    data["WVF_RH"] = range_high.to_native()
    data["WVF"] = wvf.to_native()
    data["WVF_XBBU|RH"] = signal.to_native()

    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


wvf = williams_vix_fix
