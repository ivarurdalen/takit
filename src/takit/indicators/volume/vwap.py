import narwhals as nw
from pydantic import validate_call

from takit.validation import config_dict


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def volume_weighted_average_price(ohlc: nw.DataFrame) -> nw.Series:
    """
    Volume Weighted Average Price (VWAP).

    Cumulative VWAP over the entire series. For session-based VWAP,
    pre-filter the DataFrame to the desired session.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502018-volume-weighted-average-price-vwap/

    Equations:
        1. Typical Price = (high + low + close) / 3
        2. VWAP = cumsum(Typical Price * Volume) / cumsum(Volume)

    Args:
        ohlc: DataFrame with OHLC and volume data

    Returns:
        VWAP values
    """
    missing = {"high", "low", "close", "volume"} - set(ohlc.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    result = ohlc.select(
        ((nw.col("high") + nw.col("low") + nw.col("close")) / 3 * nw.col("volume")).cum_sum().alias("_tp_vol"),
        nw.col("volume").cum_sum().alias("_cum_vol"),
    )

    vwap_series = (result.get_column("_tp_vol") / result.get_column("_cum_vol")).alias("VWAP")
    return vwap_series


vwap = volume_weighted_average_price
